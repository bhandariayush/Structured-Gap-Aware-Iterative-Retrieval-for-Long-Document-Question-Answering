import json
import time
import re
import string
import math
from collections import Counter, defaultdict

import requests
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util


INPUT_FILE = r"C:\Users\ayush\Downloads\narrativeqa-master\narrativeqa-master\processed\narrativeqa_test_qasper_like.jsonl"
OUTPUT_JSON = "structured_gap_v7_narrativeqa_results_14b.json"
OUTPUT_CSV = "structured_gap_v7_narrativeqa_results_14b.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:14b"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SAMPLES = 50
MAX_ITERATIONS = 2
FIRST_K = 6
SECOND_K = 10
SECOND_RERANK_TOP_K = 3
MAX_MEMORY_ITEMS = 12
MAX_MEMORY_CHARS = 6500
MAX_FINAL_CONTEXT_CHARS = 8500
EXTRACT_CONTEXT_CHARS = 2500

STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","by","is","are","was","were",
    "be","been","being","this","that","these","those","what","which","who","whom","when",
    "where","why","how","do","does","did","it","its","as","at","from","into","about","their",
    "them","they","we","our","can","could","should","would","may","might","than","then","there",
    "such","using","use","used","also","show","shows","shown"
}

JUDGE_PROMPT = """You are a structured evidence judge for story question answering.

Use only the provided evidence memory. Do not use outside knowledge.

Return valid JSON with exactly these keys:
- can_answer
- answer_type
- provisional_answer
- missing_items
- gap_items
- next_query

Field rules:
- can_answer: true or false
- answer_type: one of ["character","event","location","relationship","cause","result","count","timeline","other"]
- provisional_answer: short draft answer from the evidence memory, or "Not enough information"
- missing_items: list of 0 to 3 short strings describing what is still missing
- gap_items: list of objects, each with keys:
  - category: one of ["character","attribute","relation","event","location","timeline","evidence_span","count","other"]
  - target: short target entity or concept
  - slot: short attribute/relation/thing needed
  - description: short explanation of the missing evidence
- next_query: short targeted retrieval query; empty string if can_answer=true

Rules:
- If the evidence memory is sufficient, set can_answer=true, missing_items=[], gap_items=[], next_query="".
- If not sufficient, set can_answer=false and provide targeted gap items.
- Keep gap items precise and retrieval-oriented.
- Prefer concrete story details: who, what happened, where, when, why, relationship, and outcome.
- No extra text outside JSON.

Question:
{question}

Evidence memory:
{memory}
"""

EXTRACT_PROMPT = """You are selecting compact high-value evidence for story question answering.

Use only the retrieved context.

Return valid JSON with exactly these keys:
- selected_evidence

Rules:
- selected_evidence must be a list of 1 to 3 verbatim snippets copied from the retrieved context.
- Prefer snippets that directly answer the question or fill the named gap items.
- Prefer compact snippets, but keep enough detail to preserve character/event meaning.
- If there are multiple similar snippets, keep only the most directly useful ones.
- No extra text outside JSON.

Question:
{question}

Gap items:
{gap_items}

Retrieved context:
{context}
"""

FINAL_PROMPT = """/no_think
You are answering a question about a story.

Use only the provided evidence memory.

Return valid JSON with exactly these keys:
- answer
- evidence

Rules:
- answer: brief and direct.
- evidence: 1 to 3 short supporting snippets copied from the evidence memory.
- If the answer is not supported, set answer to "Not enough information" and evidence to [].
- Do not add any extra text outside JSON.

Question:
{question}

Evidence memory:
{memory}
"""

def normalize_answer(s):
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s

def tokenize(text):
    text = normalize_answer(text)
    return [t for t in text.split() if t and t not in STOPWORDS]

def get_tokens(s):
    return normalize_answer(s).split() if s else []

def f1_text(prediction, ground_truth):
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(ground_truth)

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def max_f1_over_answers(prediction, gold_answers):
    if not gold_answers:
        return 0.0
    return max(f1_text(prediction, ans) for ans in gold_answers if str(ans).strip())

def evidence_f1(pred_evidence, gold_evidence):
    pred = set(normalize_answer(x) for x in pred_evidence if str(x).strip())
    gold = set(normalize_answer(x) for x in gold_evidence if str(x).strip())

    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    overlap = len(pred & gold)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)

def safe_parse_json(text, fallback=None):
    text = text.strip()
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return fallback if fallback is not None else {}

def split_context(context):
    context = context.replace("\r\n", "\n")
    blocks = re.split(r"\n\s*\n", context)
    chunks = []

    for block in blocks:
        block = block.strip()
        if len(block) < 40:
            continue

        if len(block) <= 1200:
            chunks.append(block)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", block)
            buf = []
            cur_len = 0
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if cur_len + len(s) > 900 and buf:
                    chunks.append(" ".join(buf).strip())
                    buf = [s]
                    cur_len = len(s)
                else:
                    buf.append(s)
                    cur_len += len(s) + 1
            if buf:
                chunks.append(" ".join(buf).strip())

    return chunks

def bm25_scores(question, chunks):
    q_terms = tokenize(question)
    if not q_terms or not chunks:
        return [0.0] * len(chunks)

    chunk_terms = [tokenize(c) for c in chunks]
    N = len(chunks)
    avgdl = sum(len(ct) for ct in chunk_terms) / max(N, 1)

    df = defaultdict(int)
    for terms in chunk_terms:
        for term in set(terms):
            df[term] += 1

    scores = []
    k1 = 1.5
    b = 0.75

    for terms in chunk_terms:
        tf = Counter(terms)
        dl = len(terms)
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
            freq = tf[term]
            denom = freq + k1 * (1 - b + b * dl / max(avgdl, 1))
            score += idf * ((freq * (k1 + 1)) / denom)
        scores.append(score)

    return scores

def reciprocal_rank_fusion(rank_lists, k=60):
    fused = defaultdict(float)
    for ranked_indices in rank_lists:
        for rank, idx in enumerate(ranked_indices, start=1):
            fused[idx] += 1.0 / (k + rank)
    return fused

def chunk_bonus(chunk, query):
    q = query.lower()
    c = chunk.lower()
    extra = 0.0

    if any(x in q for x in ["who", "name", "character"]):
        if any(x in c for x in ["he", "she", "his", "her", "character", "student", "man", "woman"]):
            extra += 0.02

    if any(x in q for x in ["where", "place", "location"]):
        if any(x in c for x in ["in ", "at ", "from ", "city", "town", "house", "school"]):
            extra += 0.02

    if any(x in q for x in ["why", "because", "reason"]):
        if any(x in c for x in ["because", "so that", "in order to", "reason"]):
            extra += 0.03

    return extra

def hybrid_retrieve(query, chunks, chunk_embeddings, model, final_k=6, dense_top_n=12, bm25_top_n=12, exclude_idxs=None):
    if not chunks:
        return [], []

    exclude_idxs = set(exclude_idxs or [])

    query_embedding = model.encode(
        [query],
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=DEVICE
    )

    dense_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    dense_ranked = torch.argsort(dense_scores, descending=True).tolist()
    sparse_scores = bm25_scores(query, chunks)
    bm25_ranked = sorted(range(len(chunks)), key=lambda i: sparse_scores[i], reverse=True)

    dense_ranked = [i for i in dense_ranked if i not in exclude_idxs][:min(dense_top_n, len(chunks))]
    bm25_ranked = [i for i in bm25_ranked if i not in exclude_idxs][:min(bm25_top_n, len(chunks))]

    fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=60)
    fused_ranked = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)

    reranked = []
    for idx in fused_ranked:
        score = fused[idx] + chunk_bonus(chunks[idx], query)
        reranked.append((chunks[idx], score, idx))

    reranked.sort(key=lambda x: x[1], reverse=True)
    selected = reranked[:min(final_k, len(reranked))]
    return [x[0] for x in selected], [x[2] for x in selected]

def cross_encoder_rerank(query, chunks, reranker, top_k=3):
    if not chunks:
        return []

    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]

def call_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "think": False
    }
    start = time.perf_counter()
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    result = r.json()
    end = time.perf_counter()

    latency_sec = end - start
    if result.get("total_duration") is not None:
        latency_sec = result["total_duration"] / 1_000_000_000

    return result.get("response", "").strip(), latency_sec

def dedupe_keep_order(items):
    seen = set()
    out = []
    for x in items:
        key = normalize_answer(x)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out

def soft_compact_memory(items, max_items=MAX_MEMORY_ITEMS, max_chars=MAX_MEMORY_CHARS):
    items = dedupe_keep_order(items)

    selected = []
    total = 0

    for x in items:
        if len(selected) >= max_items:
            break
        if total + len(x) + 2 > max_chars:
            continue
        selected.append(x)
        total += len(x) + 2

    return selected

def build_gap_query(question, gap_items, next_query=""):
    phrases = []
    for g in gap_items[:2]:
        if not isinstance(g, dict):
            continue
        target = str(g.get("target", "")).strip()
        slot = str(g.get("slot", "")).strip()
        desc = str(g.get("description", "")).strip()

        if target and slot:
            phrases.append(f"{target} {slot}")
        elif target:
            phrases.append(target)
        elif slot:
            phrases.append(slot)
        elif desc:
            phrases.append(desc)

    if phrases:
        query = " ; ".join(phrases)
    elif next_query.strip():
        query = next_query.strip()
    else:
        query = question

    query = re.sub(r"\s+", " ", query).strip()
    return query[:180]

def load_jsonl(path, max_samples=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            rows.append(json.loads(line))
    return rows

def main():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
    reranker = CrossEncoder(RERANK_MODEL_NAME, device=DEVICE)

    data = load_jsonl(INPUT_FILE, max_samples=MAX_SAMPLES)
    rows = []

    for i, item in enumerate(data, 1):
        all_chunks = split_context(item["context"])
        all_chunk_embeddings = embed_model.encode(
            all_chunks,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=DEVICE
        ) if all_chunks else None

        total_latency = 0.0
        seen_idxs = set()
        evidence_memory = []
        judge_trace = []
        iteration_count = 1

        first_chunks, first_idxs = hybrid_retrieve(
            item["question"], all_chunks, all_chunk_embeddings, embed_model,
            final_k=FIRST_K, exclude_idxs=seen_idxs
        )
        seen_idxs.update(first_idxs)
        evidence_memory.extend(first_chunks)
        evidence_memory = soft_compact_memory(evidence_memory)

        memory_text = "\n\n".join(evidence_memory)
        judge_raw, lat = call_ollama(
            JUDGE_PROMPT.format(question=item["question"], memory=memory_text)
        )
        total_latency += lat

        judge = safe_parse_json(judge_raw, fallback={
            "can_answer": False,
            "answer_type": "other",
            "provisional_answer": "Not enough information",
            "missing_items": [],
            "gap_items": [],
            "next_query": ""
        })
        judge_trace.append(judge)

        can_answer = bool(judge.get("can_answer", False))
        gap_items = judge.get("gap_items", [])
        next_query = str(judge.get("next_query", "")).strip()

        if not can_answer and iteration_count < MAX_ITERATIONS:
            targeted_query = build_gap_query(item["question"], gap_items, next_query=next_query)

            second_chunks, second_idxs = hybrid_retrieve(
                targeted_query, all_chunks, all_chunk_embeddings, embed_model,
                final_k=SECOND_K, exclude_idxs=seen_idxs
            )

            if second_chunks:
                iteration_count = 2
                seen_idxs.update(second_idxs)

                reranked_second_chunks = cross_encoder_rerank(
                    targeted_query,
                    second_chunks,
                    reranker,
                    top_k=SECOND_RERANK_TOP_K
                )

                extract_raw, lat = call_ollama(
                    EXTRACT_PROMPT.format(
                        question=item["question"],
                        gap_items=json.dumps(gap_items, ensure_ascii=False),
                        context="\n\n".join(reranked_second_chunks)[:EXTRACT_CONTEXT_CHARS]
                    )
                )
                total_latency += lat

                extracted = safe_parse_json(extract_raw, fallback={"selected_evidence": reranked_second_chunks[:2]})
                selected_evidence = extracted.get("selected_evidence", [])
                if not isinstance(selected_evidence, list):
                    selected_evidence = reranked_second_chunks[:2]

                selected_evidence = selected_evidence[:3]
                evidence_memory.extend(selected_evidence)
                evidence_memory = soft_compact_memory(evidence_memory)

                memory_text = "\n\n".join(evidence_memory)
                judge_raw2, lat = call_ollama(
                    JUDGE_PROMPT.format(question=item["question"], memory=memory_text)
                )
                total_latency += lat
                judge2 = safe_parse_json(judge_raw2, fallback=judge)
                judge_trace.append(judge2)

        final_memory = "\n\n".join(
            soft_compact_memory(
                evidence_memory,
                max_items=MAX_MEMORY_ITEMS,
                max_chars=MAX_FINAL_CONTEXT_CHARS
            )
        )
        final_raw, lat = call_ollama(
            FINAL_PROMPT.format(question=item["question"], memory=final_memory)
        )
        total_latency += lat

        parsed = safe_parse_json(final_raw, fallback={"answer": final_raw.strip(), "evidence": []})
        predicted_answer = str(parsed.get("answer", "")).strip()
        predicted_evidence = parsed.get("evidence", [])
        if not isinstance(predicted_evidence, list):
            predicted_evidence = []

        gold_answers = item.get("all_answers", None) or [item["answer"]]

        rows.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "ground_truth_answer": item["answer"],
            "all_answers": gold_answers,
            "predicted_answer": predicted_answer,
            "answer_f1": max_f1_over_answers(predicted_answer, gold_answers),
            "gold_evidence": item.get("highlighted_evidence", []),
            "predicted_evidence": predicted_evidence,
            "evidence_f1": evidence_f1(predicted_evidence, item.get("highlighted_evidence", [])),
            "faithfulness": None,
            "latency_sec": total_latency,
            "iteration_count": iteration_count,
            "evidence_type": item.get("evidence_type", "text"),
            "evidence_memory": evidence_memory,
            "judge_trace": judge_trace
        })

        print(f"Done {i}/{len(data)} | iterations={iteration_count}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)

    print("\nOverall averages:")
    print(df[["answer_f1", "evidence_f1", "latency_sec", "iteration_count"]].mean(numeric_only=True))

    print("\nBy evidence_type:")
    print(df.groupby("evidence_type")[["answer_f1", "evidence_f1", "latency_sec", "iteration_count"]].mean(numeric_only=True))

    print("\nIteration usage:")
    print(df["iteration_count"].value_counts(dropna=False).sort_index())

if __name__ == "__main__":
    main()