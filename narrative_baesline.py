import json
import time
import re
import string
import math
from collections import Counter, defaultdict

import requests
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

INPUT_FILE = r"C:\Users\ayush\Downloads\narrativeqa-master\narrativeqa-master\processed\narrativeqa_test_qasper_like.jsonl"
OUTPUT_JSON = "baseline_v1_narrativeqa_results_14b.json"
OUTPUT_CSV = "baseline_v1_narrativeqa_results_14b.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:14b"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SAMPLES = 50
FIRST_K = 6
MAX_MEMORY_ITEMS = 12
MAX_FINAL_CONTEXT_CHARS = 8500

STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","by","is","are","was","were",
    "be","been","being","this","that","these","those","what","which","who","whom","when",
    "where","why","how","do","does","did","it","its","as","at","from","into","about","their",
    "them","they","we","our","can","could","should","would","may","might","than","then","there",
    "such","using","use","used","also","show","shows","shown"
}

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

def hybrid_retrieve(query, chunks, chunk_embeddings, model, final_k=6):
    if not chunks:
        return []

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

    dense_ranked = dense_ranked[:min(12, len(chunks))]
    bm25_ranked = bm25_ranked[:min(12, len(chunks))]

    fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=60)
    fused_ranked = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)

    reranked = []
    for idx in fused_ranked:
        score = fused[idx] + chunk_bonus(chunks[idx], query)
        reranked.append((chunks[idx], score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in reranked[:min(final_k, len(reranked))]]

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

def soft_compact_memory(items, max_items=MAX_MEMORY_ITEMS, max_chars=MAX_FINAL_CONTEXT_CHARS):
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

        retrieved_chunks = hybrid_retrieve(
            item["question"],
            all_chunks,
            all_chunk_embeddings,
            embed_model,
            final_k=FIRST_K
        )

        evidence_memory = soft_compact_memory(retrieved_chunks)
        final_memory = "\n\n".join(evidence_memory)

        final_raw, latency = call_ollama(
            FINAL_PROMPT.format(
                question=item["question"],
                memory=final_memory
            )
        )

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
            "latency_sec": latency,
            "iteration_count": 1,
            "evidence_type": item.get("evidence_type", "text"),
            "evidence_memory": evidence_memory
        })

        print(f"Done {i}/{len(data)}")

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

if __name__ == "__main__":
    main()