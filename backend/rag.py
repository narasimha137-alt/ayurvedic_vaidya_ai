"""
rag.py – Retrieval-Augmented Generation engine for AI Vaidya
Features:
  • Hybrid retrieval: dense (FAISS cosine) + sparse (BM25) with RRF fusion
  • Query expansion / keyword extraction via NLP
  • Cross-encoder re-ranking (optional, CPU-friendly model)
  • Structured prompt with explicit citation instruction
  • Graceful fallback if any component fails
  • Safe reload / clear operations for dynamic KB management
"""

import os
import re
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── API client (Groq) ──────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# ── models ─────────────────────────────────────────────────────────────────────
bi_encoder    = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Lightweight cross-encoder for re-ranking (runs on CPU fine)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

# ── state ──────────────────────────────────────────────────────────────────────
index:  faiss.Index | None = None
chunks: list[dict] | None  = None
bm25                       = None          # BM25Okapi instance


# ── loader ─────────────────────────────────────────────────────────────────────
def reload_vectorstore() -> None:
    """Load vectorstore from disk. Handles missing files gracefully."""
    global index, chunks, bm25

    index_path  = "vectorstore/ayurveda.index"
    chunks_path = "vectorstore/chunks.pkl"
    bm25_path   = "vectorstore/bm25.pkl"

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        print("Vectorstore files not found – starting with empty KB.")
        index  = None
        chunks = None
        bm25   = None
        return

    try:
        index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                bm25 = pickle.load(f)
        else:
            # Rebuild BM25 if missing
            from rank_bm25 import BM25Okapi
            tokenised = [c["text"].lower().split() for c in chunks]
            bm25 = BM25Okapi(tokenised)

        print(f"Vectorstore loaded: {len(chunks)} chunks, {index.ntotal} vectors.")
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        index  = None
        chunks = None
        bm25   = None


def clear_vectorstore() -> None:
    """Reset in-memory vectorstore state to empty."""
    global index, chunks, bm25
    index  = None
    chunks = None
    bm25   = None
    print("Vectorstore cleared from memory.")


# ── query helpers ──────────────────────────────────────────────────────────────
def _expand_query(question: str) -> str:
    """
    Light query expansion: strip stopwords, keep content words,
    append synonyms for common Ayurvedic terms.
    This improves BM25 recall without extra API calls.
    """
    _SYNONYMS = {
        "dosha":   "dosha vata pitta kapha",
        "herb":    "herb plant dravya aushadhi",
        "disease": "disease illness roga vikara",
        "diet":    "diet food ahara pathya",
        "digestion": "digestion agni pachana jatharagni",
        "mind":    "mind manas consciousness",
    }
    lowered = question.lower()
    extra   = []
    for key, expansion in _SYNONYMS.items():
        if key in lowered:
            extra.append(expansion)
    expanded = question + (" " + " ".join(extra) if extra else "")
    return expanded


def _dense_retrieve(query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Return (chunk_idx, score) pairs from FAISS cosine search."""
    scores, indices = index.search(query_vec, top_k)
    return list(zip(indices[0].tolist(), scores[0].tolist()))


def _sparse_retrieve(query: str, top_k: int) -> list[tuple[int, float]]:
    """Return (chunk_idx, score) pairs from BM25."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    # Normalise to [0, 1]
    max_s  = scores.max() if scores.max() > 0 else 1.0
    scores = scores / max_s
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]


def _reciprocal_rank_fusion(
    dense_hits: list[tuple[int, float]],
    sparse_hits: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion (RRF) to merge two ranked lists.
    Returns merged list sorted by fused score descending.
    """
    rrf: dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense_hits):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(sparse_hits):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank + 1)
    sorted_hits = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    return sorted_hits


def _rerank(question: str, candidates: list[dict], top_n: int) -> list[dict]:
    """
    Re-rank candidates using a cross-encoder and return top_n.
    Falls back to original order if cross-encoder fails.
    """
    try:
        pairs  = [(question, c["text"]) for c in candidates]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_n]]
    except Exception as e:
        print(f"Re-ranking skipped: {e}")
        return candidates[:top_n]


# ── public retrieval API ───────────────────────────────────────────────────────
def retrieve_relevant_chunks(
    question: str,
    top_k_initial: int = 20,    # candidates from hybrid search
    top_n_final:   int = 5,     # chunks passed to LLM after re-ranking
) -> list[dict]:
    """
    Full retrieval pipeline:
    1. Expand query
    2. Dense + sparse retrieval  →  RRF fusion
    3. Cross-encoder re-ranking
    4. Return top_n_final chunks
    """
    expanded  = _expand_query(question)
    query_vec = bi_encoder.encode([expanded], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype="float32")

    dense_hits  = _dense_retrieve(query_vec, top_k_initial)
    sparse_hits = _sparse_retrieve(expanded, top_k_initial)

    fused   = _reciprocal_rank_fusion(dense_hits, sparse_hits)
    cand_idx = [idx for idx, _ in fused[:top_k_initial]]
    candidates = [chunks[i] for i in cand_idx if 0 <= i < len(chunks)]

    return _rerank(question, candidates, top_n_final)


# ── prompt builder ─────────────────────────────────────────────────────────────
def _build_prompt(question: str, retrieved: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(retrieved, 1):
        ref = f"[Source {i}: {chunk['pdf']}, Page {chunk['page']}]"
        context_blocks.append(f"{ref}\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are AI Vaidya, an expert Ayurvedic knowledge assistant.
Your ONLY knowledge source is the context passages provided below.
Rules:
1. Answer strictly from the provided context. Do NOT add external knowledge.
2. If the context does not contain enough information, say:
   "I could not find a clear answer in the uploaded Ayurveda text."
3. Cite the source reference (e.g., [Source 1]) when you use information from it.
4. Keep your answer concise (4–6 sentences) and in plain English.
5. Use Ayurvedic terms naturally and explain them briefly on first use.

=== CONTEXT ===
{context}

=== QUESTION ===
{question}

=== ANSWER ==="""


# ── main generate function ─────────────────────────────────────────────────────
def generate_answer(question: str) -> tuple[str, list[dict]]:
    """
    Returns (answer_text, sources_list).
    sources_list items: {pdf, page, snippet}
    """
    if index is None or chunks is None or bm25 is None:
        return "Knowledge base not loaded. Please upload a PDF first.", []

    retrieved = retrieve_relevant_chunks(question)

    if not retrieved:
        return "No relevant passages found. Please try rephrasing your question.", []

    prompt = _build_prompt(question, retrieved)

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,        # low temp = factual, less hallucination
            max_tokens=400,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"LLM error: {exc}")
        answer = "The AI service is temporarily unavailable. Please try again later."
        return answer, []

    # If the LLM indicates it couldn't find the answer, don't show irrelevant sources
    if "I could not find a clear answer" in answer:
        sources = []
    else:
        sources = [
            {
                "pdf":     c["pdf"],
                "page":    c["page"],
                "snippet": c["text"][:400] + "…",
            }
            for c in retrieved
        ]

    return answer, sources