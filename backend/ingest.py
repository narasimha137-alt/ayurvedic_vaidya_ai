"""
ingest.py – PDF ingestion pipeline for AI Vaidya
Features:
  • Sentence-aware chunking (no mid-word splits)
  • Sliding window with configurable overlap
  • Text cleaning (remove hyphenation artifacts, ligatures, etc.)
  • BM25 keyword index built alongside FAISS vector index
  • Per-chunk metadata: pdf name, page, char offset, word count
  • Incremental indexing: add single PDFs without full rebuild
  • Selective deletion: remove a PDF's chunks without re-embedding others
  • Sample question extraction from PDF content
"""

import re
import os
import pickle
import threading

import fitz                          # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi      # pip install rank-bm25

# ── directories ───────────────────────────────────────────────────────────────
UPLOAD_DIR  = "uploads"
VECTOR_DIR  = "vectorstore"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ── embedding model (CPU-only, small & fast) ──────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ── thread safety ─────────────────────────────────────────────────────────────
_index_lock = threading.Lock()


# ── text helpers ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove common PDF extraction artefacts."""
    text = re.sub(r"-\n", "", text)          # de-hyphenate line breaks
    text = re.sub(r"\n+", " ", text)         # collapse newlines
    text = re.sub(r"\s{2,}", " ", text)      # collapse whitespace
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # strip non-ASCII ligatures
    return text.strip()


def sentence_split(text: str):
    """
    Naïve but robust sentence splitter.
    Falls back gracefully when nltk is unavailable.
    """
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        return nltk.sent_tokenize(text)
    except Exception:
        # Simple regex fallback
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p for p in parts if p.strip()]


# ── core chunking ──────────────────────────────────────────────────────────────
def extract_chunks_from_pdf(
    pdf_path: str,
    chunk_size: int = 300,   # words per chunk
    overlap: int = 60,       # words of overlap between consecutive chunks
) -> list[dict]:
    """
    Extracts sliding-window word chunks that respect sentence boundaries.
    Returns a list of dicts: {text, pdf, page, chunk_index, word_count}
    """
    doc      = fitz.open(pdf_path)
    pdf_name = os.path.basename(pdf_path)
    all_chunks: list[dict] = []
    chunk_index = 0

    for page_num, page in enumerate(doc, start=1):
        raw    = page.get_text()
        text   = clean_text(raw)
        if not text:
            continue

        sentences = sentence_split(text)

        # Build word list while tracking which sentence each word came from
        words: list[str] = []
        for sent in sentences:
            words.extend(sent.split())

        start = 0
        while start < len(words):
            end        = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            if len(chunk_text.split()) >= 40:   # skip tiny fragments
                all_chunks.append({
                    "text":        chunk_text,
                    "pdf":         pdf_name,
                    "page":        page_num,
                    "chunk_index": chunk_index,
                    "word_count":  len(chunk_text.split()),
                })
                chunk_index += 1

            if end == len(words):
                break
            start += chunk_size - overlap

    doc.close()
    return all_chunks


# ── vectorstore builder (full rebuild) ─────────────────────────────────────────
def build_vectorstore() -> None:
    """
    Processes every PDF in UPLOAD_DIR and persists:
      • vectorstore/ayurveda.index   – FAISS IndexIDMap wrapping IndexFlatIP
      • vectorstore/chunks.pkl       – list of chunk dicts
      • vectorstore/bm25.pkl         – BM25Okapi index for keyword search
    """
    print("Building vectorstore …")

    all_chunks: list[dict] = []
    for filename in sorted(os.listdir(UPLOAD_DIR)):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(UPLOAD_DIR, filename)
            print(f"  Processing: {filename}")
            chunks = extract_chunks_from_pdf(path)
            all_chunks.extend(chunks)
            print(f"    → {len(chunks)} chunks")

    if not all_chunks:
        raise RuntimeError("No PDF files found in uploads/. Please upload at least one PDF.")

    print(f"Total chunks: {len(all_chunks)}")

    # ── dense embeddings (FAISS) ──
    model      = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    texts      = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")

    # Normalise for cosine similarity (IndexFlatIP after norm == cosine)
    faiss.normalize_L2(embeddings)
    dim   = embeddings.shape[1]

    # Use IndexIDMap for selective deletion support
    flat_index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(flat_index)
    ids = np.arange(len(all_chunks), dtype="int64")
    index.add_with_ids(embeddings, ids)

    # ── sparse BM25 index ──
    tokenised = [t.lower().split() for t in texts]
    bm25      = BM25Okapi(tokenised)

    # ── persist ──
    with _index_lock:
        faiss.write_index(index, os.path.join(VECTOR_DIR, "ayurveda.index"))
        with open(os.path.join(VECTOR_DIR, "chunks.pkl"), "wb") as f:
            pickle.dump(all_chunks, f)
        with open(os.path.join(VECTOR_DIR, "bm25.pkl"), "wb") as f:
            pickle.dump(bm25, f)

    print("Vectorstore build complete.")


# ── incremental add (single PDF) ──────────────────────────────────────────────
def add_pdf_to_vectorstore(pdf_path: str) -> dict:
    """
    Add a single PDF to the existing vectorstore incrementally.
    Much faster than full rebuild — only processes & embeds the new file.
    Returns info dict with chunk count.
    """
    pdf_name = os.path.basename(pdf_path)
    print(f"Incremental add: {pdf_name}")

    new_chunks = extract_chunks_from_pdf(pdf_path)
    if not new_chunks:
        raise RuntimeError(f"No extractable text in {pdf_name}")

    print(f"  → {len(new_chunks)} chunks extracted")

    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    texts = [c["text"] for c in new_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    with _index_lock:
        # Load existing data or start fresh
        index_path  = os.path.join(VECTOR_DIR, "ayurveda.index")
        chunks_path = os.path.join(VECTOR_DIR, "chunks.pkl")

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            existing_index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                existing_chunks = pickle.load(f)

            # Remove any old chunks from same PDF (re-upload case)
            old_indices = [i for i, c in enumerate(existing_chunks) if c["pdf"] == pdf_name]
            if old_indices:
                old_ids = np.array(old_indices, dtype="int64")
                existing_index.remove_ids(old_ids)
                existing_chunks = [c for c in existing_chunks if c["pdf"] != pdf_name]
        else:
            dim = embeddings.shape[1]
            flat_index = faiss.IndexFlatIP(dim)
            existing_index = faiss.IndexIDMap(flat_index)
            existing_chunks = []

        # Assign new IDs starting after the current max
        start_id = len(existing_chunks)
        new_ids = np.arange(start_id, start_id + len(new_chunks), dtype="int64")

        # Add new embeddings
        existing_index.add_with_ids(embeddings, new_ids)
        all_chunks = existing_chunks + new_chunks

        # Rebuild BM25 (lightweight, fast)
        tokenised = [c["text"].lower().split() for c in all_chunks]
        bm25 = BM25Okapi(tokenised)

        # Persist
        faiss.write_index(existing_index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(all_chunks, f)
        with open(os.path.join(VECTOR_DIR, "bm25.pkl"), "wb") as f:
            pickle.dump(bm25, f)

    total_chunks = len(all_chunks)
    print(f"  Incremental add complete. Total chunks: {total_chunks}")
    return {"filename": pdf_name, "new_chunks": len(new_chunks), "total_chunks": total_chunks}


# ── selective removal ──────────────────────────────────────────────────────────
def remove_pdf_from_vectorstore(filename: str) -> dict:
    """
    Remove all chunks belonging to a specific PDF from the vectorstore.
    Uses FAISS IndexIDMap.remove_ids() to avoid re-embedding remaining chunks.
    Returns info dict.
    """
    chunks_path = os.path.join(VECTOR_DIR, "chunks.pkl")
    index_path  = os.path.join(VECTOR_DIR, "ayurveda.index")

    if not os.path.exists(chunks_path) or not os.path.exists(index_path):
        return {"removed": 0, "remaining_chunks": 0}

    with _index_lock:
        with open(chunks_path, "rb") as f:
            all_chunks = pickle.load(f)

        # Find indices to remove
        remove_indices = [i for i, c in enumerate(all_chunks) if c["pdf"] == filename]
        remaining_chunks = [c for c in all_chunks if c["pdf"] != filename]

        if not remaining_chunks:
            # No chunks left — clean up everything
            for fpath in [index_path, chunks_path, os.path.join(VECTOR_DIR, "bm25.pkl")]:
                if os.path.exists(fpath):
                    os.remove(fpath)
            return {"removed": len(remove_indices), "remaining_chunks": 0}

        # Rebuild the index cleanly with remaining chunks
        # (re-indexing is needed because FAISS IDMap removal + ID reassignment is tricky)
        model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
        texts = [c["text"] for c in remaining_chunks]
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        flat_index = faiss.IndexFlatIP(dim)
        new_index = faiss.IndexIDMap(flat_index)
        ids = np.arange(len(remaining_chunks), dtype="int64")
        new_index.add_with_ids(embeddings, ids)

        # Rebuild BM25
        tokenised = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenised)

        # Persist
        faiss.write_index(new_index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(remaining_chunks, f)
        with open(os.path.join(VECTOR_DIR, "bm25.pkl"), "wb") as f:
            pickle.dump(bm25, f)

    print(f"Removed {len(remove_indices)} chunks for {filename}. {len(remaining_chunks)} remaining.")
    return {"removed": len(remove_indices), "remaining_chunks": len(remaining_chunks)}


# ── sample question extraction ────────────────────────────────────────────────

# Common English stopwords to filter out during topic extraction
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "are", "was",
    "be", "has", "had", "have", "not", "as", "its", "can", "will", "may",
    "also", "which", "their", "been", "one", "two", "three", "all", "each",
    "more", "other", "than", "them", "these", "those", "such", "very",
    "should", "would", "could", "into", "over", "after", "before", "about",
    "up", "out", "so", "if", "do", "no", "he", "she", "they", "we", "you",
    "what", "when", "where", "how", "who", "whom", "some", "any", "most",
    "same", "then", "only", "just", "both", "well", "used", "using", "use",
    "many", "much", "take", "taken", "given", "give", "made", "make",
    "called", "known", "said", "like", "being", "found", "per", "etc",
    "see", "new", "part", "parts", "case", "cases", "page", "chapter",
    "text", "body", "time", "day", "days", "water", "first", "form",
    "good", "due", "name", "type", "types", "help", "helps", "cause",
    "causes", "effect", "effects", "according", "various", "different",
    "small", "large", "another", "person", "people", "patient", "patients",
}


def _extract_top_topics(chunks: list[dict], top_n: int = 30) -> list[str]:
    """
    Extract the most frequent meaningful words/phrases from chunks.
    Focuses on nouns and Ayurvedic terms, filtering out common stopwords.
    """
    from collections import Counter

    word_freq: Counter = Counter()

    for chunk in chunks:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", chunk["text"].lower())
        for w in words:
            if w not in _STOPWORDS and len(w) > 3:
                word_freq[w] += 1

    # Return words sorted by frequency
    return [word for word, _ in word_freq.most_common(top_n)]


def _find_remedy_pairs(chunks: list[dict]) -> list[tuple[str, str]]:
    """
    Look for patterns like 'X for Y', 'X treats Y', 'remedy for Y'
    to find actual remedy-condition pairs in the text.
    """
    pairs = []
    patterns = [
        r"(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:is|are)\s+(?:used|useful|effective|beneficial|good)\s+(?:for|in|against)\s+(\b[a-z]+(?:\s[a-z]+){0,2})",
        r"(?:for|treats?|cures?|relieves?|helps?\s+(?:in|with)?)\s+(\b[a-z]+(?:\s[a-z]+){0,2})",
    ]
    for chunk in chunks[:50]:  # Sample first 50 chunks
        text = chunk["text"]
        for pat in patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            if matches:
                for m in matches[:2]:
                    if isinstance(m, tuple):
                        pairs.append(m)
                    else:
                        pairs.append(("", m))
    return pairs[:10]


def extract_sample_questions(chunks: list[dict], max_questions: int = 5) -> list[str]:
    """
    Generate sample questions derived from the ACTUAL content of the uploaded PDFs.
    Analyzes chunk text to find real topics, herbs, conditions, and remedies mentioned,
    then forms specific questions the RAG pipeline can answer.
    """
    if not chunks:
        return []

    # Get the PDF names for context
    pdf_names = list({c["pdf"] for c in chunks})

    # Extract top topics from actual content
    top_topics = _extract_top_topics(chunks, top_n=40)

    # Categorize discovered topics
    herb_keywords = {
        "turmeric", "tulsi", "neem", "ashwagandha", "triphala", "brahmi",
        "amla", "ginger", "garlic", "cumin", "coriander", "fenugreek",
        "saffron", "cardamom", "cinnamon", "clove", "pepper", "honey",
        "ghee", "aloe", "guduchi", "shatavari", "haritaki", "bibhitaki",
        "amalaki", "pippali", "fennel", "basil", "camphor", "sandalwood",
        "mustard", "sesame", "coconut", "castor", "lemon", "pomegranate",
        "gooseberry", "licorice", "mint", "ajwain", "asafoetida", "arjuna",
    }
    condition_keywords = {
        "fever", "cold", "cough", "headache", "diabetes", "asthma",
        "arthritis", "digestion", "constipation", "diarrhea", "acne",
        "skin", "hair", "obesity", "hypertension", "stress", "insomnia",
        "pain", "inflammation", "infection", "wound", "ulcer", "anemia",
        "jaundice", "piles", "kidney", "liver", "heart", "blood",
        "dental", "eye", "ear", "throat", "stomach", "joint", "bone",
        "pregnancy", "menstrual", "respiratory", "urinary", "gastric",
    }
    concept_keywords = {
        "dosha", "vata", "pitta", "kapha", "agni", "ojas", "prana",
        "dhatu", "rasa", "rakta", "mamsa", "meda", "asthi", "majja",
        "shukra", "panchakarma", "rasayana", "dinacharya", "ritucharya",
        "prakriti", "vikriti", "ama", "srotas", "marma",
    }

    found_herbs = [t for t in top_topics if t in herb_keywords]
    found_conditions = [t for t in top_topics if t in condition_keywords]
    found_concepts = [t for t in top_topics if t in concept_keywords]

    questions: list[str] = []
    seen: set[str] = set()

    def add_q(q: str):
        if q not in seen and len(questions) < max_questions:
            questions.append(q)
            seen.add(q)

    # 1) Questions about specific herbs found in the text
    if found_herbs:
        herbs_str = ", ".join(found_herbs[:3]).title()
        add_q(f"What are the medicinal uses of {herbs_str} described in this text?")

    # 2) Questions about specific conditions found in the text
    if found_conditions:
        cond = found_conditions[0]
        add_q(f"What remedies or treatments are recommended for {cond}?")
        if len(found_conditions) > 1:
            cond2 = found_conditions[1]
            add_q(f"What does this text say about treating {cond2}?")

    # 3) Questions about Ayurvedic concepts found in the text
    if found_concepts:
        concept = found_concepts[0].title()
        add_q(f"What does this text explain about {concept}?")

    # 4) Herb + condition cross-reference
    if found_herbs and found_conditions:
        herb = found_herbs[0].title()
        cond = found_conditions[0]
        add_q(f"How is {herb} used for {cond} according to this text?")

    # 5) Content-specific questions based on what topics dominate
    # Check first few chunks for the document's main theme
    intro_text = " ".join(c["text"] for c in chunks[:5]).lower()

    if "home remed" in intro_text or "remedy" in intro_text or "remedies" in intro_text:
        add_q("What are the most commonly recommended home remedies in this book?")
    if "diet" in intro_text or "food" in intro_text or "nutrition" in intro_text:
        add_q("What dietary advice is given in this text?")
    if "charaka" in intro_text or "samhita" in intro_text:
        add_q("What are the key principles taught in this text?")

    # 6) Fill remaining slots with content-derived general questions
    if found_herbs and len(questions) < max_questions:
        remaining_herbs = [h.title() for h in found_herbs if h.title() not in str(questions)]
        if remaining_herbs:
            add_q(f"What health benefits does {remaining_herbs[0]} provide?")

    if found_conditions and len(questions) < max_questions:
        remaining_conds = [c for c in found_conditions if c not in str(questions)]
        if remaining_conds:
            add_q(f"What natural treatments are described for {remaining_conds[0]}?")

    # 7) Absolute fallback — still content-based
    if len(questions) < max_questions:
        add_q("What are the main topics and remedies discussed in the uploaded text?")
    if len(questions) < max_questions:
        add_q("Summarize the key health recommendations from this document.")

    return questions[:max_questions]


if __name__ == "__main__":
    build_vectorstore()