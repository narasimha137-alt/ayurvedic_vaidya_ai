# AI Vaidya – Ayurvedic Knowledge Q&A Backend

> Hackathon submission for **AI Fusion Challenge – Problem Statement 3**
> BMS Institute of Technology & Management

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────┐
│         Query Expansion         │  (synonym injection for Ayurvedic terms)
└────────────────┬────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
  Dense Retrieval    Sparse Retrieval
  (FAISS cosine)     (BM25 Okapi)
        │                 │
        └────────┬────────┘
                 ▼
        RRF Fusion (top-20)
                 │
                 ▼
      Cross-Encoder Re-ranking
      (ms-marco-MiniLM-L-6-v2)
                 │
                 ▼
         Top-5 Passages
                 │
                 ▼
     Structured Prompt + Context
                 │
                 ▼
      Groq LLaMA-3.1-8B-Instant
                 │
                 ▼
         Answer + Sources
```

---

## Setup

### 1. Clone / copy files
```
ai-vaidya/
├── main.py
├── ingest.py
├── rag.py
├── requirements.txt
├── .env
├── uploads/          ← put your PDFs here (auto-created)
└── vectorstore/      ← index stored here (auto-created)
```

### 2. Create `.env`
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at https://console.groq.com

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/status` | KB readiness + chunk count |
| GET | `/pdfs` | List uploaded PDFs |
| POST | `/upload` | Upload a PDF and rebuild index |
| POST | `/ask` | Ask a question (RAG pipeline) |
| DELETE | `/pdfs/{filename}` | Remove a PDF and rebuild index |

### POST `/ask` – example
```json
// Request
{ "question": "What are the three doshas in Ayurveda?" }

// Response
{
  "question": "What are the three doshas in Ayurveda?",
  "answer": "According to [Source 1], the three doshas are Vata, Pitta, and Kapha …",
  "sources": [
    { "pdf": "charaka.pdf", "page": 12, "snippet": "…" }
  ]
}
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| PDF Parsing | PyMuPDF (fitz) |
| Text Cleaning | regex + NLTK (optional) |
| Bi-encoder Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector Store | FAISS (IndexFlatIP, cosine) |
| Keyword Search | BM25 Okapi (`rank-bm25`) |
| Hybrid Fusion | Reciprocal Rank Fusion (RRF) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | LLaMA-3.1-8B-Instant via Groq API (free) |
| Web Framework | FastAPI + Uvicorn |

---

## Future Scope
- Multi-language support (Sanskrit / Hindi queries)
- Named-entity recognition for herbs, diseases, formulations
- Knowledge graph over Ayurvedic concepts
- Streaming responses via SSE
- User session history & follow-up questions