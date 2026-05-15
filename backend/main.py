"""
main.py – FastAPI backend for AI Vaidya
Features:
  • /status  – KB readiness check with processing state
  • /upload  – saves PDF, processes in background thread, returns immediately
  • /ask     – full hybrid-RAG pipeline
  • /pdfs    – list currently indexed PDFs
  • /pdfs/{filename} DELETE – remove PDF and clean vectorstore
  • /sample-questions – get suggested questions from indexed content
  • CORS left wide open for hackathon convenience (tighten in production)
  • Starts with empty knowledge base – no pre-seeded PDFs
"""

import os
import shutil
import traceback
import threading

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ingest import (
    add_pdf_to_vectorstore,
    remove_pdf_from_vectorstore,
    extract_sample_questions,
    UPLOAD_DIR,
    VECTOR_DIR,
)
from rag import generate_answer, reload_vectorstore, clear_vectorstore


# ── processing state tracking ─────────────────────────────────────────────────
processing_files: dict[str, str] = {}   # filename -> status ("processing" | "done" | "error")
processing_errors: dict[str, str] = {}  # filename -> error message
_cancelled_files: set[str] = set()      # files whose processing should be skipped
_state_lock = threading.Lock()


# ── app lifecycle ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load existing vectorstore on startup if index exists (user data from previous sessions)."""
    index_path = os.path.join(VECTOR_DIR, "ayurveda.index")
    if os.path.exists(index_path):
        try:
            reload_vectorstore()
            print("Existing vectorstore loaded.")
        except Exception as e:
            print(f"Warning – could not load existing vectorstore: {e}")
    else:
        print("No existing vectorstore found – starting with empty knowledge base.")
    yield


app = FastAPI(title="AI Vaidya API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── request / response models ──────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str


# ── background processing ──────────────────────────────────────────────────────
def _process_pdf_background(filename: str, file_path: str):
    """Run PDF ingestion in a background thread."""
    try:
        with _state_lock:
            if filename in _cancelled_files:
                _cancelled_files.discard(filename)
                processing_files.pop(filename, None)
                print(f"Processing cancelled for {filename}")
                return

        add_pdf_to_vectorstore(file_path)

        # Check again if cancelled during processing
        with _state_lock:
            if filename in _cancelled_files:
                _cancelled_files.discard(filename)
                processing_files.pop(filename, None)
                # Clean up: remove from vectorstore since it was cancelled
                remove_pdf_from_vectorstore(filename)
                print(f"Processing cancelled (post-index) for {filename}")
                return

        reload_vectorstore()

        with _state_lock:
            processing_files[filename] = "done"
        print(f"Background processing complete for {filename}")

    except Exception as exc:
        with _state_lock:
            processing_files[filename] = "error"
            processing_errors[filename] = str(exc)
        print(f"Background processing failed for {filename}: {exc}")
        traceback.print_exc()
        # Clean up the file if indexing failed
        if os.path.exists(file_path):
            os.remove(file_path)


# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "AI Vaidya backend is running 🌿"}


@app.get("/status")
def status():
    """
    Returns whether the knowledge base is ready and how many chunks are loaded.
    Also reports any files currently being processed.
    Frontend can poll this after upload to know when to enable the Q&A form.
    """
    from rag import chunks, index
    ready        = index is not None and chunks is not None
    chunk_count  = len(chunks) if chunks else 0
    pdfs_indexed = list({c["pdf"] for c in chunks}) if chunks else []

    with _state_lock:
        currently_processing = {
            f: s for f, s in processing_files.items() if s == "processing"
        }

    return {
        "ready":       ready,
        "chunk_count": chunk_count,
        "pdfs":        pdfs_indexed,
        "processing":  currently_processing,
    }


@app.get("/pdfs")
def list_pdfs():
    """List all PDFs currently in the uploads directory."""
    pdfs = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    return {"pdfs": pdfs}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accept any PDF, save it, and start background indexing.
    Returns HTTP 202 (Accepted) immediately — frontend polls /status or
    /files/{filename}/status to know when processing is done.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file (overwrite if re-uploading the same PDF)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    # Get file size for response
    file_size = os.path.getsize(file_path)

    # Mark as processing and start background thread
    with _state_lock:
        processing_files[file.filename] = "processing"
        _cancelled_files.discard(file.filename)

    thread = threading.Thread(
        target=_process_pdf_background,
        args=(file.filename, file_path),
        daemon=True,
    )
    thread.start()

    return JSONResponse(
        status_code=202,
        content={
            "message":  "PDF uploaded. Indexing started in background.",
            "filename": file.filename,
            "size":     file_size,
            "status":   "processing",
        },
    )


@app.get("/files/{filename}/status")
def file_processing_status(filename: str):
    """Check the processing status of a specific file."""
    with _state_lock:
        file_status = processing_files.get(filename)
        error = processing_errors.get(filename)

    if file_status is None:
        # Check if file exists in uploads
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            return {"filename": filename, "status": "ready"}
        raise HTTPException(status_code=404, detail="File not found.")

    response = {"filename": filename, "status": file_status}
    if file_status == "error" and error:
        response["error"] = error

    # If done, also return chunk count from that file
    if file_status == "done":
        from rag import chunks as rag_chunks
        if rag_chunks:
            file_chunks = [c for c in rag_chunks if c["pdf"] == filename]
            response["chunks"] = len(file_chunks)
        # Clean up processing state
        with _state_lock:
            processing_files.pop(filename, None)
            processing_errors.pop(filename, None)

    return response


@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Accepts a natural-language question, runs the full RAG pipeline,
    and returns the answer plus the source passages used.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    from rag import index as rag_index
    if rag_index is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not ready. Please upload a PDF first.",
        )

    try:
        answer, sources = generate_answer(question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {exc}",
        )

    return {
        "question": question,
        "answer":   answer,
        "sources":  sources,
    }


@app.get("/sample-questions")
def get_sample_questions():
    """
    Returns sample questions derived from the currently indexed content.
    Useful for suggesting what users can ask about their uploaded PDFs.
    """
    from rag import chunks as rag_chunks
    if not rag_chunks:
        return {"questions": []}

    questions = extract_sample_questions(rag_chunks, max_questions=5)
    return {"questions": questions}


@app.delete("/pdfs/{filename}")
def delete_pdf(filename: str):
    """
    Remove a PDF from the uploads dir and clean all associated data
    from the FAISS index and BM25 store.
    Also cancels in-progress processing if applicable.
    """
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Cancel any in-progress processing
    with _state_lock:
        if processing_files.get(filename) == "processing":
            _cancelled_files.add(filename)
            processing_files.pop(filename, None)
            processing_errors.pop(filename, None)
        else:
            processing_files.pop(filename, None)
            processing_errors.pop(filename, None)

    # Remove from disk
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        # Even if file is missing from disk, still try to clean vectorstore
        pass

    # Remove from vectorstore
    try:
        result = remove_pdf_from_vectorstore(filename)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"File removed but vectorstore cleanup failed: {exc}",
        )

    # Reload or clear vectorstore in memory
    remaining = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    if remaining and result.get("remaining_chunks", 0) > 0:
        try:
            reload_vectorstore()
        except Exception:
            pass
    else:
        clear_vectorstore()

    return {
        "message": f"{filename} deleted.",
        "remaining_pdfs": remaining,
        "chunks_removed": result.get("removed", 0),
        "chunks_remaining": result.get("remaining_chunks", 0),
    }