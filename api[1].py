"""
api.py
======
FastAPI backend for the Research Paper Q&A Chatbot.

Endpoints:
  POST /upload   - Upload and index PDF documents
  POST /ask      - Ask a question (with conversation memory)
  GET  /history  - Retrieve conversation history
  DELETE /reset  - Clear index and conversation history
  GET  /health   - Health check

Usage:
  $ uvicorn api:app --reload --port 8000
  $ curl -X POST /upload -F "files=@paper.pdf"
  $ curl -X POST /ask -H "Content-Type: application/json" -d '{"question":"What is BERT?"}'
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

load_dotenv()

from rag.loader import load_multiple_pdfs
from rag.chunker import chunk_and_clean
from rag.embeddings import get_embeddings
from rag.vector_store import build_vector_store, reset_collection, get_collection_stats
from rag.retriever import get_retriever
from rag.rag_chain import build_rag_chain, ask, get_memory

# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="Research Paper Q&A API",
    description="RAG-powered API for academic paper question answering with citations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (use Redis/DB in production)
_state = {
    "chain": None,
    "vector_store": None,
    "indexed_files": [],
    "conversation_history": [],
}


# ──────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    session_id: str


class UploadResponse(BaseModel):
    message: str
    files: list[str]
    total_chunks: int


class HistoryResponse(BaseModel):
    history: list[dict]
    count: int


class HealthResponse(BaseModel):
    status: str
    indexed_files: list[str]
    total_chunks: int
    chain_ready: bool


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system status and indexed document count."""
    stats = {}
    if _state["vector_store"]:
        stats = get_collection_stats(_state["vector_store"])

    return HealthResponse(
        status="ok",
        indexed_files=_state["indexed_files"],
        total_chunks=stats.get("total_chunks", 0),
        chain_ready=_state["chain"] is not None,
    )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """
    Upload one or more PDF files and index them for Q&A.

    - Validates PDF format
    - Chunks and embeds documents
    - Stores in ChromaDB
    - Rebuilds RAG chain
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided.",
        )

    pdf_files = [f for f in files if f.filename and f.filename.endswith(".pdf")]
    if not pdf_files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only PDF files are accepted.",
        )

    logger.info(f"Received {len(pdf_files)} PDF(s) for indexing")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_paths = []
        for uf in pdf_files:
            dest = Path(tmpdir) / uf.filename
            dest.write_bytes(await uf.read())
            temp_paths.append(dest)

        try:
            docs = load_multiple_pdfs(temp_paths)
            if not docs:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Could not extract text from uploaded PDFs.",
                )

            chunks = chunk_and_clean(docs)
            embeddings = get_embeddings()
            reset_collection(embeddings)
            vector_store = build_vector_store(chunks, embeddings)

            _state["vector_store"] = vector_store
            _state["indexed_files"] = [f.filename for f in pdf_files]
            _state["conversation_history"] = []

            # Rebuild chain with fresh memory
            retriever = get_retriever(vector_store)
            memory = get_memory()
            _state["chain"] = build_rag_chain(retriever, memory=memory)

        except EnvironmentError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Embedding service error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Upload processing error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Processing failed: {str(e)}",
            )

    return UploadResponse(
        message=f"Successfully indexed {len(pdf_files)} document(s).",
        files=_state["indexed_files"],
        total_chunks=len(chunks),
    )


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_question(request: AskRequest):
    """
    Ask a question about the indexed research papers.

    Returns:
    - Grounded answer (from context only)
    - Source citations (filename, page, snippet)

    Supports follow-up questions via conversation memory.
    """
    if not _state["chain"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents indexed. Upload PDFs first via POST /upload.",
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    result = ask(_state["chain"], request.question)

    # Store in conversation history
    _state["conversation_history"].append({
        "role": "user",
        "content": request.question,
    })
    _state["conversation_history"].append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        session_id=request.session_id or "default",
    )


@app.get("/history", response_model=HistoryResponse, tags=["Q&A"])
async def get_history():
    """Retrieve the full conversation history for the current session."""
    return HistoryResponse(
        history=_state["conversation_history"],
        count=len(_state["conversation_history"]),
    )


@app.delete("/reset", tags=["System"])
async def reset_system():
    """Clear the document index and conversation history."""
    if _state["vector_store"]:
        embeddings = get_embeddings()
        reset_collection(embeddings)

    _state["chain"] = None
    _state["vector_store"] = None
    _state["indexed_files"] = []
    _state["conversation_history"] = []

    logger.info("System reset complete")
    return {"message": "System reset. Upload new documents to start a new session."}
