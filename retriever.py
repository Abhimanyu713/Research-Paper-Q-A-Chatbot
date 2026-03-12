"""
rag/retriever.py
================
Semantic retrieval system for the RAG pipeline.

Retrieval Strategy:
-------------------
Similarity Search (Dense Retrieval):
  - Converts the user query to an embedding vector.
  - Finds the top_k most cosine-similar chunk vectors in ChromaDB.
  - This is "semantic retrieval" — it finds conceptually relevant chunks
    even when exact keywords don't match (e.g., "performance" matches
    chunks about "accuracy", "F1-score", "benchmark results").

Why top_k = 5?
  - k=3: Fast but misses context spread across multiple sections.
  - k=5: Sweet spot — provides enough context for multi-part questions
         without overwhelming the LLM's context window.
  - k=8+: Diminishing returns; increases hallucination risk as the LLM
          must reconcile too much potentially conflicting context.

Optional Hybrid Search (BM25 + Dense):
  - BM25 excels at exact keyword/acronym matching (e.g., "BERT", "LoRA").
  - Dense excels at semantic matching.
  - EnsembleRetriever combines both for best of both worlds.
"""

import os
from typing import List, Optional
from loguru import logger

from langchain.schema import Document, BaseRetriever
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


def get_retriever(
    vector_store: Chroma,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> BaseRetriever:
    """
    Build a standard similarity-search retriever.

    Similarity search explanation:
      - Each chunk is stored as a high-dimensional embedding vector.
      - At query time, the question is embedded into the same space.
      - ChromaDB computes cosine similarity between query and all chunk vectors.
      - The top_k highest-similarity chunks are returned.

    Semantic retrieval benefit:
      - "What method did they use to reduce model size?" retrieves chunks about
        "pruning", "quantization", "knowledge distillation" — without needing
        those exact words in the question.

    Args:
        vector_store: Populated Chroma vector store.
        top_k: Number of chunks to retrieve per query. Default: env TOP_K or 5.
        score_threshold: Minimum similarity score (0.0–1.0). Filters weak matches.

    Returns:
        LangChain BaseRetriever.
    """
    top_k = top_k or int(os.getenv("RETRIEVAL_TOP_K", "5"))
    score_threshold = score_threshold or float(
        os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.0")
    )

    logger.info(f"Building retriever | top_k={top_k} | threshold={score_threshold}")

    if score_threshold > 0:
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": top_k,
                "score_threshold": score_threshold,
            },
        )
    else:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

    return retriever


def get_hybrid_retriever(
    vector_store: Chroma,
    all_chunks: List[Document],
    top_k: Optional[int] = None,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> EnsembleRetriever:
    """
    Hybrid retriever: BM25 (sparse) + ChromaDB (dense) with weighted fusion.

    When to use:
      - Papers with many technical acronyms (BERT, GPT, LoRA, RLHF)
      - Exact match queries ("Table 3", "Algorithm 1", specific author names)
      - When dense-only retrieval misses keyword-critical chunks

    Args:
        vector_store: Populated Chroma store.
        all_chunks: All document chunks (needed for BM25 index).
        top_k: Chunks per retriever. Total returned ≤ 2 * top_k (deduplicated).
        dense_weight: Weight for semantic retriever (default: 0.6).
        sparse_weight: Weight for BM25 retriever (default: 0.4).

    Returns:
        EnsembleRetriever combining both approaches.
    """
    top_k = top_k or int(os.getenv("RETRIEVAL_TOP_K", "5"))

    logger.info(
        f"Building hybrid retriever | "
        f"dense_weight={dense_weight} | sparse_weight={sparse_weight}"
    )

    # Dense retriever (semantic)
    dense_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # Sparse retriever (BM25 keyword)
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = top_k

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[sparse_weight, dense_weight],
    )

    return ensemble


def retrieve_with_scores(
    vector_store: Chroma,
    query: str,
    top_k: Optional[int] = None,
) -> List[tuple[Document, float]]:
    """
    Retrieve chunks along with their similarity scores (for debugging/evaluation).

    Args:
        vector_store: Populated Chroma store.
        query: User's question string.
        top_k: Number of chunks to retrieve.

    Returns:
        List of (Document, score) tuples, sorted by descending similarity.
    """
    top_k = top_k or int(os.getenv("RETRIEVAL_TOP_K", "5"))
    results = vector_store.similarity_search_with_score(query, k=top_k)
    logger.debug(
        f"Retrieved {len(results)} chunks | "
        f"scores: {[round(s, 3) for _, s in results]}"
    )
    return results
