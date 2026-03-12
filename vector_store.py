"""
rag/vector_store.py
====================
ChromaDB vector store management.

Responsibilities:
- Create / load persistent ChromaDB collections
- Add document chunks with their embeddings
- Expose the store for downstream retrieval
- Support collection reset (for reindexing)

Why ChromaDB?
  - Local-first: no external server needed for development
  - Persistent: embeddings survive restarts (SQLite backend)
  - LangChain-native integration
  - Easy upgrade path to Pinecone / Weaviate / Qdrant for production
"""

import os
from typing import List, Optional
from loguru import logger

from langchain.schema import Document
from langchain_chroma import Chroma


def get_vector_store(
    embeddings,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Chroma:
    """
    Load an existing ChromaDB collection (or create one if it doesn't exist).

    Args:
        embeddings: LangChain Embeddings object (from rag/embeddings.py).
        persist_dir: Path to ChromaDB persistence directory.
        collection_name: Name of the ChromaDB collection.

    Returns:
        Chroma vector store instance.
    """
    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    collection_name = collection_name or os.getenv(
        "CHROMA_COLLECTION_NAME", "research_papers"
    )

    logger.info(
        f"Loading ChromaDB | collection='{collection_name}' | dir='{persist_dir}'"
    )

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    count = store._collection.count()
    logger.info(f"Collection '{collection_name}' has {count} existing chunks")

    return store


def add_documents_to_store(
    vector_store: Chroma,
    chunks: List[Document],
    batch_size: int = 100,
) -> None:
    """
    Add chunks to the vector store in batches (avoids API rate limits).

    Args:
        vector_store: Existing Chroma store.
        chunks: List of chunked Documents to embed and store.
        batch_size: Number of chunks per embedding API call.
    """
    if not chunks:
        logger.warning("No chunks to add — skipping.")
        return

    total = len(chunks)
    logger.info(f"Adding {total} chunks to ChromaDB in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        # Use chunk_id as the document ID to avoid duplicates on re-index
        ids = [c.metadata.get("chunk_id", f"chunk_{i+j}") for j, c in enumerate(batch)]
        vector_store.add_documents(documents=batch, ids=ids)
        logger.debug(f"Added batch {i//batch_size + 1} ({len(batch)} chunks)")

    logger.success(f"Indexed {total} chunks into ChromaDB")


def build_vector_store(
    chunks: List[Document],
    embeddings,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Chroma:
    """
    Convenience function: load store and add new chunks in one call.

    Args:
        chunks: Preprocessed document chunks.
        embeddings: Embedding model.
        persist_dir: ChromaDB persistence path.
        collection_name: Collection name.

    Returns:
        Updated Chroma vector store.
    """
    store = get_vector_store(embeddings, persist_dir, collection_name)
    add_documents_to_store(store, chunks)
    return store


def reset_collection(
    embeddings,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Chroma:
    """
    Delete and recreate the ChromaDB collection (full reindex).

    Args:
        embeddings: Embedding model.
        persist_dir: ChromaDB persistence path.
        collection_name: Collection name.

    Returns:
        Fresh empty Chroma store.
    """
    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    collection_name = collection_name or os.getenv(
        "CHROMA_COLLECTION_NAME", "research_papers"
    )

    logger.warning(f"Resetting collection '{collection_name}'...")

    # Load existing store and delete its collection
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    store.delete_collection()

    # Recreate empty
    fresh_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    logger.success(f"Collection '{collection_name}' reset successfully")
    return fresh_store


def get_collection_stats(vector_store: Chroma) -> dict:
    """
    Return basic stats about the current collection.

    Returns:
        Dict with count and collection name.
    """
    count = vector_store._collection.count()
    name = vector_store._collection.name
    return {"collection": name, "total_chunks": count}
