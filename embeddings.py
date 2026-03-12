"""
rag/embeddings.py
=================
Modular embedding pipeline supporting OpenAI and HuggingFace backends.

Design:
- Factory pattern: get_embeddings() returns the right embedder based on config.
- OpenAI text-embedding-3-small is the default (1536-dim, fast, cost-effective).
- HuggingFace multilingual model is the fallback (supports 50+ languages, free).
- Embeddings are deterministic — same text always yields same vector.

Why text-embedding-3-small?
  - 62.3% MTEB score (vs ada-002's 61.0%) at 5x lower cost
  - Supports Matryoshka Representation Learning (can reduce to 512 dims)
  - 8191 token context window (plenty for 1000-char chunks)

HuggingFace fallback:
  - paraphrase-multilingual-MiniLM-L12-v2: 50+ languages, 384-dim
  - Runs locally with no API calls
  - ~60% MTEB score — slightly weaker but free
"""

import os
from loguru import logger

from langchain_openai import OpenAIEmbeddings


def get_embeddings(provider: str | None = None):
    """
    Factory function: returns the configured embedding model.

    Args:
        provider: 'openai' | 'huggingface'. Defaults to env EMBEDDING_PROVIDER or 'openai'.

    Returns:
        LangChain-compatible Embeddings object.

    Raises:
        ValueError: If an unsupported provider is specified.
        EnvironmentError: If required API keys are missing.
    """
    provider = provider or os.getenv("EMBEDDING_PROVIDER", "openai")

    if provider == "openai":
        return _get_openai_embeddings()
    elif provider == "huggingface":
        return _get_huggingface_embeddings()
    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. "
            "Choose 'openai' or 'huggingface'."
        )


def _get_openai_embeddings() -> OpenAIEmbeddings:
    """
    OpenAI text-embedding-3-small embeddings.

    Returns:
        OpenAIEmbeddings instance.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or environment variables."
        )

    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    logger.info(f"Using OpenAI embeddings: {model}")

    return OpenAIEmbeddings(
        model=model,
        openai_api_key=api_key,
    )


def _get_huggingface_embeddings():
    """
    HuggingFace multilingual sentence-transformers embeddings.
    Downloads model on first use (~90MB).

    Returns:
        HuggingFaceEmbeddings instance.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        raise ImportError(
            "langchain-huggingface is not installed. "
            "Run: pip install langchain-huggingface sentence-transformers"
        )

    model_name = os.getenv(
        "HF_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    logger.info(f"Using HuggingFace embeddings: {model_name}")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
