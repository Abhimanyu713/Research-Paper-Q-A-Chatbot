"""
rag/chunker.py
==============
Intelligent document chunking for research papers.

Why these chunk values?
-----------------------
chunk_size = 1000 tokens (~750 words):
  - Large enough to capture complete scientific concepts, equations, and argument chains.
  - Small enough to remain semantically focused for precise retrieval.
  - Matches the "sweet spot" for OpenAI text-embedding-3-small (optimal at <512 tokens
    per chunk, but research text is dense — 1000 chars ≈ 200-250 tokens, well within limits).

chunk_overlap = 150 tokens:
  - Prevents context loss at chunk boundaries (e.g., a conclusion sentence split from
    its evidence).
  - ~15% overlap is the standard industry practice for academic text.
  - Higher overlap increases index size; lower overlap risks fragmenting reasoning chains.

RecursiveCharacterTextSplitter priority:
  \\n\\n → \\n → '. ' → ' ' → ''
  This hierarchy respects paragraph > sentence > word boundaries,
  preserving the logical structure of academic writing.
"""

import os
from typing import List
from loguru import logger

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """
    Split documents into semantically coherent chunks for embedding.

    Each chunk's metadata preserves:
        - source, file_name, page_label (from loader)
        - chunk_index: position of chunk within its source document
        - chunk_id: globally unique identifier "<filename>_p<page>_c<idx>"

    Args:
        documents: List of full-page Documents from loader.
        chunk_size: Max characters per chunk. Defaults to env CHUNK_SIZE or 1000.
        chunk_overlap: Overlap in characters. Defaults to env CHUNK_OVERLAP or 150.

    Returns:
        List of chunked Documents with enriched metadata.
    """
    chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "150"))

    logger.info(
        f"Chunking {len(documents)} pages | "
        f"chunk_size={chunk_size} | chunk_overlap={chunk_overlap}"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Priority: paragraph → sentence → word → character
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        is_separator_regex=False,
        strip_whitespace=True,
    )

    chunks = splitter.split_documents(documents)

    # Assign chunk identifiers for the citation system
    chunk_counter: dict[str, int] = {}

    for chunk in chunks:
        file_name = chunk.metadata.get("file_name", "unknown")
        page_label = chunk.metadata.get("page_label", "?")

        key = f"{file_name}_p{page_label}"
        chunk_counter[key] = chunk_counter.get(key, 0) + 1
        chunk_idx = chunk_counter[key]

        chunk.metadata["chunk_index"] = chunk_idx
        chunk.metadata["chunk_id"] = f"{file_name}_p{page_label}_c{chunk_idx}"

    logger.success(
        f"Generated {len(chunks)} chunks from {len(documents)} pages "
        f"(avg {len(chunks)/max(len(documents),1):.1f} chunks/page)"
    )
    return chunks


def filter_short_chunks(
    chunks: List[Document],
    min_length: int = 50,
) -> List[Document]:
    """
    Remove chunks that are too short to be meaningful (headers, page numbers, etc.).

    Args:
        chunks: List of chunked Documents.
        min_length: Minimum character count to keep a chunk.

    Returns:
        Filtered list of Documents.
    """
    before = len(chunks)
    filtered = [c for c in chunks if len(c.page_content.strip()) >= min_length]
    removed = before - len(filtered)

    if removed:
        logger.info(f"Removed {removed} short chunks (< {min_length} chars)")

    return filtered


def chunk_and_clean(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    min_chunk_length: int = 50,
) -> List[Document]:
    """
    Full chunking pipeline: split → filter → return.

    Args:
        documents: Raw page Documents from loader.
        chunk_size: See chunk_documents().
        chunk_overlap: See chunk_documents().
        min_chunk_length: Minimum chars to keep a chunk.

    Returns:
        Clean, indexed chunk list ready for embedding.
    """
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    chunks = filter_short_chunks(chunks, min_chunk_length)
    return chunks
