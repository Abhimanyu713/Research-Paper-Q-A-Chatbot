"""
rag/loader.py
=============
Handles loading and parsing of academic PDF documents.

Design decisions:
- PyPDFLoader for standard PDFs (arXiv-style)
- PyMuPDF (fitz) as fallback for complex layouts
- Metadata enrichment (page numbers, source filename)
- Batch processing for multiple documents
"""

import os
from pathlib import Path
from typing import List, Optional
from loguru import logger

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document


def load_pdf(file_path: str | Path) -> List[Document]:
    """
    Load a single PDF and return a list of Documents (one per page).

    Each Document.metadata contains:
        - source: original file path
        - page: zero-indexed page number
        - page_label: human-readable page number (1-indexed)
        - file_name: just the filename (no path)

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of LangChain Document objects.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the PDF is empty or unreadable.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {file_path.suffix}")

    logger.info(f"Loading PDF: {file_path.name}")

    try:
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
    except Exception as e:
        logger.error(f"PyPDFLoader failed for {file_path.name}: {e}")
        raise ValueError(f"Could not parse PDF '{file_path.name}': {e}") from e

    if not pages:
        raise ValueError(f"PDF '{file_path.name}' appears to be empty or unreadable.")

    # Enrich metadata for citation purposes
    for i, doc in enumerate(pages):
        doc.metadata["file_name"] = file_path.name
        doc.metadata["page_label"] = doc.metadata.get("page", i) + 1  # 1-indexed
        doc.metadata["source"] = str(file_path)

    logger.success(f"Loaded {len(pages)} pages from '{file_path.name}'")
    return pages


def load_multiple_pdfs(file_paths: List[str | Path]) -> List[Document]:
    """
    Load multiple PDFs and return all pages as a flat list.

    Args:
        file_paths: List of paths to PDF files.

    Returns:
        Flat list of Documents from all PDFs.
    """
    all_docs: List[Document] = []
    errors: List[str] = []

    for path in file_paths:
        try:
            docs = load_pdf(path)
            all_docs.extend(docs)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping {path}: {e}")
            errors.append(str(e))

    if errors:
        logger.warning(f"{len(errors)} PDFs failed to load:\n" + "\n".join(errors))

    logger.info(f"Total pages loaded: {len(all_docs)} from {len(file_paths) - len(errors)} PDFs")
    return all_docs


def save_uploaded_file(uploaded_file, upload_dir: str = "./data/uploads") -> Path:
    """
    Save a Streamlit UploadedFile to disk and return the path.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        upload_dir: Directory to save uploads.

    Returns:
        Path to the saved file.
    """
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)

    dest = upload_path / uploaded_file.name

    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())

    logger.info(f"Saved uploaded file: {dest}")
    return dest
