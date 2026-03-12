"""
rag/rag_chain.py
================
Core RAG chain: retrieval → prompt → LLM → structured answer with citations.

Pipeline:
  User Question
       ↓
  [ConversationalRetrievalChain]
       ↓
  Memory rewrites question using chat history
  (e.g., "What about their results?" → "What experimental results did the authors report?")
       ↓
  Retriever fetches top_k relevant chunks from ChromaDB
       ↓
  Prompt template formats: system instructions + context + chat history + question
       ↓
  LLM (GPT-4o-mini) generates grounded answer with citations
       ↓
  Source documents attached to response for UI display

Memory Strategy: ConversationBufferMemory
  - Stores full conversation turns (human + AI messages)
  - Enables natural follow-up questions
  - Limitation: grows unbounded; for production, use ConversationSummaryMemory
    or a sliding window to cap token usage.

Citation Strategy:
  - Each retrieved chunk has metadata: file_name, page_label, chunk_id
  - LLM is instructed to reference page numbers in-text
  - Source documents are returned separately for UI-level citation display
"""

import os
from typing import Optional
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import BaseRetriever


# ──────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are an expert academic research assistant helping users understand scientific papers.

Your task is to answer questions STRICTLY based on the provided context from the research papers.

RULES:
1. ONLY use information from the provided context. Do NOT use prior knowledge.
2. Always cite sources using the format: (Source: <filename>, Page <page_number>)
3. If multiple sources support an answer, cite all of them.
4. If the context does not contain the answer, say exactly: 
   "I cannot find this information in the uploaded papers."
5. Be precise and technical — the user is a researcher.
6. For numerical results, equations, or claims, always cite the exact source.
7. Structure complex answers with clear sections when appropriate.

CONTEXT FROM PAPERS:
{context}

Remember: Ground every claim in the context above. No hallucination."""

HUMAN_TEMPLATE = """{question}"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
])

# ──────────────────────────────────────────────
# Condense Question Prompt (for follow-ups)
# ──────────────────────────────────────────────
# This rewrites vague follow-up questions into standalone queries
# before retrieval, improving chunk recall for conversational questions.

CONDENSE_QUESTION_TEMPLATE = """Given the following conversation history and a follow-up question, 
rewrite the follow-up question to be a standalone question that captures full context.
Do NOT answer the question — only rewrite it.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(
    CONDENSE_QUESTION_TEMPLATE
)


# ──────────────────────────────────────────────
# LLM Factory
# ──────────────────────────────────────────────

def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    """
    Create the LLM instance.

    Args:
        model: Model name. Default: env LLM_MODEL or 'gpt-4o-mini'.
        temperature: Sampling temperature. 0.0 = deterministic. Default: 0.0.
        max_tokens: Max response tokens. Default: 1024.

    Returns:
        ChatOpenAI instance.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in environment.")

    model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = temperature if temperature is not None else float(
        os.getenv("LLM_TEMPERATURE", "0.0")
    )
    max_tokens = max_tokens or int(os.getenv("LLM_MAX_TOKENS", "1024"))

    logger.info(f"LLM: {model} | temperature={temperature} | max_tokens={max_tokens}")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,
    )


# ──────────────────────────────────────────────
# Memory Factory
# ──────────────────────────────────────────────

def get_memory() -> ConversationBufferMemory:
    """
    Create a ConversationBufferMemory instance.

    memory_key must match the chain's expected key.
    return_messages=True stores Message objects (required for chat models).

    Returns:
        ConversationBufferMemory instance.
    """
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",  # matches ConversationalRetrievalChain output
    )


# ──────────────────────────────────────────────
# RAG Chain Factory
# ──────────────────────────────────────────────

def build_rag_chain(
    retriever: BaseRetriever,
    llm: Optional[ChatOpenAI] = None,
    memory: Optional[ConversationBufferMemory] = None,
) -> ConversationalRetrievalChain:
    """
    Build the full Conversational RAG chain.

    This chain:
    1. Uses memory to condense follow-up questions into standalone queries.
    2. Retrieves relevant chunks from ChromaDB.
    3. Constructs a prompt with context + history + question.
    4. Generates a grounded answer with citations.
    5. Returns answer + source documents.

    Args:
        retriever: Configured retriever (from rag/retriever.py).
        llm: ChatOpenAI instance. Created from env if not provided.
        memory: ConversationBufferMemory. Created fresh if not provided.

    Returns:
        ConversationalRetrievalChain ready for Q&A.
    """
    llm = llm or get_llm()
    memory = memory or get_memory()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True,  # critical for citation display
        verbose=os.getenv("LOG_LEVEL", "INFO") == "DEBUG",
    )

    logger.success("RAG chain built successfully")
    return chain


# ──────────────────────────────────────────────
# Query Interface
# ──────────────────────────────────────────────

def ask(chain: ConversationalRetrievalChain, question: str) -> dict:
    """
    Submit a question to the RAG chain and return structured output.

    Args:
        chain: Built ConversationalRetrievalChain.
        question: User's natural-language question.

    Returns:
        Dict with keys:
            - answer: str — LLM-generated answer
            - sources: list of dicts with citation metadata
            - question: original question (echoed back)
    """
    if not question.strip():
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "question": question,
        }

    logger.info(f"Processing question: {question[:80]}...")

    try:
        result = chain.invoke({"question": question})
    except Exception as e:
        logger.error(f"Chain error: {e}")
        return {
            "answer": f"An error occurred while processing your question: {str(e)}",
            "sources": [],
            "question": question,
        }

    answer = result.get("answer", "")
    source_docs = result.get("source_documents", [])

    # Build citation metadata for UI display
    sources = _extract_citations(source_docs)

    logger.success(f"Answer generated | {len(sources)} citations")
    return {
        "answer": answer,
        "sources": sources,
        "question": question,
    }


def _extract_citations(source_docs) -> list[dict]:
    """
    Extract deduplicated citation metadata from retrieved source documents.

    Args:
        source_docs: List of Documents from the chain's source_documents.

    Returns:
        List of unique citation dicts with keys:
            - file_name, page, chunk_id, snippet
    """
    seen_chunks = set()
    citations = []

    for doc in source_docs:
        meta = doc.metadata
        chunk_id = meta.get("chunk_id", "unknown")

        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)

        citations.append({
            "file_name": meta.get("file_name", "Unknown document"),
            "page": meta.get("page_label", "?"),
            "chunk_id": chunk_id,
            "snippet": doc.page_content[:200].strip() + "...",
        })

    return citations
