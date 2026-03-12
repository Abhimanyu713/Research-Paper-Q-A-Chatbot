"""
evaluate.py
===========
Evaluation script for the Research Paper Q&A RAG system.

Metrics:
--------
1. Faithfulness
   Does the answer contain only information from the retrieved context?
   Measures hallucination — a faithful answer never introduces facts
   not found in the source chunks.
   Score range: 0.0 (fully hallucinated) → 1.0 (fully grounded)

2. Answer Relevance
   Is the answer relevant to the question asked?
   Measures if the LLM stayed on-topic and addressed the question.
   Score range: 0.0 (off-topic) → 1.0 (perfectly relevant)

3. Context Precision
   Were the retrieved chunks actually useful for answering the question?
   Measures retrieval quality — high precision means fewer "noise" chunks.
   Score range: 0.0 (irrelevant context) → 1.0 (perfectly relevant retrieval)

4. Context Recall (requires ground truth)
   Did the retriever find all chunks needed to answer correctly?
   Score range: 0.0 (missed all relevant chunks) → 1.0 (found all needed chunks)

Usage:
  # With RAGAS (recommended):
  $ python evaluate.py --mode ragas --pdf path/to/paper.pdf

  # Simple mode (no ground truth needed):
  $ python evaluate.py --mode simple --pdf path/to/paper.pdf
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional
from loguru import logger

from dotenv import load_dotenv
load_dotenv()


# ──────────────────────────────────────────────
# Test Dataset
# ──────────────────────────────────────────────

# Sample Q&A pairs — replace with domain-specific questions for your papers
TEST_DATASET = [
    {
        "question": "What is the main contribution of this paper?",
        "ground_truth": "The paper presents a novel method that significantly improves performance on the benchmark task.",
    },
    {
        "question": "What datasets were used in the experiments?",
        "ground_truth": "The authors evaluated on standard benchmark datasets including those commonly used in the field.",
    },
    {
        "question": "What are the key limitations of the proposed approach?",
        "ground_truth": "The authors acknowledge limitations related to computational cost and generalization to out-of-domain data.",
    },
    {
        "question": "How does the proposed method compare to baselines?",
        "ground_truth": "The proposed method outperforms baseline approaches on most metrics.",
    },
    {
        "question": "What future work do the authors suggest?",
        "ground_truth": "The authors suggest exploring extensions to multilingual settings and larger model sizes.",
    },
]


# ──────────────────────────────────────────────
# Simple Evaluation (No Ground Truth Required)
# ──────────────────────────────────────────────

def simple_evaluate(chain, questions: list[str]) -> dict:
    """
    Run simple heuristic evaluation without ground truth.

    Checks:
    - Has answer (non-empty)
    - Has citations (source documents returned)
    - No explicit "I cannot find" responses (coverage)
    - Answer length is reasonable (> 50 chars)

    Args:
        chain: Built RAG chain.
        questions: List of test questions.

    Returns:
        Dict with per-question results and aggregate scores.
    """
    from rag.rag_chain import ask

    results = []
    for q in questions:
        result = ask(chain, q)
        answer = result["answer"]
        sources = result["sources"]

        has_answer = len(answer.strip()) > 50
        has_citations = len(sources) > 0
        not_found = "cannot find" in answer.lower() or "not in the paper" in answer.lower()
        reasonable_length = 50 < len(answer) < 3000

        score = sum([
            1.0 if has_answer else 0.0,
            1.0 if has_citations else 0.0,
            0.0 if not_found else 1.0,
            1.0 if reasonable_length else 0.5,
        ]) / 4.0

        results.append({
            "question": q,
            "answer": answer[:200] + "..." if len(answer) > 200 else answer,
            "num_citations": len(sources),
            "has_answer": has_answer,
            "has_citations": has_citations,
            "is_grounded": not not_found,
            "reasonable_length": reasonable_length,
            "score": score,
        })

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0

    return {
        "mode": "simple",
        "num_questions": len(results),
        "average_score": round(avg_score, 3),
        "results": results,
    }


# ──────────────────────────────────────────────
# RAGAS Evaluation
# ──────────────────────────────────────────────

def ragas_evaluate(
    chain,
    test_data: list[dict],
    pdf_path: Optional[str] = None,
) -> dict:
    """
    Full RAGAS evaluation with faithfulness, answer relevance, context precision.

    Requires:
    - pip install ragas datasets

    Args:
        chain: Built RAG chain.
        test_data: List of {"question": str, "ground_truth": str} dicts.
        pdf_path: Path to the evaluated PDF (for logging).

    Returns:
        Dict with RAGAS metric scores.
    """
    try:
        from ragas import evaluate as ragas_eval
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        logger.error(
            "RAGAS not installed. Run: pip install ragas datasets"
        )
        return {"error": "ragas not installed"}

    from rag.rag_chain import ask

    logger.info(f"Running RAGAS evaluation on {len(test_data)} questions...")

    # Collect RAG outputs for each question
    eval_rows = []
    for item in test_data:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        result = ask(chain, question)
        answer = result["answer"]
        sources = result["sources"]

        # RAGAS expects the raw context chunks as strings
        contexts = [s["snippet"] for s in sources] if sources else ["No context found."]

        eval_rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    # Build HuggingFace Dataset for RAGAS
    dataset = Dataset.from_list(eval_rows)

    try:
        scores = ragas_eval(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        scores_dict = scores.to_pandas().mean().to_dict()
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {"error": str(e)}

    return {
        "mode": "ragas",
        "pdf": str(pdf_path) if pdf_path else "N/A",
        "num_questions": len(test_data),
        "metrics": {
            "faithfulness": round(scores_dict.get("faithfulness", 0), 3),
            "answer_relevancy": round(scores_dict.get("answer_relevancy", 0), 3),
            "context_precision": round(scores_dict.get("context_precision", 0), 3),
            "context_recall": round(scores_dict.get("context_recall", 0), 3),
        },
        "interpretation": {
            "faithfulness": "> 0.8 = low hallucination risk",
            "answer_relevancy": "> 0.8 = answers are on-topic",
            "context_precision": "> 0.7 = retrieval mostly relevant",
            "context_recall": "> 0.7 = retriever finds most needed chunks",
        },
    }


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG Q&A system")
    parser.add_argument(
        "--mode",
        choices=["simple", "ragas"],
        default="simple",
        help="Evaluation mode: 'simple' (heuristic) or 'ragas' (full metrics)",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to the PDF to evaluate against",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        sys.exit(1)

    logger.info(f"Building RAG pipeline for: {pdf_path.name}")

    # Build full pipeline
    from rag.loader import load_multiple_pdfs
    from rag.chunker import chunk_and_clean
    from rag.embeddings import get_embeddings
    from rag.vector_store import build_vector_store, reset_collection
    from rag.retriever import get_retriever
    from rag.rag_chain import build_rag_chain

    docs = load_multiple_pdfs([pdf_path])
    chunks = chunk_and_clean(docs)
    embeddings = get_embeddings()
    reset_collection(embeddings)
    vector_store = build_vector_store(chunks, embeddings)
    retriever = get_retriever(vector_store)
    chain = build_rag_chain(retriever)

    if args.mode == "simple":
        questions = [item["question"] for item in TEST_DATASET]
        results = simple_evaluate(chain, questions)
    else:
        results = ragas_evaluate(chain, TEST_DATASET, pdf_path=pdf_path)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.success(f"Evaluation complete. Results saved to: {args.output}")

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    if args.mode == "simple":
        print(f"Average Score:  {results['average_score']:.3f} / 1.000")
        print(f"Questions:      {results['num_questions']}")
    else:
        metrics = results.get("metrics", {})
        for k, v in metrics.items():
            print(f"{k:25s}: {v:.3f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
