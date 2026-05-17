"""
Evaluate RAG retrieval quality.

Measures whether the vector store returns relevant chunks for each test query.
Compares retrieval with different top_k values and chunk sizes.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.test_set import get_test_set, evaluate_answer
from rag.ingest import load_existing_index, ingest_knowledge_base, CHROMA_DB_DIR
from rag.lm_studio import LMStudioEmbedding


def evaluate_retrieval(top_k_values=(3, 5, 8)):
    """
    For each test query, retrieve top-k chunks and check if relevant
    source documents are returned.
    """
    print("Loading RAG index...")
    try:
        index = load_existing_index()
    except Exception:
        print("Building index...")
        index = ingest_knowledge_base()

    results = []

    # Expected source mapping: which source should each category retrieve?
    expected_sources = {
        "standards": "AS/NZS 1158",
        "energy": "Energy Efficiency Guidelines",
        "melbourne": "Melbourne Urban Data",
        "calculation": "AS/NZS 1158",
    }

    test_cases = get_test_set()

    for top_k in top_k_values:
        print(f"\n--- Evaluating with top_k={top_k} ---")

        retriever = index.as_retriever(similarity_top_k=top_k)

        for tc in test_cases:
            nodes = retriever.retrieve(tc["query"])

            # Check if expected source appears in retrieved chunks
            expected_src = expected_sources.get(tc["category"], "")
            source_hits = []
            for node in nodes:
                src = node.metadata.get("source", "")
                score = node.score if node.score else 0
                source_hits.append({"source": src, "score": round(score, 4)})

            # Did we get the right source in top results?
            relevant_found = any(
                expected_src.lower() in h["source"].lower()
                for h in source_hits
            )

            # Check if answer facts appear in retrieved text
            combined_text = " ".join(n.text for n in nodes)
            fact_eval = evaluate_answer(combined_text, tc)

            result = {
                "test_id": tc["id"],
                "query": tc["query"][:80],
                "category": tc["category"],
                "top_k": top_k,
                "relevant_source_found": relevant_found,
                "fact_score": fact_eval["score"],
                "sources_retrieved": [h["source"] for h in source_hits[:3]],
                "top_score": source_hits[0]["score"] if source_hits else 0,
            }
            results.append(result)

            status = "PASS" if relevant_found and fact_eval["score"] >= 0.5 else "FAIL"
            print(f"  [{status}] {tc['id']}: relevant_src={relevant_found}, fact_score={fact_eval['score']:.2f}")

    return results


def summarize_results(results):
    """Compute aggregate metrics from retrieval evaluation."""
    from collections import defaultdict

    by_topk = defaultdict(list)
    by_category = defaultdict(list)

    for r in results:
        by_topk[r["top_k"]].append(r)
        by_category[r["category"]].append(r)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)

    print("\n--- By top_k ---")
    for k, group in sorted(by_topk.items()):
        src_acc = sum(1 for r in group if r["relevant_source_found"]) / len(group) * 100
        avg_fact = sum(r["fact_score"] for r in group) / len(group) * 100
        print(f"  top_k={k}: Source accuracy={src_acc:.0f}%, Avg fact score={avg_fact:.0f}%")

    print("\n--- By category ---")
    for cat, group in sorted(by_category.items()):
        src_acc = sum(1 for r in group if r["relevant_source_found"]) / len(group) * 100
        avg_fact = sum(r["fact_score"] for r in group) / len(group) * 100
        print(f"  {cat}: Source accuracy={src_acc:.0f}%, Avg fact score={avg_fact:.0f}%")

    # Overall
    total = len(results)
    overall_src = sum(1 for r in results if r["relevant_source_found"]) / total * 100
    overall_fact = sum(r["fact_score"] for r in results) / total * 100
    print(f"\n  OVERALL: Source accuracy={overall_src:.0f}%, Avg fact score={overall_fact:.0f}%")

    return {
        "total_tests": total,
        "source_accuracy_pct": round(overall_src, 1),
        "avg_fact_score_pct": round(overall_fact, 1),
    }


if __name__ == "__main__":
    results = evaluate_retrieval()
    summary = summarize_results(results)

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "eval_rag_retrieval.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nResults saved to outputs/eval_rag_retrieval.json")
