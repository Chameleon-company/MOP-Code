"""
Compare RAG-grounded vs no-retrieval (pure LLM) answers.

This is a key HD requirement: demonstrate that RAG improves answer quality.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.test_set import get_test_set, evaluate_answer
from rag.ingest import load_existing_index, ingest_knowledge_base
from rag.query_engine import create_query_engine, SYSTEM_PROMPT
from rag.lm_studio import LMStudioLLM


NO_RAG_TEMPLATE = SYSTEM_PROMPT + """

Answer the following question using only your training knowledge.
Do NOT make up specific numbers if you are not sure.

Question: {query}

Answer:"""


def evaluate_rag_vs_norag(max_tests=10):
    """Run same queries with and without RAG retrieval."""

    # Setup RAG
    print("Loading RAG index...")
    try:
        index = load_existing_index()
    except Exception:
        index = ingest_knowledge_base()
    query_engine = create_query_engine(index=index, similarity_top_k=5)

    # Setup no-RAG (direct LLM)
    llm = LMStudioLLM(max_tokens=1024, temperature=0.1)

    test_cases = get_test_set()[:max_tests]
    results = []

    for tc in test_cases:
        print(f"\n--- {tc['id']}: {tc['query'][:60]}... ---")

        # === With RAG ===
        start = time.time()
        try:
            rag_response = query_engine.query(tc["query"])
            rag_answer = rag_response.response if hasattr(rag_response, 'response') else str(rag_response)
            rag_latency = time.time() - start
            rag_sources = []
            if hasattr(rag_response, 'source_nodes'):
                for n in rag_response.source_nodes[:3]:
                    rag_sources.append(n.metadata.get("source", "?"))
        except Exception as e:
            rag_answer = f"ERROR: {e}"
            rag_latency = time.time() - start
            rag_sources = []

        rag_eval = evaluate_answer(rag_answer, tc)

        # === Without RAG ===
        start = time.time()
        try:
            prompt = NO_RAG_TEMPLATE.format(query=tc["query"])
            norag_response = llm.complete(prompt)
            norag_answer = norag_response.text
            norag_latency = time.time() - start
        except Exception as e:
            norag_answer = f"ERROR: {e}"
            norag_latency = time.time() - start

        norag_eval = evaluate_answer(norag_answer, tc)

        result = {
            "test_id": tc["id"],
            "query": tc["query"],
            "category": tc["category"],
            "rag": {
                "score": rag_eval["score"],
                "pass": rag_eval["pass"],
                "matched_facts": rag_eval["matched_facts"],
                "missing_facts": rag_eval["missing_facts"],
                "latency_s": round(rag_latency, 2),
                "sources": rag_sources,
                "answer_preview": rag_answer[:200].replace("\n", " "),
            },
            "no_rag": {
                "score": norag_eval["score"],
                "pass": norag_eval["pass"],
                "matched_facts": norag_eval["matched_facts"],
                "missing_facts": norag_eval["missing_facts"],
                "latency_s": round(norag_latency, 2),
                "answer_preview": norag_answer[:200].replace("\n", " "),
            },
            "rag_better": rag_eval["score"] > norag_eval["score"],
            "score_diff": round(rag_eval["score"] - norag_eval["score"], 2),
        }
        results.append(result)

        print(f"  RAG:    score={rag_eval['score']:.2f} ({'PASS' if rag_eval['pass'] else 'FAIL'}) latency={rag_latency:.1f}s")
        print(f"  No-RAG: score={norag_eval['score']:.2f} ({'PASS' if norag_eval['pass'] else 'FAIL'}) latency={norag_latency:.1f}s")
        print(f"  {'>> RAG wins' if result['rag_better'] else '<< No-RAG wins or tied'}")

    return results


def summarize_rag_vs_norag(results):
    """Produce comparison summary."""
    print("\n" + "=" * 60)
    print("RAG vs NO-RAG COMPARISON")
    print("=" * 60)

    rag_scores = [r["rag"]["score"] for r in results]
    norag_scores = [r["no_rag"]["score"] for r in results]

    rag_avg = sum(rag_scores) / len(rag_scores)
    norag_avg = sum(norag_scores) / len(norag_scores)

    rag_pass = sum(1 for r in results if r["rag"]["pass"])
    norag_pass = sum(1 for r in results if r["no_rag"]["pass"])

    rag_wins = sum(1 for r in results if r["rag_better"])
    norag_wins = sum(1 for r in results if r["score_diff"] < 0)
    ties = sum(1 for r in results if r["score_diff"] == 0)

    rag_latency = sum(r["rag"]["latency_s"] for r in results) / len(results)
    norag_latency = sum(r["no_rag"]["latency_s"] for r in results) / len(results)

    print(f"\n{'Metric':<25} {'With RAG':<15} {'Without RAG':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'Avg Fact Score':<25} {rag_avg:<15.2f} {norag_avg:<15.2f} {rag_avg - norag_avg:+.2f}")
    print(f"{'Pass Rate':<25} {rag_pass}/{len(results):<12} {norag_pass}/{len(results):<12} {rag_pass - norag_pass:+d}")
    print(f"{'Avg Latency':<25} {rag_latency:<15.1f}s {norag_latency:<15.1f}s {rag_latency - norag_latency:+.1f}s")
    print(f"\n  RAG wins: {rag_wins} | No-RAG wins: {norag_wins} | Ties: {ties}")

    improvement = ((rag_avg - norag_avg) / norag_avg * 100) if norag_avg > 0 else 0
    print(f"\n  RAG improves fact accuracy by {improvement:+.1f}%")

    summary = {
        "total_tests": len(results),
        "rag_avg_score": round(rag_avg, 3),
        "norag_avg_score": round(norag_avg, 3),
        "rag_pass_rate": rag_pass,
        "norag_pass_rate": norag_pass,
        "rag_wins": rag_wins,
        "norag_wins": norag_wins,
        "ties": ties,
        "improvement_pct": round(improvement, 1),
        "rag_avg_latency_s": round(rag_latency, 2),
        "norag_avg_latency_s": round(norag_latency, 2),
    }
    return summary


if __name__ == "__main__":
    print("Running RAG vs No-RAG comparison (this will take several minutes)...")
    results = evaluate_rag_vs_norag(max_tests=10)
    summary = summarize_rag_vs_norag(results)

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "eval_rag_vs_norag.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nResults saved to outputs/eval_rag_vs_norag.json")
