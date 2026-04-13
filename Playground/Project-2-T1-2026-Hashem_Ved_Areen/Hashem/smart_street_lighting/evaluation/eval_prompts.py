"""
Evaluate different prompting strategies.

Compares zero-shot, few-shot, and chain-of-thought prompting
on the same test queries using the local LLM.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.test_set import get_test_set, evaluate_answer
from rag.lm_studio import LMStudioLLM


# ============================================================
# Prompt Templates
# ============================================================

ZERO_SHOT_TEMPLATE = """\
You are a street lighting design expert. Answer the following question accurately and concisely.

Question: {query}

Answer:"""

FEW_SHOT_TEMPLATE = """\
You are a street lighting design expert. Here are some example Q&As:

Q: What is the required average illuminance for a P1 category area?
A: According to AS/NZS 1158, P1 (major pedestrian activity areas like CBD plazas) requires an average horizontal illuminance of 14 lux, with a minimum of 7.0 lux and uniformity ratio of 0.50.

Q: What is the energy saving from LED vs HPS street lights?
A: LED street lights typically save 50-70% energy compared to High Pressure Sodium (HPS). For example, a 60W LED replaces a 175W HPS system (lamp + ballast), saving approximately 57% energy.

Q: What colour temperature is recommended for Melbourne street lighting?
A: Melbourne recommends 3000K (warm white) colour temperature to minimize ecological impact on nocturnal wildlife, per the City of Melbourne Urban Forest Strategy.

Now answer this question:
Q: {query}
A:"""

CHAIN_OF_THOUGHT_TEMPLATE = """\
You are a street lighting design expert. Think through this step by step.

Question: {query}

Let me think through this step by step:
1. First, I need to identify what standard or guideline applies.
2. Then, I'll look up the specific requirement or value.
3. Finally, I'll provide the answer with context.

Step-by-step reasoning:"""


PROMPT_STRATEGIES = {
    "zero_shot": ZERO_SHOT_TEMPLATE,
    "few_shot": FEW_SHOT_TEMPLATE,
    "chain_of_thought": CHAIN_OF_THOUGHT_TEMPLATE,
}


def evaluate_prompts(max_tests=10):
    """
    Run each test query through all 3 prompting strategies and compare.
    """
    llm = LMStudioLLM(max_tokens=512, temperature=0.1)
    test_cases = get_test_set()[:max_tests]

    results = []

    for strategy_name, template in PROMPT_STRATEGIES.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*60}")

        strategy_scores = []

        for tc in test_cases:
            prompt = template.format(query=tc["query"])

            start = time.time()
            try:
                response = llm.complete(prompt)
                answer = response.text
                latency = time.time() - start
            except Exception as e:
                answer = f"ERROR: {str(e)[:100]}"
                latency = time.time() - start

            eval_result = evaluate_answer(answer, tc)
            eval_result["strategy"] = strategy_name
            eval_result["latency_s"] = round(latency, 2)
            eval_result["answer_length"] = len(answer)
            eval_result["answer_preview"] = answer[:200].replace("\n", " ")

            results.append(eval_result)
            strategy_scores.append(eval_result["score"])

            status = "PASS" if eval_result["pass"] else "FAIL"
            print(f"  [{status}] {tc['id']}: score={eval_result['score']:.2f}, latency={latency:.1f}s")

        avg_score = sum(strategy_scores) / len(strategy_scores) if strategy_scores else 0
        print(f"\n  Average score for {strategy_name}: {avg_score:.2f}")

    return results


def summarize_prompt_results(results):
    """Compare prompting strategies side by side."""
    from collections import defaultdict

    by_strategy = defaultdict(list)
    for r in results:
        by_strategy[r["strategy"]].append(r)

    print("\n" + "=" * 60)
    print("PROMPT STRATEGY COMPARISON")
    print("=" * 60)

    print(f"\n{'Strategy':<20} {'Avg Score':<12} {'Pass Rate':<12} {'Avg Latency':<12} {'Avg Length':<12}")
    print("-" * 68)

    summary = {}
    for strategy, group in sorted(by_strategy.items()):
        avg_score = sum(r["score"] for r in group) / len(group)
        pass_rate = sum(1 for r in group if r["pass"]) / len(group) * 100
        avg_latency = sum(r["latency_s"] for r in group) / len(group)
        avg_length = sum(r["answer_length"] for r in group) / len(group)

        print(f"{strategy:<20} {avg_score:<12.2f} {pass_rate:<12.0f}% {avg_latency:<12.1f}s {avg_length:<12.0f}")

        summary[strategy] = {
            "avg_score": round(avg_score, 3),
            "pass_rate_pct": round(pass_rate, 1),
            "avg_latency_s": round(avg_latency, 2),
            "avg_answer_length": round(avg_length),
            "num_tests": len(group),
        }

    # Per-test comparison
    print("\n--- Per-Test Comparison ---")
    test_ids = sorted(set(r["test_id"] for r in results))
    print(f"{'Test ID':<10}", end="")
    for s in sorted(by_strategy.keys()):
        print(f" {s:<16}", end="")
    print()

    for tid in test_ids:
        print(f"{tid:<10}", end="")
        for s in sorted(by_strategy.keys()):
            r = next((x for x in results if x["test_id"] == tid and x["strategy"] == s), None)
            if r:
                mark = "PASS" if r["pass"] else "FAIL"
                print(f" {r['score']:.2f} ({mark}){'':<4}", end="")
            else:
                print(f" {'N/A':<16}", end="")
        print()

    return summary


if __name__ == "__main__":
    print("Evaluating prompt strategies (this will take several minutes)...")
    results = evaluate_prompts(max_tests=10)
    summary = summarize_prompt_results(results)

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "eval_prompts.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nResults saved to outputs/eval_prompts.json")
