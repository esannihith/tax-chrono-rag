import argparse
import json
import sys
from pathlib import Path

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.generation.pipeline import GenerationPipeline
from src.evaluation.generation_eval_engine import GenerationEvaluationEngine


def main():
    parser = argparse.ArgumentParser(description="Run RAG Generation Evaluation Benchmark")
    parser.add_argument("--suite", type=str, default="data/evaluation/active_suite.json", help="Path to evaluation suite")
    parser.add_argument("--sample", type=int, default=None, help="Number of cases to sample (None for all)")
    parser.add_argument("--output", type=str, default="data/evaluation/generation_experiments/v1_generation_baseline.json", help="Output JSON path")
    parser.add_argument("--top_k", type=int, default=4, help="Top-K chunks retrieved per query")
    parser.add_argument("--delay", type=float, default=0.8, help="Pacing delay between LLM calls")
    args = parser.parse_args()

    print("=" * 95)
    print("                 TEMPORAL RAG GENERATION BENCHMARK EVALUATOR")
    print("=" * 95)
    print(f"Suite File    : {args.suite}")
    print(f"Sample Limit  : {args.sample or 'Full Suite'}")
    print(f"Output Path   : {args.output}")
    print(f"Top-K Chunks  : {args.top_k}")
    print("=" * 95)

    pipeline = GenerationPipeline(indices_dir="data/indices")
    engine = GenerationEvaluationEngine(pipeline=pipeline)

    results = engine.evaluate_suite(
        suite_path=args.suite,
        max_cases=args.sample,
        top_k=args.top_k,
        pacing_delay_sec=args.delay
    )

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 95)
    print("                             GENERATION BENCHMARK SUMMARY")
    print("=" * 95)
    print(f"Total Cases Evaluated        : {results['total_cases_evaluated']}")
    print(f"Elapsed Time                 : {results['elapsed_seconds']}s")
    print(f"Mean Composite Score         : {results['mean_composite_score'] * 100:.2f}%")
    print(f"Mean Criteria Match Rate     : {results['mean_criteria_match_rate'] * 100:.2f}%")
    print(f"Mean Citation Precision      : {results['mean_citation_precision'] * 100:.2f}%")
    print(f"Mean Citation Recall         : {results['mean_citation_recall'] * 100:.2f}%")
    print(f"Mean Citation F1             : {results['mean_citation_f1'] * 100:.2f}%")
    print(f"Mean Temporal Validity Score : {results['mean_temporal_validity_score'] * 100:.2f}%")
    print(f"Negative Handling Accuracy   : {results['negative_handling_accuracy'] * 100:.2f}%")

    print("\n[Breakdown by Query Type]")
    print(f"{'Query Type':<35} | {'Count':<6} | {'Criteria Match':<16} | {'Composite Score':<16}")
    print("-" * 80)
    for qt, data in results["query_type_breakdown"].items():
        print(f"{qt:<35} | {data['count']:<6} | {data['mean_criteria_match']*100:>14.2f}% | {data['mean_composite_score']*100:>14.2f}%")

    print("\n[Breakdown by Regime]")
    print(f"{'Regime':<15} | {'Count':<6} | {'Composite Score':<16}")
    print("-" * 45)
    for reg, data in results["regime_breakdown"].items():
        print(f"{reg:<15} | {data['count']:<6} | {data['mean_composite_score']*100:>14.2f}%")
    print("=" * 95)
    print(f"Experiment artifact written to: {out_path}")


if __name__ == "__main__":
    main()
