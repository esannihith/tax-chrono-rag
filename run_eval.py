import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.indexing.vector_store import HybridVectorStore
from src.indexing.embedder import DenseEmbedder
from src.evaluation.benchmark_engine import BenchmarkEngine


def main():
    print("=" * 90)
    print("RUNNING COMPREHENSIVE RETRIEVAL & IR EVALUATION BENCHMARK")
    print("=" * 90)

    # 1. Load Vector Store & Embedder
    store = HybridVectorStore.load("data/indices")
    embedder = DenseEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    engine = BenchmarkEngine(vector_store=store, embedder=embedder)

    k_list = [5, 10, 20]
    out_dir = Path("data/evaluation")

    for suite_name in ["active_suite", "hidden_suite"]:
        suite_path = out_dir / f"{suite_name}.json"
        print(f"\n" + "=" * 90)
        print(f"EVALUATING SUITE: {suite_name.upper()} ({suite_path})")
        print("=" * 90)

        results = engine.run_evaluation(
            suite_path=str(suite_path),
            k_list=k_list,
            alpha=0.5,
            apply_normalization=True,
            apply_rule_boost=True
        )

        # Export CSV and JSON
        csv_file = out_dir / f"per_query_metrics_{suite_name}.csv"
        json_file = out_dir / f"results_{suite_name}.json"
        engine.export_per_query_csv(results["per_query_results"], str(csv_file))
        
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        mm = results["mean_metrics"]
        rb = results.get("random_baseline", {})
        corpus_size = results.get("corpus_size", 183)

        print(f"\n--- OVERALL AGGREGATED METRICS ({suite_name.upper()}) ---")
        print(f"Corpus Size: {corpus_size} chunks (95 in 1962, 88 in 2026)")
        print(f"Evaluated Cases: {results['grounded_cases']} Grounded | {results['null_cases']} Out-of-Scope (Null)")
        print("-" * 65)
        print(f"MRR (Mean Reciprocal Rank)      : {mm['mrr']:.4f}")
        print(f"HitRate@5 (Top-5 Hit Rate)      : {mm['hit_rate@5']*100:.2f}%  (Random Baseline: {rb.get('random_hit_rate@5', 0)*100:.2f}%)")
        print(f"HitRate@10 (Top-10 Hit Rate)    : {mm['hit_rate@10']*100:.2f}%  (Random Baseline: {rb.get('random_hit_rate@10', 0)*100:.2f}%)")
        print(f"HitRate@20 (Top-20 Hit Rate)    : {mm['hit_rate@20']*100:.2f}%  (Random Baseline: {rb.get('random_hit_rate@20', 0)*100:.2f}%)")
        print(f"Essential Recall@5              : {mm['essential_recall@5']*100:.2f}%  (Random Baseline: {rb.get('random_essential_recall@5', 0)*100:.2f}%)")
        print(f"Essential Recall@10             : {mm['essential_recall@10']*100:.2f}%  (Random Baseline: {rb.get('random_essential_recall@10', 0)*100:.2f}%)")
        print(f"Essential Recall@20             : {mm['essential_recall@20']*100:.2f}%  (Random Baseline: {rb.get('random_essential_recall@20', 0)*100:.2f}%)")
        print(f"Essential Precision@5           : {mm.get('essential_precision@5', 0)*100:.2f}%  (Random Baseline: {rb.get('random_essential_precision@5', 0)*100:.2f}%)")
        print(f"Full Precision@5                : {mm['precision@5']*100:.2f}%  (Random Baseline: {rb.get('random_full_precision@5', 0)*100:.2f}%)")
        print(f"Graded NDCG@5                   : {mm['ndcg@5']:.4f}")
        print(f"Graded NDCG@10                  : {mm['ndcg@10']:.4f}")
        print(f"Graded NDCG@20                  : {mm['ndcg@20']:.4f}")


        # Breakdown by Query Type
        print(f"\n--- BREAKDOWN BY QUERY TYPE ---")
        print(f"{'Query Type':35} | {'Count':5} | {'MRR':6} | {'Hit@5':7} | {'R@5':7} | {'R@10':7} | {'R@20':7} | {'NDCG@10':7}")
        print("-" * 95)
        for qt, stats in results["breakdown_by_query_type"].items():
            print(f"{qt:35} | {stats['count']:5} | {stats['mrr']:.4f} | {stats.get('hit_rate@5', 0)*100:5.1f}% | {stats['recall@5']*100:5.1f}% | {stats['recall@10']*100:5.1f}% | {stats['recall@20']*100:5.1f}% | {stats['ndcg@10']:7.4f}")

        # Breakdown by Difficulty
        print(f"\n--- BREAKDOWN BY DIFFICULTY ---")
        print(f"{'Difficulty':12} | {'Count':5} | {'MRR':6} | {'Hit@5':7} | {'R@5':7} | {'R@10':7} | {'R@20':7} | {'NDCG@10':7}")
        print("-" * 75)
        for diff, stats in results["breakdown_by_difficulty"].items():
            print(f"{diff:12} | {stats['count']:5} | {stats['mrr']:.4f} | {stats.get('hit_rate@5', 0)*100:5.1f}% | {stats['recall@5']*100:5.1f}% | {stats['recall@10']*100:5.1f}% | {stats['recall@20']*100:5.1f}% | {stats['ndcg@10']:7.4f}")

        # Breakdown by Paradigm
        print(f"\n--- BREAKDOWN BY RETRIEVAL PARADIGM ---")
        print(f"{'Paradigm':18} | {'Count':5} | {'MRR':6} | {'Hit@5':7} | {'R@5':7} | {'R@10':7} | {'R@20':7} | {'NDCG@10':7}")
        print("-" * 85)
        for para, stats in results["breakdown_by_paradigm"].items():
            print(f"{para:18} | {stats['count']:5} | {stats['mrr']:.4f} | {stats.get('hit_rate@5', 0)*100:5.1f}% | {stats['recall@5']*100:5.1f}% | {stats['recall@10']*100:5.1f}% | {stats['recall@20']*100:5.1f}% | {stats['ndcg@10']:7.4f}")

    print("\n" + "=" * 90)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 90)


if __name__ == "__main__":
    main()

