import json
import sys
from pathlib import Path
from src.indexing.pipeline import IndexingPipeline
from src.indexing.vector_store import HybridVectorStore
from src.indexing.embedder import DenseEmbedder
from src.enrichment.normalizer import TaxEntityNormalizer

def run_indexing_and_verification():
    print("=" * 80)
    print("STARTING TEMPORAL-RAG CHUNKING -> EMBEDDING & INDEXING PIPELINE")
    print("=" * 80)

    # 1. Run Pipeline
    store = IndexingPipeline.run(
        processed_data_dir="data/processed",
        indices_dir="data/indices",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Verification on Golden Dataset
    print("\n" + "=" * 80)
    print("RUNNING RETRIEVAL VERIFICATION AGAINST ACTIVE & HIDDEN SUITES")
    print("=" * 80)

    embedder = DenseEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    for suite_name in ["active_suite", "hidden_suite"]:
        suite_file = Path(f"data/evaluation/{suite_name}.json")
        with open(suite_file, "r", encoding="utf-8") as f:
            cases = json.load(f)

        hits_at_1 = 0
        hits_at_5 = 0
        hits_at_10 = 0
        total_eval_cases = 0

        for c in cases:
            gold_chunks = c.get("ground_truth_chunk_ids", [])
            if not gold_chunks:
                continue

            total_eval_cases += 1
            raw_query = c["query"]
            
            norm_query = TaxEntityNormalizer.normalize_query(raw_query)
            q_vec = embedder.embed_query(norm_query.dense_query)

            results = store.hybrid_search(
                query_vector=q_vec,
                sparse_tokens=norm_query.sparse_tokens,
                top_k=10,
                alpha=0.5,
                target_rules=norm_query.target_rules
            )
            retrieved_ids = [r.chunk_id for r in results]

            hit_1 = any(gid in retrieved_ids[:1] for gid in gold_chunks)
            hit_5 = any(gid in retrieved_ids[:5] for gid in gold_chunks)
            hit_10 = any(gid in retrieved_ids[:10] for gid in gold_chunks)

            if hit_1: hits_at_1 += 1
            if hit_5: hits_at_5 += 1
            if hit_10: hits_at_10 += 1

        recall_at_1 = (hits_at_1 / total_eval_cases) * 100 if total_eval_cases else 0
        recall_at_5 = (hits_at_5 / total_eval_cases) * 100 if total_eval_cases else 0
        recall_at_10 = (hits_at_10 / total_eval_cases) * 100 if total_eval_cases else 0

        print(f"\n--- {suite_name.upper()} RESULTS ---")
        print(f"Total Grounded Cases       : {total_eval_cases}")
        print(f"Recall@1 (Top-1 Hit Rate)  : {recall_at_1:.2f}% ({hits_at_1}/{total_eval_cases})")
        print(f"Recall@5 (Top-5 Hit Rate)  : {recall_at_5:.2f}% ({hits_at_5}/{total_eval_cases})")
        print(f"Recall@10 (Top-10 Hit Rate): {recall_at_10:.2f}% ({hits_at_10}/{total_eval_cases})")

    print("\n" + "=" * 80)
    print("INDEXING & VERIFICATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    run_indexing_and_verification()
