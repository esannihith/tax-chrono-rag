import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from src.evaluation.models import GoldenEvaluationCase
from src.evaluation.metrics import IRMetrics
from src.indexing.vector_store import HybridVectorStore
from src.indexing.embedder import DenseEmbedder
from src.enrichment.normalizer import TaxEntityNormalizer


class BenchmarkEngine:
    """Executes full evaluation across active and hidden suites, computes detailed IR metrics, and logs per-query results."""

    def __init__(self, vector_store: HybridVectorStore, embedder: DenseEmbedder):
        self.store = vector_store
        self.embedder = embedder

    def run_evaluation(
        self,
        suite_path: str,
        k_list: List[int] = [5, 10, 20],
        alpha: float = 0.5,
        apply_normalization: bool = True,
        apply_rule_boost: bool = True
    ) -> Dict[str, Any]:
        with open(suite_path, "r", encoding="utf-8") as f:
            raw_cases = json.load(f)

        cases = [GoldenEvaluationCase.model_validate(c) for c in raw_cases]
        grounded_cases = [c for c in cases if len(c.ground_truth_chunk_ids) > 0]
        null_cases = [c for c in cases if len(c.ground_truth_chunk_ids) == 0]

        per_query_results = []
        max_k = max(k_list)

        for case in grounded_cases:
            query_str = case.query
            if apply_normalization:
                norm_q = TaxEntityNormalizer.normalize_query(query_str)
                q_vec = self.embedder.embed_query(norm_q.dense_query)
                sparse_tokens = norm_q.sparse_tokens
                target_rules = norm_q.target_rules if apply_rule_boost else None
            else:
                q_vec = self.embedder.embed_query(query_str)
                sparse_tokens = query_str.split()
                target_rules = None

            search_hits = self.store.hybrid_search(
                query_vector=q_vec,
                sparse_tokens=sparse_tokens,
                top_k=max_k,
                alpha=alpha,
                target_rules=target_rules
            )
            retrieved_chunk_ids = [h.chunk_id for h in search_hits]
            relevant_chunk_ids = case.ground_truth_chunk_ids

            q_metrics = IRMetrics.evaluate_query(retrieved_chunk_ids, relevant_chunk_ids, k_list=k_list)
            
            # Find ranks of relevant chunks
            retrieved_ranks = []
            for rank, cid in enumerate(retrieved_chunk_ids, start=1):
                if cid in relevant_chunk_ids:
                    retrieved_ranks.append(rank)

            query_record = {
                "id": case.id,
                "query": case.query,
                "query_type": case.query_type.value,
                "retrieval_paradigm": case.retrieval_paradigm.value,
                "difficulty_level": case.difficulty_level.value,
                "target_regime": case.target_regime,
                "relevant_rules": case.relevant_rules,
                "num_gold_chunks": len(relevant_chunk_ids),
                "gold_chunks": relevant_chunk_ids,
                "retrieved_ranks": retrieved_ranks,
                "first_hit_rank": retrieved_ranks[0] if retrieved_ranks else None,
                "metrics": q_metrics
            }
            per_query_results.append(query_record)

        # Aggregate metrics
        num_q = len(per_query_results)
        mean_metrics = {}
        if num_q > 0:
            mean_metrics["mrr"] = sum(r["metrics"]["mrr"] for r in per_query_results) / num_q
            for k in k_list:
                mean_metrics[f"recall@{k}"] = sum(r["metrics"][f"recall@{k}"] for r in per_query_results) / num_q
                mean_metrics[f"precision@{k}"] = sum(r["metrics"][f"precision@{k}"] for r in per_query_results) / num_q
                mean_metrics[f"ndcg@{k}"] = sum(r["metrics"][f"ndcg@{k}"] for r in per_query_results) / num_q

        # Breakdown by Query Type
        by_query_type = {}
        for r in per_query_results:
            qt = r["query_type"]
            by_query_type.setdefault(qt, []).append(r)

        breakdown_qt = {}
        for qt, records in by_query_type.items():
            n = len(records)
            breakdown_qt[qt] = {
                "count": n,
                "mrr": sum(rec["metrics"]["mrr"] for rec in records) / n,
                "recall@5": sum(rec["metrics"]["recall@5"] for rec in records) / n,
                "recall@10": sum(rec["metrics"]["recall@10"] for rec in records) / n,
                "recall@20": sum(rec["metrics"]["recall@20"] for rec in records) / n,
                "ndcg@10": sum(rec["metrics"]["ndcg@10"] for rec in records) / n
            }

        # Breakdown by Difficulty
        by_diff = {}
        for r in per_query_results:
            diff = r["difficulty_level"]
            by_diff.setdefault(diff, []).append(r)

        breakdown_diff = {}
        for diff, records in by_diff.items():
            n = len(records)
            breakdown_diff[diff] = {
                "count": n,
                "mrr": sum(rec["metrics"]["mrr"] for rec in records) / n,
                "recall@5": sum(rec["metrics"]["recall@5"] for rec in records) / n,
                "recall@10": sum(rec["metrics"]["recall@10"] for rec in records) / n,
                "recall@20": sum(rec["metrics"]["recall@20"] for rec in records) / n,
                "ndcg@10": sum(rec["metrics"]["ndcg@10"] for rec in records) / n
            }

        # Breakdown by Retrieval Paradigm
        by_para = {}
        for r in per_query_results:
            para = r["retrieval_paradigm"]
            by_para.setdefault(para, []).append(r)

        breakdown_para = {}
        for para, records in by_para.items():
            n = len(records)
            breakdown_para[para] = {
                "count": n,
                "mrr": sum(rec["metrics"]["mrr"] for rec in records) / n,
                "recall@5": sum(rec["metrics"]["recall@5"] for rec in records) / n,
                "recall@10": sum(rec["metrics"]["recall@10"] for rec in records) / n,
                "recall@20": sum(rec["metrics"]["recall@20"] for rec in records) / n,
                "ndcg@10": sum(rec["metrics"]["ndcg@10"] for rec in records) / n
            }

        return {
            "suite_file": suite_path,
            "total_cases": len(cases),
            "grounded_cases": num_q,
            "null_cases": len(null_cases),
            "mean_metrics": mean_metrics,
            "breakdown_by_query_type": breakdown_qt,
            "breakdown_by_difficulty": breakdown_diff,
            "breakdown_by_paradigm": breakdown_para,
            "per_query_results": per_query_results
        }

    @staticmethod
    def export_per_query_csv(per_query_results: List[Dict[str, Any]], output_file: str):
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        
        headers = [
            "id", "query_type", "retrieval_paradigm", "difficulty_level", "target_regime",
            "mrr", "recall@5", "recall@10", "recall@20",
            "precision@5", "precision@10", "precision@20",
            "ndcg@5", "ndcg@10", "ndcg@20", "first_hit_rank", "query"
        ]
        
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in per_query_results:
                row = {
                    "id": r["id"],
                    "query_type": r["query_type"],
                    "retrieval_paradigm": r["retrieval_paradigm"],
                    "difficulty_level": r["difficulty_level"],
                    "target_regime": r["target_regime"],
                    "mrr": f"{r['metrics']['mrr']:.4f}",
                    "recall@5": f"{r['metrics']['recall@5']:.4f}",
                    "recall@10": f"{r['metrics']['recall@10']:.4f}",
                    "recall@20": f"{r['metrics']['recall@20']:.4f}",
                    "precision@5": f"{r['metrics']['precision@5']:.4f}",
                    "precision@10": f"{r['metrics']['precision@10']:.4f}",
                    "precision@20": f"{r['metrics']['precision@20']:.4f}",
                    "ndcg@5": f"{r['metrics']['ndcg@5']:.4f}",
                    "ndcg@10": f"{r['metrics']['ndcg@10']:.4f}",
                    "ndcg@20": f"{r['metrics']['ndcg@20']:.4f}",
                    "first_hit_rank": r["first_hit_rank"] if r["first_hit_rank"] else "",
                    "query": r["query"]
                }
                writer.writerow(row)
        print(f"Exported per-query CSV to {output_file}")
