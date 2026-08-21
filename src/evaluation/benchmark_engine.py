import json
import csv
import sys
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from src.evaluation.models import GoldenEvaluationCase
from src.evaluation.metrics import IRMetrics
from src.indexing.vector_store import HybridVectorStore
from src.indexing.embedder import DenseEmbedder
from src.enrichment.normalizer import TaxEntityNormalizer


class BenchmarkEngine:
    """Executes full evaluation across active and hidden suites, computes detailed IR metrics with provenance metadata."""

    def __init__(self, vector_store: HybridVectorStore, embedder: DenseEmbedder):
        self.store = vector_store
        self.embedder = embedder

    @staticmethod
    def get_git_commit_sha() -> str:
        try:
            res = subprocess.run(
                ["git", "describe", "--always", "--dirty"],
                capture_output=True,
                text=True,
                check=True
            )
            return res.stdout.strip()
        except Exception:
            try:
                res = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return res.stdout.strip()
            except Exception:
                return "uncommitted_or_non_git"

    @staticmethod
    def compute_file_sha256(filepath: str) -> str:
        p = Path(filepath)
        if not p.exists():
            return "file_not_found"
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

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
        grounded_cases = [c for c in cases if len(c.essential_chunk_ids) > 0 or len(c.ground_truth_chunk_ids) > 0]
        null_cases = [c for c in cases if len(c.essential_chunk_ids) == 0 and len(c.ground_truth_chunk_ids) == 0]

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
            essential_chunks = case.essential_chunk_ids or case.ground_truth_chunk_ids
            supporting_chunks = case.supporting_chunk_ids

            q_metrics = IRMetrics.evaluate_query(
                retrieved=retrieved_chunk_ids,
                essential_chunks=essential_chunks,
                supporting_chunks=supporting_chunks,
                k_list=k_list
            )
            
            # Find ranks of essential and relevant chunks
            retrieved_ranks = []
            for rank, cid in enumerate(retrieved_chunk_ids, start=1):
                if cid in essential_chunks or cid in supporting_chunks:
                    retrieved_ranks.append(rank)

            per_query_results.append({
                "case_id": case.id,
                "query": case.query,
                "query_type": case.query_type.value if hasattr(case.query_type, "value") else str(case.query_type),
                "difficulty_level": case.difficulty_level.value if hasattr(case.difficulty_level, "value") else str(case.difficulty_level),
                "retrieval_paradigm": case.retrieval_paradigm.value if hasattr(case.retrieval_paradigm, "value") else str(case.retrieval_paradigm),
                "target_regime": case.target_regime,
                "essential_chunk_count": len(essential_chunks),
                "supporting_chunk_count": len(supporting_chunks),
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "metrics": q_metrics,
                "first_relevant_rank": retrieved_ranks[0] if retrieved_ranks else -1
            })

        # Calculate Mean Aggregates
        mean_metrics = {}
        num_q = len(per_query_results)
        if num_q > 0:
            mean_metrics["mrr"] = sum(r["metrics"]["mrr"] for r in per_query_results) / num_q
            mean_metrics["r_precision"] = sum(r["metrics"].get("r_precision", 0) for r in per_query_results) / num_q
            mean_metrics["full_r_precision"] = sum(r["metrics"].get("full_r_precision", 0) for r in per_query_results) / num_q
            for k in k_list:
                mean_metrics[f"hit_rate@{k}"] = sum(r["metrics"][f"hit_rate@{k}"] for r in per_query_results) / num_q
                mean_metrics[f"essential_recall@{k}"] = sum(r["metrics"][f"essential_recall@{k}"] for r in per_query_results) / num_q
                mean_metrics[f"recall@{k}"] = sum(r["metrics"][f"recall@{k}"] for r in per_query_results) / num_q
                mean_metrics[f"full_recall@{k}"] = sum(r["metrics"][f"full_recall@{k}"] for r in per_query_results) / num_q
                mean_metrics[f"essential_precision@{k}"] = sum(r["metrics"][f"essential_precision@{k}"] for r in per_query_results) / num_q
                mean_metrics[f"target_bounded_precision@{k}"] = sum(r["metrics"].get(f"target_bounded_precision@{k}", 0) for r in per_query_results) / num_q
                mean_metrics[f"precision@{k}"] = sum(r["metrics"][f"precision@{k}"] for r in per_query_results) / num_q
                mean_metrics[f"full_precision@{k}"] = sum(r["metrics"][f"full_precision@{k}"] for r in per_query_results) / num_q
                mean_metrics[f"ndcg@{k}"] = sum(r["metrics"][f"ndcg@{k}"] for r in per_query_results) / num_q


        # Random Retriever Baseline calculation
        corpus_size = len(self.store.payloads) if self.store.payloads else 183
        avg_ess = float(np.mean([r["essential_chunk_count"] for r in per_query_results])) if num_q > 0 else 1.5
        avg_rel = float(np.mean([r["essential_chunk_count"] + r["supporting_chunk_count"] for r in per_query_results])) if num_q > 0 else 5.0
        random_baseline = IRMetrics.calculate_random_baseline(
            total_corpus_size=corpus_size,
            avg_essential_count=avg_ess,
            avg_relevant_count=avg_rel,
            k_list=k_list
        )

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
                "hit_rate@5": sum(rec["metrics"]["hit_rate@5"] for rec in records) / n,
                "hit_rate@10": sum(rec["metrics"]["hit_rate@10"] for rec in records) / n,
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
                "hit_rate@5": sum(rec["metrics"]["hit_rate@5"] for rec in records) / n,
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
                "hit_rate@5": sum(rec["metrics"]["hit_rate@5"] for rec in records) / n,
                "recall@5": sum(rec["metrics"]["recall@5"] for rec in records) / n,
                "recall@10": sum(rec["metrics"]["recall@10"] for rec in records) / n,
                "recall@20": sum(rec["metrics"]["recall@20"] for rec in records) / n,
                "ndcg@10": sum(rec["metrics"]["ndcg@10"] for rec in records) / n
            }

        return {
            "provenance": {
                "git_commit_sha": self.get_git_commit_sha(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "configuration": {
                    "corpus_size_chunks": corpus_size,
                    "alpha": alpha,
                    "dynamic_alpha_enabled": True,
                    "dense_embedding_model": getattr(self.embedder, "model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                    "rule_boost": 0.25 if apply_rule_boost else 0.0,
                    "k_list": k_list,
                    "apply_normalization": apply_normalization
                },
                "suite_file": suite_path,
                "suite_sha256": self.compute_file_sha256(suite_path)
            },
            "suite_file": suite_path,
            "total_cases": len(cases),
            "grounded_cases": num_q,
            "null_cases": len(null_cases),
            "corpus_size": corpus_size,
            "mean_metrics": mean_metrics,
            "random_baseline": random_baseline,
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
            "case_id", "query_type", "difficulty_level", "retrieval_paradigm", "target_regime",
            "essential_chunk_count", "supporting_chunk_count", "first_relevant_rank",
            "mrr", "hit_rate@5", "hit_rate@10", "hit_rate@20",
            "essential_recall@5", "essential_recall@10", "essential_recall@20",
            "essential_precision@5", "essential_precision@10",
            "full_precision@5", "full_precision@10", "full_precision@20",
            "ndcg@5", "ndcg@10", "ndcg@20"
        ]

        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in per_query_results:
                m = r["metrics"]
                writer.writerow([
                    r["case_id"],
                    r["query_type"],
                    r["difficulty_level"],
                    r["retrieval_paradigm"],
                    r["target_regime"],
                    r["essential_chunk_count"],
                    r["supporting_chunk_count"],
                    r["first_relevant_rank"],
                    round(m.get("mrr", 0.0), 4),
                    round(m.get("hit_rate@5", 0.0), 4),
                    round(m.get("hit_rate@10", 0.0), 4),
                    round(m.get("hit_rate@20", 0.0), 4),
                    round(m.get("essential_recall@5", 0.0), 4),
                    round(m.get("essential_recall@10", 0.0), 4),
                    round(m.get("essential_recall@20", 0.0), 4),
                    round(m.get("essential_precision@5", 0.0), 4),
                    round(m.get("essential_precision@10", 0.0), 4),
                    round(m.get("full_precision@5", 0.0), 4),
                    round(m.get("full_precision@10", 0.0), 4),
                    round(m.get("full_precision@20", 0.0), 4),
                    round(m.get("ndcg@5", 0.0), 4),
                    round(m.get("ndcg@10", 0.0), 4),
                    round(m.get("ndcg@20", 0.0), 4)
                ])
        print(f"Exported per-query CSV to {output_file}")

