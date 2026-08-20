import math
from typing import List, Set, Dict, Any, Optional


class IRMetrics:
    """Information Retrieval metrics calculator: Essential Recall@k, HitRate@k, Graded NDCG@k, Precision@k, MRR."""

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        if k <= 0 or not relevant:
            return 0.0
        ret_k = retrieved[:k]
        rel_set = set(relevant)
        hits = sum(1 for doc_id in ret_k if doc_id in rel_set)
        return hits / float(k)

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        if not relevant or k <= 0:
            return 0.0
        ret_k = retrieved[:k]
        rel_set = set(relevant)
        hits = sum(1 for doc_id in ret_k if doc_id in rel_set)
        return hits / float(len(relevant))

    @staticmethod
    def hit_rate_at_k(retrieved: List[str], essential: List[str], k: int) -> float:
        if not essential or k <= 0:
            return 0.0
        ret_k = retrieved[:k]
        ess_set = set(essential)
        return 1.0 if any(doc_id in ess_set for doc_id in ret_k) else 0.0

    @staticmethod
    def reciprocal_rank(retrieved: List[str], essential: List[str], max_k: int = 100) -> float:
        if not essential:
            return 0.0
        ess_set = set(essential)
        for rank, doc_id in enumerate(retrieved[:max_k], start=1):
            if doc_id in ess_set:
                return 1.0 / float(rank)
        return 0.0

    @staticmethod
    def graded_ndcg_at_k(
        retrieved: List[str],
        essential: List[str],
        supporting: Optional[List[str]] = None,
        k: int = 10
    ) -> float:
        """Computes Normalized Discounted Cumulative Gain with graded relevance (Essential=2, Supporting=1)."""
        if not essential and not supporting:
            return 0.0
        if k <= 0:
            return 0.0

        ess_set = set(essential or [])
        supp_set = set(supporting or []) - ess_set

        # Compute DCG
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved[:k], start=1):
            rel = 0.0
            if doc_id in ess_set:
                rel = 2.0
            elif doc_id in supp_set:
                rel = 1.0
            if rel > 0.0:
                dcg += (2.0 ** rel - 1.0) / math.log2(rank + 1.0)

        # Compute Ideal DCG (IDCG)
        ideal_rels = ([2.0] * len(ess_set) + [1.0] * len(supp_set))[:k]
        idcg = sum((2.0 ** rel - 1.0) / math.log2(rank + 1.0) for rank, rel in enumerate(ideal_rels, start=1))

        if idcg <= 0.0:
            return 0.0
        return dcg / idcg

    @classmethod
    def evaluate_query(
        cls,
        retrieved: List[str],
        essential_chunks: List[str],
        supporting_chunks: Optional[List[str]] = None,
        k_list: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        supporting_chunks = supporting_chunks or []
        all_relevant = list(set(essential_chunks + supporting_chunks))

        res = {
            "mrr": cls.reciprocal_rank(retrieved, essential_chunks if essential_chunks else all_relevant, max_k=max(k_list))
        }
        for k in k_list:
            res[f"hit_rate@{k}"] = cls.hit_rate_at_k(retrieved, essential_chunks if essential_chunks else all_relevant, k)
            res[f"essential_recall@{k}"] = cls.recall_at_k(retrieved, essential_chunks if essential_chunks else all_relevant, k)
            res[f"recall@{k}"] = cls.recall_at_k(retrieved, essential_chunks if essential_chunks else all_relevant, k)
            res[f"full_recall@{k}"] = cls.recall_at_k(retrieved, all_relevant, k)
            res[f"precision@{k}"] = cls.precision_at_k(retrieved, all_relevant, k)
            res[f"ndcg@{k}"] = cls.graded_ndcg_at_k(retrieved, essential_chunks, supporting_chunks, k)
        return res

