import math
from typing import List, Set, Dict, Any


class IRMetrics:
    """Information Retrieval metrics calculator: Precision@k, Recall@k, MRR, NDCG@k."""

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        if k <= 0:
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
    def reciprocal_rank(retrieved: List[str], relevant: List[str], max_k: int = 100) -> float:
        if not relevant:
            return 0.0
        rel_set = set(relevant)
        for rank, doc_id in enumerate(retrieved[:max_k], start=1):
            if doc_id in rel_set:
                return 1.0 / float(rank)
        return 0.0

    @staticmethod
    def dcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        if not relevant or k <= 0:
            return 0.0
        ret_k = retrieved[:k]
        rel_set = set(relevant)
        dcg = 0.0
        for rank, doc_id in enumerate(ret_k, start=1):
            if doc_id in rel_set:
                dcg += 1.0 / math.log2(rank + 1.0)
        return dcg

    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        if not relevant or k <= 0:
            return 0.0
        dcg = IRMetrics.dcg_at_k(retrieved, relevant, k)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
        if idcg <= 0.0:
            return 0.0
        return dcg / idcg

    @classmethod
    def evaluate_query(cls, retrieved: List[str], relevant: List[str], k_list: List[int] = [5, 10, 20]) -> Dict[str, float]:
        res = {
            "mrr": cls.reciprocal_rank(retrieved, relevant, max_k=max(k_list))
        }
        for k in k_list:
            res[f"precision@{k}"] = cls.precision_at_k(retrieved, relevant, k)
            res[f"recall@{k}"] = cls.recall_at_k(retrieved, relevant, k)
            res[f"ndcg@{k}"] = cls.ndcg_at_k(retrieved, relevant, k)
        return res
