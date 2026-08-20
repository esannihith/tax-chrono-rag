import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from src.indexing.payload_builder import IndexPayload

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SearchResult:
    def __init__(self, chunk_id: str, score: float, payload: IndexPayload):
        self.chunk_id = chunk_id
        self.score = score
        self.payload = payload

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": float(self.score),
            "corpus_year": self.payload.corpus_year,
            "rule_id": self.payload.rule_id,
            "statutory_path": self.payload.statutory_path,
            "display_content": self.payload.display_content,
            "metadata": self.payload.metadata
        }


class HybridVectorStore:
    """Dual-channel Hybrid Index combining Min-Max Normalized Dense Cosine Similarity with Min-Max Normalized BM25."""

    def __init__(self):
        self.payloads: List[IndexPayload] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
        self.dense_embeddings: Optional[np.ndarray] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        cleaned = re.sub(r"[^a-zA-Z0-9\(\)]", " ", text)
        return [t.lower() for t in cleaned.split() if len(t) > 1]

    def build_index(self, payloads: List[IndexPayload], dense_embeddings: np.ndarray):
        self.payloads = payloads
        self.chunk_id_to_idx = {p.chunk_id: i for i, p in enumerate(payloads)}
        self.dense_embeddings = dense_embeddings

        print("Building BM25 sparse index with regex tokenization...")
        self.tokenized_corpus = [self.tokenize(p.sparse_text) for p in payloads]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        print(f"Hybrid index built with {len(payloads)} items.")

    def dense_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        corpus_year_filter: Optional[int] = None,
        rule_filter: Optional[str] = None
    ) -> List[Tuple[int, float]]:
        if self.dense_embeddings is None:
            return []

        scores = np.dot(self.dense_embeddings, query_vector)
        valid_indices = []
        for idx, p in enumerate(self.payloads):
            if corpus_year_filter and p.corpus_year != corpus_year_filter:
                continue
            if rule_filter and p.rule_id.lower() != rule_filter.lower():
                continue
            valid_indices.append((idx, float(scores[idx])))

        valid_indices.sort(key=lambda x: x[1], reverse=True)
        return valid_indices[:top_k]

    def sparse_search(
        self,
        sparse_tokens: List[str],
        top_k: int = 10,
        corpus_year_filter: Optional[int] = None,
        rule_filter: Optional[str] = None
    ) -> List[Tuple[int, float]]:
        if self.bm25_index is None:
            return []

        tokens = [t.lower() for t in sparse_tokens if len(t) > 1]
        scores = self.bm25_index.get_scores(tokens)

        valid_indices = []
        for idx, p in enumerate(self.payloads):
            if corpus_year_filter and p.corpus_year != corpus_year_filter:
                continue
            if rule_filter and p.rule_id.lower() != rule_filter.lower():
                continue
            if scores[idx] > 0:
                valid_indices.append((idx, float(scores[idx])))

        valid_indices.sort(key=lambda x: x[1], reverse=True)
        return valid_indices[:top_k]

    def hybrid_search(
        self,
        query_vector: np.ndarray,
        sparse_tokens: List[str],
        top_k: int = 10,
        alpha: float = 0.5,
        target_rules: Optional[List[str]] = None,
        corpus_year_filter: Optional[int] = None,
        rule_filter: Optional[str] = None
    ) -> List[SearchResult]:
        if self.dense_embeddings is None or self.bm25_index is None:
            return []

        # 1. Dense Cosine Scores & Min-Max Normalization
        raw_dense = np.dot(self.dense_embeddings, query_vector)
        d_min, d_max = float(raw_dense.min()), float(raw_dense.max())
        dense_norm = (raw_dense - d_min) / (d_max - d_min + 1e-8) if (d_max - d_min) > 0 else raw_dense

        # 2. BM25 Sparse Scores & Min-Max Normalization
        tokens = [t.lower() for t in sparse_tokens if len(t) > 1]
        raw_bm25 = np.array(self.bm25_index.get_scores(tokens))
        s_min, s_max = float(raw_bm25.min()), float(raw_bm25.max())
        sparse_norm = (raw_bm25 - s_min) / (s_max - s_min + 1e-8) if (s_max - s_min) > 0 else raw_bm25

        # 3. Dynamic Alpha Shifting & Target Rule Boost
        rule_boost = np.zeros(len(self.payloads))
        effective_alpha = alpha

        if target_rules:
            # Dynamic Alpha Shift: Prioritize exact sparse rule tokens when explicit rule numbers are detected
            effective_alpha = min(alpha, 0.25)
            for tr in target_rules:
                for idx, p in enumerate(self.payloads):
                    if p.rule_id.lower() == tr.lower():
                        rule_boost[idx] = 0.25

        # 4. Combined Hybrid Formula
        combined_scores = (effective_alpha * dense_norm) + ((1.0 - effective_alpha) * sparse_norm) + rule_boost

        valid_indices = []
        for idx, p in enumerate(self.payloads):
            if corpus_year_filter and p.corpus_year != corpus_year_filter:
                continue
            if rule_filter and p.rule_id.lower() != rule_filter.lower():
                continue
            valid_indices.append((idx, float(combined_scores[idx])))

        valid_indices.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, fused_score in valid_indices[:top_k]:
            results.append(SearchResult(
                chunk_id=self.payloads[idx].chunk_id,
                score=fused_score,
                payload=self.payloads[idx]
            ))
        return results

    def save(self, index_dir: str):
        out_dir = Path(index_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "payloads.json", "w", encoding="utf-8") as f:
            json.dump([p.model_dump() for p in self.payloads], f, indent=2)

        if self.dense_embeddings is not None:
            np.save(out_dir / "dense_embeddings.npy", self.dense_embeddings)

        with open(out_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump(self.bm25_index, f)

        print(f"Hybrid vector store saved to {out_dir}")

    @classmethod
    def load(cls, index_dir: str = "data/indices") -> "HybridVectorStore":
        store = cls()
        in_dir = Path(index_dir)
        if not in_dir.is_absolute():
            # Try relative to cwd first, then fallback to PROJECT_ROOT
            if not in_dir.exists():
                in_dir = PROJECT_ROOT / in_dir

        with open(in_dir / "payloads.json", "r", encoding="utf-8") as f:
            payload_dicts = json.load(f)
            store.payloads = [IndexPayload.model_validate(d) for d in payload_dicts]
            store.chunk_id_to_idx = {p.chunk_id: i for i, p in enumerate(store.payloads)}

        dense_path = in_dir / "dense_embeddings.npy"
        if dense_path.exists():
            store.dense_embeddings = np.load(dense_path)

        bm25_path = in_dir / "bm25_index.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                store.bm25_index = pickle.load(f)

        print(f"Hybrid vector store loaded from {in_dir} ({len(store.payloads)} items).")
        return store

