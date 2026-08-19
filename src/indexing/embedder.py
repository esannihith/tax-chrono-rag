import os
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseEmbedder:
    """Generates dense vector embeddings using SentenceTransformers with batching & caching."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or "cpu"
        self.is_asymmetric = "bge" in model_name.lower() or "e5" in model_name.lower()
        print(f"Loading dense embedding model: {model_name} (asymmetric prefixes: {self.is_asymmetric})...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Dimension: {self.embedding_dim}")

    def embed_passages(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encodes a list of passage texts into normalized dense vectors."""
        formatted = [f"passage: {t}" if self.is_asymmetric else t for t in texts]
        embeddings = self.model.encode(
            formatted,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encodes a single query into a normalized dense vector."""
        formatted = f"query: {query}" if self.is_asymmetric else query
        emb = self.model.encode(
            [formatted],
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return np.array(emb[0], dtype=np.float32)
