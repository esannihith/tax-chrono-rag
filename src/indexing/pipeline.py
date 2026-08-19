import json
from pathlib import Path
from typing import List
from src.models import Chunk
from src.indexing.payload_builder import PayloadBuilder, IndexPayload
from src.indexing.embedder import DenseEmbedder
from src.indexing.vector_store import HybridVectorStore


class IndexingPipeline:
    """End-to-end pipeline: load chunks -> build payloads -> embed -> index -> persist."""

    @classmethod
    def run(
        cls,
        processed_data_dir: str = "data/processed",
        indices_dir: str = "data/indices",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> HybridVectorStore:
        data_path = Path(processed_data_dir)
        all_chunks: List[Chunk] = []

        for year in ["1962", "2026"]:
            fpath = data_path / f"chunks_{year}.json"
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    chunk_dicts = json.load(f)
                    chunks = [Chunk.model_validate(cd) for cd in chunk_dicts]
                    all_chunks.extend(chunks)
                    print(f"Loaded {len(chunks)} chunks from {fpath}")

        print(f"Total chunks to index: {len(all_chunks)}")

        # 1. Build Payloads
        print("Preparing normalized dual-channel payloads...")
        payloads = PayloadBuilder.build_all(all_chunks)

        # 2. Generate Dense Embeddings
        embedder = DenseEmbedder(model_name=model_name)
        dense_texts = [p.dense_text for p in payloads]
        embeddings = embedder.embed_passages(dense_texts, batch_size=32)

        # 3. Build & Save Hybrid Vector Store
        store = HybridVectorStore()
        store.build_index(payloads, embeddings)
        store.save(indices_dir)

        return store
