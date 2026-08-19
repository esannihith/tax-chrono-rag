from typing import Optional
from src.generation.models import GenerationOutput
from src.generation.synthesizer import GenerationSynthesizer
from src.generation.llm_client import BaseLLMProvider, get_default_llm_provider
from src.indexing.vector_store import HybridVectorStore
from src.indexing.embedder import DenseEmbedder


class GenerationPipeline:
    """End-to-end RAG Generation Pipeline for Temporal Income Tax."""

    def __init__(
        self,
        indices_dir: str = "data/indices",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: Optional[BaseLLMProvider] = None
    ):
        self.vector_store = HybridVectorStore.load(indices_dir)
        self.embedder = DenseEmbedder(model_name=model_name)
        self.synthesizer = GenerationSynthesizer(
            vector_store=self.vector_store,
            embedder=self.embedder,
            llm_provider=llm_provider or get_default_llm_provider()
        )

    def query(
        self,
        query: str,
        target_regime: str = "auto",
        persona_context: str = "Salaried individual",
        top_k: int = 5
    ) -> GenerationOutput:
        """Queries the temporal RAG pipeline and returns a structured GenerationOutput."""
        return self.synthesizer.synthesize(
            query=query,
            target_regime=target_regime,
            persona_context=persona_context,
            top_k=top_k
        )
