from typing import List, Dict, Any, Optional
from src.generation.models import GenerationInput, GenerationOutput, StatutoryCitation, RegimeDifference
from src.generation.prompter import StatutoryPrompter
from src.generation.llm_client import BaseLLMProvider, get_default_llm_provider
from src.indexing.vector_store import HybridVectorStore
from src.indexing.embedder import DenseEmbedder
from src.enrichment.normalizer import TaxEntityNormalizer


class GenerationSynthesizer:
    """Coordinates dual-stream retrieval, temporal resolution, prompt compilation, and structured output parsing."""

    def __init__(
        self,
        vector_store: HybridVectorStore,
        embedder: DenseEmbedder,
        llm_provider: Optional[BaseLLMProvider] = None
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_provider = llm_provider or get_default_llm_provider()

    def synthesize(
        self,
        query: str,
        target_regime: str = "auto",
        persona_context: str = "Salaried individual",
        top_k: int = 5
    ) -> GenerationOutput:
        # Step 1: Normalize query and extract temporal/rule anchors
        norm_q = TaxEntityNormalizer.normalize_query(query)

        # Detect regime if auto
        regime = target_regime
        if regime == "auto":
            if "2026" in query:
                regime = "2026"
            elif "1962" in query:
                regime = "1962"
            else:
                regime = "both"

        # Step 2: Retrieve relevant statutory chunks
        q_vec = self.embedder.embed_query(norm_q.dense_query)
        
        # If comparative or both, perform dual-stream retrieval for 1962 and 2026
        if regime == "both":
            hits_1962 = self.vector_store.hybrid_search(
                query_vector=q_vec,
                sparse_tokens=norm_q.sparse_tokens,
                top_k=max(2, top_k // 2),
                alpha=0.5,
                target_rules=norm_q.target_rules,
                corpus_year_filter=1962
            )
            hits_2026 = self.vector_store.hybrid_search(
                query_vector=q_vec,
                sparse_tokens=norm_q.sparse_tokens,
                top_k=max(2, top_k // 2),
                alpha=0.5,
                target_rules=norm_q.target_rules,
                corpus_year_filter=2026
            )
            hits = hits_1962 + hits_2026
        else:
            c_filter = int(regime) if regime in ["1962", "2026"] else None
            hits = self.vector_store.hybrid_search(
                query_vector=q_vec,
                sparse_tokens=norm_q.sparse_tokens,
                top_k=top_k,
                alpha=0.5,
                target_rules=norm_q.target_rules,
                corpus_year_filter=c_filter
            )

        retrieved_chunk_dicts = []
        for h in hits:
            retrieved_chunk_dicts.append({
                "chunk_id": h.chunk_id,
                "score": h.score,
                "breadcrumb": f"[{h.payload.corpus_year} > {h.payload.statutory_path}]",
                "content": h.payload.display_content or h.payload.dense_text,
                "metadata": h.payload.metadata
            })

        # Step 3: Build Generation Input
        gen_input = GenerationInput(
            query=query,
            target_regime=regime,
            persona_context=persona_context,
            resolved_ay=norm_q.detected_ay,
            resolved_fy=norm_q.detected_fy,
            retrieved_chunks=retrieved_chunk_dicts
        )

        # Step 4: Prompt Construction
        prompt = StatutoryPrompter.build_generation_prompt(gen_input)

        # Step 5: LLM Generation
        raw_res = self.llm_provider.generate(
            prompt=prompt,
            system_instruction=StatutoryPrompter.SYSTEM_INSTRUCTION
        )

        # Step 6: Parse into GenerationOutput
        citations = []
        for c in raw_res.get("statutory_citations", []):
            try:
                citations.append(StatutoryCitation(**c))
            except Exception:
                pass

        differences = []
        for d in raw_res.get("regime_differences", []):
            try:
                differences.append(RegimeDifference(**d))
            except Exception:
                pass

        output = GenerationOutput(
            query=query,
            direct_answer=raw_res.get("direct_answer", ""),
            step_by_step_reasoning=raw_res.get("step_by_step_reasoning", []),
            temporal_applicability=raw_res.get("temporal_applicability", ""),
            statutory_citations=citations,
            regime_differences=differences,
            is_out_of_scope=raw_res.get("is_out_of_scope", False),
            out_of_scope_reason=raw_res.get("out_of_scope_reason"),
            confidence_score=raw_res.get("confidence_score", 1.0),
            retrieval_metadata={
                "normalized_query": norm_q.dense_query,
                "target_regime": regime,
                "resolved_ay": norm_q.detected_ay,
                "resolved_fy": norm_q.detected_fy,
                "retrieved_chunk_ids": [h.chunk_id for h in hits]
            }
        )

        return output
