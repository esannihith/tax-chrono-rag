from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class StatutoryCitation(BaseModel):
    """Represents a verified statutory citation from retrieved legal chunks."""
    rule_id: str
    sub_rule: Optional[str] = None
    statutory_path: str
    corpus_year: int
    sections_referenced: List[str] = Field(default_factory=list)
    forms_referenced: List[str] = Field(default_factory=list)
    effective_date: Optional[str] = None
    citation_text: str = ""


class RegimeDifference(BaseModel):
    """Captures key differences between the 1962 and 2026 tax regimes for comparative queries."""
    aspect: str
    rule_1962_provision: str
    rule_2026_provision: str
    key_change_summary: str


class GenerationInput(BaseModel):
    """Input payload to the generation prompter and synthesizer."""
    query: str
    target_regime: str = "auto"  # "1962", "2026", "both", or "auto"
    persona_context: Optional[str] = "Salaried individual"
    resolved_ay: Optional[str] = None
    resolved_fy: Optional[str] = None
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)


class GenerationOutput(BaseModel):
    """Structured Pydantic response from the RAG generation pipeline."""
    query: str
    direct_answer: str
    step_by_step_reasoning: List[str] = Field(default_factory=list)
    temporal_applicability: str
    statutory_citations: List[StatutoryCitation] = Field(default_factory=list)
    regime_differences: List[RegimeDifference] = Field(default_factory=list)
    is_out_of_scope: bool = False
    out_of_scope_reason: Optional[str] = None
    confidence_score: float = 1.0
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)
