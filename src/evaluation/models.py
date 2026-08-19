from __future__ import annotations
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DifficultyLevel(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class QueryType(str, Enum):
    PERSONA_APPLICABILITY = "Persona-Specific Applicability"
    TEMPORAL_VALIDATION = "Intra-Document Temporal Validation"
    PROCEDURAL_CONDITION = "Procedural / Condition Matching"
    NEGATIVE_OUT_OF_SCOPE = "Negative / Out-of-Scope"


class RetrievalParadigm(str, Enum):
    SIMPLE_SEMANTIC = "simple_semantic"
    MULTI_HOP = "multi_hop"
    HYBRID_TEMPORAL = "hybrid_temporal"
    NEGATIVE_NULL = "negative_null"


class GoldenEvaluationCase(BaseModel):
    id: str
    query: str
    query_type: QueryType
    retrieval_paradigm: RetrievalParadigm
    difficulty_level: DifficultyLevel
    target_regime: str  # "1962" | "2026" | "both" | "none"
    persona_context: Optional[str] = None
    fy_ay_context: Optional[str] = None
    temporal_constraint: Optional[str] = None
    relevant_rules: List[str] = Field(default_factory=list)
    ground_truth_chunk_ids: List[str] = Field(default_factory=list)
    ground_truth_answer: str
    evaluation_criteria: List[str] = Field(default_factory=list)
    split: str = "active"  # "active" | "hidden"


class EvaluationSuite(BaseModel):
    suite_name: str
    total_cases: int
    cases: List[GoldenEvaluationCase]
