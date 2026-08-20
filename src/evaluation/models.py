from enum import Enum
from typing import Optional, List, Dict, Any
import re
from pydantic import BaseModel, Field, model_validator


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
    expected_fy: Optional[str] = None
    expected_ay: Optional[str] = None
    is_negative: bool = False
    relevant_rules: List[str] = Field(default_factory=list)
    essential_chunk_ids: List[str] = Field(default_factory=list)
    supporting_chunk_ids: List[str] = Field(default_factory=list)
    ground_truth_chunk_ids: List[str] = Field(default_factory=list)
    ground_truth_answer: str
    evaluation_criteria: List[str] = Field(default_factory=list)
    split: str = "active"  # "active" | "hidden"

    @model_validator(mode="after")
    def populate_defaults_and_sync(self) -> "GoldenEvaluationCase":
        # 1. Sync chunks: if essential not set but ground_truth_chunk_ids is set
        if not self.essential_chunk_ids and self.ground_truth_chunk_ids:
            # If large list (>3), first 2-3 are essential, rest are supporting
            if len(self.ground_truth_chunk_ids) > 3:
                self.essential_chunk_ids = self.ground_truth_chunk_ids[:2]
                self.supporting_chunk_ids = self.ground_truth_chunk_ids[2:]
            else:
                self.essential_chunk_ids = list(self.ground_truth_chunk_ids)
        elif self.essential_chunk_ids and not self.ground_truth_chunk_ids:
            self.ground_truth_chunk_ids = list(self.essential_chunk_ids) + list(self.supporting_chunk_ids)

        # 2. Negative detection
        if self.query_type == QueryType.NEGATIVE_OUT_OF_SCOPE or len(self.essential_chunk_ids) == 0:
            self.is_negative = True

        # 3. Derive expected AY / FY if omitted
        if not self.expected_fy or not self.expected_ay:
            text_to_check = f"{self.fy_ay_context or ''} {self.temporal_constraint or ''} {self.query}"
            fy_m = re.search(r"\b(?:FY|Financial\s+Year)\s*(\d{4})[-–/](\d{2,4})\b", text_to_check, re.IGNORECASE)
            ay_m = re.search(r"\b(?:AY|Assessment\s+Year)\s*(\d{4})[-–/](\d{2,4})\b", text_to_check, re.IGNORECASE)
            if fy_m and not self.expected_fy:
                y1 = int(fy_m.group(1))
                self.expected_fy = f"FY {y1}-{y1+1}"
                if not self.expected_ay:
                    self.expected_ay = f"AY {y1+1}-{y1+2}"
            if ay_m and not self.expected_ay:
                y1 = int(ay_m.group(1))
                self.expected_ay = f"AY {y1}-{y1+1}"
                if not self.expected_fy:
                    self.expected_fy = f"FY {y1-1}-{y1}"

        return self


class EvaluationSuite(BaseModel):
    suite_name: str
    total_cases: int
    cases: List[GoldenEvaluationCase]

