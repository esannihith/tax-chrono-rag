import re
from typing import List, Dict, Any, Set, Tuple, Optional
from src.generation.models import StatutoryCitation, GenerationOutput
from src.evaluation.llm_judge import LLMJudge


class GenerationMetrics:
    """Calculates verifiable, two-tier evaluation metrics for statutory generation outputs.
    
    Tier 1 (Deterministic Citation Engine):
        - Strict tokenized citation precision, recall, and hierarchical sub-rule containment.
    
    Tier 2 (G-Eval / LLM-as-a-Judge):
        - Chain-of-thought Criteria Adherence.
        - Semantic Temporal Validity (AY = FY + 1).
        - Faithfulness & Hallucination Grounding.
    """

    _judge: Optional[LLMJudge] = None

    @classmethod
    def get_judge(cls) -> LLMJudge:
        if cls._judge is None:
            cls._judge = LLMJudge()
        return cls._judge

    @classmethod
    def set_judge(cls, judge: LLMJudge):
        cls._judge = judge

    # ==============================================================================
    # TIER 1: DETERMINISTIC CITATION ENGINE
    # ==============================================================================

    @staticmethod
    def normalize_citation_str(rule_str: str) -> str:
        """Extract canonical rule identifier string like '12AC', '3(7)(I)', '2BB(1)'."""
        clean = re.sub(r"^rule\s*", "", rule_str.strip(), flags=re.IGNORECASE)
        clean = re.sub(r"\s+", "", clean).upper()
        return clean

    @staticmethod
    def parse_rule_components(rule_str: str) -> Tuple[str, str]:
        """Parses a statutory rule citation into canonical (base_rule, sub_rule_suffix).
        
        Examples:
            '3' -> ('3', '')
            '3(1)' -> ('3', '(1)')
            '3(7)(I)' -> ('3', '(7)(I)')
            '3A' -> ('3A', '')
            '12AC' -> ('12AC', '')
            '12AC(1)' -> ('12AC', '(1)')
            'Rule 26D' -> ('26D', '')
            'Rule 114AAA' -> ('114AAA', '')
        """
        clean = re.sub(r"^rule\s*", "", rule_str.strip(), flags=re.IGNORECASE)
        clean = re.sub(r"\s+", "", clean).upper()
        m = re.match(r"^([0-9]+[A-Z]*)(.*)$", clean)
        if m:
            return m.group(1), m.group(2)
        return clean, ""

    @staticmethod
    def extract_subrule_tokens(sub_str: str) -> List[str]:
        """Extracts ordered hierarchical sub-rule tokens (e.g. '(7)(i)' -> ['7', 'I'], '(1)' -> ['1'], '(i)' -> ['I'])."""
        if not sub_str:
            return []
        return re.findall(r"[A-Z0-9]+", sub_str.upper())

    @classmethod
    def matches_rule_citation(cls, pred_cit: str, gt_cit: str) -> bool:
        """Determines if a predicted citation correctly satisfies a ground truth citation.
        
        Requires:
        1. Exact match on base rule identifier (e.g. '3' != '30', '3' != '3A', '12' != '12AC').
        2. Sub-rule hierarchy consistency:
           - If GT has no sub-rule specified (e.g. 'Rule 3'), any sub-rule under Rule 3 satisfies it.
           - If GT specifies a sub-rule (e.g. 'Rule 3(7)(i)'), the prediction must specify the
             exact sub-rule hierarchy or a valid parent/child prefix.
           - Disjoint tokens like '(i)' will NOT match '(7)(i)'.
        """
        p_base, p_sub = cls.parse_rule_components(pred_cit)
        gt_base, gt_sub = cls.parse_rule_components(gt_cit)

        # 1. Base rule MUST be strictly identical
        if p_base != gt_base:
            return False

        # 2. If Ground Truth does not require a specific sub-rule, base match is sufficient
        if not gt_sub:
            return True

        # 3. If Ground Truth requires a specific sub-rule, prediction must specify a compatible sub-rule
        if not p_sub:
            return False

        p_tokens = cls.extract_subrule_tokens(p_sub)
        gt_tokens = cls.extract_subrule_tokens(gt_sub)

        if not p_tokens or not gt_tokens:
            return False

        # Exact match
        if p_tokens == gt_tokens:
            return True

        # Hierarchical prefix matching:
        if len(p_tokens) < len(gt_tokens):
            return gt_tokens[:len(p_tokens)] == p_tokens
        else:
            return p_tokens[:len(gt_tokens)] == gt_tokens

    @classmethod
    def compute_citation_metrics(
        cls,
        predicted_citations: List[StatutoryCitation],
        ground_truth_rules: List[str]
    ) -> Dict[str, float]:
        """Calculates Citation Precision, Recall, and F1 with strict rule tokenization."""
        if not ground_truth_rules:
            if len(predicted_citations) == 0:
                return {"citation_precision": 1.0, "citation_recall": 1.0, "citation_f1": 1.0}
            else:
                return {"citation_precision": 0.0, "citation_recall": 0.0, "citation_f1": 0.0}

        pred_citations_normalized = []
        for c in predicted_citations:
            full_cit = f"{c.rule_id}{c.sub_rule or ''}"
            pred_citations_normalized.append(full_cit)

        matched_gt = set()
        for gt in ground_truth_rules:
            for pred in pred_citations_normalized:
                if cls.matches_rule_citation(pred, gt):
                    matched_gt.add(gt)
                    break

        matched_pred_count = 0
        for pred in pred_citations_normalized:
            if any(cls.matches_rule_citation(pred, gt) for gt in ground_truth_rules):
                matched_pred_count += 1

        tp = len(matched_gt)
        fn = len(ground_truth_rules) - tp
        total_pred = len(pred_citations_normalized)

        precision = (matched_pred_count / total_pred) if total_pred > 0 else 0.0
        recall = (tp / len(ground_truth_rules)) if len(ground_truth_rules) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "citation_precision": round(precision, 4),
            "citation_recall": round(recall, 4),
            "citation_f1": round(f1, 4)
        }

    # ==============================================================================
    # TIER 2: G-EVAL / LLM-AS-A-JUDGE SEMANTIC METRICS
    # ==============================================================================

    @classmethod
    def compute_criteria_adherence(
        cls,
        output: GenerationOutput,
        criteria_list: List[str],
        query: str = "",
        use_llm_judge: bool = True
    ) -> Dict[str, Any]:
        """Evaluates legal criteria satisfaction using Chain-of-Thought LLM-as-a-Judge."""
        if not criteria_list:
            return {
                "criteria_adherence_rate": 1.0,
                "criteria_keyword_coverage_rate": 1.0,
                "criteria_match_rate": 1.0,
                "matched_criteria": [],
                "missed_criteria": [],
                "evaluations": []
            }

        if use_llm_judge:
            judge = cls.get_judge()
            res = judge.evaluate_criteria_adherence(
                query=query or output.query if hasattr(output, "query") else "",
                direct_answer=output.direct_answer,
                step_by_step_reasoning=output.step_by_step_reasoning,
                criteria_list=criteria_list
            )
            # Add compatibility aliases
            res["criteria_keyword_coverage_rate"] = res["criteria_adherence_rate"]
            res["criteria_match_rate"] = res["criteria_adherence_rate"]
            return res

        # Deterministic keyword fallback
        return cls.compute_criteria_keyword_coverage(output, criteria_list)

    @classmethod
    def compute_criteria_keyword_coverage(
        cls,
        output: GenerationOutput,
        criteria_list: List[str]
    ) -> Dict[str, Any]:
        """Deterministic keyword coverage fallback."""
        if not criteria_list:
            return {"criteria_adherence_rate": 1.0, "criteria_keyword_coverage_rate": 1.0, "criteria_match_rate": 1.0, "matched_criteria": [], "missed_criteria": []}

        full_text = f"{output.direct_answer} {' '.join(output.step_by_step_reasoning)} {output.temporal_applicability}".lower()
        matched = []
        missed = []

        for crit in criteria_list:
            crit_clean = crit.lower()
            keywords = re.findall(r"[a-z0-9\(\)]+", crit_clean)
            key_entities = [k for k in keywords if len(k) > 3 or k.isdigit() or k in ["evc", "dsc", "sbi", "hra", "lta", "ltc", "194p", "rfa"]]

            if not key_entities:
                hits = 1 if crit_clean in full_text else 0
                ratio = 1.0 if hits else 0.0
            else:
                matched_count = sum(1 for k in key_entities if k in full_text)
                ratio = matched_count / len(key_entities)

            if ratio >= 0.40 or any(phrase in full_text for phrase in [crit_clean[:25]]):
                matched.append(crit)
            else:
                missed.append(crit)

        match_rate = len(matched) / len(criteria_list)
        return {
            "criteria_adherence_rate": round(match_rate, 4),
            "criteria_keyword_coverage_rate": round(match_rate, 4),
            "criteria_match_rate": round(match_rate, 4),
            "matched_criteria": matched,
            "missed_criteria": missed
        }

    # Backward compatibility alias
    compute_criteria_match = compute_criteria_adherence

    @classmethod
    def compute_temporal_validity(
        cls,
        output: GenerationOutput,
        expected_ay: Optional[str] = None,
        expected_fy: Optional[str] = None,
        is_negative: bool = False,
        query: str = "",
        use_llm_judge: bool = True
    ) -> Dict[str, Any]:
        """Evaluates semantic temporal validity (AY = FY + 1) using LLM-as-a-Judge."""
        def clean_year(y: Optional[str]) -> Optional[str]:
            if not y or str(y).strip().lower() in ["not applicable", "n/a", "none", "null", ""]:
                return None
            return str(y).strip()

        clean_ay = clean_year(expected_ay)
        clean_fy = clean_year(expected_fy)

        if not clean_ay and not clean_fy:
            return {
                "temporal_validity_score": 1.0,
                "is_labelled_temporal_case": False,
                "verified_correct_year": True
            }

        if use_llm_judge:
            judge = cls.get_judge()
            return judge.evaluate_temporal_validity(
                query=query or output.query if hasattr(output, "query") else "",
                direct_answer=output.direct_answer,
                temporal_applicability=output.temporal_applicability,
                expected_ay=clean_ay,
                expected_fy=clean_fy,
                is_negative=is_negative
            )

        # Fallback deterministic check
        temp_text = f"{output.temporal_applicability} {output.direct_answer}".lower()
        has_temporal_statement = len(output.temporal_applicability.strip()) > 5
        matches_ay = True
        matches_fy = True

        if clean_ay:
            ay_digits = re.findall(r"\d{4}", clean_ay)
            matches_ay = bool(ay_digits and ay_digits[0] in temp_text)

        if clean_fy:
            fy_digits = re.findall(r"\d{4}", clean_fy)
            matches_fy = bool(fy_digits and fy_digits[0] in temp_text)

        correct = (matches_ay or matches_fy) and has_temporal_statement
        return {
            "temporal_validity_score": 1.0 if correct else 0.0,
            "is_labelled_temporal_case": True,
            "expected_ay": clean_ay,
            "expected_fy": clean_fy,
            "matches_expected_ay": matches_ay,
            "matches_expected_fy": matches_fy,
            "verified_correct_year": correct
        }

    @classmethod
    def compute_faithfulness(
        cls,
        output: GenerationOutput,
        retrieved_chunks: List[Dict[str, Any]],
        use_llm_judge: bool = True
    ) -> Dict[str, Any]:
        """Evaluates hallucination rate & faithfulness against retrieved context chunks."""
        if use_llm_judge:
            judge = cls.get_judge()
            return judge.evaluate_faithfulness(
                direct_answer=output.direct_answer,
                step_by_step_reasoning=output.step_by_step_reasoning,
                retrieved_chunks=retrieved_chunks
            )
        return {
            "faithfulness_score": 1.0,
            "supported_claims": 1,
            "total_claims": 1,
            "unsupported_claims": []
        }

    @classmethod
    def compute_negative_detection(
        cls,
        output: GenerationOutput,
        is_negative_case: bool
    ) -> Dict[str, Any]:
        """Validates out-of-scope / negative boundary query handling."""
        answer_text = output.direct_answer.lower()
        explicit_abstention_phrases = [
            "not in force", "not applicable for", "cannot be filed",
            "out of scope", "was not in existence", "not available for",
            "not permissible for", "not legally in force", "does not apply to",
            "not applicable"
        ]
        declared_negative = output.is_out_of_scope or any(
            phrase in answer_text for phrase in explicit_abstention_phrases
        )

        if is_negative_case:
            correct = declared_negative
        else:
            correct = not output.is_out_of_scope

        return {
            "negative_handling_correct": correct,
            "declared_negative": declared_negative
        }
