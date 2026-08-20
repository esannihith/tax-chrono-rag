import re
from typing import List, Dict, Any, Set, Tuple
from src.generation.models import StatutoryCitation, GenerationOutput


class GenerationMetrics:
    """Calculates granular evaluation metrics for statutory generation outputs."""

    @staticmethod
    def normalize_rule_str(rule_str: str) -> str:
        """Extract canonical rule identifier like '12AC', '3', '2BB', '280'."""
        match = re.search(r"(?:rule\s*)?([0-9]+[a-zA-Z]*)", rule_str, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return rule_str.strip().upper()

    @classmethod
    def compute_citation_metrics(
        cls,
        predicted_citations: List[StatutoryCitation],
        ground_truth_rules: List[str]
    ) -> Dict[str, float]:
        """Calculates Citation Precision, Recall, and F1."""
        if not ground_truth_rules:
            # If no ground truth rules (e.g. pure negative query)
            precision = 1.0 if len(predicted_citations) == 0 else 0.5
            recall = 1.0
            f1 = 1.0 if precision == 1.0 else 0.67
            return {"citation_precision": precision, "citation_recall": recall, "citation_f1": f1}

        pred_set = {cls.normalize_rule_str(c.rule_id) for c in predicted_citations}
        gt_set = {cls.normalize_rule_str(r) for r in ground_truth_rules}

        tp = len(pred_set.intersection(gt_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "citation_precision": round(precision, 4),
            "citation_recall": round(recall, 4),
            "citation_f1": round(f1, 4)
        }

    @classmethod
    def compute_criteria_match(
        cls,
        output: GenerationOutput,
        criteria_list: List[str]
    ) -> Dict[str, Any]:
        """Evaluates semantic adherence against golden evaluation criteria."""
        if not criteria_list:
            return {"criteria_match_rate": 1.0, "matched_criteria": [], "missed_criteria": []}

        full_text = f"{output.direct_answer} {' '.join(output.step_by_step_reasoning)} {output.temporal_applicability}".lower()

        matched = []
        missed = []

        for crit in criteria_list:
            crit_clean = crit.lower()
            # Extract key terms/numbers from criterion
            keywords = re.findall(r"[a-z0-9\(\)]+", crit_clean)
            key_entities = [k for k in keywords if len(k) > 3 or k.isdigit() or k in ["evc", "dsc", "sbi", "hra", "lta", "ltc", "194p"]]

            if not key_entities:
                hits = 1 if crit_clean in full_text else 0
                ratio = 1.0 if hits else 0.0
            else:
                matched_count = sum(1 for k in key_entities if k in full_text)
                ratio = matched_count / len(key_entities)

            if ratio >= 0.50 or any(phrase in full_text for phrase in [crit_clean[:20]]):
                matched.append(crit)
            else:
                missed.append(crit)

        match_rate = len(matched) / len(criteria_list)
        return {
            "criteria_match_rate": round(match_rate, 4),
            "matched_criteria": matched,
            "missed_criteria": missed
        }

    @classmethod
    def compute_negative_detection(
        cls,
        output: GenerationOutput,
        is_negative_case: bool
    ) -> Dict[str, Any]:
        """Validates out-of-scope / negative boundary query handling."""
        answer_text = output.direct_answer.lower()
        declared_negative = output.is_out_of_scope or any(
            phrase in answer_text for phrase in [
                "not in force", "not applicable", "not permissible", "cannot be filed",
                "out of scope", "was not in existence", "not available", "prior to"
            ]
        )

        if is_negative_case:
            correct = declared_negative
        else:
            correct = not output.is_out_of_scope

        return {
            "negative_handling_correct": correct,
            "declared_negative": declared_negative
        }

    @classmethod
    def compute_temporal_validity(
        cls,
        output: GenerationOutput
    ) -> Dict[str, Any]:
        """Validates accuracy of temporal statements and effective date citations."""
        temp_text = output.temporal_applicability.lower()
        has_temporal_statement = len(output.temporal_applicability.strip()) > 10

        # Check for presence of AY/FY or effective date mentions
        has_date_or_year = bool(re.search(r"(?:ay|fy|assessment\s+year|financial\s+year|[0-9]{4}-[0-9]{2,4}|[0-9]{2}-[0-9]{2}-[0-9]{4})", temp_text))

        score = 1.0 if (has_temporal_statement and has_date_or_year) else (0.5 if has_temporal_statement else 0.0)

        return {
            "temporal_validity_score": score,
            "has_date_or_year": has_date_or_year
        }
