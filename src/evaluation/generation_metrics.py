import re
from typing import List, Dict, Any, Set, Tuple, Optional
from src.generation.models import StatutoryCitation, GenerationOutput


class GenerationMetrics:
    """Calculates granular, verifiable evaluation metrics for statutory generation outputs."""

    @staticmethod
    def normalize_citation_str(rule_str: str) -> str:
        """Extract canonical rule identifier like '12AC', '3(7)(I)', '2BB(1)'."""
        clean = re.sub(r"^rule\s*", "", rule_str.strip(), flags=re.IGNORECASE)
        # Normalize whitespace
        clean = re.sub(r"\s+", "", clean).upper()
        return clean

    @classmethod
    def compute_citation_metrics(
        cls,
        predicted_citations: List[StatutoryCitation],
        ground_truth_rules: List[str]
    ) -> Dict[str, float]:
        """Calculates Citation Precision, Recall, and F1."""
        if not ground_truth_rules:
            # If no ground truth rules (e.g. out-of-scope query)
            if len(predicted_citations) == 0:
                return {"citation_precision": 1.0, "citation_recall": 1.0, "citation_f1": 1.0}
            else:
                # Hallucinated citations when none apply
                return {"citation_precision": 0.0, "citation_recall": 0.0, "citation_f1": 0.0}

        pred_set = set()
        for c in predicted_citations:
            full_cit = f"{c.rule_id}{c.sub_rule or ''}"
            pred_set.add(cls.normalize_citation_str(full_cit))
            pred_set.add(cls.normalize_citation_str(c.rule_id))

        gt_set = {cls.normalize_citation_str(r) for r in ground_truth_rules}

        # Check hits: either exact sub-rule match or base rule match
        tp_rules = set()
        for gt in gt_set:
            if gt in pred_set:
                tp_rules.add(gt)
            else:
                # Check base rule prefix match
                gt_base = re.match(r"^[0-9]+[A-Z]*", gt)
                if gt_base and any(p.startswith(gt_base.group(0)) for p in pred_set):
                    tp_rules.add(gt)

        tp = len(tp_rules)
        # Unique predicted rule bases
        pred_base_count = len({re.match(r"^[0-9]+[A-Z]*", p).group(0) for p in pred_set if re.match(r"^[0-9]+[A-Z]*", p)})
        fp = max(0, pred_base_count - tp)
        fn = len(gt_set - tp_rules)

        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(pred_set) == 0 and len(gt_set) == 0 else 0.0)
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
        """Evaluates entity and key-phrase adherence against golden evaluation criteria."""
        if not criteria_list:
            return {"criteria_match_rate": 1.0, "matched_criteria": [], "missed_criteria": []}

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

            if ratio >= 0.50 or any(phrase in full_text for phrase in [crit_clean[:25]]):
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
        """Validates out-of-scope / negative boundary query handling without false-positive substrings."""
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

    @classmethod
    def compute_temporal_validity(
        cls,
        output: GenerationOutput,
        expected_ay: Optional[str] = None,
        expected_fy: Optional[str] = None
    ) -> Dict[str, Any]:
        """Strictly validates whether the stated AY/FY matches the expected ground-truth year."""
        temp_text = f"{output.temporal_applicability} {output.direct_answer}".lower()
        has_temporal_statement = len(output.temporal_applicability.strip()) > 10

        # If expected AY/FY is specified, verify exact match
        if expected_ay or expected_fy:
            matches_ay = False
            matches_fy = False

            if expected_ay:
                ay_digits = re.findall(r"\d{4}", expected_ay)
                if ay_digits:
                    matches_ay = ay_digits[0] in temp_text

            if expected_fy:
                fy_digits = re.findall(r"\d{4}", expected_fy)
                if fy_digits:
                    matches_fy = fy_digits[0] in temp_text

            if expected_ay and expected_fy:
                correct = (matches_ay or matches_fy) and has_temporal_statement
            elif expected_ay:
                correct = matches_ay and has_temporal_statement
            else:
                correct = matches_fy and has_temporal_statement

            score = 1.0 if correct else 0.0
            return {
                "temporal_validity_score": score,
                "expected_ay": expected_ay,
                "expected_fy": expected_fy,
                "verified_correct_year": correct
            }

        # Fallback if no specific expected year in case: check for presence of AY/FY
        has_date_or_year = bool(re.search(r"(?:ay|fy|assessment\s+year|financial\s+year|[0-9]{4}-[0-9]{2,4})", temp_text))
        score = 1.0 if (has_temporal_statement and has_date_or_year) else (0.5 if has_temporal_statement else 0.0)

        return {
            "temporal_validity_score": score,
            "has_date_or_year": has_date_or_year
        }

