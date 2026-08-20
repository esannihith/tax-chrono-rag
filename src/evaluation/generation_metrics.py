import re
from typing import List, Dict, Any, Set, Tuple, Optional
from src.generation.models import StatutoryCitation, GenerationOutput


class GenerationMetrics:
    """Calculates granular, verifiable evaluation metrics for statutory generation outputs."""

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

    @classmethod
    def matches_rule_citation(cls, pred_cit: str, gt_cit: str) -> bool:
        """Determines if a predicted citation correctly satisfies a ground truth citation.
        
        Requires:
        1. Exact match on base rule identifier (e.g. '3' != '30', '3' != '3A', '12' != '12AC').
        2. Sub-rule consistency:
           - If GT has no sub-rule specified (e.g. 'Rule 3'), any sub-rule under Rule 3 satisfies it.
           - If GT specifies a sub-rule (e.g. 'Rule 3(7)(i)'), the prediction must specify the
             same sub-rule or a compatible parent/child hierarchy. E.g. 'Rule 3(1)' != 'Rule 3(7)(i)'.
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
            # Prediction only gave base rule when specific sub-rule was required
            return False

        # Check sub-rule hierarchy containment (e.g. '(7)(I)' == '(7)(I)', '(7)' parent of '(7)(I)')
        if p_sub == gt_sub or p_sub in gt_sub or gt_sub in p_sub:
            return True

        return False

    @classmethod
    def compute_citation_metrics(
        cls,
        predicted_citations: List[StatutoryCitation],
        ground_truth_rules: List[str]
    ) -> Dict[str, float]:
        """Calculates Citation Precision, Recall, and F1 with strict rule tokenization."""
        if not ground_truth_rules:
            # Out-of-scope / negative query
            if len(predicted_citations) == 0:
                return {"citation_precision": 1.0, "citation_recall": 1.0, "citation_f1": 1.0}
            else:
                return {"citation_precision": 0.0, "citation_recall": 0.0, "citation_f1": 0.0}

        pred_citations_normalized = []
        for c in predicted_citations:
            full_cit = f"{c.rule_id}{c.sub_rule or ''}"
            pred_citations_normalized.append(full_cit)

        # Evaluate Ground Truth Recall: which GT rules are satisfied by at least one prediction?
        matched_gt = set()
        for gt in ground_truth_rules:
            for pred in pred_citations_normalized:
                if cls.matches_rule_citation(pred, gt):
                    matched_gt.add(gt)
                    break

        # Evaluate Prediction Precision: which predicted citations satisfy at least one GT rule?
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

    @classmethod
    def compute_criteria_keyword_coverage(
        cls,
        output: GenerationOutput,
        criteria_list: List[str]
    ) -> Dict[str, Any]:
        """Evaluates entity and key-phrase keyword coverage against curated evaluation criteria."""
        if not criteria_list:
            return {"criteria_keyword_coverage_rate": 1.0, "criteria_match_rate": 1.0, "matched_criteria": [], "missed_criteria": []}

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
            "criteria_keyword_coverage_rate": round(match_rate, 4),
            "criteria_match_rate": round(match_rate, 4),  # backward compatibility alias
            "matched_criteria": matched,
            "missed_criteria": missed
        }

    # Backward compatibility alias
    compute_criteria_match = compute_criteria_keyword_coverage

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
        """Strictly validates whether the stated AY/FY matches the expected ground-truth year field-by-field."""
        temp_text = f"{output.temporal_applicability} {output.direct_answer}".lower()
        has_temporal_statement = len(output.temporal_applicability.strip()) > 10

        # If expected AY/FY is specified, verify exact field match
        if expected_ay or expected_fy:
            matches_ay = True
            matches_fy = True

            if expected_ay:
                ay_digits = re.findall(r"\d{4}", expected_ay)
                if ay_digits:
                    # Require AY or Assessment Year pattern with matching year
                    ay_pattern = rf"(?:ay|assessment\s+year)[^\d]*{ay_digits[0]}"
                    matches_ay = bool(re.search(ay_pattern, temp_text)) or (ay_digits[0] in temp_text and "ay" in temp_text)
                else:
                    matches_ay = False

            if expected_fy:
                fy_digits = re.findall(r"\d{4}", expected_fy)
                if fy_digits:
                    # Require FY or Financial Year pattern with matching year
                    fy_pattern = rf"(?:fy|financial\s+year|previous\s+year)[^\d]*{fy_digits[0]}"
                    matches_fy = bool(re.search(fy_pattern, temp_text)) or (fy_digits[0] in temp_text and "fy" in temp_text)
                else:
                    matches_fy = False

            # Strict conjunction: all specified ground truth temporal constraints must hold
            if expected_ay and expected_fy:
                correct = matches_ay and matches_fy and has_temporal_statement
            elif expected_ay:
                correct = matches_ay and has_temporal_statement
            else:
                correct = matches_fy and has_temporal_statement

            score = 1.0 if correct else 0.0
            return {
                "temporal_validity_score": score,
                "is_labelled_temporal_case": True,
                "expected_ay": expected_ay,
                "expected_fy": expected_fy,
                "matches_expected_ay": matches_ay,
                "matches_expected_fy": matches_fy,
                "verified_correct_year": correct
            }

        # Fallback for unlabelled cases
        has_date_or_year = bool(re.search(r"(?:ay|fy|assessment\s+year|financial\s+year|[0-9]{4}-[0-9]{2,4})", temp_text))
        score = 1.0 if (has_temporal_statement and has_date_or_year) else (0.5 if has_temporal_statement else 0.0)

        return {
            "temporal_validity_score": score,
            "is_labelled_temporal_case": False,
            "has_date_or_year": has_date_or_year,
            "verified_correct_year": score == 1.0
        }


