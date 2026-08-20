import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from src.evaluation.models import GoldenEvaluationCase, QueryType, DifficultyLevel
from src.evaluation.generation_metrics import GenerationMetrics
from src.generation.pipeline import GenerationPipeline
from src.generation.models import GenerationOutput


class GenerationEvaluationEngine:
    """Automated benchmark engine for evaluating RAG generation quality across golden evaluation suites."""

    def __init__(self, pipeline: GenerationPipeline):
        self.pipeline = pipeline

    def evaluate_suite(
        self,
        suite_path: str,
        max_cases: Optional[int] = None,
        top_k: int = 4,
        pacing_delay_sec: float = 0.5
    ) -> Dict[str, Any]:
        """Executes generation evaluation across the specified suite and computes aggregated metrics."""
        with open(suite_path, "r", encoding="utf-8") as f:
            raw_cases = json.load(f)

        cases = [GoldenEvaluationCase.model_validate(c) for c in raw_cases]
        if max_cases:
            cases = cases[:max_cases]

        print(f"\nStarting generation evaluation for {len(cases)} cases...", flush=True)

        per_case_results = []
        start_time = time.time()

        for idx, case in enumerate(cases, 1):
            print(f"[{idx}/{len(cases)}] Evaluating ({case.id}): {case.query[:65]}...", flush=True)
            
            try:
                output: GenerationOutput = self.pipeline.query(
                    query=case.query,
                    target_regime=case.target_regime,
                    persona_context=case.persona_context,
                    top_k=top_k
                )

                # 1. Citation Metrics
                cit_metrics = GenerationMetrics.compute_citation_metrics(
                    predicted_citations=output.statutory_citations,
                    ground_truth_rules=case.relevant_rules
                )

                # 2. Criteria Match Metrics
                crit_metrics = GenerationMetrics.compute_criteria_match(
                    output=output,
                    criteria_list=case.evaluation_criteria
                )

                # 3. Temporal Validity
                temp_metrics = GenerationMetrics.compute_temporal_validity(output)

                # 4. Negative / Out-of-scope Handling
                is_negative_case = (case.query_type == QueryType.NEGATIVE_OUT_OF_SCOPE or len(case.ground_truth_chunk_ids) == 0)
                neg_metrics = GenerationMetrics.compute_negative_detection(
                    output=output,
                    is_negative_case=is_negative_case
                )

                # Composite score per case
                case_score = (
                    0.35 * crit_metrics["criteria_match_rate"] +
                    0.30 * cit_metrics["citation_recall"] +
                    0.20 * cit_metrics["citation_precision"] +
                    0.15 * temp_metrics["temporal_validity_score"]
                )

                per_case_results.append({
                    "case_id": case.id,
                    "query": case.query,
                    "query_type": case.query_type.value if hasattr(case.query_type, "value") else str(case.query_type),
                    "difficulty_level": case.difficulty_level.value if hasattr(case.difficulty_level, "value") else str(case.difficulty_level),
                    "target_regime": case.target_regime,
                    "direct_answer": output.direct_answer,
                    "temporal_applicability": output.temporal_applicability,
                    "citations": [c.model_dump() for c in output.statutory_citations],
                    "citation_precision": cit_metrics["citation_precision"],
                    "citation_recall": cit_metrics["citation_recall"],
                    "citation_f1": cit_metrics["citation_f1"],
                    "criteria_match_rate": crit_metrics["criteria_match_rate"],
                    "matched_criteria": crit_metrics["matched_criteria"],
                    "missed_criteria": crit_metrics["missed_criteria"],
                    "temporal_validity_score": temp_metrics["temporal_validity_score"],
                    "negative_handling_correct": neg_metrics["negative_handling_correct"],
                    "is_out_of_scope": output.is_out_of_scope,
                    "composite_score": round(case_score, 4)
                })

            except Exception as e:
                print(f"  [ERROR] Case {case.id} failed: {e}")
                per_case_results.append({
                    "case_id": case.id,
                    "query": case.query,
                    "query_type": case.query_type.value if hasattr(case.query_type, "value") else str(case.query_type),
                    "difficulty_level": case.difficulty_level.value if hasattr(case.difficulty_level, "value") else str(case.difficulty_level),
                    "target_regime": case.target_regime,
                    "error": str(e),
                    "composite_score": 0.0,
                    "citation_precision": 0.0,
                    "citation_recall": 0.0,
                    "citation_f1": 0.0,
                    "criteria_match_rate": 0.0,
                    "temporal_validity_score": 0.0,
                    "negative_handling_correct": False
                })

            if pacing_delay_sec > 0:
                time.sleep(pacing_delay_sec)

        elapsed = time.time() - start_time

        # Calculate Aggregates
        mean_cit_prec = float(np.mean([r.get("citation_precision", 0) for r in per_case_results]))
        mean_cit_rec = float(np.mean([r.get("citation_recall", 0) for r in per_case_results]))
        mean_cit_f1 = float(np.mean([r.get("citation_f1", 0) for r in per_case_results]))
        mean_crit_match = float(np.mean([r.get("criteria_match_rate", 0) for r in per_case_results]))
        mean_temp_val = float(np.mean([r.get("temporal_validity_score", 0) for r in per_case_results]))
        mean_composite = float(np.mean([r.get("composite_score", 0) for r in per_case_results]))
        neg_correct_count = sum(1 for r in per_case_results if r.get("negative_handling_correct", False))
        neg_accuracy = neg_correct_count / len(per_case_results) if per_case_results else 0.0

        # Breakdown by Query Type
        query_type_breakdown = {}
        for r in per_case_results:
            qt = r.get("query_type", "Unknown")
            if qt not in query_type_breakdown:
                query_type_breakdown[qt] = {"count": 0, "scores": [], "criteria_matches": []}
            query_type_breakdown[qt]["count"] += 1
            query_type_breakdown[qt]["scores"].append(r.get("composite_score", 0))
            query_type_breakdown[qt]["criteria_matches"].append(r.get("criteria_match_rate", 0))

        qt_summary = {
            qt: {
                "count": data["count"],
                "mean_composite_score": round(float(np.mean(data["scores"])), 4),
                "mean_criteria_match": round(float(np.mean(data["criteria_matches"])), 4)
            }
            for qt, data in query_type_breakdown.items()
        }

        # Breakdown by Target Regime
        regime_breakdown = {}
        for r in per_case_results:
            reg = r.get("target_regime", "Unknown")
            if reg not in regime_breakdown:
                regime_breakdown[reg] = {"count": 0, "scores": []}
            regime_breakdown[reg]["count"] += 1
            regime_breakdown[reg]["scores"].append(r.get("composite_score", 0))

        reg_summary = {
            reg: {
                "count": data["count"],
                "mean_composite_score": round(float(np.mean(data["scores"])), 4)
            }
            for reg, data in regime_breakdown.items()
        }

        summary = {
            "suite": Path(suite_path).name,
            "total_cases_evaluated": len(per_case_results),
            "elapsed_seconds": round(elapsed, 2),
            "mean_composite_score": round(mean_composite, 4),
            "mean_criteria_match_rate": round(mean_crit_match, 4),
            "mean_citation_precision": round(mean_cit_prec, 4),
            "mean_citation_recall": round(mean_cit_rec, 4),
            "mean_citation_f1": round(mean_cit_f1, 4),
            "mean_temporal_validity_score": round(mean_temp_val, 4),
            "negative_handling_accuracy": round(neg_accuracy, 4),
            "query_type_breakdown": qt_summary,
            "regime_breakdown": reg_summary,
            "per_case_results": per_case_results
        }

        return summary
