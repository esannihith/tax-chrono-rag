import json
import time
import hashlib
import subprocess
from datetime import datetime, timezone
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

    @staticmethod
    def get_git_commit_sha() -> str:
        try:
            res = subprocess.run(
                ["git", "describe", "--always", "--dirty"],
                capture_output=True,
                text=True,
                check=True
            )
            return res.stdout.strip()
        except Exception:
            try:
                res = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return res.stdout.strip()
            except Exception:
                return "uncommitted_or_non_git"

    @staticmethod
    def compute_file_sha256(filepath: str) -> str:
        p = Path(filepath)
        if not p.exists():
            return "file_not_found"
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def evaluate_suite(
        self,
        suite_path: str,
        max_cases: Optional[int] = None,
        top_k: int = 4,
        pacing_delay_sec: float = 0.5,
        stratified: bool = True
    ) -> Dict[str, Any]:
        """Executes generation evaluation across the specified suite and computes aggregated metrics."""
        with open(suite_path, "r", encoding="utf-8") as f:
            raw_cases = json.load(f)

        cases = [GoldenEvaluationCase.model_validate(c) for c in raw_cases]
        if max_cases and max_cases < len(cases):
            if stratified:
                # Stratified sample across query types
                by_type = {}
                for c in cases:
                    by_type.setdefault(c.query_type.value, []).append(c)
                per_type_count = max(1, max_cases // len(by_type))
                sampled_cases = []
                for qtype, qcases in by_type.items():
                    sampled_cases.extend(qcases[:per_type_count])
                cases = sampled_cases[:max_cases]
            else:
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

                # 2. Criteria Keyword Coverage Metrics
                crit_metrics = GenerationMetrics.compute_criteria_keyword_coverage(
                    output=output,
                    criteria_list=case.evaluation_criteria
                )

                # 3. Strict Temporal Validity
                temp_metrics = GenerationMetrics.compute_temporal_validity(
                    output=output,
                    expected_ay=case.expected_ay,
                    expected_fy=case.expected_fy
                )

                # 4. Negative / Out-of-scope Handling
                neg_metrics = GenerationMetrics.compute_negative_detection(
                    output=output,
                    is_negative_case=case.is_negative
                )

                # Composite score per case
                case_score = (
                    0.35 * crit_metrics["criteria_keyword_coverage_rate"] +
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
                    "expected_ay": case.expected_ay,
                    "expected_fy": case.expected_fy,
                    "is_negative": case.is_negative,
                    "direct_answer": output.direct_answer,
                    "temporal_applicability": output.temporal_applicability,
                    "citations": [c.model_dump() for c in output.statutory_citations],
                    "citation_precision": cit_metrics["citation_precision"],
                    "citation_recall": cit_metrics["citation_recall"],
                    "citation_f1": cit_metrics["citation_f1"],
                    "criteria_keyword_coverage_rate": crit_metrics["criteria_keyword_coverage_rate"],
                    "criteria_match_rate": crit_metrics["criteria_keyword_coverage_rate"],
                    "matched_criteria": crit_metrics["matched_criteria"],
                    "missed_criteria": crit_metrics["missed_criteria"],
                    "temporal_validity_score": temp_metrics["temporal_validity_score"],
                    "is_labelled_temporal_case": temp_metrics.get("is_labelled_temporal_case", False),
                    "verified_correct_year": temp_metrics.get("verified_correct_year", False),
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
                    "criteria_keyword_coverage_rate": 0.0,
                    "criteria_match_rate": 0.0,
                    "temporal_validity_score": 0.0,
                    "is_labelled_temporal_case": False,
                    "verified_correct_year": False,
                    "negative_handling_correct": False
                })

            if pacing_delay_sec > 0:
                time.sleep(pacing_delay_sec)

        elapsed = time.time() - start_time

        # Calculate Aggregates
        mean_cit_prec = float(np.mean([r.get("citation_precision", 0) for r in per_case_results]))
        mean_cit_rec = float(np.mean([r.get("citation_recall", 0) for r in per_case_results]))
        mean_cit_f1 = float(np.mean([r.get("citation_f1", 0) for r in per_case_results]))
        mean_crit_cov = float(np.mean([r.get("criteria_keyword_coverage_rate", 0) for r in per_case_results]))
        mean_temp_val = float(np.mean([r.get("temporal_validity_score", 0) for r in per_case_results]))
        mean_composite = float(np.mean([r.get("composite_score", 0) for r in per_case_results]))
        
        # Strict temporal validation subset
        labelled_temp_cases = [r for r in per_case_results if r.get("is_labelled_temporal_case", False)]
        strict_temp_acc = (
            sum(1 for r in labelled_temp_cases if r.get("verified_correct_year", False)) / len(labelled_temp_cases)
        ) if labelled_temp_cases else 1.0

        # Negative / Abstention accuracy
        neg_cases = [r for r in per_case_results if r.get("is_negative", False)]
        neg_accuracy = (sum(1 for r in neg_cases if r.get("negative_handling_correct", False)) / len(neg_cases)) if neg_cases else 1.0

        # Breakdown by Query Type
        query_type_breakdown = {}
        for r in per_case_results:
            qt = r.get("query_type", "Unknown")
            if qt not in query_type_breakdown:
                query_type_breakdown[qt] = {"count": 0, "scores": [], "criteria_matches": []}
            query_type_breakdown[qt]["count"] += 1
            query_type_breakdown[qt]["scores"].append(r.get("composite_score", 0))
            query_type_breakdown[qt]["criteria_matches"].append(r.get("criteria_keyword_coverage_rate", 0))

        qt_summary = {
            qt: {
                "count": data["count"],
                "mean_composite_score": round(float(np.mean(data["scores"])), 4),
                "mean_criteria_keyword_coverage": round(float(np.mean(data["criteria_matches"])), 4)
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
            "provenance": {
                "git_commit_sha": self.get_git_commit_sha(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "suite_file": suite_path,
                "suite_sha256": self.compute_file_sha256(suite_path),
                "top_k": top_k
            },
            "suite": Path(suite_path).name,
            "total_cases_evaluated": len(per_case_results),
            "labelled_temporal_cases_count": len(labelled_temp_cases),
            "unlabelled_temporal_cases_count": len(per_case_results) - len(labelled_temp_cases),
            "negative_cases_count": len(neg_cases),
            "elapsed_seconds": round(elapsed, 2),
            "mean_composite_score": round(mean_composite, 4),
            "mean_criteria_keyword_coverage": round(mean_crit_cov, 4),
            "mean_citation_precision": round(mean_cit_prec, 4),
            "mean_citation_recall": round(mean_cit_rec, 4),
            "mean_citation_f1": round(mean_cit_f1, 4),
            "mean_temporal_validity_score": round(mean_temp_val, 4),
            "strict_temporal_accuracy": round(strict_temp_acc, 4),
            "negative_abstention_accuracy": round(neg_accuracy, 4),
            "query_type_breakdown": qt_summary,
            "regime_breakdown": reg_summary,
            "per_case_results": per_case_results
        }

        return summary


