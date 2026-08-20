import pytest
from src.evaluation.metrics import IRMetrics
from src.evaluation.generation_metrics import GenerationMetrics
from src.generation.models import StatutoryCitation, GenerationOutput


def test_ir_metrics_essential_and_graded_ndcg():
    retrieved = ['chunk_A', 'chunk_B', 'chunk_C', 'chunk_D', 'chunk_E']
    essential = ['chunk_B']
    supporting = ['chunk_C', 'chunk_Z']

    metrics = IRMetrics.evaluate_query(
        retrieved=retrieved,
        essential_chunks=essential,
        supporting_chunks=supporting,
        k_list=[5, 10]
    )

    # chunk_B is at rank 2 -> MRR = 0.5
    assert metrics['mrr'] == 0.5
    # Essential recall@5: 1/1 = 1.0
    assert metrics['essential_recall@5'] == 1.0
    assert metrics['hit_rate@5'] == 1.0
    # Full recall@5: 2 hits (B, C) out of 3 relevant (B, C, Z) = 2/3 = 0.6667
    assert round(metrics['full_recall@5'], 4) == 0.6667
    # Graded NDCG should be positive and bounded by 1.0
    assert 0.0 < metrics['ndcg@5'] <= 1.0


def test_generation_temporal_validity_strict_checking():
    # 1. Correct AY
    out_correct = GenerationOutput(
        query='What is the perquisite valuation under Rule 3 for AY 2024-25?',
        direct_answer='Applicable for AY 2024-25 under Rule 3.',
        step_by_step_reasoning=['Step 1'],
        temporal_applicability='Applicable for Assessment Year 2024-25 (Financial Year 2023-24).',
        statutory_citations=[],
        regime_differences=[],
        is_out_of_scope=False,
        confidence_score=0.9
    )
    res_correct = GenerationMetrics.compute_temporal_validity(
        output=out_correct,
        expected_ay='AY 2024-25',
        expected_fy='FY 2023-24'
    )
    assert res_correct['temporal_validity_score'] == 1.0
    assert res_correct['verified_correct_year'] is True

    # 2. Wrong AY
    out_wrong = GenerationOutput(
        query='What is the perquisite valuation under Rule 3 for AY 2024-25?',
        direct_answer='Applicable for AY 2021-22 under Rule 3.',
        step_by_step_reasoning=['Step 1'],
        temporal_applicability='Applicable for Assessment Year 2021-22.',
        statutory_citations=[],
        regime_differences=[],
        is_out_of_scope=False,
        confidence_score=0.9
    )
    res_wrong = GenerationMetrics.compute_temporal_validity(
        output=out_wrong,
        expected_ay='AY 2024-25',
        expected_fy='FY 2023-24'
    )
    assert res_wrong['temporal_validity_score'] == 0.0
    assert res_wrong['verified_correct_year'] is False


def test_generation_negative_detection():
    # 1. True negative correctly declared
    out_neg = GenerationOutput(
        query='Can I file ITR-U under Rule 12AC for AY 2020-21?',
        direct_answer='Rule 12AC was not in force for AY 2020-21 and cannot be filed for that year.',
        step_by_step_reasoning=['Step 1'],
        temporal_applicability='Rule 12AC not in force prior to AY 2022-23.',
        statutory_citations=[],
        regime_differences=[],
        is_out_of_scope=True,
        confidence_score=0.95
    )
    res_neg = GenerationMetrics.compute_negative_detection(output=out_neg, is_negative_case=True)
    assert res_neg['negative_handling_correct'] is True
    assert res_neg['declared_negative'] is True

    # 2. Positive case containing incidental text should NOT be marked out-of-scope falsely
    out_pos = GenerationOutput(
        query='What is the accommodation perquisite under Rule 3(1)?',
        direct_answer='Perquisite value is Rs. 50,000 under Rule 3(1).',
        step_by_step_reasoning=['Step 1: Computed prior to deductions.'],
        temporal_applicability='Applicable for AY 2024-25.',
        statutory_citations=[],
        regime_differences=[],
        is_out_of_scope=False,
        confidence_score=0.9
    )
    res_pos = GenerationMetrics.compute_negative_detection(output=out_pos, is_negative_case=False)
    assert res_pos['negative_handling_correct'] is True
    assert res_pos['declared_negative'] is False


def test_generation_citation_metrics_on_empty_ground_truth():
    # Model correctly provides no citations on out-of-scope query
    res_clean = GenerationMetrics.compute_citation_metrics(
        predicted_citations=[],
        ground_truth_rules=[]
    )
    assert res_clean['citation_precision'] == 1.0
    assert res_clean['citation_recall'] == 1.0
    assert res_clean['citation_f1'] == 1.0

    # Model hallucinates citations on out-of-scope query
    hallucinated = [StatutoryCitation(rule_id='12AC', sub_rule='(1)', statutory_path='Rule 12AC', corpus_year=1962)]
    res_hallucinated = GenerationMetrics.compute_citation_metrics(
        predicted_citations=hallucinated,
        ground_truth_rules=[]
    )
    assert res_hallucinated['citation_precision'] == 0.0
    assert res_hallucinated['citation_f1'] == 0.0


def test_citation_tokenization_regression_suite():
    """Regression test covering the four citation prefix / containment false positives:
    1. Rule 30 != Rule 3
    2. Rule 3A != Rule 3
    3. Rule 12AC != Rule 12
    4. Rule 3(1) != Rule 3(7)(i)
    """
    # 1. Rule 30 predicting when Rule 3 is ground truth
    pred_30 = [StatutoryCitation(rule_id="30", sub_rule="", statutory_path="Rule 30", corpus_year=1962)]
    res_30 = GenerationMetrics.compute_citation_metrics(pred_30, ["Rule 3"])
    assert res_30["citation_recall"] == 0.0
    assert res_30["citation_precision"] == 0.0

    # 2. Rule 3A predicting when Rule 3 is ground truth
    pred_3a = [StatutoryCitation(rule_id="3A", sub_rule="", statutory_path="Rule 3A", corpus_year=1962)]
    res_3a = GenerationMetrics.compute_citation_metrics(pred_3a, ["Rule 3"])
    assert res_3a["citation_recall"] == 0.0
    assert res_3a["citation_precision"] == 0.0

    # 3. Rule 12AC predicting when Rule 12 is ground truth
    pred_12ac = [StatutoryCitation(rule_id="12AC", sub_rule="", statutory_path="Rule 12AC", corpus_year=1962)]
    res_12ac = GenerationMetrics.compute_citation_metrics(pred_12ac, ["Rule 12"])
    assert res_12ac["citation_recall"] == 0.0
    assert res_12ac["citation_precision"] == 0.0

    # 4. Rule 3(1) predicting when Rule 3(7)(i) is ground truth
    pred_3_1 = [StatutoryCitation(rule_id="3", sub_rule="(1)", statutory_path="Rule 3(1)", corpus_year=1962)]
    res_3_1 = GenerationMetrics.compute_citation_metrics(pred_3_1, ["Rule 3(7)(i)"])
    assert res_3_1["citation_recall"] == 0.0
    assert res_3_1["citation_precision"] == 0.0

    # 5. Exact sub-rule match should succeed
    pred_3_7_i = [StatutoryCitation(rule_id="3", sub_rule="(7)(i)", statutory_path="Rule 3(7)(i)", corpus_year=1962)]
    res_3_7_i = GenerationMetrics.compute_citation_metrics(pred_3_7_i, ["Rule 3(7)(i)"])
    assert res_3_7_i["citation_recall"] == 1.0
    assert res_3_7_i["citation_precision"] == 1.0
    assert res_3_7_i["citation_f1"] == 1.0


def test_criteria_keyword_coverage_metrics():
    output = GenerationOutput(
        query="What is the accommodation perquisite?",
        direct_answer="Perquisite value is 10% of salary for population exceeding 40 lakh under Rule 3.",
        step_by_step_reasoning=["Step 1: Check population tier."],
        temporal_applicability="Applicable for AY 2024-25 under Notification No. 65/2023.",
        statutory_citations=[],
        regime_differences=[],
        is_out_of_scope=False,
        confidence_score=0.95
    )
    criteria = [
        "10% of salary for population > 40 lakh",
        "Notification No. 65/2023 effective 01-09-2023",
        "Completely irrelevant criterion that should fail"
    ]
    res = GenerationMetrics.compute_criteria_keyword_coverage(output, criteria)
    assert res["criteria_keyword_coverage_rate"] == round(2 / 3, 4)
    assert len(res["matched_criteria"]) == 2
    assert len(res["missed_criteria"]) == 1

