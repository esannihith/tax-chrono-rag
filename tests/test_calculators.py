import pytest
from src.mcp.calculators import PerquisiteCalculator


def test_rent_free_accommodation_post_2023():
    # AY 2024-25 (Post Notification 65/2023)
    # Tier >40L -> 10%
    res = PerquisiteCalculator.calculate_rent_free_accommodation(
        salary_for_period=1200000.0,
        assessment_year="AY 2024-25",
        population_tier=">40L",
        is_owned_by_employer=True,
        amount_recovered_from_employee=10000.0
    )
    # 10% of 12L = 120k - 10k = 110k
    assert res["unfurnished_value"] == 120000.0
    assert res["taxable_perquisite_value"] == 110000.0
    assert "Notification No. 65/2023" in res["statutory_citation"]
    assert res["source_chunk_id"] == "chunk_1962_rule_3_rule_3_table_1_115"


def test_rent_free_accommodation_pre_2023_historical():
    # AY 2022-23 (Pre Notification 65/2023)
    # Tier >25L -> 15%
    res = PerquisiteCalculator.calculate_rent_free_accommodation(
        salary_for_period=1000000.0,
        assessment_year="AY 2022-23",
        population_tier=">25L",
        is_owned_by_employer=True,
        amount_recovered_from_employee=0.0
    )
    # 15% of 10L = 150k
    assert res["unfurnished_value"] == 150000.0
    assert res["taxable_perquisite_value"] == 150000.0
    assert "Pre-2023" in res["statutory_citation"]


def test_interest_free_loan_threshold_statutory_logic():
    # 1. Original loan 18k -> exempt
    res_exempt = PerquisiteCalculator.calculate_interest_free_loan(
        max_outstanding_monthly_balance=18000.0,
        original_aggregate_loan_amount=18000.0,
        sbi_benchmark_rate_pct=9.0,
        actual_interest_rate_charged_pct=4.0
    )
    assert res_exempt["is_exempt"] is True
    assert res_exempt["taxable_perquisite_value"] == 0.0

    # 2. Original loan 500k, paid down to 18k balance -> NOT exempt because original was >20k
    res_not_exempt = PerquisiteCalculator.calculate_interest_free_loan(
        max_outstanding_monthly_balance=18000.0,
        original_aggregate_loan_amount=500000.0,
        sbi_benchmark_rate_pct=9.0,
        actual_interest_rate_charged_pct=4.0,
        months_outstanding=12
    )
    assert res_not_exempt["is_exempt"] is False
    # 18k * 5% = 900
    assert res_not_exempt["taxable_perquisite_value"] == 900.0


def test_interest_free_loan_medical_proviso_reimbursement():
    # Medical loan with zero insurance reimbursement -> 100% exempt
    res_pure_medical = PerquisiteCalculator.calculate_interest_free_loan(
        max_outstanding_monthly_balance=300000.0,
        is_medical_treatment_specified_disease=True,
        medical_insurance_reimbursement=0.0,
        sbi_benchmark_rate_pct=10.0,
        actual_interest_rate_charged_pct=2.0
    )
    assert res_pure_medical["is_exempt"] is True
    assert res_pure_medical["taxable_perquisite_value"] == 0.0

    # Medical loan with 100k insurance reimbursement -> perquisite is taxed ONLY on the 100k reimbursed portion
    res_partial_reimbursed = PerquisiteCalculator.calculate_interest_free_loan(
        max_outstanding_monthly_balance=300000.0,
        is_medical_treatment_specified_disease=True,
        medical_insurance_reimbursement=100000.0,
        sbi_benchmark_rate_pct=10.0,
        actual_interest_rate_charged_pct=2.0,
        months_outstanding=12
    )
    assert res_partial_reimbursed["is_exempt"] is False
    # 100k * 8% = 8000
    assert res_partial_reimbursed["taxable_perquisite_value"] == 8000.0
    assert res_partial_reimbursed["max_balance_considered"] == 100000.0


def test_motor_car_employee_owned_proration():
    # Car <= 1.6L (rate 1800/mo), used 6 months, running expenses incurred = 25000, recovered = 0
    # Allowable exemption = 1800 * 6 = 10800 -> Taxable = 25000 - 10800 = 14200
    res = PerquisiteCalculator.calculate_motor_car(
        cubic_capacity_cc=1400,
        is_employer_owned_or_hired=False,
        actual_running_expense_incurred=25000.0,
        months_used=6,
        amount_recovered=0.0
    )
    assert res["reimbursement_amount"] == 25000.0
    assert res["statutory_deduction"] == 10800.0
    assert res["taxable_perquisite_value"] == 14200.0


def test_free_food_exemption():
    # 200 meals at Rs. 90/meal with Rs. 10 recovered
    # Taxable per meal = 90 - 50 - 10 = 30 -> Total 200 * 30 = 6000
    res = PerquisiteCalculator.calculate_free_food(
        cost_per_meal=90.0,
        num_meals_per_day=1,
        working_days=200,
        amount_recovered_per_meal=10.0
    )
    assert res["taxable_per_meal"] == 30.0
    assert res["taxable_perquisite_value"] == 6000.0
    assert res["source_chunk_id"] == "chunk_1962_rule_3_rule_3_subrule_7_49"
