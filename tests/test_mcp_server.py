import pytest
from src.mcp.calculators import PerquisiteCalculator
from src.mcp.server import (
    resolve_tax_year,
    get_rule_details,
    search_tax_rules,
    compare_regimes,
    verify_effective_date,
    calculate_perquisite
)


def test_resolve_tax_year():
    res = resolve_tax_year("FY 2023-24")
    assert "2023" in str(res["resolved_financial_year"])
    assert "2024" in str(res["resolved_assessment_year"])
    assert res["default_applicable_regime"] == "1962"

    res_draft = resolve_tax_year("AY 2026-27 draft rules")
    assert res_draft["default_applicable_regime"] == "2026"


def test_get_rule_details():
    res = get_rule_details("3", regime="1962")
    assert res["found"] is True
    assert res["rule_id"] == "3"
    assert res["total_sub_chunks"] >= 20

    res_missing = get_rule_details("9999", regime="1962")
    assert res_missing["found"] is False


def test_search_tax_rules():
    res = search_tax_rules(query="rent free accommodation census population", top_k=3)
    assert res["total_retrieved"] > 0
    assert "results" in res


def test_compare_regimes():
    res = compare_regimes(topic="perquisite car motor vehicle", top_k_per_regime=1)
    assert len(res["comparison"]["income_tax_rules_1962"]) > 0
    assert len(res["comparison"]["draft_income_tax_rules_2026"]) > 0
    assert "Statutory comparison for topic 'perquisite car motor vehicle'" in res["key_transition_summary"]


def test_verify_effective_date():
    res_invalid = verify_effective_date(rule_id="12AC", date_or_ay="AY 2020-21")
    assert res_invalid["is_in_force_for_period"] is False
    assert "NOT in force for periods prior to AY 2022-23" in res_invalid["scope_note"]

    res_valid = verify_effective_date(rule_id="12AC", date_or_ay="AY 2023-24")
    assert res_valid["is_in_force_for_period"] is True

    # Rule 26D
    res_26d_pre = verify_effective_date(rule_id="26D", date_or_ay="AY 2020-21")
    assert res_26d_pre["is_in_force_for_period"] is False

    res_26d_post = verify_effective_date(rule_id="26D", date_or_ay="AY 2023-24")
    assert res_26d_post["is_in_force_for_period"] is True


def test_perquisite_calculators():
    # 1. RFA Post 2023
    rfa = PerquisiteCalculator.calculate_rent_free_accommodation(
        salary_for_period=1200000.0,
        assessment_year="AY 2024-25",
        population_tier=">40L",
        is_owned_by_employer=True,
        is_furnished=True,
        furniture_original_cost=100000.0,
        amount_recovered_from_employee=15000.0
    )
    # 10% of 12L = 120,000 + 10% of 100,000 = 10,000 -> Gross 130,000 - 15,000 = 115,000
    assert rfa["unfurnished_value"] == 120000.0
    assert rfa["furniture_value"] == 10000.0
    assert rfa["taxable_perquisite_value"] == 115000.0
    assert rfa["source_chunk_id"] == "chunk_1962_rule_3_rule_3_table_1_115"

    # 2. Car Perk
    car = PerquisiteCalculator.calculate_motor_car(
        cubic_capacity_cc=1500,
        is_employer_owned_or_hired=True,
        is_employer_expenses_met=True,
        is_driver_provided=False,
        months_used=10
    )
    # 1800 * 10 = 18000
    assert car["taxable_perquisite_value"] == 18000.0

    # 3. Loan Perk
    loan = PerquisiteCalculator.calculate_interest_free_loan(
        max_outstanding_monthly_balance=400000.0,
        original_aggregate_loan_amount=400000.0,
        sbi_benchmark_rate_pct=8.5,
        actual_interest_rate_charged_pct=3.5,
        months_outstanding=12
    )
    # 400,000 * 5% = 20,000
    assert loan["taxable_perquisite_value"] == 20000.0

