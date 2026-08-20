import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv(Path(__file__).parent / ".env")

from src.mcp.server import mcp, search_tax_rules, get_rule_details, compare_regimes, resolve_tax_year, verify_effective_date, calculate_perquisite


def run_self_test():
    """Runs a quick sanity check on all MCP tools to ensure zero serialization issues."""
    print("=" * 80)
    print("RUNNING MCP TOOLS INTEGRITY SELF-TEST")
    print("=" * 80)

    # 1. Test Year Resolver
    print("\n[1/6] Testing `resolve_tax_year`...")
    y_res = resolve_tax_year("FY 2023-24")
    print(f"  Result: FY={y_res['resolved_financial_year']} | AY={y_res['resolved_assessment_year']}")
    assert "2023" in str(y_res["resolved_financial_year"])
    assert "2024" in str(y_res["resolved_assessment_year"])

    # 2. Test Rule Lookup
    print("\n[2/6] Testing `get_rule_details`...")
    r_res = get_rule_details(rule_id="3", regime="1962")
    print(f"  Result: Rule {r_res['rule_id']} ({r_res.get('rule_title', '')}) | Found: {r_res['found']} | Sub-chunks: {r_res['total_sub_chunks']}")
    assert r_res["found"] is True

    # 3. Test Search
    print("\n[3/6] Testing `search_tax_rules`...")
    s_res = search_tax_rules(query="interest free loan valuation", top_k=2)
    print(f"  Result: Retrieved {s_res['total_retrieved']} items | Top match path: {s_res['results'][0]['statutory_path']}")
    assert s_res["total_retrieved"] > 0

    # 4. Test Regime Compare
    print("\n[4/6] Testing `compare_regimes`...")
    c_res = compare_regimes(topic="rent free accommodation", top_k_per_regime=1)
    print(f"  Result: 1962 items: {len(c_res['comparison']['income_tax_rules_1962'])} | 2026 items: {len(c_res['comparison']['draft_income_tax_rules_2026'])}")
    assert len(c_res["comparison"]["income_tax_rules_1962"]) > 0

    # 5. Test Effective Date Verifier
    print("\n[5/6] Testing `verify_effective_date`...")
    eff_res = verify_effective_date(rule_id="12AC", date_or_ay="AY 2020-21")
    print(f"  Result: Rule 12AC in force for AY 2020-21: {eff_res['is_in_force_for_period']} | Note: {eff_res['scope_note']}")
    assert eff_res["is_in_force_for_period"] is False

    # 6. Test Perquisite Calculators
    print("\n[6/6] Testing `calculate_perquisite`...")
    # RFA Test
    rfa_calc = calculate_perquisite(
        perquisite_type="rent_free_accommodation",
        parameters={"salary_for_period": 1000000.0, "population_tier": ">40L", "is_owned_by_employer": True, "amount_recovered": 20000.0}
    )
    print(f"  RFA Taxable Value (Salary 10L, Tier >40L @ 10% - 20k recovered): Rs. {rfa_calc['taxable_perquisite_value']:,.2f}")
    assert rfa_calc["taxable_perquisite_value"] == 80000.0

    # Car Perk Test
    car_calc = calculate_perquisite(
        perquisite_type="motor_car",
        parameters={"cubic_capacity_cc": 1800, "is_employer_owned_or_hired": True, "is_employer_expenses_met": True, "is_driver_provided": True, "months_used": 12}
    )
    print(f"  Car Perk (>1.6L @ 2400/mo + Driver @ 900/mo = 3300 * 12): Rs. {car_calc['taxable_perquisite_value']:,.2f}")
    assert car_calc["taxable_perquisite_value"] == 39600.0

    # Loan Perk Test
    loan_calc = calculate_perquisite(
        perquisite_type="interest_free_loan",
        parameters={"max_outstanding_monthly_balance": 500000.0, "sbi_benchmark_rate_pct": 9.0, "actual_interest_rate_charged_pct": 4.0, "months_outstanding": 12}
    )
    print(f"  Loan Perk (5L @ 5% rate spread = 25k): Rs. {loan_calc['taxable_perquisite_value']:,.2f}")
    assert loan_calc["taxable_perquisite_value"] == 25000.0

    print("\n" + "=" * 80)
    print("ALL MCP TOOLS PASSED INTEGRITY VERIFICATION!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Temporal-RAG FastMCP Server Runner")
    parser.add_argument("--transport", choices=["stdio", "sse", "test"], default="stdio", help="MCP transport mode (stdio, sse, or test)")
    parser.add_argument("--port", type=int, default=8000, help="Port for SSE transport (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for SSE transport (default: 127.0.0.1)")
    parser.add_argument("--analyze-telemetry", action="store_true", help="Print statistical summary report of MCP query telemetry")
    parser.add_argument("--export-telemetry", type=str, default=None, help="Export logged queries to standard RAG evaluation JSON suite path")

    args = parser.parse_args()

    if args.analyze_telemetry:
        from src.evaluation.telemetry_analyzer import print_telemetry_summary
        print_telemetry_summary()
    elif args.export_telemetry:
        from src.evaluation.telemetry_analyzer import MCPTelemetryAnalyzer
        analyzer = MCPTelemetryAnalyzer()
        res = analyzer.export_to_eval_suite(args.export_telemetry)
        print(f"Exported {res['exported_cases_count']} evaluation cases to {res['output_path']}")
    elif args.transport == "test":
        run_self_test()
    elif args.transport == "sse":
        print(f"Starting FastMCP Server on http://{args.host}:{args.port}/sse ...")
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
