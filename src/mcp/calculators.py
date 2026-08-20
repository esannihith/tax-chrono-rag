import re
from typing import Dict, Any, Optional
from src.enrichment.temporal_registry import STATUTORY_EFFECTIVE_DATE_REGISTRY


class PerquisiteCalculator:
    """Deterministic computational engine for Indian Income Tax perquisite valuations under Rule 3 (1962 Rules) and Draft 2026 Rules.
    
    Fully temporal-aware, supporting historical pre-Notification 65/2023 rates, post-2023 rates, and draft 2026 rules with source chunk provenance.
    """

    @staticmethod
    def _extract_ay_year(ay_str: str) -> int:
        """Extracts the start year of Assessment Year (e.g. 'AY 2024-25' -> 2024)."""
        match = re.search(r"\b(\d{4})\b", ay_str)
        if match:
            return int(match.group(1))
        return 2024

    @staticmethod
    def calculate_rent_free_accommodation(
        salary_for_period: float,
        assessment_year: str = "AY 2024-25",
        is_government_employee: bool = False,
        government_license_fee: float = 0.0,
        is_owned_by_employer: bool = True,
        population_tier: str = ">40L",  # ">40L", "15L-40L", "<=15L", or historical ">25L", "10L-25L", "<=10L"
        actual_lease_rent_paid: float = 0.0,
        is_furnished: bool = False,
        furniture_original_cost: float = 0.0,
        furniture_annual_hire_charges: float = 0.0,
        amount_recovered_from_employee: float = 0.0,
        regime: str = "1962"
    ) -> Dict[str, Any]:
        """Calculates taxable perquisite value of Rent-Free / Concessional Accommodation under Rule 3(1)."""
        breakdown = []
        ay_year = PerquisiteCalculator._extract_ay_year(assessment_year)

        if is_government_employee:
            unfurnished_val = government_license_fee
            breakdown.append(f"Government License Fee specified: Rs. {government_license_fee:,.2f}")
            source_citation = "Rule 3(1) Table 1 (Serial 1)"
            source_chunk_id = "chunk_1962_rule_3_rule_3_table_1_115"
        else:
            if regime == "2026":
                # Draft 2026 Simplified Schedule
                rate = 0.10 if population_tier in [">40L", ">25L"] else 0.07
                breakdown.append(f"Draft 2026 Rules Rate: {rate*100:.1f}% of salary (Tier: {population_tier})")
                base_val = salary_for_period * rate
                if not is_owned_by_employer:
                    unfurnished_val = min(actual_lease_rent_paid, base_val)
                    breakdown.append(f"Leased accommodation: Min(Actual Rent Rs. {actual_lease_rent_paid:,.2f}, Base Rs. {base_val:,.2f}) = Rs. {unfurnished_val:,.2f}")
                else:
                    unfurnished_val = base_val
                source_citation = "Draft Income-tax Rules 2026 Schedule on Accommodation"
                source_chunk_id = "chunk_2026_rule_15_rule_15_subrule_1_1"
            else:
                # 1962 Rules: Check Notification 65/2023 boundary from authoritative registry
                amd_eff_ay = STATUTORY_EFFECTIVE_DATE_REGISTRY["3"]["amendments"][0]["effective_ay"]
                if ay_year >= amd_eff_ay:
                    # Post-Notification 65/2023 Rates

                    if is_owned_by_employer:
                        tier_rates = {">40L": 0.10, "15L-40L": 0.075, "<=15L": 0.05, ">25L": 0.10, "10L-25L": 0.075, "<=10L": 0.05}
                        rate = tier_rates.get(population_tier, 0.10)
                        unfurnished_val = salary_for_period * rate
                        breakdown.append(f"Post-Notification 65/2023 ({assessment_year}) Employer-owned in tier '{population_tier}': {rate*100:.1f}% of salary (Rs. {salary_for_period:,.2f}) = Rs. {unfurnished_val:,.2f}")
                    else:
                        ten_pct_salary = salary_for_period * 0.10
                        unfurnished_val = min(actual_lease_rent_paid, ten_pct_salary)
                        breakdown.append(f"Post-Notification 65/2023 ({assessment_year}) Leased: Min(Actual Rent Rs. {actual_lease_rent_paid:,.2f}, 10% Salary Rs. {ten_pct_salary:,.2f}) = Rs. {unfurnished_val:,.2f}")
                    source_citation = "Rule 3(1) Table 1 (Amended by Notification No. 65/2023 w.e.f. 01-09-2023)"
                    source_chunk_id = "chunk_1962_rule_3_rule_3_table_1_115"
                else:
                    # Pre-Notification 65/2023 Historical Rates (AY 2023-24 and earlier)
                    if is_owned_by_employer:
                        hist_rates = {">25L": 0.15, ">40L": 0.15, "10L-25L": 0.10, "15L-40L": 0.10, "<=10L": 0.075, "<=15L": 0.075}
                        rate = hist_rates.get(population_tier, 0.15)
                        unfurnished_val = salary_for_period * rate
                        breakdown.append(f"Pre-Notification 65/2023 Historical ({assessment_year}) Employer-owned in tier '{population_tier}': {rate*100:.1f}% of salary (Rs. {salary_for_period:,.2f}) = Rs. {unfurnished_val:,.2f}")
                    else:
                        fifteen_pct_salary = salary_for_period * 0.15
                        unfurnished_val = min(actual_lease_rent_paid, fifteen_pct_salary)
                        breakdown.append(f"Pre-Notification 65/2023 Historical ({assessment_year}) Leased: Min(Actual Rent Rs. {actual_lease_rent_paid:,.2f}, 15% Salary Rs. {fifteen_pct_salary:,.2f}) = Rs. {unfurnished_val:,.2f}")
                    source_citation = "Rule 3(1) Table 1 (Historical Pre-2023 Amendment)"
                    source_chunk_id = "chunk_1962_rule_3_rule_3_table_1_115"

        # Furniture additions
        furniture_addition = 0.0
        if is_furnished:
            if furniture_original_cost > 0:
                furniture_addition += furniture_original_cost * 0.10
                breakdown.append(f"Owned Furniture Addition: 10% of Rs. {furniture_original_cost:,.2f} = Rs. {furniture_original_cost*0.10:,.2f}")
            if furniture_annual_hire_charges > 0:
                furniture_addition += furniture_annual_hire_charges
                breakdown.append(f"Hired Furniture Hire Charges: Rs. {furniture_annual_hire_charges:,.2f}")

        gross_value = unfurnished_val + furniture_addition
        taxable_perquisite = max(0.0, gross_value - amount_recovered_from_employee)

        if amount_recovered_from_employee > 0:
            breakdown.append(f"Less: Amount recovered from employee: Rs. {amount_recovered_from_employee:,.2f}")

        breakdown.append(f"Final Taxable Accommodation Perquisite: Rs. {taxable_perquisite:,.2f}")

        return {
            "perquisite_type": "rent_free_accommodation",
            "statutory_rule": "Rule 3(1)",
            "regime": regime,
            "assessment_year": assessment_year,
            "salary_considered": salary_for_period,
            "unfurnished_value": round(unfurnished_val, 2),
            "furniture_value": round(furniture_addition, 2),
            "gross_perquisite_value": round(gross_value, 2),
            "amount_recovered": round(amount_recovered_from_employee, 2),
            "taxable_perquisite_value": round(taxable_perquisite, 2),
            "statutory_citation": source_citation,
            "source_chunk_id": source_chunk_id,
            "computation_steps": breakdown
        }

    @staticmethod
    def calculate_motor_car(
        assessment_year: str = "AY 2024-25",
        cubic_capacity_cc: int = 1500,
        is_employer_owned_or_hired: bool = True,
        is_employer_expenses_met: bool = True,
        usage_type: str = "mixed",  # "official", "personal", "mixed"
        is_driver_provided: bool = False,
        actual_running_expense_incurred: float = 0.0,
        car_original_cost: float = 0.0,
        amount_recovered: float = 0.0,
        months_used: int = 12,
        regime: str = "1962"
    ) -> Dict[str, Any]:
        """Calculates taxable perquisite value of Motor Car under Rule 3(2)."""
        breakdown = []
        source_citation = "Rule 3(2) Table 2"
        source_chunk_id = "chunk_1962_rule_3_rule_3_table_2_116"

        if usage_type == "official":
            return {
                "perquisite_type": "motor_car",
                "statutory_rule": "Rule 3(2)",
                "regime": regime,
                "assessment_year": assessment_year,
                "taxable_perquisite_value": 0.0,
                "statutory_citation": source_citation,
                "source_chunk_id": source_chunk_id,
                "computation_steps": [
                    "Car used 100% exclusively for official performance.",
                    "Subject to logbook maintenance and employer certification.",
                    "Taxable Perquisite = Rs. 0.00"
                ]
            }

        if usage_type == "personal":
            annual_depreciation = car_original_cost * 0.10
            gross_val = (actual_running_expense_incurred + (annual_depreciation * (months_used / 12)))
            taxable_val = max(0.0, gross_val - amount_recovered)
            return {
                "perquisite_type": "motor_car",
                "statutory_rule": "Rule 3(2)",
                "regime": regime,
                "assessment_year": assessment_year,
                "usage_type": "personal",
                "gross_perquisite_value": round(gross_val, 2),
                "amount_recovered": round(amount_recovered, 2),
                "taxable_perquisite_value": round(taxable_val, 2),
                "statutory_citation": source_citation,
                "source_chunk_id": source_chunk_id,
                "computation_steps": [
                    f"100% Personal Use: Running Expenses incurred (Rs. {actual_running_expense_incurred:,.2f}) + 10% Wear & Tear prorated ({months_used} mos = Rs. {annual_depreciation*(months_used/12):,.2f})",
                    f"Gross = Rs. {gross_val:,.2f} less recovered Rs. {amount_recovered:,.2f} = Rs. {taxable_val:,.2f}"
                ]
            }

        # Mixed Use (Partly Official, Partly Personal)
        is_higher_cc = cubic_capacity_cc > 1600
        cc_label = "> 1.6 Litres (1600 cc)" if is_higher_cc else "<= 1.6 Litres (1600 cc)"

        if is_employer_owned_or_hired and is_employer_expenses_met:
            monthly_car_val = 2400.0 if is_higher_cc else 1800.0
            breakdown.append(f"Car owned & maintained by employer ({cc_label}): Rs. {monthly_car_val:,.2f}/month")
        elif is_employer_owned_or_hired and not is_employer_expenses_met:
            monthly_car_val = 900.0 if is_higher_cc else 600.0
            breakdown.append(f"Car owned by employer, running expenses met by employee ({cc_label}): Rs. {monthly_car_val:,.2f}/month")
        else:
            # Employee owned, employer reimbursed: actual running expenses incurred less allowable exemption
            exemption_rate = 2400.0 if is_higher_cc else 1800.0
            allowable_exemption = exemption_rate * months_used
            driver_exemption = (900.0 * months_used) if is_driver_provided else 0.0
            total_allowable = allowable_exemption + driver_exemption
            taxable_val = max(0.0, actual_running_expense_incurred - total_allowable - amount_recovered)
            return {
                "perquisite_type": "motor_car",
                "statutory_rule": "Rule 3(2)",
                "regime": regime,
                "assessment_year": assessment_year,
                "usage_type": "mixed_employee_owned",
                "reimbursement_amount": round(actual_running_expense_incurred, 2),
                "statutory_deduction": round(total_allowable, 2),
                "taxable_perquisite_value": round(taxable_val, 2),
                "statutory_citation": source_citation,
                "source_chunk_id": source_chunk_id,
                "computation_steps": [
                    f"Employee-owned car: Actual employer reimbursement Rs. {actual_running_expense_incurred:,.2f}",
                    f"Less statutory exemption ({months_used} mos @ Rs. {exemption_rate:,.2f} + driver Rs. {driver_exemption:,.2f}): Rs. {total_allowable:,.2f}",
                    f"Taxable value = Rs. {taxable_val:,.2f}"
                ]
            }

        monthly_driver_val = 900.0 if is_driver_provided else 0.0
        if is_driver_provided:
            breakdown.append("Chauffeur / Driver provided: Rs. 900.00/month")

        total_monthly = monthly_car_val + monthly_driver_val
        annual_perk = total_monthly * months_used
        taxable_val = max(0.0, annual_perk - amount_recovered)

        breakdown.append(f"Total Monthly Rate = Rs. {total_monthly:,.2f}/month * {months_used} months = Rs. {annual_perk:,.2f}")
        if amount_recovered > 0:
            breakdown.append(f"Less recovered: Rs. {amount_recovered:,.2f}")
        breakdown.append(f"Final Taxable Motor Car Perquisite: Rs. {taxable_val:,.2f}")

        return {
            "perquisite_type": "motor_car",
            "statutory_rule": "Rule 3(2)",
            "regime": regime,
            "assessment_year": assessment_year,
            "cubic_capacity_cc": cubic_capacity_cc,
            "tier_category": cc_label,
            "monthly_rate": total_monthly,
            "months_used": months_used,
            "annual_gross_value": round(annual_perk, 2),
            "amount_recovered": round(amount_recovered, 2),
            "taxable_perquisite_value": round(taxable_val, 2),
            "statutory_citation": source_citation,
            "source_chunk_id": source_chunk_id,
            "computation_steps": breakdown
        }

    @staticmethod
    def calculate_interest_free_loan(
        max_outstanding_monthly_balance: float,
        original_aggregate_loan_amount: Optional[float] = None,
        assessment_year: str = "AY 2024-25",
        actual_interest_rate_charged_pct: float = 0.0,
        sbi_benchmark_rate_pct: float = 8.5,
        is_medical_treatment_specified_disease: bool = False,
        medical_insurance_reimbursement: float = 0.0,
        is_aggregate_loans_under_20k: bool = False,
        months_outstanding: int = 12,
        regime: str = "1962"
    ) -> Dict[str, Any]:
        """Calculates taxable perquisite value of Concessional / Interest-Free Loan under Rule 3(7)(i)."""
        breakdown = []
        source_citation = "Rule 3(7)(i)"
        source_chunk_id = "chunk_1962_rule_3_rule_3_subrule_7_49"

        # Threshold check: Statute states aggregate original loans not exceeding Rs. 20,000
        orig_loan = original_aggregate_loan_amount if original_aggregate_loan_amount is not None else max_outstanding_monthly_balance
        if is_aggregate_loans_under_20k or orig_loan <= 20000.0:
            return {
                "perquisite_type": "interest_free_loan",
                "statutory_rule": "Rule 3(7)(i)",
                "regime": regime,
                "assessment_year": assessment_year,
                "is_exempt": True,
                "exemption_reason": f"Aggregate original loan amount (Rs. {orig_loan:,.2f}) does not exceed statutory threshold of Rs. 20,000.",
                "taxable_perquisite_value": 0.0,
                "statutory_citation": "Rule 3(7)(i) Proviso 1",
                "source_chunk_id": source_chunk_id,
                "computation_steps": [
                    f"Original aggregate loan amount Rs. {orig_loan:,.2f} <= statutory Rs. 20,000 threshold.",
                    "Exempt under Rule 3(7)(i) proviso 1.",
                    "Taxable Perquisite = Rs. 0.00"
                ]
            }

        rate_diff = max(0.0, sbi_benchmark_rate_pct - actual_interest_rate_charged_pct)
        breakdown.append(f"SBI Benchmark Lending Rate (1st day of relevant FY): {sbi_benchmark_rate_pct:.2f}%")
        breakdown.append(f"Actual Rate Charged by Employer: {actual_interest_rate_charged_pct:.2f}%")
        breakdown.append(f"Concessional Interest Spread: {rate_diff:.2f}%")

        # Medical treatment proviso handling
        if is_medical_treatment_specified_disease:
            if medical_insurance_reimbursement <= 0.0:
                return {
                    "perquisite_type": "interest_free_loan",
                    "statutory_rule": "Rule 3(7)(i)",
                    "regime": regime,
                    "assessment_year": assessment_year,
                    "is_exempt": True,
                    "exemption_reason": "Loan given for medical treatment of specified diseases under Rule 3A (no insurance reimbursement).",
                    "taxable_perquisite_value": 0.0,
                    "statutory_citation": "Rule 3(7)(i) Proviso 2",
                    "source_chunk_id": source_chunk_id,
                    "computation_steps": [
                        "Loan granted for medical treatment of specified diseases under Rule 3A.",
                        "Zero insurance reimbursement received; 100% exempt under Rule 3(7)(i) proviso 2.",
                        "Taxable Perquisite = Rs. 0.00"
                    ]
                }
            else:
                # Proviso specifies perquisite is taxable only to the extent of reimbursed portion
                taxable_base = min(max_outstanding_monthly_balance, medical_insurance_reimbursement)
                annual_perk = (taxable_base * (rate_diff / 100.0)) * (months_outstanding / 12)
                breakdown.append(f"Medical Loan with Insurance Reimbursement: Taxable base is reimbursed amount Rs. {taxable_base:,.2f}")
                breakdown.append(f"Perquisite on Rs. {taxable_base:,.2f} for {months_outstanding} months = Rs. {annual_perk:,.2f}")
                return {
                    "perquisite_type": "interest_free_loan",
                    "statutory_rule": "Rule 3(7)(i)",
                    "regime": regime,
                    "assessment_year": assessment_year,
                    "is_exempt": False,
                    "max_balance_considered": taxable_base,
                    "sbi_rate_pct": sbi_benchmark_rate_pct,
                    "employer_rate_pct": actual_interest_rate_charged_pct,
                    "rate_spread_pct": round(rate_diff, 2),
                    "months_outstanding": months_outstanding,
                    "taxable_perquisite_value": round(annual_perk, 2),
                    "statutory_citation": "Rule 3(7)(i) Proviso 2",
                    "source_chunk_id": source_chunk_id,
                    "computation_steps": breakdown
                }

        annual_perk = (max_outstanding_monthly_balance * (rate_diff / 100.0)) * (months_outstanding / 12)
        breakdown.append(f"Perquisite on Rs. {max_outstanding_monthly_balance:,.2f} for {months_outstanding} months = Rs. {annual_perk:,.2f}")

        return {
            "perquisite_type": "interest_free_loan",
            "statutory_rule": "Rule 3(7)(i)",
            "regime": regime,
            "assessment_year": assessment_year,
            "is_exempt": False,
            "max_balance_considered": max_outstanding_monthly_balance,
            "sbi_rate_pct": sbi_benchmark_rate_pct,
            "employer_rate_pct": actual_interest_rate_charged_pct,
            "rate_spread_pct": round(rate_diff, 2),
            "months_outstanding": months_outstanding,
            "taxable_perquisite_value": round(annual_perk, 2),
            "statutory_citation": source_citation,
            "source_chunk_id": source_chunk_id,
            "computation_steps": breakdown
        }


    @staticmethod
    def calculate_free_food(
        cost_per_meal: float,
        assessment_year: str = "AY 2024-25",
        num_meals_per_day: int = 1,
        working_days: int = 250,
        amount_recovered_per_meal: float = 0.0,
        is_remote_area_or_offshore: bool = False,
        is_tea_or_snacks: bool = False,
        regime: str = "1962"
    ) -> Dict[str, Any]:
        """Calculates taxable perquisite value of Free Food & Beverages under Rule 3(7)(iii)."""
        source_citation = "Rule 3(7)(iii)"
        source_chunk_id = "chunk_1962_rule_3_rule_3_subrule_7_49"

        if is_remote_area_or_offshore or is_tea_or_snacks:
            return {
                "perquisite_type": "free_food",
                "statutory_rule": "Rule 3(7)(iii)",
                "regime": regime,
                "assessment_year": assessment_year,
                "is_exempt": True,
                "taxable_perquisite_value": 0.0,
                "statutory_citation": source_citation,
                "source_chunk_id": source_chunk_id,
                "computation_steps": [
                    "Food provided in remote area/offshore or light tea/snacks during working hours is 100% exempt.",
                    "Taxable Perquisite = Rs. 0.00"
                ]
            }

        statutory_exemption_per_meal = 50.0
        taxable_cost_per_meal = max(0.0, (cost_per_meal - statutory_exemption_per_meal) - amount_recovered_per_meal)
        total_meals = num_meals_per_day * working_days
        total_taxable_value = taxable_cost_per_meal * total_meals

        return {
            "perquisite_type": "free_food",
            "statutory_rule": "Rule 3(7)(iii)",
            "regime": regime,
            "assessment_year": assessment_year,
            "cost_per_meal": cost_per_meal,
            "statutory_exemption_per_meal": statutory_exemption_per_meal,
            "amount_recovered_per_meal": amount_recovered_per_meal,
            "taxable_per_meal": round(taxable_cost_per_meal, 2),
            "total_meals": total_meals,
            "taxable_perquisite_value": round(total_taxable_value, 2),
            "statutory_citation": source_citation,
            "source_chunk_id": source_chunk_id,
            "computation_steps": [
                f"Meal Cost: Rs. {cost_per_meal:.2f} - Statutory Exemption Rs. 50.00 - Recovered Rs. {amount_recovered_per_meal:.2f} = Rs. {taxable_cost_per_meal:.2f}/meal",
                f"Total meals ({working_days} days * {num_meals_per_day} meals = {total_meals}) * Rs. {taxable_cost_per_meal:.2f} = Rs. {total_taxable_value:,.2f}"
            ]
        }

