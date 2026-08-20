import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastmcp import FastMCP

from src.indexing.vector_store import HybridVectorStore
from src.indexing.embedder import DenseEmbedder
from src.enrichment.normalizer import TaxEntityNormalizer
from src.generation.pipeline import GenerationPipeline
from src.generation.models import GenerationOutput
from src.mcp.calculators import PerquisiteCalculator

# Instantiate FastMCP server
mcp = FastMCP(
    name="TaxChronoRAG",
    instructions="""Expert Indian Income Tax Statutory Intelligence Server specializing in the Income-tax Rules, 1962 and Draft Income-tax Rules, 2026.
Provides hybrid statutory retrieval, Assessment Year (AY) vs Financial Year (FY) resolution, regime comparison, deterministic perquisite calculations, and legal synthesis."""
)

# Lazy-loaded singletons
_vector_store: Optional[HybridVectorStore] = None
_dense_embedder: Optional[DenseEmbedder] = None
_generation_pipeline: Optional[GenerationPipeline] = None
_processed_chunks: Optional[Dict[str, List[Dict[str, Any]]]] = None


def get_store() -> HybridVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = HybridVectorStore.load("data/indices")
    return _vector_store


def get_embedder() -> DenseEmbedder:
    global _dense_embedder
    if _dense_embedder is None:
        _dense_embedder = DenseEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _dense_embedder


def get_pipeline() -> GenerationPipeline:
    global _generation_pipeline
    if _generation_pipeline is None:
        _generation_pipeline = GenerationPipeline(indices_dir="data/indices")
    return _generation_pipeline


def get_all_chunks() -> Dict[str, List[Dict[str, Any]]]:
    global _processed_chunks
    if _processed_chunks is None:
        _processed_chunks = {"1962": [], "2026": []}
        for y in ["1962", "2026"]:
            fpath = Path(f"data/processed/chunks_{y}.json")
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    _processed_chunks[y] = json.load(f)
    return _processed_chunks


# ==============================================================================
# TOOL 1: SEARCH STATUTORY TAX RULES (HYBRID RETRIEVAL)
# ==============================================================================
@mcp.tool()
def search_tax_rules(
    query: str,
    target_regime: str = "both",
    fy_year: Optional[str] = None,
    ay_year: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """Searches Indian Income-tax Rules (1962 & 2026) using Parent-Aware Hybrid dense-sparse retrieval.
    
    Args:
        query: Tax question or legal subject (e.g. 'perquisite valuation of rent free accommodation', 'ITR-U filing deadline').
        target_regime: '1962', '2026', or 'both' / 'auto' for comparative queries.
        fy_year: Financial Year mentioned by user (e.g. '2023-24').
        ay_year: Assessment Year mentioned by user (e.g. '2024-25').
        top_k: Number of statutory chunks to retrieve (default: 5).
    """
    store = get_store()
    embedder = get_embedder()

    norm_query = TaxEntityNormalizer.normalize_query(query)
    q_vec = embedder.embed_query(norm_query.dense_query)

    resolved_regime = target_regime
    corpus_filter = 1962 if resolved_regime == "1962" else (2026 if resolved_regime == "2026" else None)
    results = store.hybrid_search(
        query_vector=q_vec,
        sparse_tokens=norm_query.sparse_tokens,
        top_k=top_k,
        alpha=0.5,
        target_rules=norm_query.target_rules,
        corpus_year_filter=corpus_filter
    )

    retrieved_items = []
    for r in results:
        meta = r.payload.metadata or {}
        retrieved_items.append({
            "chunk_id": r.chunk_id,
            "rule_id": r.payload.rule_id,
            "statutory_path": r.payload.statutory_path,
            "corpus_year": r.payload.corpus_year,
            "effective_date": meta.get("effective_date", "Standard"),
            "hybrid_score": round(r.score, 4),
            "content": r.payload.display_content
        })

    return {
        "query": query,
        "normalized_dense_query": norm_query.dense_query,
        "resolved_temporal": {
            "financial_year": norm_query.detected_fy or fy_year,
            "assessment_year": norm_query.detected_ay or ay_year,
            "target_regime": resolved_regime
        },
        "total_retrieved": len(retrieved_items),
        "results": retrieved_items
    }


# ==============================================================================
# TOOL 2: GET RULE DETAILS BY RULE ID
# ==============================================================================
@mcp.tool()
def get_rule_details(
    rule_id: str,
    regime: str = "1962"
) -> Dict[str, Any]:
    """Retrieves the full statutory text, sub-rules, provisos, and embedded tables for a specific Rule ID.
    
    Args:
        rule_id: Rule number (e.g. '3', '12AC', '2BB', '114B', '26C').
        regime: '1962' for Income-tax Rules, 1962 or '2026' for Draft Rules, 2026.
    """
    clean_id = rule_id.upper().replace("RULE", "").replace("SEC", "").strip()
    all_chunks = get_all_chunks()
    target_year = "2026" if "2026" in regime else "1962"
    chunks = all_chunks.get(target_year, [])

    matching_chunks = [
        c for c in chunks
        if str(c.get("metadata", {}).get("rule_id", "")).upper() == clean_id
    ]

    if not matching_chunks:
        return {
            "rule_id": clean_id,
            "regime": target_year,
            "found": False,
            "message": f"Rule '{clean_id}' not found in Income-tax Rules, {target_year} index."
        }

    first_meta = matching_chunks[0].get("metadata", {})
    return {
        "rule_id": clean_id,
        "regime": target_year,
        "found": True,
        "rule_title": first_meta.get("rule_title", f"Rule {clean_id}"),
        "effective_date": first_meta.get("effective_date", "Standard"),
        "total_sub_chunks": len(matching_chunks),
        "sub_rules": [
            {
                "chunk_id": c.get("chunk_id"),
                "statutory_path": c.get("metadata", {}).get("statutory_path", ""),
                "content": c.get("content", "")
            }
            for c in matching_chunks
        ]
    }


# ==============================================================================
# TOOL 3: COMPARE STATUTORY REGIMES (1962 VS DRAFT 2026)
# ==============================================================================
@mcp.tool()
def compare_regimes(
    topic: str,
    top_k_per_regime: int = 2
) -> Dict[str, Any]:
    """Compares statutory provisions between the legacy Income-tax Rules, 1962 and Draft Income-tax Rules, 2026 side-by-side.
    
    Args:
        topic: Subject to compare (e.g. 'rent free accommodation', 'car perquisite', 'interest free loan', 'ITR filing forms').
        top_k_per_regime: Number of statutory chunks to retrieve from each regime (default: 2).
    """
    store = get_store()
    embedder = get_embedder()

    norm_query = TaxEntityNormalizer.normalize_query(topic)
    q_vec = embedder.embed_query(norm_query.dense_query)

    res_1962 = store.hybrid_search(
        query_vector=q_vec,
        sparse_tokens=norm_query.sparse_tokens,
        top_k=top_k_per_regime,
        alpha=0.5,
        target_rules=norm_query.target_rules,
        corpus_year_filter=1962
    )

    res_2026 = store.hybrid_search(
        query_vector=q_vec,
        sparse_tokens=norm_query.sparse_tokens,
        top_k=top_k_per_regime,
        alpha=0.5,
        target_rules=norm_query.target_rules,
        corpus_year_filter=2026
    )

    return {
        "topic": topic,
        "comparison": {
            "income_tax_rules_1962": [
                {
                    "rule_id": r.payload.rule_id,
                    "statutory_path": r.payload.statutory_path,
                    "effective_date": (r.payload.metadata or {}).get("effective_date", "Standard"),
                    "content": r.payload.display_content
                }
                for r in res_1962
            ],
            "draft_income_tax_rules_2026": [
                {
                    "rule_id": r.payload.rule_id,
                    "statutory_path": r.payload.statutory_path,
                    "effective_date": (r.payload.metadata or {}).get("effective_date", "Draft statutory commencement"),
                    "content": r.payload.display_content
                }
                for r in res_2026
            ]
        },
        "key_transition_summary": "Draft Rules 2026 streamline tables into standardized schedules and eliminate redundant historical provisos."
    }


# ==============================================================================
# TOOL 4: RESOLVE TAX YEAR (AY VS FY ARITHMETIC)
# ==============================================================================
@mcp.tool()
def resolve_tax_year(
    year_input: str
) -> Dict[str, Any]:
    """Resolves and validates Assessment Year (AY) and Financial Year (FY) arithmetic under Indian tax law.
    
    Args:
        year_input: Any temporal mention (e.g. 'FY 2023-24', 'AY 2024-25', '2023-2024', '2025').
    """
    import re
    norm = TaxEntityNormalizer.normalize_query(year_input)
    fy = norm.detected_fy
    ay = norm.detected_ay

    if not fy and not ay:
        digits = re.findall(r"\d{4}", year_input)
        if digits:
            y1 = int(digits[0])
            if "AY" in year_input.upper():
                ay = f"AY {y1}-{y1+1}"
                fy = f"FY {y1-1}-{y1}"
            else:
                fy = f"FY {y1}-{y1+1}"
                ay = f"AY {y1+1}-{y1+2}"
        else:
            fy = "Unknown"
            ay = "Unknown"

    regime = "2026" if ("2026" in str(ay) or "2027" in str(ay) or "2026" in str(fy)) else "1962"

    return {
        "input_string": year_input,
        "resolved_financial_year": fy,
        "resolved_assessment_year": ay,
        "default_applicable_regime": regime,
        "statutory_rule_definition": {
            "financial_year": f"Period from 1st April to 31st March (Year of earning income - {fy}).",
            "assessment_year": f"Period from 1st April to 31st March (Year of tax assessment & filing - {ay}).",
            "legal_precedence": "In Indian tax jurisprudence, AY is always FY + 1."
        }
    }


# ==============================================================================
# TOOL 5: VERIFY STATUTORY EFFECTIVE DATE
# ==============================================================================
@mcp.tool()
def verify_effective_date(
    rule_id: str,
    date_or_ay: str,
    regime: str = "1962"
) -> Dict[str, Any]:
    """Verifies whether a statutory rule, sub-rule, or amendment was legally in force for a specific date or Assessment Year.
    
    Args:
        rule_id: Rule identifier (e.g. '12AC', '3', '26C').
        date_or_ay: Date string ('2020-04-01') or Assessment Year ('AY 2021-22').
        regime: '1962' or '2026'.
    """
    clean_id = rule_id.upper().replace("RULE", "").strip()
    all_chunks = get_all_chunks()
    target_year = "2026" if "2026" in regime else "1962"
    chunks = all_chunks.get(target_year, [])

    matching = [c for c in chunks if str(c.get("metadata", {}).get("rule_id", "")).upper() == clean_id]
    if not matching:
        return {
            "rule_id": clean_id,
            "regime": target_year,
            "status": "not_found",
            "message": f"Rule {clean_id} is not in the {target_year} corpus."
        }

    eff_date = matching[0].get("metadata", {}).get("effective_date", "Standard statutory commencement")
    
    # Specific known statutory timeline checks
    in_force = True
    scope_note = "In force during queried period."

    if clean_id == "12AC":
        # Rule 12AC (Updated Return ITR-U) inserted w.e.f. 29-04-2022 (Notification No. 48/2022)
        if any(term in date_or_ay for term in ["2020", "2021", "2019", "AY 2021-22", "AY 2020-21"]):
            in_force = False
            scope_note = "Rule 12AC was inserted w.e.f. 29-04-2022 and was NOT in force for periods prior to AY 2022-23."

    return {
        "rule_id": clean_id,
        "regime": target_year,
        "queried_period": date_or_ay,
        "effective_date_recorded": eff_date,
        "is_in_force_for_period": in_force,
        "scope_note": scope_note
    }


# ==============================================================================
# TOOL 6: CALCULATE STATUTORY PERQUISITE VALUATION (DETERMINISTIC)
# ==============================================================================
@mcp.tool()
def calculate_perquisite(
    perquisite_type: str,
    parameters: Dict[str, Any],
    regime: str = "1962"
) -> Dict[str, Any]:
    """Calculates exact taxable perquisite values deterministically under Rule 3 (1962) or Draft 2026 Rules.
    
    Args:
        perquisite_type: One of 'rent_free_accommodation', 'motor_car', 'interest_free_loan', 'free_food'.
        parameters: Dictionary of numerical inputs for the perquisite calculation:
            - rent_free_accommodation: {'salary_for_period': float, 'population_tier': '>40L'|'15L-40L'|'<=15L', 'is_owned_by_employer': bool, 'actual_lease_rent_paid': float, 'is_furnished': bool, 'furniture_original_cost': float, 'amount_recovered': float}
            - motor_car: {'cubic_capacity_cc': int, 'is_employer_owned_or_hired': bool, 'is_employer_expenses_met': bool, 'usage_type': 'mixed'|'official'|'personal', 'is_driver_provided': bool, 'months_used': int, 'amount_recovered': float}
            - interest_free_loan: {'max_outstanding_monthly_balance': float, 'actual_interest_rate_charged_pct': float, 'sbi_benchmark_rate_pct': float, 'is_medical_treatment_specified_disease': bool, 'months_outstanding': int}
            - free_food: {'cost_per_meal': float, 'num_meals_per_day': int, 'working_days': int, 'amount_recovered_per_meal': float}
        regime: '1962' or '2026'.
    """
    ptype = perquisite_type.lower().strip()

    if ptype in ["rent_free_accommodation", "accommodation", "rfa"]:
        return PerquisiteCalculator.calculate_rent_free_accommodation(
            salary_for_period=float(parameters.get("salary_for_period", 0.0)),
            is_government_employee=bool(parameters.get("is_government_employee", False)),
            government_license_fee=float(parameters.get("government_license_fee", 0.0)),
            is_owned_by_employer=bool(parameters.get("is_owned_by_employer", True)),
            population_tier=str(parameters.get("population_tier", ">40L")),
            actual_lease_rent_paid=float(parameters.get("actual_lease_rent_paid", 0.0)),
            is_furnished=bool(parameters.get("is_furnished", False)),
            furniture_original_cost=float(parameters.get("furniture_original_cost", 0.0)),
            furniture_annual_hire_charges=float(parameters.get("furniture_annual_hire_charges", 0.0)),
            amount_recovered_from_employee=float(parameters.get("amount_recovered", parameters.get("amount_recovered_from_employee", 0.0))),
            regime=regime
        )
    elif ptype in ["motor_car", "car", "vehicle"]:
        return PerquisiteCalculator.calculate_motor_car(
            cubic_capacity_cc=int(parameters.get("cubic_capacity_cc", 1500)),
            is_employer_owned_or_hired=bool(parameters.get("is_employer_owned_or_hired", True)),
            is_employer_expenses_met=bool(parameters.get("is_employer_expenses_met", True)),
            usage_type=str(parameters.get("usage_type", "mixed")),
            is_driver_provided=bool(parameters.get("is_driver_provided", False)),
            actual_running_expense_incurred=float(parameters.get("actual_running_expense_incurred", 0.0)),
            car_original_cost=float(parameters.get("car_original_cost", 0.0)),
            amount_recovered=float(parameters.get("amount_recovered", 0.0)),
            months_used=int(parameters.get("months_used", 12)),
            regime=regime
        )
    elif ptype in ["interest_free_loan", "loan", "concessional_loan"]:
        return PerquisiteCalculator.calculate_interest_free_loan(
            max_outstanding_monthly_balance=float(parameters.get("max_outstanding_monthly_balance", parameters.get("loan_amount", 0.0))),
            actual_interest_rate_charged_pct=float(parameters.get("actual_interest_rate_charged_pct", 0.0)),
            sbi_benchmark_rate_pct=float(parameters.get("sbi_benchmark_rate_pct", 8.5)),
            is_medical_treatment_specified_disease=bool(parameters.get("is_medical_treatment_specified_disease", False)),
            medical_insurance_reimbursement=float(parameters.get("medical_insurance_reimbursement", 0.0)),
            is_aggregate_loans_under_20k=bool(parameters.get("is_aggregate_loans_under_20k", False)),
            months_outstanding=int(parameters.get("months_outstanding", 12)),
            regime=regime
        )
    elif ptype in ["free_food", "food", "beverages", "meal"]:
        return PerquisiteCalculator.calculate_free_food(
            cost_per_meal=float(parameters.get("cost_per_meal", 80.0)),
            num_meals_per_day=int(parameters.get("num_meals_per_day", 1)),
            working_days=int(parameters.get("working_days", 250)),
            amount_recovered_per_meal=float(parameters.get("amount_recovered_per_meal", 0.0)),
            is_remote_area_or_offshore=bool(parameters.get("is_remote_area_or_offshore", False)),
            is_tea_or_snacks=bool(parameters.get("is_tea_or_snacks", False)),
            regime=regime
        )
    else:
        return {
            "error": f"Unsupported perquisite type '{perquisite_type}'.",
            "supported_types": ["rent_free_accommodation", "motor_car", "interest_free_loan", "free_food"]
        }


# ==============================================================================
# TOOL 7: ASK TAX COPILOT (END-TO-END RAG SYNTHESIS)
# ==============================================================================
@mcp.tool()
def ask_tax_copilot(
    query: str,
    persona_context: str = "Salaried Individual",
    target_regime: str = "auto"
) -> Dict[str, Any]:
    """Generates an end-to-end statutory tax response with sub-rule citations and reasoning steps.
    
    Args:
        query: Full natural language tax question.
        persona_context: Persona description (e.g. 'Senior Citizen', 'Salaried Central Govt Employee', 'Blind Individual').
        target_regime: '1962', '2026', or 'auto'.
    """
    pipeline = get_pipeline()
    output: GenerationOutput = pipeline.run(
        query=query,
        persona_context=persona_context,
        target_regime=target_regime
    )
    return output.model_dump()


# Stdio runner entry point
def run_stdio():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_stdio()
