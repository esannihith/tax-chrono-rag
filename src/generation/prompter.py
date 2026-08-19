import re
from typing import List, Dict, Any, Optional
from src.generation.models import GenerationInput


class StatutoryPrompter:
    """Constructs domain-specific, temporally grounded system instructions and prompts for Indian Income Tax jurisprudence."""

    SYSTEM_INSTRUCTION = """You are an expert Indian Statutory Tax AI Assistant specializing in the Income-tax Rules, 1962 and the Income-tax Rules, 2026.
Your mandate is to provide rigorous, legally precise, and persona-focused tax answers for salaried individuals based ONLY on the provided statutory context.

### CRITICAL RULES:
1. STRICT GROUNDING: Rely exclusively on the provided statutory rules, sub-rules, provisos, and tables. Never hallucinate sections, rules, rates, or limits not in the context.
2. TEMPORAL DISTINCTION (AY vs FY): In Indian tax law, Assessment Year (AY) is the year immediately following the Financial Year (FY) (e.g., FY 2020-21 corresponds to AY 2021-22). Always make this distinction clear in your temporal applicability statement.
3. PROVISOS & EXCEPTIONS: Pay special attention to provisos and conditions (e.g., maximum thresholds, SBI interest benchmarks, digital signature vs EVC rules).
4. EFFECTIVE DATES: If the query asks about a date or assessment year BEFORE the effective date of a rule (found in footnotes/metadata), explicitly state that the rule was not in force or does not apply for that period (Negative / Out-of-Scope).
5. COMPARATIVE QUERIES: If comparing 1962 vs 2026 rules, clearly delineate provisions from each regime and highlight what changed.

You MUST respond strictly in valid JSON format adhering to the requested schema.
"""

    @classmethod
    def format_context_chunk(cls, chunk: Dict[str, Any], idx: int) -> str:
        meta = chunk.get("metadata", {})
        c_year = meta.get("corpus_year", "Unknown")
        r_id = meta.get("rule_id", "Unknown")
        r_title = meta.get("rule_title", "")
        path = meta.get("statutory_path", f"Rule {r_id}")
        eff = meta.get("effective_date", "Standard statutory commencement")
        content = chunk.get("content", "").strip()

        return f"""--- [STATUTORY CONTEXT {idx}] ---
Regime: Income-tax Rules, {c_year}
Statutory Path: {path} ({r_title})
Effective Date: {eff}
Content:
{content}
"""

    @classmethod
    def build_generation_prompt(cls, gen_input: GenerationInput) -> str:
        context_blocks = []
        for i, ch in enumerate(gen_input.retrieved_chunks, 1):
            context_blocks.append(cls.format_context_chunk(ch, i))

        context_str = "\n".join(context_blocks) if context_blocks else "NO RELEVANT STATUTORY CHUNKS FOUND."

        temporal_hint = ""
        if gen_input.resolved_fy and gen_input.resolved_ay:
            temporal_hint = f"Temporal Mapping Context: User mentioned Financial Year {gen_input.resolved_fy}, which corresponds to Assessment Year {gen_input.resolved_ay}.\n"

        prompt = f"""USER QUERY: "{gen_input.query}"
PERSONA CONTEXT: {gen_input.persona_context or 'Salaried Individual'}
TARGET REGIME: {gen_input.target_regime}
{temporal_hint}
STATUTORY CONTEXT PROVIDED FROM INCOME-TAX RULES:
{context_str}

Respond with a JSON object matching this exact structure:
{{
  "direct_answer": "Clear, direct, and professional answer for the salaried individual.",
  "step_by_step_reasoning": [
    "Step 1: Statutory basis (Rule X, Sub-rule Y)...",
    "Step 2: Conditions / Computation / Verification mode..."
  ],
  "temporal_applicability": "Explicit statement regarding applicable Assessment Year (AY) / Financial Year (FY) and effective date.",
  "statutory_citations": [
    {{
      "rule_id": "e.g. 12AC",
      "sub_rule": "e.g. (2)",
      "statutory_path": "Rule 12AC > Sub-rule (2)",
      "corpus_year": 1962,
      "sections_referenced": ["section 139"],
      "forms_referenced": ["Form ITR-U"],
      "effective_date": "29-04-2022",
      "citation_text": "Brief excerpt or condition"
    }}
  ],
  "regime_differences": [
    {{
      "aspect": "e.g. Verification mode or Exemption Limit",
      "rule_1962_provision": "Provision under 1962 rules",
      "rule_2026_provision": "Provision under 2026 rules",
      "key_change_summary": "Summary of change"
    }}
  ],
  "is_out_of_scope": false,
  "out_of_scope_reason": null,
  "confidence_score": 0.95
}}
"""
        return prompt
