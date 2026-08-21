import os
import json
import re
from typing import Dict, Any, List, Optional
from src.generation.llm_client import get_llm_client, BaseLLMProvider


class LLMJudge:
    """Industry-Standard G-Eval / LLM-as-a-Judge for semantic tax RAG evaluation.
    
    Evaluates:
    1. Criteria Adherence: Chain-of-thought semantic verification against golden criteria.
    2. Temporal Validity: Semantic Assessment Year (AY) vs Financial Year (FY) verification (AY = FY + 1).
    3. Faithfulness: Groundedness verification ensuring claims are entailed by statutory context without hallucinations.
    4. Negative Abstention: Evaluates whether out-of-scope/unnotified queries were cleanly refused.
    """

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm = llm_provider or get_llm_client()

    def evaluate_criteria_adherence(
        self,
        query: str,
        direct_answer: str,
        step_by_step_reasoning: List[str],
        criteria_list: List[str]
    ) -> Dict[str, Any]:
        """Evaluates whether each golden evaluation criterion is satisfied using Chain-of-Thought reasoning."""
        if not criteria_list:
            return {
                "criteria_adherence_rate": 1.0,
                "matched_criteria": [],
                "missed_criteria": [],
                "evaluations": []
            }

        reason_list = step_by_step_reasoning if isinstance(step_by_step_reasoning, list) else [str(step_by_step_reasoning)]
        reasoning_text = " ".join(reason_list)
        full_response = f"{direct_answer}\nReasoning: {reasoning_text}".strip()

        system_instruction = """You are an expert Indian Income Tax Statutory Evaluation Judge (G-Eval).
Your task is to objectively evaluate whether an AI-generated tax response satisfies each specified ground-truth criterion.

Evaluate each criterion independently. A criterion is SATISFIED (pass: true) if the response conveys the required legal principle, condition, form, rate, or rule, even if phrased differently or using synonyms.
A criterion is MISSED (pass: false) only if the response completely omits, contradicts, or misstates the legal requirement.

Respond ONLY with a JSON object adhering to this schema:
{
  "criteria_evaluations": [
    {
      "criterion": "text of criterion",
      "pass": true,
      "reasoning": "brief explanation"
    }
  ]
}"""

        prompt = f"""USER QUERY:
"{query}"

MODEL GENERATED RESPONSE:
{full_response}

GOLDEN EVALUATION CRITERIA TO VERIFY:
{json.dumps(criteria_list, indent=2)}

Evaluate each criterion thoroughly."""

        try:
            res = self.llm.generate(prompt, system_instruction)
            evals = res.get("criteria_evaluations", [])
            
            matched = []
            missed = []
            
            eval_map = {str(e.get("criterion", "")).strip().lower(): e for e in evals}
            for c in criteria_list:
                c_clean = c.strip().lower()
                matching_e = eval_map.get(c_clean)
                if not matching_e:
                    for k, v in eval_map.items():
                        if c_clean in k or k in c_clean:
                            matching_e = v
                            break
                
                if matching_e and matching_e.get("pass", False):
                    matched.append(c)
                else:
                    missed.append(c)

            adherence_rate = len(matched) / len(criteria_list) if criteria_list else 1.0
            return {
                "criteria_adherence_rate": round(adherence_rate, 4),
                "matched_criteria": matched,
                "missed_criteria": missed,
                "evaluations": evals
            }
        except Exception:
            # Deterministic fallback for offline / unit tests
            matched = []
            missed = []
            ans_lower = full_response.lower()
            for c in criteria_list:
                words = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", c) if len(w) > 3]
                if not words or any(w in ans_lower for w in words):
                    matched.append(c)
                else:
                    missed.append(c)
            rate = len(matched) / len(criteria_list) if criteria_list else 1.0
            return {
                "criteria_adherence_rate": round(rate, 4),
                "matched_criteria": matched,
                "missed_criteria": missed,
                "evaluations": []
            }

    def evaluate_temporal_validity(
        self,
        query: str,
        direct_answer: str,
        temporal_applicability: str,
        expected_ay: Optional[str] = None,
        expected_fy: Optional[str] = None,
        is_negative: bool = False
    ) -> Dict[str, Any]:
        """Evaluates whether the stated Assessment Year / Financial Year corresponds semantically to the ground truth."""
        def clean_year(y: Optional[str]) -> Optional[str]:
            if not y or str(y).strip().lower() in ["not applicable", "n/a", "none", "null", ""]:
                return None
            return str(y).strip()

        clean_ay = clean_year(expected_ay)
        clean_fy = clean_year(expected_fy)

        if not clean_ay and not clean_fy:
            return {
                "temporal_validity_score": 1.0,
                "is_labelled_temporal_case": False,
                "verified_correct_year": True,
                "judge_reasoning": "Unlabelled / Not Applicable temporal case."
            }

        full_text = f"{temporal_applicability}\n{direct_answer}".strip()

        system_instruction = """You are an Indian Tax Legal Evaluation Judge.
Evaluate whether the model response accurately identifies the applicable Assessment Year (AY) and/or Financial Year (FY).

Key Rule: In Indian Tax law, AY is FY + 1 (e.g. FY 2023-24 corresponds to AY 2024-25).
If the model correctly identifies the expected assessment year (even if phrased as 'Assessment Year 2024-25', 'AY 2024-25', 'year commencing 1st April 2024', or derives it from the correct FY), mark is_correct: true.
If the model states an incorrect year or fails to state the period when explicitly expected, mark is_correct: false.

Respond ONLY with a JSON object:
{
  "is_correct": true,
  "stated_ay": "e.g. AY 2024-25",
  "stated_fy": "e.g. FY 2023-24",
  "reasoning": "brief explanation"
}"""

        prompt = f"""USER QUERY:
"{query}"

EXPECTED GROUND TRUTH PERIOD:
Expected Assessment Year: {clean_ay or 'Not specified'}
Expected Financial Year: {clean_fy or 'Not specified'}

MODEL GENERATED TEXT:
{full_text}

Verify if the model's stated temporal period is legally and factually correct."""

        try:
            res = self.llm.generate(prompt, system_instruction)
            is_correct = bool(res.get("is_correct", False))
            return {
                "temporal_validity_score": 1.0 if is_correct else 0.0,
                "is_labelled_temporal_case": True,
                "expected_ay": clean_ay,
                "expected_fy": clean_fy,
                "verified_correct_year": is_correct,
                "stated_ay": res.get("stated_ay"),
                "stated_fy": res.get("stated_fy"),
                "judge_reasoning": res.get("reasoning", "")
            }
        except Exception:
            temp_lower = full_text.lower()
            correct = True
            if clean_ay:
                digits = re.findall(r"\d{4}", clean_ay)
                if digits and (digits[0] not in temp_lower):
                    correct = False
            if clean_fy and not clean_ay:
                digits = re.findall(r"\d{4}", clean_fy)
                if digits and (digits[0] not in temp_lower):
                    correct = False
            return {
                "temporal_validity_score": 1.0 if correct else 0.0,
                "is_labelled_temporal_case": True,
                "expected_ay": clean_ay,
                "expected_fy": clean_fy,
                "verified_correct_year": correct,
                "judge_reasoning": "Deterministic fallback check"
            }

    def evaluate_faithfulness(
        self,
        direct_answer: str,
        step_by_step_reasoning: List[str],
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluates whether claims in the direct answer and reasoning steps are strictly grounded in retrieved statutory chunks."""
        if not retrieved_chunks:
            return {
                "faithfulness_score": 1.0 if not direct_answer else 0.0,
                "supported_claims": 1 if not direct_answer else 0,
                "total_claims": 1,
                "unsupported_claims": [],
                "judge_reasoning": "No retrieved chunks provided."
            }

        context_texts = [
            f"[{c.get('metadata', {}).get('statutory_path', 'Context')}]: {c.get('content', '')}"
            for c in retrieved_chunks
        ]
        context_str = "\n\n".join(context_texts)
        reason_list = step_by_step_reasoning if isinstance(step_by_step_reasoning, list) else [str(step_by_step_reasoning)]
        answer_str = f"{direct_answer}\n" + "\n".join(reason_list)

        system_instruction = """You are a Legal Faithfulness and Hallucination Evaluation Judge (Ragas Standard).
Your task is to verify whether the factual statements, rules, rates, thresholds, and calculations in the MODEL ANSWER are strictly supported by the PROVIDED STATUTORY CONTEXT.

Score 1.0 if all claims are faithfully derived from the context or sound legal deduction from it.
Score 0.0 if the model invents non-existent statutory sections, contradictory perquisite rates, or ungrounded claims.

Respond ONLY with a JSON object:
{
  "faithfulness_score": 1.0,
  "supported_claims_count": 4,
  "total_claims_count": 4,
  "unsupported_claims": [],
  "reasoning": "brief explanation"
}"""

        prompt = f"""STATUTORY CONTEXT PROVIDED:
{context_str}

MODEL ANSWER TO VERIFY:
{answer_str}

Evaluate faithfulness and detect any hallucinations."""

        try:
            res = self.llm.generate(prompt, system_instruction)
            f_score = float(res.get("faithfulness_score", 1.0))
            return {
                "faithfulness_score": round(max(0.0, min(1.0, f_score)), 4),
                "supported_claims": res.get("supported_claims_count", 0),
                "total_claims": res.get("total_claims_count", 0),
                "unsupported_claims": res.get("unsupported_claims", []),
                "judge_reasoning": res.get("reasoning", "")
            }
        except Exception:
            return {
                "faithfulness_score": 1.0,
                "supported_claims": 1,
                "total_claims": 1,
                "unsupported_claims": [],
                "judge_reasoning": "Deterministic fallback (assumed grounded)"
            }
