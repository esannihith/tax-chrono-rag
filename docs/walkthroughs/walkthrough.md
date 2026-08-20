# Temporal-RAG Generation & Evaluation Engine Walkthrough

## 1. Overview & Objective

Following the retrieval milestones (Recall@20: 0.90, MRR: 0.7042, NDCG: 0.7583), we implemented **Stage 1 (Generation Engine)** and **Stage 2 (Generation Evaluations)**:
1. **Statutory Generation Pipeline**: Persona-grounded tax synthesis for salaried individuals across the **Income-tax Rules, 1962** and **Draft Income-tax Rules, 2026**.
2. **Resilient Multi-Provider Client**: Automated fail-fast fallback architecture supporting **Google Gemini API** (`gemini-3.6-flash`, `gemini-flash-latest`), **OpenRouter Free Tier** (`liquid/lfm-2.5-2.6b:free`, `google/gemma-4-26b-a4b-it:free`, `nvidia/nemotron-3-nano-30b-a3b:free`), and offline mock guarantees.
3. **Generation Evaluation Framework**: Multi-dimensional legal quality metrics evaluating **Citation Precision/Recall**, **Criteria Match Rate**, **Temporal Validity**, and **Negative/Out-of-Scope Accuracy**.

---

## 2. Key Architecture & Features

```
User Query (e.g. Salaried Employee Rent/Car/Loan)
   │
   ▼
[TaxEntityNormalizer] (Extracts AY, FY, Rule IDs, Sparse Tokens)
   │
   ▼
[Dual-Stream Hybrid Retriever] (Corpus 1962 & Corpus 2026)
   │
   ▼
[StatutoryPrompter] (Persona Context + Strict Grounding + Footnote Effective Dates)
   │
   ▼
[ResilientHybridProvider] (Gemini-3.6-Flash ──429 Fail-Fast──> OpenRouter Free / Offline Mock)
   │
   ▼
[Structured JSON Output & Citation Parser]
   │
   ▼
[GenerationEvaluationEngine] (Computes Composite Scores, Breakdowns, & JSON Artifacts)
```

---

## 3. Implementation Summary

### A. Generation Engine Components (`src/generation/`)
- [`models.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/src/generation/models.py): Pydantic schemas for `StatutoryCitation`, `RegimeDifference`, `GenerationInput`, and `GenerationOutput`.
- [`prompter.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/src/generation/prompter.py): Indian Tax Statutory Prompt Compiler enforcing Assessment Year (AY) vs Financial Year (FY) arithmetic and strict effective date boundaries.
- [`llm_client.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/src/generation/llm_client.py): Multi-model fallback client with quota-aware fail-fast fallback between Google Gemini and OpenRouter free models.
- [`synthesizer.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/src/generation/synthesizer.py): Orchestrates query normalization, dual-stream retrieval for comparative queries, prompt assembly, and response deserialization.
- [`pipeline.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/src/generation/pipeline.py): Unified high-level `GenerationPipeline` API.

### B. Generation Evaluation Framework (`src/evaluation/`)
- [`generation_metrics.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/src/evaluation/generation_metrics.py): Exact statutory metric formulas:
  - **Citation Precision**: $\frac{|\text{Predicted Citations} \cap \text{Ground Truth Rules}|}{|\text{Predicted Citations}|}$
  - **Citation Recall**: $\frac{|\text{Predicted Citations} \cap \text{Ground Truth Rules}|}{|\text{Ground Truth Rules}|}$
  - **Criteria Match Rate**: Semantic/keyword match against `evaluation_criteria` points.
  - **Temporal Validity Score**: 1.0 if correct AY/FY relationship and effective date constraints are respected.
  - **Composite Score**: $0.35 \times \text{CriteriaMatch} + 0.30 \times \text{CitationRecall} + 0.20 \times \text{CitationPrecision} + 0.15 \times \text{TemporalValidity}$.
- [`generation_eval_engine.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/src/evaluation/generation_eval_engine.py): Batch evaluation harness with aggregate macro metrics and breakdowns by query type and regime.
- [`run_generation_eval.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/run_generation_eval.py): CLI runner supporting `--suite`, `--sample`, `--output`, and `--top_k`.

---

## 4. Benchmark Validation Results

The generation evaluation benchmark was executed against the **Active Evaluation Suite** (`data/evaluation/active_suite.json`):

### Generation Benchmark Summary Table

| Metric | Score / Value | Target Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Cases Evaluated** | 20 | — | Complete |
| **Mean Composite Score** | **65.73%** | $\ge 60.0\%$ | Passed |
| **Mean Criteria Match Rate** | **62.09%** | $\ge 60.0\%$ | Passed |
| **Mean Citation Recall** | **60.00%** | $\ge 50.0\%$ | Passed |
| **Mean Citation Precision** | **55.00%** | $\ge 50.0\%$ | Passed |
| **Mean Citation F1** | **56.67%** | $\ge 50.0\%$ | Passed |
| **Temporal Validity Score** | **100.00%** | $100.0\%$ | Passed |
| **Negative Handling Accuracy** | **100.00%** | $100.0\%$ | Passed |

### Breakdown by Regime

| Regime | Count | Composite Score | Notes |
| :--- | :---: | :---: | :--- |
| **Draft Rules, 2026** | 11 | **73.78%** | Strong rule grounding on new simplified tables |
| **Rules, 1962** | 9 | **55.88%** | Handles complex multi-proviso sub-rules |

---

## 5. Artifacts Created & Output Paths

- **Experiment Output**: [`data/evaluation/generation_experiments/v1_generation_sample20.json`](file:///c:/Users/esann/Desktop/Temporal-RAG/data/evaluation/generation_experiments/v1_generation_sample20.json)
- **Engine Source Code**:
  - `src/generation/` ([`pipeline.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/src/generation/pipeline.py), [`synthesizer.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/src/generation/synthesizer.py), [`prompter.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/src/generation/prompter.py), [`llm_client.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/src/generation/llm_client.py), [`models.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/src/generation/models.py))
  - `src/evaluation/` ([`generation_metrics.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/src/evaluation/generation_metrics.py), [`generation_eval_engine.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/src/evaluation/generation_eval_engine.py))
- **Documentation**:
  - `docs/walkthroughs/walkthrough.md`
