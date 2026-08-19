# Walkthrough: Golden Evaluation Dataset Generation

We have generated, verified, and stratified a **120-case synthetic rich golden dataset** for Temporal-RAG benchmarking across the salaried persona in Indian tax law.

---

## 1. Summary of Accomplishments

1. **120-Case Golden Dataset Created**:
   - Strictly grounded in the 56 rule documents and 1,177 AST chunks.
   - **Active Suite (75%)**: 90 test cases in [`data/evaluation/active_suite.json`](file:///c:/Users/esann/Desktop/Temporal-RAG/data/evaluation/active_suite.json).
   - **Hidden Suite (25%)**: 30 test cases in [`data/evaluation/hidden_suite.json`](file:///c:/Users/esann/Desktop/Temporal-RAG/data/evaluation/hidden_suite.json).
   - **Consolidated Master Dataset**: 120 test cases in [`data/evaluation/golden_full.json`](file:///c:/Users/esann/Desktop/Temporal-RAG/data/evaluation/golden_full.json).
2. **Stratified Split on Query Types**:
   - **Persona-Specific Applicability**: 24 Active, 8 Hidden (Total: 32)
   - **Intra-Document Temporal Validation**: 24 Active, 8 Hidden (Total: 32)
   - **Procedural / Condition Matching**: 24 Active, 8 Hidden (Total: 32)
   - **Negative / Out-of-Scope (Null)**: 18 Active, 6 Hidden (Total: 24)
3. **Domain Refinements Implemented**:
   - **AY vs. FY Confusion Resolution**: Test cases evaluating queries with user-provided FY mapping to rule-specified AY (e.g. FY 2019-20 $\rightarrow$ AY 2020-21 for Rule 12AC).
   - **Nested Table Multi-Hops**: Multi-hop queries traversing complex tabular hierarchies (e.g. Rule 12AC Sl. No. 1(2)(B) EVC verification without DSC).
   - **Temporal Boundary Negatives**: Edge cases testing dates prior to rule notification (e.g. pre-29-04-2022 or AY 2017-18 for Rule 12AC) alongside domain negatives (MAT u/s 115JB, crypto TDS u/s 194S, GST input credits).
   - **Difficulty Grading**: Balanced across `Easy` (38), `Medium` (42), and `Hard` (40).
4. **Validation**:
   - 100% of non-null `ground_truth_chunk_ids` are verified against `data/processed/chunks_1962.json` and `chunks_2026.json`.
   - Zero hallucinated IDs or unresolvable legal references.

---

## 2. Sample Golden Evaluation Cases

### Sample 1: Intra-Document Temporal Validation with FY/AY Resolution
```json
{
  "id": "GOLD-TEMP-001",
  "query": "I realized I missed reporting my salary bonus for Financial Year (FY) 2019-20. Can I file an Updated Return of Income (ITR-U) under Rule 12AC of the 1962 rules?",
  "query_type": "Intra-Document Temporal Validation",
  "retrieval_paradigm": "hybrid_temporal",
  "difficulty_level": "Medium",
  "target_regime": "1962",
  "persona_context": "Salaried employee wanting to correct omitted income from FY 2019-20",
  "fy_ay_context": "User asks for FY 2019-20 (which corresponds to Assessment Year 2020-21)",
  "temporal_constraint": "Rule 12AC w.e.f. 29-04-2022 applies to AY commencing 1st April 2020 and subsequent years",
  "relevant_rules": ["Rule 12AC"],
  "ground_truth_chunk_ids": [
    "chunk_1962_rule_12AC_rule_12AC_subrule_1_1",
    "chunk_1962_rule_12AC_rule_12AC_subrule_2_2"
  ],
  "ground_truth_answer": "Yes. Financial Year (FY) 2019-20 corresponds to Assessment Year (AY) 2020-21. Under Rule 12AC(1) of the Income Tax Rules 1962 (inserted w.e.f. 29-04-2022), the return of income under section 139(8A) in Form ITR-U is eligible for the assessment year commencing on 1st April 2020 (AY 2020-21) and subsequent assessment years. Therefore, FY 2019-20 is eligible.",
  "evaluation_criteria": [
    "Maps FY 2019-20 to AY 2020-21",
    "Cites Rule 12AC(1)",
    "Confirms AY 2020-21 is the first eligible assessment year under Rule 12AC",
    "Identifies Form ITR-U u/s 139(8A)"
  ],
  "split": "active"
}
```

### Sample 2: Procedural Condition Matching with Nested Table Multi-Hop
```json
{
  "id": "GOLD-PROC-001",
  "query": "I am a salaried individual filing an Updated Return (ITR-U) under Rule 12AC. My accounts are not required to be audited, and I do NOT have a Digital Signature Certificate (DSC). How am I permitted to verify my return according to the table in Rule 12AC?",
  "query_type": "Procedural / Condition Matching",
  "retrieval_paradigm": "multi_hop",
  "difficulty_level": "Medium",
  "target_regime": "1962",
  "persona_context": "Individual salaried taxpayer with no business audit and no digital signature certificate",
  "relevant_rules": ["Rule 12AC"],
  "ground_truth_chunk_ids": [
    "chunk_1962_rule_12AC_rule_12AC_subrule_1_1",
    "chunk_1962_rule_12AC_rule_12AC_table_1_7"
  ],
  "ground_truth_answer": "Under Rule 12AC Table (Sl. No. 1(2)), for an individual in whose case accounts are not required to be audited, the return in Form ITR-U may be verified either: (A) Electronically under digital signature, OR (B) Transmitting the data electronically in the return under electronic verification code (EVC). Since you do not have a DSC, you are permitted to verify it by transmitting data electronically under EVC.",
  "evaluation_criteria": [
    "Traverses Rule 12AC Table Sl. No. 1(2)",
    "Identifies Sub-item (B) EVC verification",
    "Confirms DSC is not mandatory for unaudited individual"
  ],
  "split": "active"
}
```

### Sample 3: Negative / Out-of-Scope with Temporal Boundary Testing
```json
{
  "id": "GOLD-NEG-006",
  "query": "Can I submit Form ITR-U under Rule 12AC on 15th January 2021 for my salaried income?",
  "query_type": "Negative / Out-of-Scope",
  "retrieval_paradigm": "negative_null",
  "difficulty_level": "Hard",
  "target_regime": "none",
  "persona_context": "Temporal boundary query testing date prior to rule notification",
  "temporal_constraint": "Rule 12AC came into force only on 29-04-2022",
  "relevant_rules": ["Rule 12AC"],
  "ground_truth_chunk_ids": [],
  "ground_truth_answer": "This request is temporally invalid. Rule 12AC and Form ITR-U were inserted by the Income-tax (Eleventh Amendment) Rules, 2022 and came into force only on 29th April, 2022. On 15th January 2021, Rule 12AC and Form ITR-U did not legally exist.",
  "evaluation_criteria": [
    "Identifies temporal boundary violation",
    "States Rule 12AC came into force on 29-04-2022",
    "Confirms Rule 12AC did not exist on 15-01-2021"
  ],
  "split": "active"
}
```

---

## 3. Code Modules in `src/evaluation/`

- [`src/evaluation/models.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/src/evaluation/models.py): Pydantic data schemas (`GoldenEvaluationCase`, `EvaluationSuite`, `DifficultyLevel`, `QueryType`, `RetrievalParadigm`).
- [`src/evaluation/split_manager.py`](file:///c:/Users/esann/Desktop/Temporal-RAG/src/evaluation/split_manager.py): Stratified dataset splitting and suite export manager.
