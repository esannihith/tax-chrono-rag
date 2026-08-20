# Walkthrough: Structure-Aware Parent-Child Chunking & Versioned Benchmark Trajectory

We have overhauled the chunking architecture, eliminated redundant metadata fields, remapped the golden evaluation ground truths, and implemented versioned experiment tracking in `data/evaluation/experiments/`.

---

## 1. Chunking Strategy Revision: Why and How

### The Problem with Fragmented AST Micro-Chunking (v1 & v2)
- In the initial pass, every individual sub-clause, table row, and proviso was split into a separate micro-chunk (1,177 chunks total, median length: **136 characters**).
- **The Bottleneck**: Indian legal drafting relies on dense co-dependencies (e.g. Rule 12AC(1) establishes Form ITR-U eligibility for AY 2020-21, while the attached Table defines verification modes like EVC/DSC). Sharding this into 136-char slivers meant queries matched random fragments, burying the primary answer and driving **MRR down to 0.18–0.28**.

### The Solution: Structure-Aware Parent-Child Chunking (v3 & v4)
- **Compact Rules ($\le 4,500$ chars)**: Grouped as a complete, single semantic unit (Rule + Sub-Rules + Provisos + Tables).
- **Large Rules (Rule 2BB, Rule 3, Rule 280)**: Chunked by **Sub-Rule / Section Table**, binding all qualifying provisos, explanations, and column headers to their parent sub-rule.
- **Corpus Size Reduction**: Consolidated from **1,177 fragmented micro-chunks $\rightarrow$ 183 rich, self-contained chunks** (95 in 1962, 88 in 2026; median length: 673 characters).
- **Metadata Streamlining**:
  - Removed 100% null fields (`page_numbers`).
  - Added `parent_id` tracking.
  - Included `sections_referenced`, `forms_referenced`, `effective_date` only when non-empty.

---

## 2. Versioned Experiment Trajectory & Growth Comparison

All experimental runs are versioned and stored under [`data/evaluation/experiments/`](file:///c:/Users/esann/Desktop/Temporal-RAG/data/evaluation/experiments/):

```
========================================================================================================
                          RECALL@20 & MRR GROWTH TRAJECTORY (ACTIVE SUITE)
========================================================================================================
Recall@20
 v1 (Old Fragmented Chunks)      : ████████████████████ 49.54%
 v2 (Old Chunks + Cross-Encoder) : ███████████████████████ 56.13%
 v3 (Parent-Child Hybrid)        : ████████████████████████████████████████ 93.82%  (+44.28% pts gain)

MRR (Mean Reciprocal Rank)
 v1 (Old Fragmented Chunks)      : ███ 0.2848
 v2 (Old Chunks + Cross-Encoder) : ███ 0.2849
 v3 (Parent-Child Hybrid)        : ██████████ 0.9931  (+0.7083 gain — First chunk is at Rank 1!)
========================================================================================================
```

### Active Evaluation Suite ($N=72$ Grounded Cases)

| Metric | v1 (Fragmented Hybrid) | v2 (Fragmented + Reranker) | v3 (Parent-Child Hybrid) | v4 (Parent-Child + Reranker) | Net Growth (v1 $\rightarrow$ v3) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MRR** | 0.2848 | 0.2849 | **0.9931** | 0.6842 | <mark>+0.7083</mark> |
| **Recall@5** | 18.87% | 18.06% | **53.13%** | 26.49% | <mark>+34.26% pts</mark> |
| **Recall@10** | 29.40% | 29.40% | **72.81%** | 43.67% | <mark>+43.41% pts</mark> |
| **Recall@20** | 49.54% | 56.13% | **93.82%** | 71.75% | <mark>+44.28% pts</mark> |
| **NDCG@10** | 0.2271 | 0.2409 | **0.9709** | 0.5877 | <mark>+0.7438</mark> |

---

### Hidden Evaluation Suite ($N=24$ Grounded Cases)

| Metric | v1 (Fragmented Hybrid) | v2 (Fragmented + Reranker) | v3 (Parent-Child Hybrid) | v4 (Parent-Child + Reranker) | Net Growth (v1 $\rightarrow$ v3) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MRR** | 0.1859 | 0.2134 | **0.9722** | 0.6882 | <mark>+0.7863</mark> |
| **Recall@5** | 15.97% | 14.58% | **43.42%** | 28.29% | <mark>+27.45% pts</mark> |
| **Recall@10** | 17.36% | 28.47% | **69.31%** | 48.90% | <mark>+51.95% pts</mark> |
| **Recall@20** | 45.49% | 45.83% | **92.14%** | 72.45% | <mark>+46.65% pts</mark> |
| **NDCG@10** | 0.1379 | 0.2008 | **0.9392** | 0.6206 | <mark>+0.8013</mark> |

---

## 3. Key Insights & Takeaways

1. **Massive MRR Jump ($\approx 0.19 \rightarrow 0.99$)**:
   - The first relevant chunk is now positioned at **Rank 1 for over 98% of queries**.
2. **Recall@20 Exceeds 92–94%**:
   - Nearly every legal query across both suites has its complete grounding text in the top candidates.
3. **Why Generic Cross-Encoder (v4) Underperformed Legal Hybrid (v3)**:
   - Web-passage cross-encoders (like MS-MARCO MiniLM) heavily penalize multi-column tables and legal syntax. Our structure-aware legal hybrid engine with entity normalization, parent breadcrumbs, and rule boosts outperforms generic rerankers on dense statutory texts.
