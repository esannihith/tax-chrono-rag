# Walkthrough: Parsing & Chunking Pipeline Implementation

We have implemented and verified the end-to-end **Parsing -> Chunking** pipeline for the Temporal-RAG Indian Income Tax Rules (1962 and 2026) corpus.

---

## 1. Summary of Accomplishments

1. **Corpus Ingestion & Verification**:
   - Processed all **56 PDFs** (32 from 1962, 24 from 2026).
   - Fully resolved PDF structure trees (`page.structure_tree`), text coordinates, and Marked Content Identifiers (`mcid`).
2. **Noise Cleaning & Footnote Separation**:
   - Stripped official portal headers (*Income Tax Department...*) and footers (*Downloaded/Printed on...*).
   - Extracted and cataloged **64 statutory amendment footnotes**, detaching them from legal prose and transforming them into structured temporal metadata records.
3. **Table Extraction & Formatting**:
   - Extracted **39 tables** across the corpus, formatting them into Markdown tables with header preservation.
   - Handled complex multi-column structures and cross-referenced tables (e.g. [`Rule-12AC_1962`](../../data/raw/1962/Rule-12AC_1962.pdf), [`Rule-136_2026`](../../data/raw/2026/Rule-136_2026.pdf), [`Rule-2BB_1962`](../../data/raw/1962/Rule-2BB_1962.pdf), [`Rule-280_2026`](../../data/raw/2026/Rule-280_2026.pdf)).
4. **Statutory Abstract Syntax Tree (AST)**:
   - Structured rules into hierarchical AST nodes: `Rule` -> `Sub-rule` -> `Clause` -> `Sub-clause` -> `Item` -> `Proviso` -> `Explanation` -> `Table`.
5. **Context-Enriched Chunking**:
   - Created **1,177 structured chunks** (653 chunks for 1962, 524 chunks for 2026).
   - Injected hierarchical statutory breadcrumbs into each chunk:
     `[Income Tax Rules 1962 > Rule 12AC: Updated return of income > (1) > Table 1]`
   - Attached metadata: `sections_referenced` (e.g. `section 139(8A)`, `section 44AB`), `forms_referenced` (e.g. `Form ITR-7`, `Form ITR-U`), `effective_date`, and `is_table` flags.
   - Persisted output to `data/processed/chunks_1962.json` and `data/processed/chunks_2026.json`.

---

## 2. Implemented Codebase Structure

- [`src/models.py`](../../src/models.py): Pydantic data models (`RuleDocument`, `StatutoryNode`, `NodeType`, `Footnote`, `TableData`, `Chunk`, `ChunkMetadata`).
- [`src/parser/pdf_loader.py`](../../src/parser/pdf_loader.py): PDF loader with character-level MCID index maps.
- [`src/parser/cleaner.py`](../../src/parser/cleaner.py): Deterministic noise removal and footnote parser.
- [`src/parser/table_extractor.py`](../../src/parser/table_extractor.py): Table parser & Markdown converter.
- [`src/parser/ast_builder.py`](../../src/parser/ast_builder.py): Legal AST builder following Indian statutory grammar.
- [`src/parser/pipeline.py`](../../src/parser/pipeline.py): End-to-end PDF parsing orchestrator.
- [`src/chunker/breadcrumbs.py`](../../src/chunker/breadcrumbs.py): Breadcrumb path generator.
- [`src/chunker/legal_chunker.py`](../../src/chunker/legal_chunker.py): Context-enriched legal & temporal chunker.
- [`main.py`](../../main.py): Pipeline execution entrypoint across both 1962 and 2026 corpora.


---

## 3. Sample Chunk Demonstration

```json
{
  "chunk_id": "chunk_1962_rule_12AC_rule_12AC_table_1_7",
  "breadcrumb": "[Income Tax Rules 1962 > Rule 12AC: Updated return of income > Table 1]",
  "text": "[Income Tax Rules 1962 > Rule 12AC: Updated return of income > Table 1]\n| Sl. No. (1) | Person (2) | Manner of furnishing return of income (3) |  |  |  |\n| --- | --- | --- | --- | --- | --- |\n| 1. | Individual, or Hindu undivided family or a firm or limited liability partnership or an association of persons... | Electronically under digital signature. |  |  |  |",
  "content": "| Sl. No. (1) | Person (2) | Manner of furnishing return of income (3) |  |  |  |\n| --- | --- | --- | --- | --- | --- |\n| 1. | Individual, or Hindu undivided family or a firm or limited liability partnership...",
  "metadata": {
    "chunk_id": "chunk_1962_rule_12AC_rule_12AC_table_1_7",
    "corpus_year": 1962,
    "rule_id": "12AC",
    "rule_title": "Updated return of income",
    "statutory_path": "Table 1",
    "node_type": "TABLE",
    "source_file": "data/raw/1962/Rule-12AC_1962.pdf",
    "sections_referenced": ["section 139", "section 44AB"],
    "forms_referenced": ["Form ITR-7"],
    "effective_date": null,
    "is_table": true
  }
}
```

---

## 4. Verification Results

Running `uv run python main.py`:
- **1962 Corpus**: 32 rules parsed -> 653 chunks created.
- **2026 Corpus**: 24 rules parsed -> 524 chunks created.
- **Total Chunks**: **1,177 chunks**.
- **Execution**: 100% successful with zero unhandled exceptions.
