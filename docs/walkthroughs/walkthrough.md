# MCP Query Telemetry & Evaluation Logging Walkthrough

## 1. Accomplishments Overview

We designed, implemented, and verified a **Zero-Latency Query Telemetry & Evaluation Logging System** for the FastMCP Server (`TaxChronoRAG`):

1. **Zero-Latency Asynchronous Telemetry Engine ([`src/mcp/telemetry.py`](../../src/mcp/telemetry.py))**:
   - Uses an in-memory thread-safe FIFO queue (`queue.Queue`) and a dedicated background daemon worker (`MCPTelemetryWriter`).
   - Tool execution threads enqueue structured events via non-blocking `put_nowait()`, incurring **$< 0.01\text{ ms}$ overhead** per invocation.
   - Flushes batches to `data/logs/mcp_telemetry.jsonl` every 10 events or every 1.0 second.
   - Fault-tolerant error boundaries prevent any telemetry or I/O failure from disrupting MCP tool calls.

2. **Telemetry Hooks Integrated Across All 7 FastMCP Tools ([`src/mcp/server.py`](../../src/mcp/server.py))**:
   - `search_tax_rules`: Captures input query, parameters, retrieved chunk IDs, full statutory paths (`Rule > Sub-rule > Proviso`), resolved AY/FY, and execution latency.
   - `get_rule_details`: Captures rule ID, target regime, sub-rule chunk IDs, and latency.
   - `compare_regimes`: Captures comparative topic, dual-stream retrieved chunks for 1962 & 2026, and latency.
   - `resolve_tax_year`: Captures year input, resolved Financial Year, and Assessment Year.
   - `verify_effective_date`: Captures rule ID, queried date/AY, and in-force verification status.
   - `calculate_perquisite`: Captures perquisite type, parameters, and taxable value.
   - `ask_tax_copilot`: Captures prompt query, persona context, generated statutory citations, and latency.

3. **Telemetry Analyzer & Evaluation Suite Exporter ([`src/evaluation/telemetry_analyzer.py`](../../src/evaluation/telemetry_analyzer.py))**:
   - **`--analyze-telemetry`**: Displays summary statistics of tool calls, latency distributions (mean, p95, min, max), and retrieval frequency heatmaps.
   - **`--export-telemetry <path>`**: Automatically converts collected real user queries into golden benchmark test suites (`data/evaluation/active_suite.json` format) for automated offline RAG regression testing.

4. **Automated Verification & Testing**:
   - **Pytest Suite ([`tests/test_telemetry.py`](../../tests/test_telemetry.py))**: Sub-millisecond enqueue benchmarks and evaluation export integrity verified (100% pass rate).
   - Full test suite passing across all 8 tests in `tests/`.


---

## 2. CLI Usage Reference

```bash
# Print live query & retrieval telemetry statistics
uv run python run_mcp.py --analyze-telemetry

# Export collected user queries to standard RAG evaluation JSON suite
uv run python run_mcp.py --export-telemetry data/evaluation/telemetry_suite.json

# Run all automated unit tests
uv run --with pytest pytest -o pythonpath=. tests/
```
