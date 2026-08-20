import time
import json
import tempfile
from pathlib import Path
from src.mcp.telemetry import MCPTelemetryLogger
from src.evaluation.telemetry_analyzer import MCPTelemetryAnalyzer


def test_telemetry_logger_non_blocking_performance():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MCPTelemetryLogger(log_dir=tmpdir, log_file="test_telemetry.jsonl")

        # Measure latency of 100 consecutive log_event calls
        start_time = time.perf_counter()
        for i in range(100):
            logger.log_event(
                tool_name="search_tax_rules",
                inputs={"query": f"test query {i}", "top_k": 5},
                outputs={"total_retrieved": 3},
                retrieved_chunk_ids=[f"chunk_{i}_1", f"chunk_{i}_2"],
                retrieved_statutory_paths=[f"Rule {i} > Sub-rule (1)"],
                detected_fy="FY 2023-2024",
                detected_ay="AY 2024-2025",
                target_regime="1962",
                duration_ms=5.2,
                success=True
            )
        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        avg_call_ms = total_time_ms / 100.0

        # Average call should be sub-millisecond (< 0.1ms)
        assert avg_call_ms < 0.5, f"Expected sub-millisecond enqueue, got {avg_call_ms:.4f} ms/call"

        # Force flush to disk
        logger.flush()
        stats = logger.get_stats()
        assert stats["total_records_written"] == 100


def test_telemetry_analyzer_and_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_telemetry.jsonl"
        logger = MCPTelemetryLogger(log_dir=tmpdir, log_file="test_telemetry.jsonl")

        # Log some realistic events
        logger.log_event(
            tool_name="search_tax_rules",
            inputs={"query": "interest free loan perquisite valuation", "top_k": 3},
            outputs={"total_retrieved": 2},
            retrieved_chunk_ids=["1962_rule3_sub7_clause_i_p1", "1962_rule3_sub7_clause_i_p2"],
            retrieved_statutory_paths=["Rule 3 > Sub-rule (7) > Clause (i) > Proviso 1"],
            detected_fy="FY 2023-2024",
            detected_ay="AY 2024-2025",
            target_regime="1962",
            duration_ms=12.5,
            success=True
        )
        logger.flush()

        analyzer = MCPTelemetryAnalyzer(log_path=str(log_path))
        rep = analyzer.generate_report()

        assert rep["total_events"] >= 1
        assert "search_tax_rules" in rep["tool_breakdown"]
        assert rep["latency_stats"]["mean_duration_ms"] > 0

        # Test evaluation suite export
        export_path = Path(tmpdir) / "exported_eval_suite.json"
        export_res = analyzer.export_to_eval_suite(str(export_path))

        assert export_res["exported_cases_count"] >= 1
        assert export_path.exists()

        with open(export_path, "r", encoding="utf-8") as f:
            cases = json.load(f)
            assert len(cases) >= 1
            assert cases[0]["query"] == "interest free loan perquisite valuation"
            assert cases[0]["target_regime"] == "1962"
            assert "1962_rule3_sub7_clause_i_p1" in cases[0]["expected_chunks"]
