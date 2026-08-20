import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter


class MCPTelemetryAnalyzer:
    """Parses and analyzes MCP query telemetry logs for offline RAG benchmarking."""

    def __init__(self, log_path: str = "data/logs/mcp_telemetry.jsonl"):
        self.log_path = Path(log_path)

    def load_events(self) -> List[Dict[str, Any]]:
        """Loads all recorded telemetry events from JSONL."""
        if not self.log_path.exists():
            return []
        events = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events

    def generate_report(self) -> Dict[str, Any]:
        """Generates statistical breakdown of MCP tool calls and retrieval patterns."""
        events = self.load_events()
        if not events:
            return {
                "total_events": 0,
                "message": f"No telemetry events found at {self.log_path}"
            }

        tool_counts = Counter()
        latencies = []
        retrieved_chunk_counts = Counter()
        retrieved_rule_counts = Counter()
        queries_logged = []

        for e in events:
            tool = e.get("tool_name", "unknown")
            tool_counts[tool] += 1
            lat = e.get("duration_ms", 0.0)
            if lat > 0:
                latencies.append(lat)

            # Chunks
            for cid in e.get("retrieved_chunk_ids", []):
                retrieved_chunk_counts[cid] += 1
                # Extract rule ID
                if "rule" in cid.lower():
                    retrieved_rule_counts[cid.split("_")[1]] += 1

            # Queries
            inputs = e.get("inputs", {})
            q = inputs.get("query") or inputs.get("topic")
            if q:
                queries_logged.append({
                    "tool": tool,
                    "query": q,
                    "target_regime": e.get("temporal_metadata", {}).get("target_regime"),
                    "duration_ms": lat
                })

        avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0

        return {
            "total_events": len(events),
            "tool_breakdown": dict(tool_counts.most_common()),
            "latency_stats": {
                "mean_duration_ms": round(avg_latency, 2),
                "p95_duration_ms": round(p95_latency, 2),
                "min_duration_ms": round(min(latencies), 2) if latencies else 0.0,
                "max_duration_ms": round(max(latencies), 2) if latencies else 0.0
            },
            "top_retrieved_chunks": dict(retrieved_chunk_counts.most_common(10)),
            "top_retrieved_rules": dict(retrieved_rule_counts.most_common(10)),
            "total_queries_captured": len(queries_logged),
            "sample_queries": queries_logged[:5]
        }

    def export_to_eval_suite(
        self,
        output_suite_path: str = "data/evaluation/telemetry_suite.json",
        min_citations: int = 1
    ) -> Dict[str, Any]:
        """Converts logged user queries into standard RAG evaluation test cases."""
        events = self.load_events()
        out_path = Path(output_suite_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        eval_cases = []
        seen_queries = set()

        for idx, e in enumerate(events):
            tool = e.get("tool_name")
            if tool not in ["search_tax_rules", "ask_tax_copilot", "compare_regimes"]:
                continue

            inputs = e.get("inputs", {})
            q = inputs.get("query") or inputs.get("topic")
            if not q or q in seen_queries:
                continue
            seen_queries.add(q)

            temp = e.get("temporal_metadata", {})
            chunk_ids = e.get("retrieved_chunk_ids", [])
            stat_paths = e.get("retrieved_statutory_paths", [])

            # Derive target rules
            target_rules = []
            for sp in stat_paths:
                if "Rule" in sp:
                    parts = sp.split(">")
                    target_rules.append(parts[0].strip())

            eval_case = {
                "id": f"telemetry_q_{len(eval_cases)+1:03d}",
                "query": q,
                "persona": inputs.get("persona_context", "Salaried Individual"),
                "query_type": "telemetry_user_query",
                "target_regime": temp.get("target_regime", "1962"),
                "financial_year": temp.get("detected_fy"),
                "assessment_year": temp.get("detected_ay"),
                "expected_chunks": chunk_ids[:5],
                "expected_statutory_paths": stat_paths[:5],
                "evaluation_criteria": [
                    f"Grounded response retrieved using {tool}.",
                    "Cites applicable statutory sub-rules and provisos."
                ]
            }
            eval_cases.append(eval_case)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(eval_cases, f, indent=2, ensure_ascii=False)

        return {
            "exported_cases_count": len(eval_cases),
            "output_path": str(out_path)
        }


def print_telemetry_summary():
    """Prints a formatted report of MCP telemetry."""
    analyzer = MCPTelemetryAnalyzer()
    rep = analyzer.generate_report()

    print("=" * 80)
    print("MCP SERVER QUERY & RETRIEVAL TELEMETRY REPORT")
    print("=" * 80)

    if rep.get("total_events", 0) == 0:
        print("No telemetry events recorded yet. Run some MCP queries first!")
        print("=" * 80)
        return

    print(f"Total Tool Invocations: {rep['total_events']}")
    print("\nTool Invocation Breakdown:")
    for tool, count in rep.get("tool_breakdown", {}).items():
        print(f"  - {tool:<28}: {count:>4} calls")

    lat = rep.get("latency_stats", {})
    print(f"\nExecution Latency Metrics:")
    print(f"  - Mean Latency : {lat.get('mean_duration_ms', 0)} ms")
    print(f"  - P95 Latency  : {lat.get('p95_duration_ms', 0)} ms")
    print(f"  - Min / Max    : {lat.get('min_duration_ms', 0)} ms / {lat.get('max_duration_ms', 0)} ms")

    print("\nTop Retrieved Statutory Chunks:")
    for cid, count in rep.get("top_retrieved_chunks", {}).items():
        print(f"  - {cid:<35}: {count:>3} hits")

    print("=" * 80)


if __name__ == "__main__":
    print_telemetry_summary()
