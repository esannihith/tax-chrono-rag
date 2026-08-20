import os
import time
import json
import uuid
import queue
import atexit
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional


class MCPTelemetryLogger:
    """Thread-safe asynchronous telemetry logger for MCP tool invocations.
    
    Uses an in-memory non-blocking queue and a background daemon worker thread
    to batch-append telemetry records to JSONL on disk without blocking tool responses.
    """


    def __init__(self, log_dir: str = "data/logs", log_file: str = "mcp_telemetry.jsonl"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / log_file

        self.queue: queue.Queue = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Start daemon background writer thread
        self.worker_thread = threading.Thread(
            target=self._writer_loop,
            name="MCPTelemetryWriter",
            daemon=True
        )
        self.worker_thread.start()

        # Register clean shutdown flush
        atexit.register(self.shutdown)

    def _writer_loop(self):
        """Continuous daemon loop that drains the queue and writes to JSONL."""
        batch = []
        last_flush = time.time()

        while not self._stop_event.is_set():
            try:
                # Wait up to 500ms for new log items
                item = self.queue.get(timeout=0.5)
                batch.append(item)
                self.queue.task_done()
            except queue.Empty:
                pass

            # Flush if batch size >= 10 or every 1.0 second
            now = time.time()
            if batch and (len(batch) >= 10 or (now - last_flush) >= 1.0 or self._stop_event.is_set()):
                self._flush_batch(batch)
                batch = []
                last_flush = now

        # Final drain on shutdown
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                batch.append(item)
                self.queue.task_done()
            except queue.Empty:
                break
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: List[Dict[str, Any]]):
        """Appends a batch of records to the JSONL log file safely."""
        try:
            with self._lock:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    for entry in batch:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            # Telemetry errors must never crash the server
            pass

    def log_event(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        retrieved_chunk_ids: Optional[List[str]] = None,
        retrieved_statutory_paths: Optional[List[str]] = None,
        detected_fy: Optional[str] = None,
        detected_ay: Optional[str] = None,
        target_regime: Optional[str] = None,
        duration_ms: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Enqueues a telemetry record non-blockingly to the in-memory queue."""
        record = {

            "log_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_name": tool_name,
            "inputs": inputs,
            "retrieved_chunk_ids": retrieved_chunk_ids or [],
            "retrieved_statutory_paths": retrieved_statutory_paths or [],
            "temporal_metadata": {
                "detected_fy": detected_fy,
                "detected_ay": detected_ay,
                "target_regime": target_regime
            },
            "duration_ms": round(duration_ms, 2),
            "success": success,
            "error_message": error_message,
            "output_summary": self._summarize_output(outputs),
            "extra_metadata": metadata or {}
        }

        try:
            # Non-blocking put; drops record if queue is completely saturated under extreme load
            self.queue.put_nowait(record)
        except queue.Full:
            pass

    def _summarize_output(self, outputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarizes output for lightweight storage while retaining evaluation ground truth."""
        if not outputs or not isinstance(outputs, dict):
            return {}

        summary = {}
        if "total_retrieved" in outputs:
            summary["total_retrieved"] = outputs["total_retrieved"]
        if "direct_answer" in outputs:
            summary["direct_answer_preview"] = str(outputs["direct_answer"])[:200]
        if "statutory_citations" in outputs:
            summary["citations_count"] = len(outputs["statutory_citations"])
            summary["citations"] = [
                f"Rule {c.get('rule_id')}{c.get('sub_rule', '')}"
                for c in outputs.get("statutory_citations", [])
            ]
        if "taxable_perquisite_value" in outputs:
            summary["taxable_perquisite_value"] = outputs["taxable_perquisite_value"]
        if "found" in outputs:
            summary["found"] = outputs["found"]
        if "is_in_force_for_period" in outputs:
            summary["is_in_force_for_period"] = outputs["is_in_force_for_period"]
        return summary

    def get_stats(self) -> Dict[str, Any]:
        """Returns log queue and file statistics."""
        with self._lock:
            file_size_bytes = self.log_file.stat().st_size if self.log_file.exists() else 0
            line_count = 0
            if self.log_file.exists():
                try:
                    with open(self.log_file, "r", encoding="utf-8") as f:
                        line_count = sum(1 for _ in f)
                except Exception:
                    pass

        return {
            "log_path": str(self.log_file),
            "total_records_written": line_count,
            "file_size_bytes": file_size_bytes,
            "queue_pending_items": self.queue.qsize()
        }

    def flush(self):
        """Forces an immediate flush of any queued items to disk."""
        batch = []
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                batch.append(item)
                self.queue.task_done()
            except queue.Empty:
                break
        if batch:
            self._flush_batch(batch)

    def shutdown(self):
        """Flushes remaining queue items before process exit."""
        self._stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)


# Global singleton instance for application use
telemetry_logger = MCPTelemetryLogger()
