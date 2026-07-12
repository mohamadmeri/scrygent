"""Executes a batch of queries against the Scrygent graph for benchmarking.

Reads queries from a JSONL file (format: {"query": "...", "csv_path": "..."}),
runs them through the compiler, and outputs latency and success metrics.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from scrygent.graph.builder import build_graph
from scrygent.models.state import AgentState

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_single_query(graph: Any, query: str, csv_path: Path) -> dict[str, Any]:
    """Invokes the graph and returns execution metadata."""
    initial_state = AgentState(
        original_csv_path=csv_path,
        current_csv_path=csv_path,
        user_query=query,
    )
    payload = initial_state.model_dump(mode="json")

    start_time = time.time()
    final_state_dict = graph.invoke(payload)
    duration_ms = int((time.time() - start_time) * 1000)

    final_state = AgentState.model_validate(final_state_dict)

    return {
        "query": query,
        "status": final_state.execution_status,
        "duration_ms": duration_ms,
        "steps_executed": len(final_state.execution_trace),
        "error": final_state.error_log[-1] if final_state.error_log else None,
    }


def main() -> None:
    """Entry point of the script."""
    parser = argparse.ArgumentParser(description="Run Scrygent benchmark queries.")
    parser.add_argument("input", type=Path, help="Path to a JSONL file containing queries.")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("scripts/benchmark_results.jsonl"), help="Output file for results."
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Benchmark file not found: {args.input}")

    logger.info("Compiling LangGraph...")
    graph = build_graph()

    results = []
    total_start = time.time()

    with args.input.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    for i, item in enumerate(lines):
        query = item["query"]
        csv_path = Path(item["csv_path"])

        print(f"[{i + 1}/{len(lines)}] Executing: {query[:50]}...")
        result = run_single_query(graph, query, csv_path)
        results.append(result)

        # Append to output file incrementally to prevent data loss on crash
        with args.output.open("a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result) + "\n")

    total_duration = time.time() - total_start
    success_count = sum(1 for r in results if r["status"] == "complete")

    print("\n--- BENCHMARK SUMMARY ---")
    print(f"Total Queries: {len(results)}")
    print(f"Successful: {success_count} ({success_count / len(results) * 100:.1f}%)")
    print(f"Total Time: {total_duration:.2f}s")
    print(f"Avg Latency: {sum(r['duration_ms'] for r in results) / len(results):.0f}ms")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
