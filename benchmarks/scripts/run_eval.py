"""Core evaluation harness for Scrygent benchmarks."""

import argparse
import csv
import json
import logging
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from scrygent.graph.builder import build_graph
from scrygent.models.state import AgentState

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages the checkpoint.json file for resumable evaluation."""

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "checkpoint.json"
        self.completed: set[str] = set()
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    self.completed = set(json.load(f).get("completed", []))
            except json.JSONDecodeError:
                logger.warning("Corrupted checkpoint file. Starting fresh.")

    def mark_complete(self, item_id: str) -> None:
        self.completed.add(item_id)
        self._save()

    def _save(self) -> None:
        # Atomic write
        temp_path = self.path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump({"completed": list(self.completed)}, f, indent=2)
        temp_path.replace(self.path)


def normalize_answer(s: Any) -> str:
    """Normalizes a string, number, or list for robust comparison."""
    if isinstance(s, (list, tuple)):
        return ", ".join(sorted([normalize_answer(x) for x in s]))
    if isinstance(s, bool):
        return str(s).lower()
    if isinstance(s, (int, float)):
        return f"{float(s):.4f}"

    s_str = str(s).lower().strip()
    s_str = re.sub(r"\b(a|an|the)\b", " ", s_str)
    s_str = re.sub(r"[^\w\s]", "", s_str)
    s_str = re.sub(r"\s+", " ", s_str).strip()

    return s_str


def evaluate_answer(agent_answer: Any, gold_answer: Any) -> tuple[float, str]:
    """Returns (score, match_type) for robust comparison."""
    norm_agent = normalize_answer(agent_answer)
    norm_gold = normalize_answer(gold_answer)

    if norm_agent == norm_gold:
        return 1.0, "EXACT_MATCH"

    # Attempt numeric proximity check for scalars only
    if not isinstance(agent_answer, (list, tuple)) and not isinstance(gold_answer, (list, tuple)):
        try:
            agent_nums = re.findall(r"-?\d+\.?\d*", str(agent_answer).replace(",", ""))
            gold_nums = re.findall(r"-?\d+\.?\d*", str(gold_answer).replace(",", ""))
            if agent_nums and gold_nums:
                if abs(float(agent_nums[0]) - float(gold_nums[0])) <= 1e-2:
                    return 1.0, "NUMERIC_MATCH"
        except ValueError, IndexError:
            pass

    return 0.0, "NO_MATCH"


def save_failure_trace(output_dir: Path, item: dict[str, Any], final_state: dict[str, Any] | None, exception: Exception | None) -> None:
    """Saves a detailed JSON trace for failed queries."""
    failures_dir = output_dir / "failures"
    failures_dir.mkdir(exist_ok=True)
    trace_path = failures_dir / f"{item['id']}.json"

    serializable_state = {}
    if final_state:
        for k, v in final_state.items():
            try:
                # Pydantic v2 native serialization
                if hasattr(v, "model_dump"):
                    serializable_state[k] = v.model_dump(mode="json")
                else:
                    json.dumps(v)  # Test serializability
                    serializable_state[k] = v
            except TypeError, ValueError:
                serializable_state[k] = str(v)

    trace = {
        "id": item["id"],
        "query": item["query"],
        "gold_answer": item["gold_answer"],
        "csv_path": item["csv_path"],
        "final_state": serializable_state,
        "exception": str(exception) if exception else None,
    }

    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, default=str)


def run_evaluation(manifest_path: Path, output_dir: Path, limit: int | None) -> None:
    """Executes the evaluation loop."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    checkpoint = CheckpointManager(output_dir)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = [json.loads(line) for line in f]

    if limit:
        manifest = manifest[:limit]

    pending = [item for item in manifest if item["id"] not in checkpoint.completed]

    if not pending:
        logger.info("✅ All items in manifest are already completed. Exiting.")
        return

    logger.info("Initializing Scrygent Graph...")
    graph = build_graph()

    if not predictions_path.exists():
        with open(predictions_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id",
                "query",
                "gold_answer",
                "agent_answer",
                "score",
                "match_type",
                "status",
                "latency_s",
                "retries",
                "failure_category",
            ])

    def handle_keyboard_interrupt(signum: int, frame: Any) -> None:
        tqdm.write("\n[!] KeyboardInterrupt received. Saving checkpoint and exiting...")
        checkpoint._save()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_keyboard_interrupt)

    pbar = tqdm(pending, desc="Evaluating", unit="query")
    for item in pbar:
        agent_answer: Any = None
        status: str = "unknown"
        latency: float = 0.0
        retries: int = 0
        failure_category: str = "Unknown"
        exception: Exception | None = None
        final_state: dict[str, Any] | None = None
        match_type: str = "NO_MATCH"

        try:
            state = AgentState(
                original_csv_path=Path(item["csv_path"]), current_csv_path=Path(item["csv_path"]), user_query=item["query"], eval_mode=True
            )

            start_time = time.perf_counter()
            final_state = graph.invoke(state.model_dump())  # type: ignore[call-overload]
            latency = time.perf_counter() - start_time

            if final_state is not None:
                status = final_state.get("execution_status", "aborted")
                retries = final_state.get("retry_count", 0)

                if not final_state.get("plan"):
                    failure_category = "Planner"
                elif not final_state.get("final_report"):
                    failure_category = "Reporter"
                elif final_state.get("error_log"):
                    failure_category = "Executor"

                report = final_state.get("final_report")
                if report:
                    agent_answer = report.get("answer") if isinstance(report, dict) else getattr(report, "answer", None)
                elif final_state.get("error_log"):
                    agent_answer = "ERROR: " + str(final_state["error_log"][-1])
                else:
                    agent_answer = "ERROR: No answer generated"

        except Exception as e:
            exception = e
            status = "crashed"
            failure_category = "Graph"
            agent_answer = f"CRASH: {str(e)}"
            logger.exception("Crash on item %s", item["id"])

        score, match_type = evaluate_answer(agent_answer, item["gold_answer"])

        if score < 1.0:
            save_failure_trace(output_dir, item, final_state, exception)

        with open(predictions_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                item["id"],
                item["query"],
                item["gold_answer"],
                str(agent_answer),
                score,
                match_type,
                status,
                f"{latency:.4f}",
                retries,
                failure_category,
            ])

        checkpoint.mark_complete(item["id"])


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Run Scrygent Benchmark Evaluation.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--limit", type=int, default=None, help="Limit items for smoke tests")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_evaluation(Path(args.manifest), Path(args.output_dir), args.limit)


if __name__ == "__main__":
    main()
