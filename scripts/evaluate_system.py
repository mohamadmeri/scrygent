"""Evaluation Harness for Scrygent.

Executes a batch of benchmark queries, captures granular telemetry,
evaluates results against gold standards, and generates comprehensive
reports for deep-dive post-mortem analysis.
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scrygent.graph.builder import build_graph
from scrygent.models.state import AgentState

# Silence noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("langchain_core").setLevel(logging.WARNING)

# CONFIGURATION & SETUP
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EVAL_DIR = Path("eval_reports") / f"run_{RUN_TIMESTAMP}"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = EVAL_DIR / "summary.csv"
DETAILED_JSON = EVAL_DIR / "detailed_results.json"
POST_MORTEM_DIR = EVAL_DIR / "post_mortems"
POST_MORTEM_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# SCORING & EVALUATION LOGIC
def normalize_answer(answer: Any) -> Any:
    """Normalize an answer for exact/set matching (lowercase, strip, remove punctuation)."""
    if isinstance(answer, list):
        return sorted([normalize_answer(a) for a in answer])
    if isinstance(answer, (int, float)):
        return round(float(answer), 4)
    if isinstance(answer, str):
        return re.sub(r"[^\w\s.-]", "", answer).lower().strip()
    return str(answer).lower().strip()


def evaluate_answer(agent_answer: str, gold_answer: str) -> dict[str, Any]:
    """Evaluates the agent's answer against the gold standard."""
    if not agent_answer or agent_answer in ("ABORTED", "None", "null"):
        return {"match_type": "FAIL_ABORTED", "score": 0.0}

    norm_agent = normalize_answer(agent_answer)
    norm_gold = normalize_answer(gold_answer)

    # 1. Exact Match (for scalars, strings)
    if norm_agent == norm_gold:
        return {"match_type": "EXACT_MATCH", "score": 1.0}

    # 2. Set Match (for lists, e.g., "top 4 salaries")
    if isinstance(norm_agent, list) and isinstance(norm_gold, list):
        if set(norm_agent) == set(norm_gold):
            return {"match_type": "SET_MATCH", "score": 1.0}
        # Partial credit for overlapping sets
        overlap = len(set(norm_agent) & set(norm_gold))
        return {"match_type": "PARTIAL_SET_MATCH", "score": overlap / len(norm_gold) if norm_gold else 0.0}

    # 3. Numeric proximity (for floats)
    try:
        # Ensure neither variable is a list before converting
        if not isinstance(norm_agent, list) and not isinstance(norm_gold, list):
            if abs(float(norm_agent) - float(norm_gold)) < 0.01:
                return {"match_type": "NUMERIC_MATCH", "score": 1.0}
    except (ValueError, TypeError):
        pass

    return {"match_type": "NO_MATCH", "score": 0.0}


def categorize_failure(state: AgentState) -> str:
    """Categorizes the reason for failure based on state and error logs."""
    if state.execution_status == "complete":
        return "SUCCESS"

    errors = " ".join(state.error_log).lower()

    if "429" in errors or "rate limit" in errors:
        return "RATE_LIMITED"
    if "tool call validation failed" in errors or "did not match schema" in errors:
        return "PLANNER_SCHEMA_ERROR"
    if "sort.column" in errors and "not a metric alias" in errors:
        return "PLANNER_LOGIC_ERROR"
    if "permanently failed execution after" in errors:
        return "EXECUTOR_EXHAUSTED"
    if "compiler pipeline failed" in errors:
        return "PLANNER_COMPILATION_FAILED"

    return "UNKNOWN_ERROR"


# POST-MORTEM GENERATION
def generate_post_mortem(
    state: AgentState,
    q_id: str,
    query: str,
    gold: str,
    eval_result: dict,  # type: ignore
    elapsed: float,
) -> Path:
    """Generates a highly structured Markdown post-mortem for a single query."""
    agent_answer = "ABORTED"
    if state.final_report:
        agent_answer = getattr(state.final_report, "answer", getattr(state.final_report, "primary_answer", "None"))

    lines = [
        f"# Post-Mortem: {q_id}",
        f"**Status:** `{eval_result['failure_category']}` | **Score:** `{eval_result['score']}` ({eval_result['match_type']})",
        f"**Latency:** `{elapsed:.2f}s`",
        "---",
        f"**Query:** `{query}`",
        f"**Gold Answer:** `{gold}`",
        f"**Agent Answer:** `{agent_answer}`",
        "---",
        "## 1. Profiler Insights",
    ]

    if state.data_profile:
        lines.append(f"- **Query-Specific Matches:** `{state.data_profile.query_specific_matches}`")
        lines.append(f"- **Regex Skeletons:** `{list(state.data_profile.regex_skeletons.keys())}`")
        lines.append(f"- **Missing Detailed Stats:** `{state.data_profile.missing_detailed_stats}`")
    else:
        lines.append("- *No profile data available.*")

    lines.append("\n## 2. Compiler Pipeline")
    lines.append(f"- **Lazy Fetch Triggered:** `{state.has_replanned}`")

    if state.plan:
        for i, step in enumerate(state.plan.steps):
            lines.append(f"\n### Step {i}: `{step.tool_name.value}`")
            lines.append(f"> **Rationale:** {step.rationale}")
            lines.append("\n**Emitted Parameters:**")
            lines.append("```json\n" + json.dumps(step.parameters, indent=2) + "\n```")
    else:
        lines.append("\n*No plan was generated.*")

    lines.append("\n## 3. Execution Trace")
    if not state.execution_trace:
        lines.append("*No execution trace available.*")
    else:
        for trace in state.execution_trace:
            icon = "✅" if trace.status == "success" else "❌"
            duration = f"({trace.duration_ms}ms)" if trace.duration_ms else ""
            lines.append(f"- {icon} **{trace.tool_name.value}** {duration}")
            if trace.error:
                lines.append(f"  - *Exception:* `{trace.error}`")
            if trace.summary:
                lines.append(f"  - *Summary:* {trace.summary}")

    if state.error_log:
        lines.append("\n## 4. Self-Healing / Error Logs")
        for err in state.error_log:
            lines.append(f"- ⚠️ `{err}`")

    file_path = POST_MORTEM_DIR / f"{q_id}.md"
    file_path.write_text("\n".join(lines), encoding="utf-8")
    return file_path


# MAIN EXECUTION LOOP
def main() -> None:
    """Main Entry point."""
    parser = argparse.ArgumentParser(description="Run Scrygent production-grade evaluation.")
    parser.add_argument("input", type=Path, help="Path to a JSONL file containing sample queries.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of queries to run (for testing).")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file %s not found.", args.input)
        return

    logger.info("🚀 Starting Evaluation Run. Artifacts will be saved to: %s", EVAL_DIR)

    with args.input.open("r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        queries = queries[: args.limit]
        logger.info("Limited to %d queries for testing.", len(queries))

    graph = build_graph()
    results = []

    for i, item in enumerate(queries):
        q_id = item.get("id", f"Q{i}")
        query = item["query"]
        csv_path = Path(item["csv_path"])
        gold = item.get("gold_answer", "N/A")

        logger.info("[%d/%d] Evaluating %s: %s...", i + 1, len(queries), q_id, query[:50])

        if not csv_path.exists():
            logger.error("❌ CSV NOT FOUND: %s", csv_path)
            continue

        state = AgentState(original_csv_path=csv_path, current_csv_path=csv_path, user_query=query, eval_mode=True)

        start_time = time.time()
        try:
            raw_state = graph.invoke(state.model_dump())  # type: ignore
            final_state = AgentState.model_validate(raw_state)
            elapsed = time.time() - start_time

            agent_answer = "ABORTED"
            if final_state.final_report:
                agent_answer = getattr(
                    final_state.final_report, "answer", getattr(final_state.final_report, "primary_answer", "None")
                )

            eval_result = evaluate_answer(str(agent_answer), str(gold))
            eval_result["failure_category"] = categorize_failure(final_state)

            generate_post_mortem(final_state, q_id, query, str(gold), eval_result, elapsed)

            results.append({
                "id": q_id,
                "status": final_state.execution_status,
                "category": eval_result["failure_category"],
                "match_type": eval_result["match_type"],
                "score": eval_result["score"],
                "latency_sec": round(elapsed, 2),
                "retries": final_state.retry_count,
                "agent_answer": str(agent_answer),
                "gold_answer": str(gold),
            })

            icon = "✅" if eval_result["score"] == 1.0 else "⚠️" if eval_result["score"] > 0 else "❌"
            logger.info(
                "  %s %s (Score: %.1f, Latency: %.1fs)", icon, eval_result["match_type"], eval_result["score"], elapsed
            )

        except Exception as e:
            logger.error("💥 CRASH on %s: %s", q_id, e)
            results.append({
                "id": q_id,
                "status": "crashed",
                "category": "SYSTEM_CRASH",
                "match_type": "FAIL_CRASH",
                "score": 0.0,
                "latency_sec": 0.0,
                "retries": 0,
            })

        # Strict pacing to protect Groq Free Tier
        time.sleep(3)

    # FINAL REPORTING
    with DETAILED_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    df.to_csv(SUMMARY_CSV, index=False)

    total = len(results)
    success = sum(1 for r in results if r["score"] == 1.0)
    avg_latency = df["latency_sec"].mean()
    categories = df["category"].value_counts().to_dict()

    print("\n" + "=" * 70)
    print("🎉 EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total Queries:       {total}")
    print(f"Exact/Set Matches:   {success} ({success / total * 100:.1f}%)")
    print(f"Average Latency:     {avg_latency:.2f}s")
    print("-" * 70)
    print("Failure Breakdown:")
    for cat, count in categories.items():
        if cat != "SUCCESS":
            print(f"  - {cat:<25} {count}")
    print("=" * 70)
    print(f"📁 View detailed post-mortems in: {POST_MORTEM_DIR}")
    print(f"📊 View summary data in:          {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
