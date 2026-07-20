"""Hybrid deterministic execution node.

This node dispatches validated IR payloads to the deterministic tool suite.
It handles specialized data-access patterns (e.g., DataFrame injection for
analyze_data), manages the multi-step composition state transitions, and
executes the internal self-healing correction chain on runtime failures.
"""

import json
import logging
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from ..contracts import ToolName
from ..core.llm_factory import get_structured_llm
from ..core.resilience import ServiceExhaustedError, resilient_call
from ..models.registry import TOOL_PARAM_MODELS
from ..models.state import AgentState
from ..models.step_models import StepRecord
from ..prompts.executor import CORRECTION_SYSTEM_PROMPT
from ..prompts.schema_formatter import TOOL_SPECIFICATIONS
from ..tools.analyze_data import analyze_data
from ..tools.arithmetic import derive_column, evaluate_metrics
from ..tools.io import load_csv
from ..tools.statistics import correlation, detect_outliers, regression, request_column_stats
from ..tools.visualization import generate_plot
from ..tools.wrangling import filter_dataset, normalize_column, reset_dataset

logger = logging.getLogger(__name__)

_TOOL_DISPATCHER = {
    ToolName.ANALYZE_DATA: analyze_data,
    ToolName.FILTER_DATASET: filter_dataset,
    ToolName.NORMALIZE_COLUMN: normalize_column,
    ToolName.RESET_DATASET: reset_dataset,
    ToolName.CORRELATION: correlation,
    ToolName.REGRESSION: regression,
    ToolName.DETECT_OUTLIERS: detect_outliers,
    ToolName.REQUEST_COLUMN_STATS: request_column_stats,
    ToolName.GENERATE_PLOT: generate_plot,
    ToolName.DERIVE_COLUMN: derive_column,
    ToolName.EVALUATE_METRICS: evaluate_metrics,
}

MAX_RETRIES = 2


def extract_single_tool_spec(tool_name: str) -> str:
    """Extracts the isolated Markdown schema for a specific tool.

    This prevents the LLM correction chain from being distracted by
    irrelevant tool schemas, reducing token usage and hallucination risk.
    """
    lines = TOOL_SPECIFICATIONS.split("\n")
    capture = []
    started = False

    for line in lines:
        if line.strip().startswith("## ") and tool_name in line:
            started = True
            capture.append(line)
            continue
        if started:
            if line.strip().startswith("## ") or line.strip().startswith("---"):
                break
            capture.append(line)

    if "SHARED FILTER SCHEMA" in TOOL_SPECIFICATIONS:
        shared_part = TOOL_SPECIFICATIONS.split("---")[-1]
        capture.append("\n---" + shared_part)

    return "\n".join(capture)


def _run_correction_chain(tool_name: ToolName, failed_params: dict[str, Any], error_message: str) -> Any:
    """Invokes the LLM to repair a failed tool payload.

    This internal loop isolates the self-healing mechanism from the main
    LangGraph state, allowing rapid parameter correction without triggering
    a full graph back-edge.
    """
    logger.info("Triggering LLM Correction Chain for tool '%s'", tool_name)

    target_schema = TOOL_PARAM_MODELS[tool_name]
    structured_llm = get_structured_llm(pydantic_schema=target_schema)

    prompt = ChatPromptTemplate.from_messages([
        ("system", CORRECTION_SYSTEM_PROMPT),
        ("human", "Regenerate the corrected parameters now."),
    ])

    chain = prompt | structured_llm

    tool_string_identifier = tool_name.value if hasattr(tool_name, "value") else str(tool_name)
    isolated_markdown_spec = extract_single_tool_spec(tool_string_identifier)

    corrected_model = resilient_call(
        lambda: chain.invoke({
            "tool_specs": isolated_markdown_spec,
            "tool_name": tool_string_identifier,
            "failed_params": json.dumps(failed_params, indent=2),
            "error_message": error_message,
        }),
        service="Executor (Correction Chain)",
    )

    return corrected_model.model_dump()


def run_executor_node(state: AgentState) -> dict[str, Any]:
    """Dispatches the current step to the deterministic tool suite.

    Handles specialized kwargs injection for state-mutating tools and
    triggers the internal self-healing correction chain on validation
    or runtime failures.
    """
    logger.info("--- NODE: EXECUTOR (Step %d) ---", state.current_step_index)

    if not state.plan or state.current_step_index >= len(state.plan.steps):
        logger.error("Executor invoked but no valid steps remain.")
        return {"execution_status": "aborted"}

    step = state.plan.steps[state.current_step_index]

    # Hard session-level guard: Prevent infinite lazy-fetch loops.
    if step.tool_name == ToolName.REQUEST_COLUMN_STATS and state.has_replanned:
        error_msg = (
            f"Step {step.step_id}: request_column_stats was invoked a second "
            "time in this session. Only one lazy-fetch replan is permitted "
            "per query. Rejecting."
        )
        logger.error(error_msg)
        record = StepRecord(
            step_id=step.step_id,
            tool_name=step.tool_name,
            status="failed",
            error=error_msg,
            duration_ms=0,
        )
        return {
            "execution_status": "aborted",
            "execution_trace": state.execution_trace + [record],
            "error_log": state.error_log + [error_msg],
        }

    current_parameters = step.parameters.copy()
    retry_count = 0
    last_error_msg = ""

    while retry_count <= MAX_RETRIES:
        start_time = time.time()
        try:
            kwargs = current_parameters.copy()

            # Specialized kwargs injection based on tool data-access patterns.
            # analyze_data requires the loaded DataFrame, reset_dataset requires
            # the immutable original path, and evaluate_metrics requires no path.
            # All other tools consume the active current_csv_path.
            if step.tool_name == ToolName.ANALYZE_DATA:
                kwargs["df"] = load_csv(state.current_csv_path)
            elif step.tool_name == ToolName.RESET_DATASET:
                kwargs["original_csv_path"] = state.original_csv_path
            elif step.tool_name != ToolName.EVALUATE_METRICS:
                kwargs["current_csv_path"] = state.current_csv_path

            # Lazy-fetch validation: Ensure the Planner only requests stats
            # for columns that were explicitly omitted during initial profiling.
            if step.tool_name == ToolName.REQUEST_COLUMN_STATS:
                requested_columns = kwargs.get("columns", [])
                missing_set = set(state.data_profile.missing_detailed_stats) if state.data_profile else set()
                invalid_columns = [c for c in requested_columns if c not in missing_set]
                if invalid_columns:
                    # Do not dump the entire missing_set list.
                    # Large lists cause the LLM to overcorrect and request all of them.
                    # Instead, point it to the data profile for the exact strings.
                    raise ValueError(
                        f"request_column_stats rejected: column(s) {invalid_columns} are not in the "
                        f"'missing_detailed_stats' list. Please check the data profile and use the "
                        f"EXACT column names (including emojis and spacing) from that list."
                    )

            logger.info("Dispatching %s (Attempt %d/%d)...", step.tool_name, retry_count + 1, MAX_RETRIES + 1)
            tool_func = _TOOL_DISPATCHER[step.tool_name]
            result = tool_func(**kwargs)  # type: ignore
            duration_ms = int((time.time() - start_time) * 1000)

            # Lazy-fetch special case: Enrich the profile and trigger a one-time
            # constrained re-plan loop. The Planner is strictly forbidden from
            # requesting multiple lazy fetches in a single session.
            if step.tool_name == ToolName.REQUEST_COLUMN_STATS:
                logger.info("Lazy fetch completed. Triggering one-time Re-Plan loop.")

                new_profile = state.data_profile.model_copy(deep=True)  # type: ignore
                fetched_stats = result.get("detailed_stats", {})

                new_profile.detailed_stats.update(fetched_stats)
                new_profile.missing_detailed_stats = [
                    c for c in new_profile.missing_detailed_stats if c not in fetched_stats
                ]

                record = StepRecord(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    status="success",
                    summary=f"Lazy-fetched {len(fetched_stats)} column(s): {sorted(fetched_stats)}",
                    duration_ms=duration_ms,
                )

                return {
                    "data_profile": new_profile,
                    "execution_status": "replan",
                    "has_replanned": True,
                    "execution_trace": state.execution_trace + [record],
                }

            # Standard execution success path
            new_outputs = state.step_outputs.copy()
            new_outputs[step.step_id] = result

            # Multi-step composition: Update the active CSV path if the tool
            # performed a state-mutating wrangling operation.
            new_csv_path = result.get("current_csv_path", state.current_csv_path)
            record = StepRecord(
                step_id=step.step_id,
                tool_name=step.tool_name,
                status="success",
                duration_ms=duration_ms,
            )

            next_index = state.current_step_index + 1
            next_status = "complete" if next_index >= len(state.plan.steps) else "running"

            # Persist the corrected parameters back to the plan so the UI trace is accurate.
            new_plan = state.plan.model_copy(deep=True)
            new_plan.steps[state.current_step_index].parameters = current_parameters

            return {
                "plan": new_plan,
                "step_outputs": new_outputs,
                "current_csv_path": new_csv_path,
                "current_step_index": next_index,
                "execution_status": next_status,
                "execution_trace": state.execution_trace + [record],
            }

        except Exception as exc:
            last_error_msg = f"Runtime error in {step.tool_name}: {str(exc)}"
            logger.warning("Execution attempt %d failed. Error: %s", retry_count + 1, last_error_msg)

            retry_count += 1
            if retry_count <= MAX_RETRIES:
                try:
                    current_parameters = _run_correction_chain(
                        tool_name=step.tool_name, failed_params=current_parameters, error_message=last_error_msg
                    )
                except ServiceExhaustedError as service_exc:
                    # The correction engine itself is rate-limited. We cannot
                    # burn remaining self-healing attempts on a dead service.
                    error_msg = (
                        f"Step {step.step_id}: {service_exc.service} is temporarily "
                        f"unavailable (rate limited after {service_exc.attempts} attempts) "
                        "while attempting to repair a failed step."
                    )
                    logger.error(error_msg)
                    record = StepRecord(
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        status="failed",
                        error=error_msg,
                        duration_ms=0,
                    )
                    return {
                        "execution_status": "aborted",
                        "execution_trace": state.execution_trace + [record],
                        "error_log": state.error_log + [error_msg],
                    }
                except Exception as repair_exc:
                    logger.error("Self-healing correction engine itself broke: %s", str(repair_exc))
                    break

    logger.error(
        "Step %s (%s) permanently failed execution after %d attempts.", step.step_id, step.tool_name, MAX_RETRIES + 1
    )
    record = StepRecord(
        step_id=step.step_id,
        tool_name=step.tool_name,
        status="failed",
        error=last_error_msg,
        duration_ms=0,
    )
    return {
        "execution_status": "aborted",
        "execution_trace": state.execution_trace + [record],
        "error_log": state.error_log + [last_error_msg],
    }
