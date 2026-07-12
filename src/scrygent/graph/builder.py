"""LangGraph orchestration and state routing.

Wires the four core nodes (profiler -> planner -> executor -> reporter)
with conditional routing driven entirely by AgentState.execution_status.
This module contains no checkpointer; Scrygent is a single-pass,
fire-and-forget engine where UI-side caching handles persistence.
"""

import logging
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..agents.executor_node import run_executor_node
from ..agents.planner_node import run_planner_node
from ..agents.profiler_node import run_profiler_node
from ..agents.reporter_node import run_reporter_node
from ..models.state import AgentState

logger = logging.getLogger(__name__)


def _abort_node(state: AgentState) -> dict[str, Any]:
    """Terminal abort handler.

    Provides an explicit, loggable terminal step for abort paths instead
    of routing directly to END, ensuring the abort reason is visible in
    trace and debug outputs.
    """
    logger.error(
        "--- GRAPH ABORTED --- last error: %s",
        state.error_log[-1] if state.error_log else "unknown",
    )
    return {}


def _route_after_profiler(state: AgentState) -> str:
    """Routes execution based on the Profiler's completion status."""
    return "abort" if state.execution_status == "aborted" else "planner"


def _route_after_planner(state: AgentState) -> str:
    """Routes execution based on the Planner's compilation status."""
    return "abort" if state.execution_status == "aborted" else "executor"


def _route_after_executor(state: AgentState) -> str:
    """Routes execution based on the Executor's step-level status.

    Enforces the one-shot lazy-fetch replan constraint and the
    deterministic step-iteration loop.
    """
    status = state.execution_status
    if status == "running":
        return "executor"
    if status == "replan":
        return "planner"
    if status == "complete":
        return "reporter"
    if status == "aborted":
        return "abort"

    # Defensive routing: Fail loudly on unrecognized states to prevent
    # silent infinite loops or misrouted execution.
    raise ValueError(
        f"Executor produced unroutable execution_status: '{status}'. "
        "Expected one of: running, replan, complete, aborted."
    )


def build_graph() -> CompiledStateGraph[AgentState]:
    """Constructs and compiles the Scrygent LangGraph application.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    graph = StateGraph(AgentState)

    graph.add_node("profiler", run_profiler_node)
    graph.add_node("planner", run_planner_node)
    graph.add_node("executor", run_executor_node)
    graph.add_node("reporter", run_reporter_node)
    graph.add_node("abort", _abort_node)

    graph.set_entry_point("profiler")

    graph.add_conditional_edges(
        "profiler",
        _route_after_profiler,
        {"planner": "planner", "abort": "abort"},
    )
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"executor": "executor", "abort": "abort"},
    )
    graph.add_conditional_edges(
        "executor",
        _route_after_executor,
        {
            "executor": "executor",
            "planner": "planner",
            "reporter": "reporter",
            "abort": "abort",
        },
    )

    graph.add_edge("reporter", END)
    graph.add_edge("abort", END)

    return graph.compile()
