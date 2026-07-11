"""
LangGraph Graph Builder.

Wires the four nodes (profiler -> planner -> executor -> reporter) with
conditional routing driven entirely by AgentState.execution_status.

Routing contract (matches ARCHITECTURE.md + the one-shot lazy-fetch
replan extension):
    profiler  -> planner   (always, unless profiler aborted)
    planner   -> executor  (always, unless planner aborted)
    executor  -> executor  (execution_status == "running": more steps remain)
    executor  -> planner   (execution_status == "replan": ONE-TIME lazy-fetch
                             re-plan only -- guarded by state.has_replanned in
                             executor_node.py so this edge can only fire once
                             per session)
    executor  -> reporter  (execution_status == "complete")
    executor  -> abort     (execution_status == "aborted")
    reporter  -> END       (always; reporter itself sets complete/aborted
                             but has no outgoing branches -- it's terminal)

NOTE: this file assumes AgentState.has_replanned (bool, default False) has
been added per the earlier review. If it hasn't landed yet in state.py,
the "replan" edge below has no protection against a second lazy-fetch
loop and executor_node.py's own guard is the only thing stopping it.
"""

import logging

from langgraph.graph import StateGraph, END

from ..models.state import AgentState
from ..agents.profiler_node import run_profiler_node
from ..agents.planner_node import run_planner_node
from ..agents.executor_node import run_executor_node
from ..agents.reporter_node import run_reporter_node

logger = logging.getLogger(__name__)


def _abort_node(state: AgentState) -> dict:
    """
    Terminal abort handler. Does not mutate state further -- by the time
    we're here, execution_status is already "aborted" and error_log
    already carries the reason. This node exists only so the graph has
    an explicit, loggable terminal step instead of routing straight to
    END from multiple places, which would make the abort path invisible
    in trace/debug output.
    """
    logger.error(
        "--- GRAPH ABORTED --- last error: %s",
        state.error_log[-1] if state.error_log else "unknown",
    )
    return {}


def _route_after_profiler(state: AgentState) -> str:
    return "abort" if state.execution_status == "aborted" else "planner"


def _route_after_planner(state: AgentState) -> str:
    return "abort" if state.execution_status == "aborted" else "executor"


def _route_after_executor(state: AgentState) -> str:
    status = state.execution_status
    if status == "running":
        return "executor"
    if status == "replan":
        return "planner"
    if status == "complete":
        return "reporter"
    if status == "aborted":
        return "abort"
    # Defensive: any other value is a programming error, not a valid
    # routing decision. Fail loudly instead of silently defaulting
    # somewhere -- an unhandled status here means a node returned
    # something the router was never taught about.
    raise ValueError(
        f"Executor produced unroutable execution_status: '{status}'. "
        "Expected one of: running, replan, complete, aborted."
    )


def build_graph():
    """
    Constructs and compiles the Scrygent LangGraph application.
    No checkpointer is used -- per DESIGN.md, this is a single-pass
    fire-and-forget engine; st.session_state handles UI-side caching.
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
