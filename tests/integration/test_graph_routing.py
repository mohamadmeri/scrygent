"""Integration tests for LangGraph routing and state transitions.

This module verifies that the compiled graph wires nodes correctly and
routes execution based on `AgentState.execution_status` transitions. It
aggressively tests for infinite loop prevention and fail-fast behavior
on hallucinated state machine statuses.
"""

from unittest.mock import MagicMock

import pytest
from langgraph.graph.state import CompiledStateGraph

from scrygent.graph.builder import build_graph
from scrygent.models.state import AgentState


@pytest.fixture
def mock_nodes(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Mock all graph nodes to isolate routing logic from business logic."""
    nodes = {
        "profiler": MagicMock(return_value={"execution_status": "running"}),
        "planner": MagicMock(return_value={"execution_status": "running"}),
        "executor": MagicMock(return_value={"execution_status": "complete"}),
        "reporter": MagicMock(return_value={"execution_status": "complete"}),
        "abort": MagicMock(return_value={}),
    }
    monkeypatch.setattr("scrygent.graph.builder.run_profiler_node", nodes["profiler"])
    monkeypatch.setattr("scrygent.graph.builder.run_planner_node", nodes["planner"])
    monkeypatch.setattr("scrygent.graph.builder.run_executor_node", nodes["executor"])
    monkeypatch.setattr("scrygent.graph.builder.run_reporter_node", nodes["reporter"])
    monkeypatch.setattr("scrygent.graph.builder._abort_node", nodes["abort"])
    return nodes


class TestGraphRouting:
    """Tests validating the structural integrity and routing of the compiled graph."""

    def test_graph_routes_success_path_profiler_planner_executor_reporter(
        self, valid_agent_state: AgentState, mock_nodes: dict[str, MagicMock]
    ) -> None:
        """Inject a standard execution flow where all nodes return success.

        Asserts the graph strictly follows the path: profiler -> planner ->
        executor -> reporter, and never touches the abort node.
        """
        graph = build_graph()
        assert isinstance(graph, CompiledStateGraph)

        final_state = graph.invoke(valid_agent_state)

        mock_nodes["profiler"].assert_called_once()
        mock_nodes["planner"].assert_called_once()
        mock_nodes["executor"].assert_called_once()
        mock_nodes["reporter"].assert_called_once()
        mock_nodes["abort"].assert_not_called()

        assert final_state["execution_status"] == "complete"

    def test_graph_routes_to_abort_on_profiler_failure(
        self, valid_agent_state: AgentState, mock_nodes: dict[str, MagicMock]
    ) -> None:
        """Inject an aborted status from the Profiler.

        Asserts the graph skips all downstream processing and routes immediately
        to the abort node to halt the pipeline.
        """
        mock_nodes["profiler"].return_value = {"execution_status": "aborted", "error_log": ["Profiler failed"]}

        graph = build_graph()
        final_state = graph.invoke(valid_agent_state)

        mock_nodes["profiler"].assert_called_once()
        mock_nodes["planner"].assert_not_called()
        mock_nodes["executor"].assert_not_called()
        mock_nodes["reporter"].assert_not_called()
        mock_nodes["abort"].assert_called_once()

        assert final_state["execution_status"] == "aborted"

    def test_graph_routes_replan_loop_exactly_once(
        self, valid_agent_state: AgentState, mock_nodes: dict[str, MagicMock]
    ) -> None:
        """Inject a 'replan' status from the Executor, followed by 'complete'.

        Asserts the graph correctly routes back to the Planner and then back
        to the Executor, verifying the constrained lazy-fetch back-edge.
        """
        mock_nodes["executor"].side_effect = [
            {"execution_status": "replan", "has_replanned": True},
            {"execution_status": "complete"},
        ]

        graph = build_graph()
        final_state = graph.invoke(valid_agent_state)

        mock_nodes["profiler"].assert_called_once()
        assert mock_nodes["planner"].call_count == 2
        assert mock_nodes["executor"].call_count == 2
        mock_nodes["reporter"].assert_called_once()
        mock_nodes["abort"].assert_not_called()

        assert final_state["execution_status"] == "complete"
        assert final_state["has_replanned"] is True

    def test_router_raises_valueerror_on_unroutable_status(self, valid_agent_state: AgentState) -> None:
        """Inject a hallucinated 'failed' status directly into the router function.

        The state schema (Pydantic Literal) prevents invalid statuses from ever
        reaching the router during a real graph invocation. We test the router
        function directly to ensure it fails fast if bypassed.
        """
        from scrygent.graph.builder import _route_after_executor

        valid_agent_state.execution_status = "failed"  # type: ignore[assignment]

        with pytest.raises(ValueError) as exc_info:
            _route_after_executor(valid_agent_state)

        assert "Executor produced unroutable execution_status: 'failed'." in str(exc_info.value)
        assert "Expected one of: running, replan, complete, aborted." in str(exc_info.value)
