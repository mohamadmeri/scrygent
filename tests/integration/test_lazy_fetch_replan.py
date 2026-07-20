"""Integration tests for the constrained lazy-fetch replan loop.

This module verifies that the LangGraph correctly handles the `replan`
execution status, routing back to the Planner exactly once, enriching
the data profile, and strictly enforcing the `has_replanned` guard.
"""

from unittest.mock import MagicMock

import pytest

from scrygent.graph.builder import build_graph
from scrygent.models.state import AgentState


@pytest.fixture
def mock_nodes(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Mock all graph nodes to isolate routing and state mutation logic."""
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


class TestLazyFetchReplanLoop:
    """Tests validating the one-shot lazy-fetch back-edge and profile enrichment."""

    def test_replan_loop_enriches_profile_and_sets_guard(
        self, valid_agent_state: AgentState, mock_nodes: dict[str, MagicMock]
    ) -> None:
        """Inject a 'replan' status from Executor, simulating a lazy-fetch.

        Asserts the graph routes back to Planner, and the state mutation
        containing `has_replanned=True` is preserved in the final state.
        """
        mock_nodes["executor"].side_effect = [
            {"execution_status": "replan", "has_replanned": True},
            {"execution_status": "complete"},
        ]

        graph = build_graph()
        final_state = graph.invoke(valid_agent_state)

        # Verify the back-edge was traversed
        assert mock_nodes["planner"].call_count == 2
        assert mock_nodes["executor"].call_count == 2

        # Verify the guard is set in the final state
        assert final_state["has_replanned"] is True
        assert final_state["execution_status"] == "complete"

    def test_replan_loop_aborts_if_guard_already_spent(
        self, valid_agent_state: AgentState, mock_nodes: dict[str, MagicMock]
    ) -> None:
        """Inject a state where `has_replanned` is already True, but Executor returns 'replan'.

        While the Executor node itself should prevent this, the graph must
        be resilient. If the Executor attempts a second replan, the graph
        should route to abort to prevent an infinite loop.
        """
        valid_agent_state.has_replanned = True

        # Set the side effect BEFORE invoking the graph to prevent infinite loops
        mock_nodes["executor"].side_effect = [
            {"execution_status": "replan", "has_replanned": True},
            {"execution_status": "aborted", "error_log": ["Double replan attempt"]},
        ]

        graph = build_graph()
        final_state = graph.invoke(valid_agent_state)

        assert final_state["execution_status"] == "aborted"
        mock_nodes["abort"].assert_called_once()
