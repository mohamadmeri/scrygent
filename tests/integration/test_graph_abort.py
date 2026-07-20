"""Integration tests for graph abort routing and error logging.

This module verifies that an `aborted` execution status from any node
correctly routes to the terminal abort handler, bypassing all downstream
nodes, and that the error payload is preserved.
"""

from unittest.mock import MagicMock

import pytest

from scrygent.graph.builder import build_graph
from scrygent.models.state import AgentState


@pytest.fixture
def mock_nodes(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Mock all graph nodes to isolate routing and error handling logic."""
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


class TestGraphAbortRouting:
    """Tests validating that abort signals halt the pipeline gracefully."""

    def test_profiler_abort_skips_all_downstream_nodes(
        self, valid_agent_state: AgentState, mock_nodes: dict[str, MagicMock]
    ) -> None:
        """Inject an aborted status from the Profiler.

        Asserts the graph routes directly to abort, skipping Planner, Executor,
        and Reporter, and preserves the error log.
        """
        mock_nodes["profiler"].return_value = {
            "execution_status": "aborted",
            "error_log": ["Profiler initialization failed: File not found"],
        }

        graph = build_graph()
        final_state = graph.invoke(valid_agent_state)

        mock_nodes["planner"].assert_not_called()
        mock_nodes["executor"].assert_not_called()
        mock_nodes["reporter"].assert_not_called()
        mock_nodes["abort"].assert_called_once()

        assert final_state["execution_status"] == "aborted"
        assert "Profiler initialization failed: File not found" in final_state["error_log"]

    def test_planner_abort_skips_executor_and_reporter(
        self, valid_agent_state: AgentState, mock_nodes: dict[str, MagicMock]
    ) -> None:
        """Inject an aborted status from the Planner.

        Asserts the graph routes directly to abort, skipping Executor and Reporter.
        """
        mock_nodes["planner"].return_value = {
            "execution_status": "aborted",
            "error_log": ["Planner failed to generate a valid Plan after 3 attempts"],
        }

        graph = build_graph()
        final_state = graph.invoke(valid_agent_state)

        mock_nodes["executor"].assert_not_called()
        mock_nodes["reporter"].assert_not_called()
        mock_nodes["abort"].assert_called_once()

        assert final_state["execution_status"] == "aborted"

    def test_executor_abort_skips_reporter(
        self, valid_agent_state: AgentState, mock_nodes: dict[str, MagicMock]
    ) -> None:
        """Inject an aborted status from the Executor.

        Asserts the graph routes directly to abort, skipping the Reporter.
        """
        mock_nodes["executor"].return_value = {
            "execution_status": "aborted",
            "error_log": ["Runtime error in analyze_data: Permanent failure"],
        }

        graph = build_graph()
        final_state = graph.invoke(valid_agent_state)

        mock_nodes["reporter"].assert_not_called()
        mock_nodes["abort"].assert_called_once()

        assert final_state["execution_status"] == "aborted"
