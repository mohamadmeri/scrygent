"""Destructive test suite for the LangGraph orchestration and routing logic.

This module aggressively tests the graph construction and conditional
routing functions. It ensures that execution statuses route to the exact
correct nodes, and that hallucinated or unrecognized statuses trigger an
immediate fail-fast ValueError to prevent silent infinite loops.
"""

import pytest
from langgraph.graph.state import CompiledStateGraph

from scrygent.graph.builder import _route_after_executor, _route_after_planner, _route_after_profiler, build_graph
from scrygent.models.state import AgentState


class TestGraphCompilation:
    """Tests validating the structural integrity of the compiled graph."""

    def test_build_graph_compiles_successfully_without_errors(self) -> None:
        """Invoke the graph builder.

        Asserts the function returns a compiled LangGraph object without raising
        any missing-edge or node-configuration exceptions.
        """
        graph = build_graph()
        assert isinstance(graph, CompiledStateGraph)


class TestProfilerRouting:
    """Tests validating the routing logic after the Profiler node."""

    def test_routes_to_planner_on_success(self, valid_agent_state: AgentState) -> None:
        """Inject a state with `execution_status="running"`.

        Asserts the router returns "planner" to continue the pipeline.
        """
        valid_agent_state.execution_status = "running"
        assert _route_after_profiler(valid_agent_state) == "planner"

    def test_routes_to_abort_on_failure(self, valid_agent_state: AgentState) -> None:
        """Inject a state with `execution_status="aborted"`.

        Asserts the router returns "abort" to halt the pipeline gracefully.
        """
        valid_agent_state.execution_status = "aborted"
        assert _route_after_profiler(valid_agent_state) == "abort"


class TestPlannerRouting:
    """Tests validating the routing logic after the Planner node."""

    def test_routes_to_executor_on_success(self, valid_agent_state: AgentState) -> None:
        """Inject a state with `execution_status="running"`.

        Asserts the router returns "executor" to begin step dispatch.
        """
        valid_agent_state.execution_status = "running"
        assert _route_after_planner(valid_agent_state) == "executor"

    def test_routes_to_abort_on_failure(self, valid_agent_state: AgentState) -> None:
        """Inject a state with `execution_status="aborted"`.

        Asserts the router returns "abort" to halt the pipeline gracefully.
        """
        valid_agent_state.execution_status = "aborted"
        assert _route_after_planner(valid_agent_state) == "abort"


class TestExecutorRouting:
    """Tests validating the complex routing logic after the Executor node."""

    def test_routes_to_executor_on_running_status(self, valid_agent_state: AgentState) -> None:
        """Inject a state with `execution_status="running"`.

        Asserts the router returns "executor" to loop to the next step.
        """
        valid_agent_state.execution_status = "running"
        assert _route_after_executor(valid_agent_state) == "executor"

    def test_routes_to_planner_on_replan_status(self, valid_agent_state: AgentState) -> None:
        """Inject a state with `execution_status="replan"`.

        Asserts the router returns "planner" to trigger the constrained lazy-fetch loop.
        """
        valid_agent_state.execution_status = "replan"
        assert _route_after_executor(valid_agent_state) == "planner"

    def test_routes_to_reporter_on_complete_status(self, valid_agent_state: AgentState) -> None:
        """Inject a state with `execution_status="complete"`.

        Asserts the router returns "reporter" to begin final synthesis.
        """
        valid_agent_state.execution_status = "complete"
        assert _route_after_executor(valid_agent_state) == "reporter"

    def test_routes_to_abort_on_aborted_status(self, valid_agent_state: AgentState) -> None:
        """Inject a state with `execution_status="aborted"`.

        Asserts the router returns "abort" to halt the pipeline gracefully.
        """
        valid_agent_state.execution_status = "aborted"
        assert _route_after_executor(valid_agent_state) == "abort"

    def test_rejects_hallucinated_status_with_exact_error(self, valid_agent_state: AgentState) -> None:
        """Inject a state with a hallucinated `execution_status="failed"`.

        The router must raise a ValueError to prevent silent misrouting or
        infinite loops on unrecognized state machine transitions.
        """
        valid_agent_state.execution_status = "failed"  # type: ignore[assignment]

        with pytest.raises(ValueError) as exc_info:
            _route_after_executor(valid_agent_state)

        assert "Executor produced unroutable execution_status: 'failed'." in str(exc_info.value)
        assert "Expected one of: running, replan, complete, aborted." in str(exc_info.value)
