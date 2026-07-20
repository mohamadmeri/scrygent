"""Destructive test suite for the 3-Pass Compiler Planner Node.

This module aggressively tests the LLM-driven planning pipeline. It ensures
that missing profiles, malformed LLM JSON outputs, and rate-limit exhaustion
are handled gracefully, and that the internal 3-pass self-healing loop
correctly triggers and recovers without crashing the graph.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from scrygent.agents.planner_node import run_planner_node
from scrygent.contracts.tool_names import ToolName
from scrygent.core.resilience import ServiceExhaustedError
from scrygent.models.abstract_step_models import AbstractStep, DraftPlan
from scrygent.models.outputs import CSVProfile
from scrygent.models.state import AgentState
from scrygent.models.step_models import Plan, Step


def _make_validation_error() -> ValidationError:
    """Helper to generate a real Pydantic ValidationError for the mock."""
    try:
        Plan.model_validate({"steps": [{"step_id": "s1", "tool_name": "fake_tool"}]})
    except ValidationError as e:
        return e
    raise RuntimeError("Failed to generate validation error")


@pytest.fixture
def valid_draft_plan() -> DraftPlan:
    """Provide a valid DraftPlan for Pass 1 and Pass 2."""
    return DraftPlan(
        rationale="Filter and aggregate",
        steps=[AbstractStep(step_id="step_1", tool_name=ToolName.ANALYZE_DATA, intent_description="Get avg age")],
    )


@pytest.fixture
def valid_final_plan() -> Plan:
    """Provide a valid strict Plan for Pass 3."""
    return Plan(
        steps=[
            Step(
                step_id="step_1",
                rationale="Get avg age",
                tool_name=ToolName.ANALYZE_DATA,
                parameters={"metrics": [{"column": "age", "aggregation": "mean", "alias": "avg_age"}]},
            )
        ]
    )


@pytest.fixture
def planner_state(valid_agent_state: AgentState) -> AgentState:
    """Provide an AgentState with a valid data_profile to prevent early aborts."""
    valid_agent_state.data_profile = CSVProfile(
        row_count=10, global_schema={"age": "int64"}, detailed_stats={"age": {"mean": 30.0}}, missing_detailed_stats=[]
    )
    return valid_agent_state


@pytest.fixture(autouse=True)
def mock_planner_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock network calls and file I/O to isolate the compiler logic."""
    monkeypatch.setattr("scrygent.agents.planner_node.retrieve_experience", lambda q: "No experience.")
    monkeypatch.setattr("scrygent.agents.planner_node.get_structured_llm", lambda **kwargs: MagicMock())
    monkeypatch.setattr("scrygent.agents.planner_node._dump_plan_debug", lambda *args: None)


class TestRunPlannerNodeExecution:
    """Tests validating the end-to-end compilation and state mutation."""

    def test_use_case_compiles_three_passes_and_returns_running_status(
        self,
        planner_state: AgentState,
        valid_draft_plan: DraftPlan,
        valid_final_plan: Plan,
        resilient_call_mock: Any,
    ) -> None:
        """Inject a valid AgentState and mock all 3 LLM passes to succeed.

        Asserts the node returns the exact final Plan object and sets the
        `execution_status` to `"running"`.
        """
        with resilient_call_mock([valid_draft_plan, valid_draft_plan, valid_final_plan]):
            result = run_planner_node(planner_state)

        assert result["execution_status"] == "running"
        assert isinstance(result["plan"], Plan)
        assert len(result["plan"].steps) == 1


class TestRunPlannerNodeFailures:
    """Tests validating the graceful failure modes and exact error payloads."""

    def test_aborts_with_exact_error_when_data_profile_is_missing(self, valid_agent_state: AgentState) -> None:
        """Inject a state with `data_profile=None`.

        The node must immediately abort and return an exact error message
        preventing the LLM from being invoked with null context.
        """
        valid_agent_state.data_profile = None
        result = run_planner_node(valid_agent_state)

        assert result["execution_status"] == "aborted"
        assert "Planner invoked but data_profile is missing from state." in result["error_log"]

    def test_aborts_on_service_exhausted_error(
        self,
        planner_state: AgentState,
        resilient_call_mock: Any,
    ) -> None:
        """Mock `resilient_call` to raise `ServiceExhaustedError` on Pass 1.

        The node must catch the rate-limit failure and return an aborted status
        with an exact message containing the service name.
        """
        error = ServiceExhaustedError(service="Planner (Parser Pass)", attempts=3, last_error=Exception("429"))

        with resilient_call_mock([error]):
            result = run_planner_node(planner_state)

        assert result["execution_status"] == "aborted"
        assert any("Planner (Parser Pass) is temporarily unavailable" in msg for msg in result["error_log"])

    def test_aborts_on_unexpected_exception(
        self,
        planner_state: AgentState,
        resilient_call_mock: Any,
    ) -> None:
        """Mock `resilient_call` to raise a standard `RuntimeError`.

        The node must catch unexpected errors and wrap them in an aborted status.
        """
        with resilient_call_mock([RuntimeError("LLM returned 500")]):
            result = run_planner_node(planner_state)

        assert result["execution_status"] == "aborted"
        assert "Compiler Pipeline failed: LLM returned 500" in result["error_log"]


class TestPlannerSelfHealingLoop:
    """Tests validating the internal Pass 3 correction loop."""

    def test_self_heals_pass_3_validation_error_on_first_retry(
        self,
        planner_state: AgentState,
        valid_draft_plan: DraftPlan,
        valid_final_plan: Plan,
        resilient_call_mock: Any,
    ) -> None:
        """Inject a `ValidationError` on the first Pass 3 attempt, then success.

        The node must catch the schema failure, append the error context to the
        prompt, retry, and ultimately return a successful `"running"` status.
        """
        val_error = _make_validation_error()

        with resilient_call_mock([valid_draft_plan, valid_draft_plan, val_error, valid_final_plan]) as script:
            result = run_planner_node(planner_state)

        assert result["execution_status"] == "running"
        assert isinstance(result["plan"], Plan)
        # 2 calls for Pass 1/2, 2 calls for Pass 3 (1 fail + 1 success)
        assert script.call_count == 4

    def test_aborts_after_max_self_healing_retries(
        self,
        planner_state: AgentState,
        valid_draft_plan: DraftPlan,
        resilient_call_mock: Any,
    ) -> None:
        """Inject a `ValidationError` on all Pass 3 attempts.

        The node must exhaust its 2 retries (3 total attempts) and then abort
        with an exact error message containing the final validation failure.
        """
        val_error = _make_validation_error()

        # Pass 1 (success), Pass 2 (success), Pass 3 Attempt 1 (fail), Attempt 2 (fail), Attempt 3 (fail)
        with resilient_call_mock([valid_draft_plan, valid_draft_plan, val_error, val_error, val_error]):
            result = run_planner_node(planner_state)

        assert result["execution_status"] == "aborted"
        assert any("Planner failed to generate a valid Plan after 3 attempts" in msg for msg in result["error_log"])
        assert any("Last error: Field" in msg for msg in result["error_log"])
