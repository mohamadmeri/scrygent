"""Destructive and functional test suite for the final synthesis Reporter Node.

This module aggressively tests the output synthesis and semantic memory
commitment. It ensures that eval mode enforces the strict DirectAnswer
schema, successful executions are committed to Qdrant, and rate-limit
exhaustion triggers an immediate graceful abort.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from scrygent.agents.reporter_node import run_reporter_node
from scrygent.contracts.tool_names import ToolName
from scrygent.core.resilience import ServiceExhaustedError
from scrygent.models.outputs import AnalysisReport, DirectAnswer
from scrygent.models.state import AgentState
from scrygent.models.step_models import Plan, Step


@pytest.fixture(autouse=True)
def mock_reporter_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock network calls and LLM initialization to isolate synthesis logic."""
    monkeypatch.setattr("scrygent.agents.reporter_node.get_structured_llm", lambda **kwargs: MagicMock())
    monkeypatch.setattr("scrygent.agents.reporter_node.commit_experience", lambda q, p: None)


@pytest.fixture
def completed_state(valid_agent_state: AgentState) -> AgentState:
    """Provide an AgentState primed with completed step outputs and a plan."""
    valid_agent_state.execution_status = "running"
    valid_agent_state.step_outputs = {"step_1": {"result": 42.0}}
    valid_agent_state.plan = Plan(
        steps=[Step(step_id="step_1", rationale="Test", tool_name=ToolName.ANALYZE_DATA, parameters={})]
    )
    return valid_agent_state


class TestRunReporterNodeExecution:
    """Tests validating the output routing and schema enforcement."""

    def test_use_case_synthesizes_standard_analysis_report(
        self, completed_state: AgentState, resilient_call_mock: Any
    ) -> None:
        """Inject a valid state with `eval_mode=False`.

        Asserts the node returns an `AnalysisReport` object and sets the
        `execution_status` to `"complete"`.
        """
        mock_report = AnalysisReport(primary_answer="The answer is 42.")
        with resilient_call_mock([mock_report]):
            result = run_reporter_node(completed_state)

        assert result["execution_status"] == "complete"
        assert isinstance(result["final_report"], AnalysisReport)

    def test_use_case_synthesizes_eval_mode_direct_answer(
        self, completed_state: AgentState, resilient_call_mock: Any
    ) -> None:
        """Inject a valid state with `eval_mode=True`.

        Asserts the node returns a `DirectAnswer` object, dropping narrative.
        """
        completed_state.eval_mode = True
        mock_answer = DirectAnswer(answer="42.0")
        with resilient_call_mock([mock_answer]):
            result = run_reporter_node(completed_state)

        assert result["execution_status"] == "complete"
        assert isinstance(result["final_report"], DirectAnswer)


class TestRunReporterNodeMemory:
    """Tests validating the integration with the semantic memory engine."""

    def test_commits_successful_plan_to_memory(
        self, completed_state: AgentState, resilient_call_mock: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inject a valid state that successfully synthesizes.

        Asserts `commit_experience` is invoked exactly once with the exact
        user query and Plan object.
        """
        mock_commit = MagicMock()
        monkeypatch.setattr("scrygent.agents.reporter_node.commit_experience", mock_commit)

        with resilient_call_mock([AnalysisReport(primary_answer="Done.")]):
            run_reporter_node(completed_state)

        mock_commit.assert_called_once_with(completed_state.user_query, completed_state.plan)

    def test_skips_memory_commit_if_plan_is_missing(
        self, completed_state: AgentState, resilient_call_mock: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inject a state missing the `plan` field.

        Asserts `commit_experience` is never called, preventing NoneType
        exceptions in the memory layer.
        """
        completed_state.plan = None
        mock_commit = MagicMock()
        monkeypatch.setattr("scrygent.agents.reporter_node.commit_experience", mock_commit)

        with resilient_call_mock([AnalysisReport(primary_answer="Done.")]):
            run_reporter_node(completed_state)

        mock_commit.assert_not_called()


class TestRunReporterNodeFailures:
    """Tests validating the graceful failure modes and exact error payloads."""

    def test_aborts_on_service_exhausted_error(self, completed_state: AgentState, resilient_call_mock: Any) -> None:
        """Mock `resilient_call` to raise `ServiceExhaustedError`.

        The node must catch the rate-limit failure and return an aborted status
        with an exact message containing the service name.
        """
        error = ServiceExhaustedError(service="Reporter", attempts=3, last_error=Exception("429"))

        with resilient_call_mock([error]):
            result = run_reporter_node(completed_state)

        assert result["execution_status"] == "aborted"
        assert (
            "Reporter is temporarily unavailable (rate limited after 3 attempts) while synthesizing the final report."
            in result["error_log"][-1]
        )

    def test_aborts_on_unexpected_exception(self, completed_state: AgentState, resilient_call_mock: Any) -> None:
        """Mock `resilient_call` to raise a standard `RuntimeError`.

        The node must catch unexpected errors and wrap them in an aborted status.
        """
        with resilient_call_mock([RuntimeError("JSON parse error")]):
            result = run_reporter_node(completed_state)

        assert result["execution_status"] == "aborted"
        assert "Reporter failed: JSON parse error" in result["error_log"][-1]
