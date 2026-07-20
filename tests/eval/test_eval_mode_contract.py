"""Eval mode contract tests.

This module verifies the architectural guarantees of `eval_mode=True`:
1. The `DirectAnswer` schema strictly forbids narrative and plots.
2. The Reporter Node correctly selects the `DirectAnswer` schema and prompt
   when `eval_mode` is active.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from scrygent.agents.reporter_node import run_reporter_node
from scrygent.contracts.tool_names import ToolName
from scrygent.models.outputs import AnalysisReport, DirectAnswer
from scrygent.models.state import AgentState
from scrygent.models.step_models import Plan, Step


@pytest.fixture
def eval_state(valid_agent_state: AgentState) -> AgentState:
    """Provide an AgentState explicitly configured for eval mode."""
    valid_agent_state.eval_mode = True
    valid_agent_state.execution_status = "running"
    valid_agent_state.step_outputs = {"step_1": {"result": 42.0}}
    valid_agent_state.plan = Plan(
        steps=[Step(step_id="step_1", rationale="Test", tool_name=ToolName.ANALYZE_DATA, parameters={})]
    )
    return valid_agent_state


class TestEvalModeSchemaContract:
    """Tests validating the strict structural boundaries of the DirectAnswer schema."""

    def test_direct_answer_schema_drops_narrative_and_plots(self) -> None:
        """Inspect the `DirectAnswer` Pydantic model fields.

        Asserts the schema strictly forbids `additional_insights` and `plots`,
        ensuring benchmark mode can never leak narrative artifacts.
        """
        fields = DirectAnswer.model_fields

        assert "answer" in fields
        assert "additional_insights" not in fields
        assert "plots" not in fields

    def test_analysis_report_schema_allows_narrative_and_plots(self) -> None:
        """Inspect the `AnalysisReport` Pydantic model fields.

        Asserts the standard schema includes narrative fields, proving
        the two modes are structurally distinct.
        """
        fields = AnalysisReport.model_fields

        assert "primary_answer" in fields
        assert "additional_insights" in fields
        assert "plots" in fields


class TestEvalModeNodeRouting:
    """Tests validating that the Reporter Node respects the eval_mode flag."""

    def test_reporter_selects_direct_answer_schema_when_eval_mode_is_true(
        self, eval_state: AgentState, monkeypatch: pytest.MonkeyPatch, resilient_call_mock: Any
    ) -> None:
        """Inject an AgentState with `eval_mode=True`.

        Asserts `get_structured_llm` is invoked with the `DirectAnswer` schema,
        guaranteeing the LLM is physically constrained from generating narrative.
        """
        mock_get_llm = MagicMock()
        monkeypatch.setattr("scrygent.agents.reporter_node.get_structured_llm", mock_get_llm)
        monkeypatch.setattr("scrygent.agents.reporter_node.commit_experience", lambda q, p: None)

        with resilient_call_mock([DirectAnswer(answer="42.0")]):
            run_reporter_node(eval_state)

        mock_get_llm.assert_called_once()
        _, kwargs = mock_get_llm.call_args
        assert kwargs["pydantic_schema"] is DirectAnswer

    def test_reporter_selects_analysis_report_schema_when_eval_mode_is_false(
        self, valid_agent_state: AgentState, monkeypatch: pytest.MonkeyPatch, resilient_call_mock: Any
    ) -> None:
        """Inject an AgentState with `eval_mode=False`.

        Asserts `get_structured_llm` is invoked with the `AnalysisReport` schema.
        """
        valid_agent_state.execution_status = "running"
        valid_agent_state.step_outputs = {"step_1": {"result": 42.0}}
        valid_agent_state.plan = Plan(
            steps=[Step(step_id="step_1", rationale="Test", tool_name=ToolName.ANALYZE_DATA, parameters={})]
        )

        mock_get_llm = MagicMock()
        monkeypatch.setattr("scrygent.agents.reporter_node.get_structured_llm", mock_get_llm)
        monkeypatch.setattr("scrygent.agents.reporter_node.commit_experience", lambda q, p: None)

        with resilient_call_mock([AnalysisReport(primary_answer="42.0")]):
            run_reporter_node(valid_agent_state)

        mock_get_llm.assert_called_once()
        _, kwargs = mock_get_llm.call_args
        assert kwargs["pydantic_schema"] is AnalysisReport
