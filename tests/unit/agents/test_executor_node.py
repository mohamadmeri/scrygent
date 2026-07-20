"""Destructive test suite for the hybrid deterministic Executor Node.

This module aggressively tests the tool dispatching, multi-step composition,
and internal self-healing correction chain. It ensures that kwarg injection
is precise, CSV swapping is strict, lazy-fetch guards hold firm, and runtime
failures are either corrected or aborted with exact error payloads.
"""

from unittest.mock import MagicMock

import pytest

from scrygent.agents.executor_node import _TOOL_DISPATCHER, run_executor_node
from scrygent.contracts.tool_names import ToolName
from scrygent.core.resilience import ServiceExhaustedError
from scrygent.models.outputs import CSVProfile
from scrygent.models.state import AgentState
from scrygent.models.step_models import Plan, Step


@pytest.fixture
def executor_state(valid_agent_state: AgentState) -> AgentState:
    """Provide an AgentState primed with a single-step plan for execution."""
    valid_agent_state.execution_status = "running"
    valid_agent_state.current_step_index = 0
    valid_agent_state.plan = Plan(
        steps=[
            Step(
                step_id="step_1",
                rationale="Test",
                tool_name=ToolName.ANALYZE_DATA,
                parameters={"metrics": [{"column": "age", "aggregation": "mean", "alias": "avg_age"}]},
            )
        ]
    )
    valid_agent_state.data_profile = CSVProfile(
        row_count=10, global_schema={"age": "int64"}, detailed_stats={}, missing_detailed_stats=["age"]
    )
    return valid_agent_state


@pytest.fixture
def mock_correction_chain(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock the LLM correction chain to isolate retry logic."""
    mock = MagicMock(return_value={"corrected": True})
    monkeypatch.setattr("scrygent.agents.executor_node._run_correction_chain", mock)
    return mock


class TestExecutorDispatchAndInjection:
    """Tests validating the exact kwargs injection and state mutations per tool type."""

    def test_use_case_executes_analyze_data_with_loaded_dataframe(
        self, executor_state: AgentState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inject a plan step for `analyze_data`.

        Asserts the Executor loads the CSV into a DataFrame and passes it via `df`
        rather than `current_csv_path`, and advances the step index.
        """
        mock_tool = MagicMock(return_value={"result": 42.0})
        monkeypatch.setitem(_TOOL_DISPATCHER, ToolName.ANALYZE_DATA, mock_tool)

        result = run_executor_node(executor_state)

        mock_tool.assert_called_once()
        _, kwargs = mock_tool.call_args
        assert "df" in kwargs
        assert "current_csv_path" not in kwargs

        assert result["execution_status"] == "complete"
        assert result["current_step_index"] == 1

    def test_use_case_executes_filter_dataset_and_swaps_csv_path(
        self, executor_state: AgentState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inject a plan step for `filter_dataset`.

        Asserts the Executor passes `current_csv_path` and updates the state's
        `current_csv_path` to the new path returned by the tool.
        """
        assert executor_state.plan is not None
        executor_state.plan.steps[0].tool_name = ToolName.FILTER_DATASET
        executor_state.plan.steps[0].parameters = {"filters": [{"column": "age", "operator": ">", "value": 20}]}

        new_path = "/tmp/filtered.csv"
        mock_tool = MagicMock(return_value={"current_csv_path": new_path, "row_count": 5})
        monkeypatch.setitem(_TOOL_DISPATCHER, ToolName.FILTER_DATASET, mock_tool)

        result = run_executor_node(executor_state)

        _, kwargs = mock_tool.call_args
        # executor_node passes the Path object directly
        assert kwargs["current_csv_path"] == executor_state.current_csv_path
        assert result["current_csv_path"] == new_path


class TestExecutorSelfHealing:
    """Tests validating the internal correction loop on tool failures."""

    def test_self_heals_on_tool_failure_and_succeeds_on_retry(
        self, executor_state: AgentState, monkeypatch: pytest.MonkeyPatch, mock_correction_chain: MagicMock
    ) -> None:
        """Inject a tool that fails once, then succeeds on retry.

        Asserts the Executor catches the exception, invokes the correction chain,
        and retries the tool with the repaired parameters.
        """
        mock_tool = MagicMock(side_effect=[ValueError("Bad param"), {"result": 42.0}])
        monkeypatch.setitem(_TOOL_DISPATCHER, ToolName.ANALYZE_DATA, mock_tool)
        mock_correction_chain.return_value = {
            "metrics": [{"column": "fare", "aggregation": "mean", "alias": "avg_fare"}]
        }

        result = run_executor_node(executor_state)

        assert mock_tool.call_count == 2
        mock_correction_chain.assert_called_once()
        assert result["execution_status"] == "complete"
        # Verify the corrected params were persisted to the plan
        assert result["plan"].steps[0].parameters == {
            "metrics": [{"column": "fare", "aggregation": "mean", "alias": "avg_fare"}]
        }

    def test_aborts_after_max_self_healing_retries(
        self, executor_state: AgentState, monkeypatch: pytest.MonkeyPatch, mock_correction_chain: MagicMock
    ) -> None:
        """Inject a tool that always fails.

        Asserts the Executor exhausts its 2 retries (3 total attempts) and aborts
        with an exact error message containing the final runtime error.
        """
        mock_tool = MagicMock(side_effect=ValueError("Permanent failure"))
        monkeypatch.setitem(_TOOL_DISPATCHER, ToolName.ANALYZE_DATA, mock_tool)

        result = run_executor_node(executor_state)

        assert mock_tool.call_count == 3
        assert result["execution_status"] == "aborted"
        assert "Runtime error in analyze_data: Permanent failure" in result["error_log"][-1]

    def test_aborts_on_service_exhausted_error_during_correction(
        self, executor_state: AgentState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inject a tool failure followed by a `ServiceExhaustedError` in the correction chain.

        Asserts the Executor catches the rate-limit failure in the correction chain
        and aborts immediately without burning remaining retries.
        """
        mock_tool = MagicMock(side_effect=ValueError("Tool broke"))
        monkeypatch.setitem(_TOOL_DISPATCHER, ToolName.ANALYZE_DATA, mock_tool)

        service_err = ServiceExhaustedError(
            service="Executor (Correction Chain)", attempts=3, last_error=Exception("429")
        )
        monkeypatch.setattr("scrygent.agents.executor_node._run_correction_chain", MagicMock(side_effect=service_err))

        result = run_executor_node(executor_state)

        assert mock_tool.call_count == 1  # Should not retry after rate limit
        assert result["execution_status"] == "aborted"
        assert "Executor (Correction Chain) is temporarily unavailable" in result["error_log"][-1]


class TestExecutorLazyFetch:
    """Tests validating the strict guards and routing of the `request_column_stats` tool."""

    def test_lazy_fetch_aborts_if_replan_guard_already_spent(self, executor_state: AgentState) -> None:
        """Inject a lazy-fetch step when `has_replanned` is already True.

        The Executor must hard-abort immediately to prevent infinite lazy-fetch loops.
        """
        assert executor_state.plan is not None
        executor_state.has_replanned = True
        executor_state.plan.steps[0].tool_name = ToolName.REQUEST_COLUMN_STATS
        executor_state.plan.steps[0].parameters = {"columns": ["age"]}

        result = run_executor_node(executor_state)

        assert result["execution_status"] == "aborted"
        assert "request_column_stats was invoked a second time" in result["error_log"][-1]

    def test_lazy_fetch_rejects_columns_not_in_missing_stats(
        self, executor_state: AgentState, monkeypatch: pytest.MonkeyPatch, mock_correction_chain: MagicMock
    ) -> None:
        """Inject a lazy-fetch step requesting a column ('fare') not in `missing_detailed_stats`.

        The Executor must reject the parameter to prevent the LLM from fetching
        already-known stats, triggering the self-healing loop.
        """
        assert executor_state.plan is not None
        executor_state.plan.steps[0].tool_name = ToolName.REQUEST_COLUMN_STATS
        executor_state.plan.steps[0].parameters = {"columns": ["fare"]}

        mock_tool = MagicMock(return_value={"detailed_stats": {"fare": {}}})
        monkeypatch.setitem(_TOOL_DISPATCHER, ToolName.REQUEST_COLUMN_STATS, mock_tool)
        mock_correction_chain.return_value = {"columns": ["age"]}

        run_executor_node(executor_state)

        mock_correction_chain.assert_called_once()
        assert "fare" in mock_correction_chain.call_args.kwargs["error_message"]

    def test_lazy_fetch_success_triggers_replan_and_updates_profile(
        self, executor_state: AgentState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inject a valid lazy-fetch step for 'age'.

        Asserts the Executor updates the `data_profile`, sets `has_replanned=True`,
        and sets `execution_status` to `"replan"` to route the graph back to the Planner.
        """
        assert executor_state.plan is not None
        executor_state.plan.steps[0].tool_name = ToolName.REQUEST_COLUMN_STATS
        executor_state.plan.steps[0].parameters = {"columns": ["age"]}

        mock_tool = MagicMock(return_value={"detailed_stats": {"age": {"mean": 30.0}}})
        monkeypatch.setitem(_TOOL_DISPATCHER, ToolName.REQUEST_COLUMN_STATS, mock_tool)

        result = run_executor_node(executor_state)

        assert result["execution_status"] == "replan"
        assert result["has_replanned"] is True
        assert "age" in result["data_profile"].detailed_stats
        assert "age" not in result["data_profile"].missing_detailed_stats
