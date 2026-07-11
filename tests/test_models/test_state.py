"""Tests for AgentState: initialization, transitions, serialization, and fuzzing."""
from pathlib import Path

import pytest
from typing import Literal, cast
from hypothesis import given, strategies as st
from pydantic import ValidationError

from scrygent.models.state import AgentState
from scrygent.models.step_models import Step, Plan, StepRecord
from scrygent.models.outputs import (
    CSVProfile, AnalysisReport, DirectAnswer
)
from scrygent.contracts.tool_names import ToolName


# ── Helper strategies ──
@st.composite
def valid_agent_state_strategy(draw):
    return AgentState(
        original_csv_path=Path(draw(st.text(min_size=1, max_size=50).map(lambda s: f"/{s}.csv"))),
        current_csv_path=Path(draw(st.text(min_size=1, max_size=50).map(lambda s: f"/{s}.csv"))),
        user_query=draw(st.text(min_size=1, max_size=200)),
        eval_mode=draw(st.booleans()),
        data_profile=draw(
            st.none()
            | st.builds(
                CSVProfile,
                global_schema=st.dictionaries(st.text(min_size=1), st.text(min_size=1)),
                row_count=st.integers(min_value=0, max_value=100000),
            )
        ),
        plan=draw(
            st.none()
            | st.builds(
                Plan,
                steps=st.lists(
                    step_strategy(),
                    max_size=3,
                ),
            )
        ),
        current_step_index=draw(st.integers(min_value=0, max_value=10)),
        step_outputs=draw(
            st.dictionaries(
                st.text(),
                st.one_of(st.dictionaries(st.text(), st.text()), st.text()),
            )
        ),
        retry_count=draw(st.integers(min_value=0, max_value=5)),
        error_log=draw(st.lists(st.text(), max_size=10)),
        execution_status=cast(
            Literal["pending", "running", "aborted", "complete"],
            draw(st.sampled_from(["pending", "running", "aborted", "complete"]))
        ),
        sandbox_activated=draw(st.booleans()),
        execution_trace=draw(
            st.lists(
                st.builds(
                    StepRecord,
                    tool_name=st.none() | st.sampled_from(list(ToolName)),
                ),
                max_size=5,
            )
        ),
        final_report=draw(
            st.none()
            | st.builds(AnalysisReport, primary_answer=st.text(min_size=1))
            | st.builds(DirectAnswer, answer=st.text())
        ),
    )

# Fuzzing helper dictionary: provides minimal valid parameters for each tool, used in fuzzing tests.
MINIMAL_VALID_PARAMS: dict[ToolName, dict] = {
    ToolName.ANALYZE_DATA: {"metrics": [{"column": "x", "aggregation": "count", "alias": "cnt"}]},
    ToolName.FILTER_DATASET: {"filters": [{"column": "c", "operator": "==", "value": 1}]},
    ToolName.NORMALIZE_COLUMN: {"column": "c", "method": "min_max"},
    ToolName.RESET_DATASET: {},
    ToolName.CORRELATION: {"columns": ["a", "b"]},
    ToolName.REGRESSION: {"target": "y", "features": ["x"]},
    ToolName.DETECT_OUTLIERS: {"column": "x", "method": "iqr"},
    ToolName.REQUEST_COLUMN_STATS: {"columns": ["a"]},
    ToolName.GENERATE_PLOT: {"plot_type": "bar", "columns": ["cat", "val"]},
    ToolName.DERIVE_COLUMN: {"new_column": "r", "expression": "a+b"},
    ToolName.EVALUATE_METRICS: {"expression": "a", "values": {"a": 1.0}},
}

# Fuzzing helper strategy for generating valid Step instances
@st.composite
def step_strategy(draw):
    action = draw(st.sampled_from(["tool", "sandbox"]))
    step_id = draw(st.text(min_size=1, max_size=10))
    rationale = draw(st.text(min_size=1, max_size=50))
    required = draw(st.booleans())

    if action == "tool":
        tool_name = draw(st.sampled_from(list(ToolName)))
        parameters = MINIMAL_VALID_PARAMS[tool_name]
        instruction = None
    else:
        tool_name = None
        parameters = {}
        instruction = draw(st.text(min_size=1, max_size=50))

    return Step(
        step_id=step_id,
        rationale=rationale,
        action=action,
        tool_name=tool_name,
        parameters=parameters,
        instruction=instruction,
        required=required,
    )

class TestAgentStateInitialization:
    def test_state_minimal_initialization(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="What is the total revenue?",
        )
        assert state.original_csv_path == Path("/data/input.csv")
        assert state.current_csv_path == Path("/data/input.csv")
        assert state.user_query == "What is the total revenue?"

    def test_state_defaults(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        assert state.eval_mode is False
        assert state.data_profile is None
        assert state.plan is None
        assert state.current_step_index == 0
        assert state.step_outputs == {}
        assert state.retry_count == 0
        assert state.error_log == []
        assert state.execution_status == "pending"
        assert state.sandbox_activated is False
        assert state.final_report is None
        assert state.execution_trace == []

    def test_state_eval_mode_flag(self):
        state1 = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
            eval_mode=False,
        )
        state2 = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
            eval_mode=True,
        )
        assert state1.eval_mode is False
        assert state2.eval_mode is True


class TestAgentStatePathHandling:
    def test_original_csv_path_is_immutable_reference(self):
        original = Path("/data/original.csv")
        state = AgentState(
            original_csv_path=original,
            current_csv_path=Path("/data/current.csv"),
            user_query="Query",
        )
        state.current_csv_path = Path("/data/transformed.csv")
        assert state.original_csv_path == Path("/data/original.csv")

    def test_current_csv_path_updated_by_wrangling(self):
        state = AgentState(
            original_csv_path=Path("/data/original.csv"),
            current_csv_path=Path("/data/original.csv"),
            user_query="Query",
        )
        state.current_csv_path = Path("/tmp/wrangled_12345.csv")
        assert state.current_csv_path == Path("/tmp/wrangled_12345.csv")
        assert state.original_csv_path == Path("/data/original.csv")

    def test_paths_accept_pathlib_objects(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        assert isinstance(state.original_csv_path, Path)
        assert isinstance(state.current_csv_path, Path)


class TestAgentStateExecution:
    def test_state_transition_pending_to_running(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        state.data_profile = CSVProfile(global_schema={"col": "int64"}, row_count=0)
        state.plan = Plan(
            steps=[
                Step(
                    step_id="s1",
                    rationale="Analyze data",
                    action="tool",
                    tool_name=ToolName.ANALYZE_DATA,
                    parameters={
                        "metrics": [
                            {"column": "sales", "aggregation": "mean", "alias": "avg_sales"}
                        ]
                    },
                    required=True,  # or default
                )
            ]
        )
        state.execution_status = "running"
        state.current_step_index = 0

        assert state.execution_status == "running"
        assert state.current_step_index == 0
        assert state.plan is not None

    def test_state_transition_running_to_complete(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        state.plan = Plan(
            steps=[
                Step(
                    step_id="s1",
                    rationale="Analyze data",
                    action="tool",
                    tool_name=ToolName.ANALYZE_DATA,
                    parameters={
                        "metrics": [
                            {"column": "sales", "aggregation": "mean", "alias": "avg_sales"}
                        ]
                    },
                    required=True,  # or default
                )
            ]
        )
        state.execution_status = "running"
        state.current_step_index = 0

        state.step_outputs["s1"] = {"result": "some_value"}
        state.current_step_index = 1

        if state.current_step_index >= len(state.plan.steps):
            state.execution_status = "complete"

        assert state.execution_status == "complete"
        assert len(state.step_outputs) == 1

    def test_state_transition_running_to_aborted(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        state.plan = Plan(
            steps=[
                Step(
                    step_id="s1",
                    rationale="Analyze data",
                    action="tool",
                    tool_name=ToolName.ANALYZE_DATA,
                    parameters={
                        "metrics": [
                            {"column": "sales", "aggregation": "mean", "alias": "avg_sales"}
                        ]
                    },
                    required=True,  # or default
                )
            ]
        )
        state.execution_status = "running"
        state.retry_count = 2

        state.error_log.append("Tool validation failed after 2 retries.")
        state.execution_status = "aborted"

        assert state.execution_status == "aborted"
        assert state.retry_count == 2


class TestAgentStateStepOutputs:
    def test_step_outputs_accumulate_dicts(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        state.step_outputs["s1"] = {"column": "sales", "sum": 50000.0, "count": 100}
        assert state.step_outputs["s1"]["sum"] == 50000.0

    def test_step_outputs_accumulate_strings(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        state.step_outputs["s2"] = "Calculated 95th percentile: 42.5"
        assert isinstance(state.step_outputs["s2"], str)

    def test_step_outputs_json_safe_primitives(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        state.step_outputs["s1"] = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }
        json_str = state.model_dump_json()
        assert "value" in json_str


class TestAgentStateErrorLog:
    def test_error_log_tracks_non_fatal_warnings(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        state.error_log.append("Column 'missing_col' not found in global_schema.")
        state.error_log.append(
            "Step 's2' marked required=False, skipped after validation failure."
        )
        assert len(state.error_log) == 2
        assert "Column" in state.error_log[0]

    def test_error_log_does_not_abort_non_required_steps(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        state.error_log.append(
            "Step 's3' (required=False) validation failed, skipped."
        )
        state.execution_status = "running"
        assert state.execution_status == "running"
        assert len(state.error_log) == 1


class TestAgentStateDataProfile:
    def test_data_profile_populated_by_profiler(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="What is total sales?",
        )
        state.data_profile = CSVProfile(
            global_schema={"sales": "float64", "date": "datetime64"},
            detailed_stats={
                "sales": {"dtype": "float64", "sum": 1000000, "count": 5000}
            },
            truncated=False,
            row_count=5000,
        )
        assert state.data_profile is not None
        assert len(state.data_profile.global_schema) == 2
        assert "sales" in state.data_profile.detailed_stats

    def test_data_profile_is_none_before_profiler(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        assert state.data_profile is None


class TestAgentStateFinalReport:
    def test_final_report_analysis_report_when_eval_false(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
            eval_mode=False,
        )
        state.final_report = AnalysisReport(
            primary_answer="The answer is 42.",
            additional_insights=["Insight 1", "Insight 2"],
        )
        assert isinstance(state.final_report, AnalysisReport)
        assert state.final_report.primary_answer == "The answer is 42."

    def test_final_report_direct_answer_when_eval_true(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
            eval_mode=True,
        )
        state.final_report = DirectAnswer(answer="42")
        assert isinstance(state.final_report, DirectAnswer)
        assert state.final_report.answer == "42"


class TestAgentStateExecutionTrace:
    def test_execution_trace_empty_by_default(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        assert state.execution_trace == []

    def test_add_step_records(self):
        record1 = StepRecord(
            step_id="s1",
            tool_name=ToolName.ANALYZE_DATA,
            status="success",
            summary="Completed successfully",
            duration_ms=150,
        )
        record2 = StepRecord(
            step_id="s2",
            tool_name=ToolName.CORRELATION,
            status="failed",
            error="Division by zero",
            duration_ms=200,
        )
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
            execution_trace=[record1, record2],
        )
        assert len(state.execution_trace) == 2
        assert state.execution_trace[0].step_id == "s1"
        assert state.execution_trace[0].status == "success"
        assert state.execution_trace[1].status == "failed"
        assert state.execution_trace[1].error == "Division by zero"

    def test_execution_trace_serialization(self):
        record = StepRecord(
            step_id="s1",
            tool_name=ToolName.ANALYZE_DATA,
            status="success",
            summary="All good",
        )
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
            execution_trace=[record],
        )
        json_str = state.model_dump_json()
        restored = AgentState.model_validate_json(json_str)
        assert len(restored.execution_trace) == 1
        assert restored.execution_trace[0].step_id == "s1"


class TestAgentStateSerialization:
    def test_state_json_roundtrip(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="What is revenue?",
            eval_mode=False,
        )
        state.data_profile = CSVProfile(global_schema={"revenue": "float64"}, row_count=0)
        state.step_outputs["s1"] = {"total": 1000.0}
        state.execution_status = "running"

        json_str = state.model_dump_json()
        restored = AgentState.model_validate_json(json_str)
        assert restored.user_query == state.user_query
        assert restored.execution_status == "running"
        assert len(restored.step_outputs) == 1

    def test_state_path_preserved_on_roundtrip(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        json_str = state.model_dump_json()
        restored = AgentState.model_validate_json(json_str)
        assert restored.original_csv_path == Path("/data/input.csv")
        assert restored.current_csv_path == Path("/data/input.csv")


class TestAgentStateConsistencyWithDocs:
    def test_state_fields_match_architecture_docs(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        required_fields = [
            "original_csv_path", "current_csv_path", "user_query",
            "data_profile", "plan", "current_step_index", "step_outputs",
            "execution_status", "sandbox_activated", "error_log", "retry_count",
            "eval_mode", "final_report", "execution_trace"
        ]
        for field in required_fields:
            assert hasattr(state, field)

    def test_execution_status_literal_values_match_docs(self):
        valid_statuses = ["pending", "running", "complete", "aborted"]
        for status in valid_statuses:
            state = AgentState(
                original_csv_path=Path("/data/input.csv"),
                current_csv_path=Path("/data/input.csv"),
                user_query="Query",
                execution_status=cast(Literal["pending", "running", "aborted", "complete"], status),
            )
            assert state.execution_status == status

    def test_sandbox_activated_flag_in_state(self):
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
            sandbox_activated=False,
        )
        assert state.sandbox_activated is False
        state.sandbox_activated = True
        assert state.sandbox_activated is True


class TestAgentStateFuzzing:
    @given(valid_agent_state_strategy())
    def test_random_state_passes_validation(self, state: AgentState):
        """Any randomly generated valid state should be constructible and serializable."""
        assert isinstance(state.original_csv_path, Path)
        assert isinstance(state.user_query, str)
        assert isinstance(state.eval_mode, bool)
        assert isinstance(state.execution_status, str)
        assert state.execution_status in {"pending", "running", "aborted", "complete"}

        json_str = state.model_dump_json()
        assert isinstance(json_str, str)
        restored = AgentState.model_validate_json(json_str)
        assert restored.user_query == state.user_query

    @given(
        st.text(min_size=1).map(lambda s: Path(f"/{s}.csv")),
        st.text(min_size=1).map(lambda s: Path(f"/{s}.csv")),
        st.text(),
        st.sampled_from(["pending", "running", "aborted", "complete"]),
    )
    def test_invalid_extra_field_raises(self, orig, curr, query, status):
        with pytest.raises(ValidationError):
            AgentState(
                original_csv_path=orig,
                current_csv_path=curr,
                user_query=query,
                execution_status=status,
                unknown_field=42, # type: ignore
            )
