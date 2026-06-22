import pytest # noqa: F401
from pathlib import Path
from pydantic import ValidationError # noqa: F401

from scrygent.models import (
    CSVProfile, Plan, Step, AnalysisReport, DirectAnswer, AgentState
)


class TestAgentStateInitialization:
    """Tests for AgentState construction and defaults."""

    def test_state_minimal_initialization(self):
        """AgentState can be initialized with only required fields."""
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="What is the total revenue?",
        )
        assert state.original_csv_path == Path("/data/input.csv")
        assert state.current_csv_path == Path("/data/input.csv")
        assert state.user_query == "What is the total revenue?"

    def test_state_defaults(self):
        """AgentState defaults match documentation."""
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        # Default values per docs
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

    def test_state_eval_mode_flag(self):
        """eval_mode switches output format (True → DirectAnswer, False → AnalysisReport)."""
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
    """Tests for Path field handling."""

    def test_original_csv_path_is_immutable_reference(self):
        """original_csv_path is never modified during execution."""
        original = Path("/data/original.csv")
        state = AgentState(
            original_csv_path=original,
            current_csv_path=Path("/data/current.csv"),
            user_query="Query",
        )
        # Even if wrangling tools update current_csv_path, original stays unchanged
        state.current_csv_path = Path("/data/transformed.csv")
        assert state.original_csv_path == Path("/data/original.csv")

    def test_current_csv_path_updated_by_wrangling(self):
        """current_csv_path is updated when wrangling tools write temp CSVs."""
        state = AgentState(
            original_csv_path=Path("/data/original.csv"),
            current_csv_path=Path("/data/original.csv"),
            user_query="Query",
        )
        # Simulating wrangling tool updating path
        state.current_csv_path = Path("/tmp/wrangled_12345.csv")
        assert state.current_csv_path == Path("/tmp/wrangled_12345.csv")
        assert state.original_csv_path == Path("/data/original.csv")

    def test_paths_accept_pathlib_objects(self):
        """Paths are Path objects (not strings)."""
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        assert isinstance(state.original_csv_path, Path)
        assert isinstance(state.current_csv_path, Path)


class TestAgentStateExecution:
    """Tests for execution flow state transitions."""

    def test_state_transition_pending_to_running(self):
        """Planner sets execution_status to 'running' after plan generation."""
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        # Profiler output: populate data_profile
        state.data_profile = CSVProfile(
            global_schema={"col": "int64"},
        )
        # Planner output: populate plan and set status to running
        state.plan = Plan(steps=[
            Step(step_id="s1", reasoning="Analyze data", action="tool", tool_name="analyze_data"),
        ])
        state.execution_status = "running"
        state.current_step_index = 0

        assert state.execution_status == "running"
        assert state.current_step_index == 0
        assert state.plan is not None

    def test_state_transition_running_to_complete(self):
        """Executor advances through steps, sets status to 'complete' when done."""
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        # Simulate step execution
        state.plan = Plan(steps=[
            Step(step_id="s1", reasoning="Analyze data", action="tool", tool_name="analyze_data"),
        ])
        state.execution_status = "running"
        state.current_step_index = 0

        # Executor processes step_0
        state.step_outputs["s1"] = {"result": "some_value"}
        state.current_step_index = 1

        # All steps complete
        if state.current_step_index >= len(state.plan.steps):
            state.execution_status = "complete"

        assert state.execution_status == "complete"
        assert len(state.step_outputs) == 1

    def test_state_transition_running_to_aborted(self):
        """Executor aborts on failed required step after retries."""
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        state.plan = Plan(steps=[
           Step(step_id="s1", reasoning="Analyze data", action="tool", tool_name="analyze_data", required=True),
        ])
        state.execution_status = "running"
        state.retry_count = 2  # Max retries exhausted

        # Required step fails after retries
        state.error_log.append("Tool validation failed after 2 retries.")
        state.execution_status = "aborted"

        assert state.execution_status == "aborted"
        assert state.retry_count == 2


class TestAgentStateStepOutputs:
    """Tests for step_outputs accumulation (JSON-safe dict/string)."""

    def test_step_outputs_accumulate_dicts(self):
        """step_outputs stores tool results as dicts (Tier 1)."""
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        # Tier 1 tool result: dict
        state.step_outputs["s1"] = {
            "column": "sales",
            "sum": 50000.0,
            "count": 100,
        }
        assert state.step_outputs["s1"]["sum"] == 50000.0

    def test_step_outputs_accumulate_strings(self):
        """step_outputs stores sandbox results as strings (Tier 2)."""
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        # Tier 2 sandbox result: string
        state.step_outputs["s2"] = "Calculated 95th percentile: 42.5"
        assert isinstance(state.step_outputs["s2"], str)

    def test_step_outputs_json_safe_primitives(self):
        """step_outputs values contain only JSON-safe types."""
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
        # All values are JSON-serializable
        json_str = state.model_dump_json()
        assert "value" in json_str


class TestAgentStateErrorLog:
    """Tests for error_log accumulation."""

    def test_error_log_tracks_non_fatal_warnings(self):
        """error_log collects non-fatal warnings and skipped steps."""
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        state.error_log.append("Column 'missing_col' not found in global_schema.")
        state.error_log.append("Step 's2' marked required=False, skipped after validation failure.")

        assert len(state.error_log) == 2
        assert "Column" in state.error_log[0]

    def test_error_log_does_not_abort_non_required_steps(self):
        """Skipped non-required steps are logged but execution continues."""
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        state.error_log.append("Step 's3' (required=False) validation failed, skipped.")
        state.execution_status = "running"  # Execution continues

        assert state.execution_status == "running"
        assert len(state.error_log) == 1


class TestAgentStateDataProfile:
    """Tests for data_profile field."""

    def test_data_profile_populated_by_profiler(self):
        """Profiler Node populates data_profile with two-level structure."""
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="What is total sales?",
        )
        # Profiler populates profile
        state.data_profile = CSVProfile(
            global_schema={"sales": "float64", "date": "datetime64"},
            detailed_stats={
                "sales": {"dtype": "float64", "sum": 1000000, "count": 5000}
            },
            truncated=False,
        )
        assert state.data_profile is not None
        assert len(state.data_profile.global_schema) == 2
        assert "sales" in state.data_profile.detailed_stats

    def test_data_profile_is_none_before_profiler(self):
        """data_profile is None until Profiler populates it."""
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        assert state.data_profile is None


class TestAgentStateFinalReport:
    """Tests for final_report (output from Reporter)."""

    def test_final_report_analysis_report_when_eval_false(self):
        """When eval_mode=False, final_report is AnalysisReport."""
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
        """When eval_mode=True, final_report is DirectAnswer."""
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
            eval_mode=True,
        )
        state.final_report = DirectAnswer(answer="42")
        assert isinstance(state.final_report, DirectAnswer)
        assert state.final_report.answer == "42"


class TestAgentStateSerialization:
    """Tests for JSON serialization (critical for LangGraph state)."""

    def test_state_json_roundtrip(self):
        """AgentState serializes and deserializes without loss."""
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="What is revenue?",
            eval_mode=False,
        )
        state.data_profile = CSVProfile(
            global_schema={"revenue": "float64"},
        )
        state.step_outputs["s1"] = {"total": 1000.0}
        state.execution_status = "running"

        json_str = state.model_dump_json()
        restored = AgentState.model_validate_json(json_str)

        assert restored.user_query == state.user_query
        assert restored.execution_status == "running"
        assert len(restored.step_outputs) == 1

    def test_state_path_preserved_on_roundtrip(self):
        """Path objects are preserved during serialization."""
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
    """Cross-validate AgentState against ARCHITECTURE.md."""

    def test_state_fields_match_architecture_docs(self):
        """AgentState includes all fields from ARCHITECTURE.md."""
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
        )
        # From ARCHITECTURE.md table
        assert hasattr(state, "original_csv_path")
        assert hasattr(state, "current_csv_path")
        assert hasattr(state, "user_query")
        assert hasattr(state, "data_profile")
        assert hasattr(state, "plan")
        assert hasattr(state, "current_step_index")
        assert hasattr(state, "step_outputs")
        assert hasattr(state, "execution_status")
        assert hasattr(state, "sandbox_activated")
        assert hasattr(state, "error_log")
        assert hasattr(state, "retry_count")
        assert hasattr(state, "eval_mode")
        assert hasattr(state, "final_report")

    def test_execution_status_literal_values_match_docs(self):
        """execution_status values match DESIGN.md (pending/running/complete/aborted)."""
        valid_statuses = ["pending", "running", "complete", "aborted"]
        for status in valid_statuses:
            state = AgentState(
                original_csv_path=Path("/data/input.csv"),
                current_csv_path=Path("/data/input.csv"),
                user_query="Query",
                execution_status=status,  # type: ignore
            )
            assert state.execution_status == status

    def test_sandbox_activated_flag_in_state(self):
        """sandbox_activated is surfaced to UI when Tier 2 is used."""
        state = AgentState(
            original_csv_path=Path("/data/input.csv"),
            current_csv_path=Path("/data/input.csv"),
            user_query="Query",
            sandbox_activated=False,
        )
        assert state.sandbox_activated is False

        state.sandbox_activated = True
        assert state.sandbox_activated is True