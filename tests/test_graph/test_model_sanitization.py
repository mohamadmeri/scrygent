import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from scrygent.models.base_model import ScrygentBaseModel, _recursive_sanitize
from scrygent.models.schemas import (
    Step, Plan, CSVProfile, PlotMetadata, AnalysisReport, DirectAnswer,
)
from scrygent.models.state import AgentState


# Helper minimal models for direct sanitizer testing
class DummyModel(ScrygentBaseModel):
    value: object
    items: list[object] = []
    mapping: dict[str, object] = {}


class TestRecursiveSanitizer:
    """Unit tests for _recursive_sanitize, the engine behind ScrygentBaseModel."""

    def test_int_numpy_to_int(self):
        assert _recursive_sanitize(np.int64(5)) == 5
        assert isinstance(_recursive_sanitize(np.int64(5)), int)

    def test_float_numpy_to_float(self):
        assert _recursive_sanitize(np.float64(3.14)) == 3.14
        assert isinstance(_recursive_sanitize(np.float64(3.14)), float)

    def test_nan_to_none(self):
        assert _recursive_sanitize(np.float64(np.nan)) is None

    def test_inf_to_none(self):
        assert _recursive_sanitize(np.float64(np.inf)) is None
        assert _recursive_sanitize(np.float64(-np.inf)) is None

    def test_bool_numpy_to_bool(self):
        assert _recursive_sanitize(np.bool_(True)) is True
        assert isinstance(_recursive_sanitize(np.bool_(False)), bool)

    def test_ndarray_to_list(self):
        arr = np.array([1, 2, 3])
        result = _recursive_sanitize(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_recursive_dict(self):
        data = {
            "a": np.int64(1),
            "b": {"c": np.float64(2.5), "d": np.float64(np.nan)},
            "e": [np.int64(3), np.float64(np.inf)]
        }
        clean = _recursive_sanitize(data)
        assert clean == {
            "a": 1,
            "b": {"c": 2.5, "d": None},
            "e": [3, None],
        }

    def test_pandas_timestamp_to_isoformat(self):
        ts = pd.Timestamp("2024-01-01 12:00:00")
        result = _recursive_sanitize(ts)
        assert result == "2024-01-01T12:00:00"

    def test_pandas_na_to_none(self):
        assert _recursive_sanitize(pd.NA) is None

    def test_plain_python_passthrough(self):
        assert _recursive_sanitize("hello") == "hello"
        assert _recursive_sanitize(42) == 42
        assert _recursive_sanitize(3.14) == 3.14
        assert _recursive_sanitize(True) is True
        assert _recursive_sanitize(None) is None

    def test_empty_structures(self):
        assert _recursive_sanitize([]) == []
        assert _recursive_sanitize({}) == {}

    def test_mixed_list(self):
        mixed = [np.int64(1), "two", np.float64(3.0), None, np.bool_(True)]
        assert _recursive_sanitize(mixed) == [1, "two", 3.0, None, True]


class TestScrygentBaseModel:
    """Verify that model_validate triggers sanitization."""

    def test_sanitization_on_construction(self):
        m = DummyModel(value=np.int64(42))
        assert m.value == 42
        assert type(m.value) is int

    def test_sanitization_on_validate(self):
        m = DummyModel.model_validate({"value": np.float64(3.14), "items": [np.float64(np.nan)]})
        assert m.value == 3.14
        assert m.items[0] is None

    def test_nested_dict_sanitization(self):
        m = DummyModel.model_validate({
            "value": 1,
            "mapping": {"score": np.float64(99.9), "flag": np.bool_(True)}
        })
        assert m.mapping["score"] == 99.9
        assert m.mapping["flag"] is True

    def test_json_dump_after_sanitization(self):
        m = DummyModel.model_validate({"value": np.int64(10)})
        json_str = m.model_dump_json()
        data = json.loads(json_str)
        assert data["value"] == 10  # not numpy integer


# Step model tests
class TestStepModel:
    def test_minimal_tool_step(self):
        step = Step(step_id="1", reasoning="Test", action="tool", tool_name="analyze_data")
        assert step.step_id == "1"
        assert step.tool_name == "analyze_data"
        assert step.required is True
        assert step.parameters == {}

    def test_tool_step_missing_tool_name(self):
        with pytest.raises(ValidationError):
            Step(step_id="1", reasoning="Test", action="tool")

    def test_sandbox_step_missing_instruction(self):
        with pytest.raises(ValidationError):
            Step(step_id="2", reasoning="Test", action="sandbox")

    def test_sandbox_step_valid(self):
        step = Step(step_id="2", reasoning="Test", action="sandbox", instruction="Do something")
        assert step.instruction == "Do something"
        assert step.tool_name is None

    def test_parameters_accept_nested_json(self):
        params = {
            "filter": {"col": "age", "op": ">", "val": 30},
            "agg": "mean"
        }
        step = Step(step_id="3", reasoning="Complex", action="tool", tool_name="analyze_data", parameters=params)
        assert step.parameters == params

    def test_required_defaults_to_true(self):
        step = Step(step_id="4", reasoning="T", action="tool", tool_name="t")
        assert step.required is True

    def test_reasoning_field_present(self):
        step = Step(step_id="5", reasoning="Because", action="tool", tool_name="t")
        assert step.reasoning == "Because"

    def test_json_roundtrip(self):
        step = Step(step_id="1", reasoning="Roundtrip", action="tool", tool_name="correlation", parameters={"col": "x"})
        json_str = step.model_dump_json()
        restored = Step.model_validate_json(json_str)
        assert restored == step

    def test_sanitization_in_parameters(self):
        # Parameters with numpy types should be cleaned
        step = Step.model_validate({
            "step_id": "1",
            "reasoning": "test",
            "action": "tool",
            "tool_name": "analyze_data",
            "parameters": {"value": np.float64(3.14), "flag": np.bool_(True)}
        })
        assert step.parameters["value"] == 3.14
        assert type(step.parameters["value"]) is float
        assert step.parameters["flag"] is True


# Plan model tests
class TestPlanModel:
    def test_empty_plan(self):
        plan = Plan(steps=[])
        assert plan.steps == []

    def test_multiple_steps_order_preserved(self):
        steps = [
            Step(step_id="1", reasoning="First", action="tool", tool_name="analyze_data"),
            Step(step_id="2", reasoning="Second", action="sandbox", instruction="do it"),
        ]
        plan = Plan(steps=steps)
        assert len(plan.steps) == 2
        assert plan.steps[0].step_id == "1"

    def test_json_roundtrip(self):
        plan = Plan(steps=[
            Step(step_id="1", reasoning="A", action="tool", tool_name="t"),
        ])
        json_str = plan.model_dump_json()
        restored = Plan.model_validate_json(json_str)
        assert restored.steps[0].step_id == "1"


# CSVProfile model tests
class TestCSVProfileModel:
    def test_minimal_profile(self):
        profile = CSVProfile(global_schema={"col": "int64"})
        assert profile.global_schema == {"col": "int64"}
        assert profile.detailed_stats == {}
        assert profile.row_sample == []
        assert profile.truncated is False

    def test_full_profile_with_stats(self):
        profile = CSVProfile(
            global_schema={"a": "int64", "b": "float64"},
            detailed_stats={
                "a": {"dtype": "int64", "null_rate": 0.0, "unique_count": 100, "min": 1.0, "max": 100.0, "mean": 50.0},
                "b": {"dtype": "float64", "null_rate": 0.1, "unique_count": 50, "min": 0.5, "max": 99.5, "mean": 40.2},
            },
            truncated=True,
            row_sample=[{"a": 1, "b": 0.5}, {"a": 2, "b": None}],
        )
        assert profile.truncated is True
        assert len(profile.detailed_stats) == 2
        assert profile.row_sample[1]["b"] is None

    def test_numpy_sanitization_in_detailed_stats(self):
        # Direct construction with numpy types should be sanitized via model_validate
        raw = {
            "global_schema": {"col": "float64"},
            "detailed_stats": {
                "col": {
                    "dtype": "float64",
                    "null_rate": np.float64(0.05),
                    "unique_count": np.int64(10),
                    "min": np.float64(0.0),
                    "max": np.float64(np.inf),  # will become None
                    "mean": np.float64(np.nan),  # will become None
                }
            }
        }
        profile = CSVProfile.model_validate(raw)
        stats = profile.detailed_stats["col"]
        assert stats["null_rate"] == 0.05
        assert isinstance(stats["null_rate"], float)
        assert stats["unique_count"] == 10
        assert isinstance(stats["unique_count"], int)
        assert stats["min"] == 0.0
        assert stats["max"] is None  # inf → None
        assert stats["mean"] is None  # nan → None

    def test_truncated_flag(self):
        profile = CSVProfile(global_schema={"a": "int64"}, truncated=True)
        assert profile.truncated is True

    def test_json_roundtrip(self):
        profile = CSVProfile(
            global_schema={"col": "float64"},
            detailed_stats={"col": {"dtype": "float64", "mean": 1.0}},
            row_sample=[{"col": 1.0}],
        )
        json_str = profile.model_dump_json()
        restored = CSVProfile.model_validate_json(json_str)
        assert restored.global_schema == profile.global_schema

    def test_row_sample_accepts_none(self):
        profile = CSVProfile(global_schema={"a": "int64"}, row_sample=[{"a": None}])
        assert profile.row_sample[0]["a"] is None


# PlotMetadata & AnalysisReport
class TestPlotMetadataModel:
    def test_plot_metadata(self):
        pm = PlotMetadata(file_path="/tmp/plot.png", description="A nice plot")
        assert pm.file_path == "/tmp/plot.png"
        assert not pm.file_path.startswith("data:image")

    def test_json_roundtrip(self):
        pm = PlotMetadata(file_path="/tmp/x.png", description="desc")
        json_str = pm.model_dump_json()
        restored = PlotMetadata.model_validate_json(json_str)
        assert restored.file_path == pm.file_path


class TestAnalysisReportModel:
    def test_primary_answer_required(self):
        with pytest.raises(ValidationError):
            AnalysisReport.model_validate({})

    def test_valid_report(self):
        report = AnalysisReport(primary_answer="Answer.")
        assert report.primary_answer == "Answer."
        assert report.additional_insights is None
        assert report.plots == []

    def test_full_report(self):
        report = AnalysisReport(
            primary_answer="42",
            additional_insights=["insight1", "insight2"],
            plots=[PlotMetadata(file_path="/tmp/a.png", description="graph")],
        )
        assert report.additional_insights
        assert len(report.additional_insights) == 2  
        assert len(report.plots) == 1

    def test_json_roundtrip(self):
        report = AnalysisReport(primary_answer="Answer", plots=[PlotMetadata(file_path="/tmp/x.png", description="d")])
        json_str = report.model_dump_json()
        restored = AnalysisReport.model_validate_json(json_str)
        assert restored.primary_answer == report.primary_answer


# DirectAnswer model
class TestDirectAnswerModel:
    def test_answer_string(self):
        da = DirectAnswer(answer="42")
        assert da.answer == "42"

    def test_only_answer_field(self):
        assert set(DirectAnswer.model_fields.keys()) == {"answer"}

    def test_json_roundtrip(self):
        da = DirectAnswer(answer="hello")
        json_str = da.model_dump_json()
        restored = DirectAnswer.model_validate_json(json_str)
        assert restored.answer == "hello"


# AgentState model tests
class TestAgentStateModel:
    def test_minimal_state(self):
        state = AgentState(
            original_csv_path=Path("/data/test.csv"),
            current_csv_path=Path("/data/test.csv"),
            user_query="query",
        )
        assert state.execution_status == "pending"
        assert state.eval_mode is False

    def test_state_accepts_plan_and_profile(self):
        profile = CSVProfile(global_schema={"col": "int64"})
        plan = Plan(steps=[Step(step_id="1", reasoning="T", action="tool", tool_name="t")])
        state = AgentState(
            original_csv_path=Path("/data/test.csv"),
            current_csv_path=Path("/data/test.csv"),
            user_query="q",
            data_profile=profile,
            plan=plan,
            execution_status="running",
        )
        assert state.data_profile == profile
        assert state.plan is not None
        assert state.plan.steps[0].step_id == "1"

    def test_state_final_report_union(self):
        # AnalysisReport
        state = AgentState(
            original_csv_path=Path("/data/test.csv"),
            current_csv_path=Path("/data/test.csv"),
            user_query="q",
            final_report=AnalysisReport(primary_answer="ans"),
        )
        assert isinstance(state.final_report, AnalysisReport)
        # DirectAnswer
        state2 = AgentState(
            original_csv_path=Path("/data/test.csv"),
            current_csv_path=Path("/data/test.csv"),
            user_query="q",
            final_report=DirectAnswer(answer="42"),
        )
        assert isinstance(state2.final_report, DirectAnswer)

    def test_path_handling(self):
        state = AgentState(
            original_csv_path=Path("/data/orig.csv"),
            current_csv_path=Path("/data/curr.csv"),
            user_query="q",
        )
        assert state.original_csv_path == Path("/data/orig.csv")
        assert state.current_csv_path == Path("/data/curr.csv")

    def test_json_roundtrip_full(self):
        state = AgentState(
            original_csv_path=Path("/data/orig.csv"),
            current_csv_path=Path("/data/curr.csv"),
            user_query="What is the mean?",
            eval_mode=False,
            data_profile=CSVProfile(global_schema={"a": "int64"}),
            plan=Plan(steps=[Step(step_id="1", reasoning="R", action="tool", tool_name="analyze_data")]),
            step_outputs={"1": {"result": 42}},
            execution_status="complete",
            final_report=AnalysisReport(primary_answer="42"),
        )
        json_str = state.model_dump_json()
        restored = AgentState.model_validate_json(json_str)
        assert restored.user_query == "What is the mean?"
        assert restored.execution_status == "complete"
        assert restored.step_outputs["1"] == {"result": 42}
        if isinstance(restored.final_report, AnalysisReport):
            assert restored.final_report.primary_answer == "42"

    def test_sanitization_in_step_outputs(self):
        # step_outputs may contain numpy scalars from tools; should be sanitized
        state = AgentState.model_validate({
            "original_csv_path": "/data/test.csv",
            "current_csv_path": "/data/test.csv",
            "user_query": "q",
            "step_outputs": {
                "s1": {"mean": np.float64(5.5), "count": np.int64(10)}
            }
        })
        out = state.step_outputs["s1"]
        if isinstance(out, dict):
            assert out["mean"] == 5.5
            assert isinstance(out["mean"], float)
            assert out["count"] == 10
            assert isinstance(out["count"], int)

    def test_error_log_is_list(self):
        state = AgentState(
            original_csv_path=Path("/data/test.csv"),
            current_csv_path=Path("/data/test.csv"),
            user_query="q",
            error_log=["warning1", "warning2"],
        )
        assert state.error_log == ["warning1", "warning2"]


# Stress tests: large payloads, nesting, edge cases
class TestModelStress:
    def test_many_columns_in_profile(self):
        # Simulate wide CSV (200 columns)
        global_schema = {f"col_{i}": "float64" for i in range(200)}
        detailed_stats = {f"col_{i}": {"dtype": "float64", "mean": float(i)} for i in range(15)}
        profile = CSVProfile(global_schema=global_schema, detailed_stats=detailed_stats, truncated=True)
        assert len(profile.global_schema) == 200
        assert len(profile.detailed_stats) == 15

    def test_deeply_nested_parameters(self):
        params = {
            "filter": {
                "conditions": [
                    {"col": "a", "op": ">", "val": 1},
                    {"col": "b", "op": "<", "val": np.float64(10.5)}
                ]
            }
        }
        step = Step.model_validate({
            "step_id": "1",
            "reasoning": "deep",
            "action": "tool",
            "tool_name": "analyze_data",
            "parameters": params
        })
        # The inner numpy float should be sanitized
        assert step.parameters["filter"]["conditions"][1]["val"] == 10.5
        assert type(step.parameters["filter"]["conditions"][1]["val"]) is float

    def test_row_sample_with_all_nulls(self):
        sample: list[dict[str, str | int | float | bool | None]] = [{"a": None, "b": None}]
        assert sample is not None
        profile = CSVProfile(global_schema={"a": "int64", "b": "float64"}, row_sample=sample)
        assert profile.row_sample[0]["a"] is None

    def test_empty_plan(self):
        plan = Plan(steps=[])
        assert plan.steps == []

    def test_agent_state_large_error_log(self):
        log = [f"error {i}" for i in range(100)]
        state = AgentState(
            original_csv_path=Path("/data/test.csv"),
            current_csv_path=Path("/data/test.csv"),
            user_query="q",
            error_log=log,
        )
        assert len(state.error_log) == 100

    def test_serialization_with_all_models_nested(self):
        # Build a fully populated AgentState and verify JSON roundtrip
        state = AgentState(
            original_csv_path=Path("/data/orig.csv"),
            current_csv_path=Path("/data/curr.csv"),
            user_query="q",
            data_profile=CSVProfile(
                global_schema={"col": "int64"},
                detailed_stats={"col": {"dtype": "int64", "mean": np.float64(3.14)}},
            ),
            plan=Plan(steps=[Step(step_id="1", reasoning="R", action="tool", tool_name="t")]),
            step_outputs={"1": {"value": 42}},
            final_report=AnalysisReport(primary_answer="42"),
        )
        json_str = state.model_dump_json()
        assert '"mean":3.14' in json_str  # sanitized float
        restored = AgentState.model_validate_json(json_str)
        assert restored.data_profile is not None
        assert restored.data_profile.detailed_stats["col"]["mean"] == 3.14
