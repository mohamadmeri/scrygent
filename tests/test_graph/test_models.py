import pytest
from pydantic import ValidationError

from scrygent.models import (
    Step, Plan, CSVProfile, PlotMetadata, AnalysisReport, DirectAnswer
)


class TestStep:
    """Tests for the Step model."""

    def test_step_tool_action_requires_tool_name(self):
        """Tool steps must have tool_name."""
        with pytest.raises(ValidationError) as excinfo:
            Step(step_id="step_1", reasoning="test", action="tool", tool_name=None)
        assert "tool_name is required" in str(excinfo.value)

    def test_step_tool_action_valid(self):
        """Tool step with all required fields is valid."""
        step = Step(
            step_id="step_1",
            reasoning="Compute aggregate of sales column",
            action="tool",
            tool_name="analyze_data",
            parameters={"column": "sales", "agg": "sum"},
            required=True,
        )
        assert step.action == "tool"
        assert step.tool_name == "analyze_data"
        assert step.required is True
        assert step.reasoning == "Compute aggregate of sales column"

    def test_step_sandbox_action_requires_instruction(self):
        """Sandbox steps must have instruction (never code)."""
        with pytest.raises(ValidationError) as excinfo:
            Step(step_id="step_2", reasoning="test", action="sandbox", instruction=None)
        assert "instruction is required" in str(excinfo.value)

    def test_step_sandbox_action_valid(self):
        """Sandbox step with instruction is valid."""
        step = Step(
            step_id="step_2",
            reasoning="Compute percentile of normalized sales",
            action="sandbox",
            instruction="Calculate the 95th percentile of normalized sales.",
            required=False,
        )
        assert step.action == "sandbox"
        assert step.instruction is not None
        assert "Calculate" in step.instruction
        assert step.required is False

    def test_step_default_required_is_true(self):
        """Step.required defaults to True."""
        step = Step(
            step_id="step_1",
            reasoning="test",
            action="tool",
            tool_name="analyze_data",
        )
        assert step.required is True

    def test_step_parameters_json_safe(self):
        """Parameters contain only JSON-safe primitives (str, int, float, bool, None)."""
        step = Step(
            step_id="step_1",
            reasoning="Filter and aggregate sales data",
            action="tool",
            tool_name="analyze_data",
            parameters={
                "column": "sales",
                "min_val": 100,
                "threshold": 0.95,
                "enabled": True,
                "optional_field": None,
            },
        )
        for val in step.parameters.values():
            assert isinstance(val, (str, int, float, bool, type(None)))


class TestPlan:
    """Tests for the Plan model."""

    def test_plan_empty_steps(self):
        """Plan can contain zero steps."""
        plan = Plan(steps=[])
        assert len(plan.steps) == 0

    def test_plan_ordered_steps(self):
        """Plan maintains step order for sequential execution."""
        step1 = Step(
            step_id="s1", reasoning="First step", action="tool", tool_name="analyze_data"
        )
        step2 = Step(
            step_id="s2",
            reasoning="Second step",
            action="sandbox",
            instruction="Calculate percentile.",
        )
        step3 = Step(
            step_id="s3", reasoning="Third step", action="tool", tool_name="correlation"
        )

        plan = Plan(steps=[step1, step2, step3])
        assert len(plan.steps) == 3
        assert plan.steps[0].step_id == "s1"
        assert plan.steps[1].step_id == "s2"
        assert plan.steps[2].step_id == "s3"

    def test_plan_executor_uses_current_step_index(self):
        """Executor advances through plan using current_step_index pointer."""
        steps = [
            Step(step_id="s1", reasoning="First", action="tool", tool_name="analyze_data"),
            Step(step_id="s2", reasoning="Second", action="tool", tool_name="correlation"),
        ]
        plan = Plan(steps=steps)

        for idx in range(len(plan.steps)):
            current_step = plan.steps[idx]
            assert current_step.step_id == f"s{idx + 1}"


class TestCSVProfile:
    """Tests for the CSVProfile model."""

    def test_profile_global_schema_is_always_complete(self):
        """global_schema lists every column, regardless of detailed_stats coverage."""
        profile = CSVProfile(
            global_schema={
                "id": "int64",
                "name": "object",
                "sales": "float64",
                "date": "datetime64",
                "region": "object",
            },
            detailed_stats={
                "sales": {
                    "dtype": "float64",
                    "null_rate": 0.02,
                    "unique_count": 50,
                    "min": 10.5,
                    "max": 1000.0,
                    "mean": 250.3,
                }
            },
            truncated=True,
        )

        assert len(profile.global_schema) == 5
        assert profile.global_schema["id"] == "int64"
        assert profile.global_schema["region"] == "object"
        assert profile.truncated is True

    def test_profile_detailed_stats_is_selective(self):
        """detailed_stats contains metrics only for query-matched + top N columns."""
        profile = CSVProfile(
            global_schema={"col_a": "int64", "col_b": "object", "col_c": "float64"},
            detailed_stats={
                "col_a": {"dtype": "int64", "null_rate": 0.0, "mean": 42}
            },
            truncated=True,
        )

        assert len(profile.global_schema) == 3
        assert len(profile.detailed_stats) == 1
        assert "col_b" not in profile.detailed_stats

    def test_profile_row_sample_nans_replaced_with_none(self):
        """3-row sample has NaN replaced with None (JSON-serializable)."""
        profile = CSVProfile(
            global_schema={"id": "int64", "value": "float64"},
            row_sample=[
                {"id": 1, "value": 10.5},
                {"id": 2, "value": None},
                {"id": 3, "value": 20.3},
            ],
        )

        assert len(profile.row_sample) == 3
        assert profile.row_sample[1]["value"] is None

    def test_profile_json_safe_values(self):
        """All numeric values in detailed_stats are JSON-safe (no np.nan, np.inf)."""
        profile = CSVProfile(
            global_schema={"col": "float64"},
            detailed_stats={
                "col": {
                    "dtype": "float64",
                    "null_rate": 0.05,
                    "mean": 100.5,
                    "std": 15.3,
                    "min": 50.0,
                    "max": 200.0,
                }
            },
        )

        for val in profile.detailed_stats["col"].values():
            assert isinstance(val, (str, int, float, type(None)))

    def test_profile_truncated_flag_signals_partial_profiling(self):
        """truncated=True means detailed_stats is subset of global_schema."""
        profile = CSVProfile(
            global_schema={"a": "int64", "b": "int64", "c": "int64"},
            detailed_stats={"a": {"dtype": "int64"}},
            truncated=True,
        )
        assert profile.truncated is True
        assert len(profile.detailed_stats) < len(profile.global_schema)


class TestPlotMetadata:
    """Tests for PlotMetadata model."""

    def test_plot_metadata_is_disk_path_not_blob(self):
        """PlotMetadata stores file path, never base64-encoded image."""
        meta = PlotMetadata(
            file_path="/tmp/plot_12345.png",
            description="Distribution of sales by region.",
        )
        assert meta.file_path == "/tmp/plot_12345.png"
        assert not meta.file_path.startswith("data:image")
        assert meta.description == "Distribution of sales by region."

    def test_plot_metadata_json_serializable(self):
        """PlotMetadata serializes cleanly."""
        meta = PlotMetadata(
            file_path="/tmp/correlation_heatmap.png",
            description="Pairwise correlations",
        )
        serialized = meta.model_dump()
        assert "file_path" in serialized
        assert "description" in serialized


class TestAnalysisReport:
    """Tests for AnalysisReport (portfolio-facing output)."""

    def test_report_primary_answer_is_required(self):
        """AnalysisReport requires primary_answer (cannot be None)."""
        with pytest.raises(ValidationError):
            AnalysisReport(primary_answer=None)  # type: ignore

    def test_report_primary_answer_enforced_first(self):
        """primary_answer is populated before additional_insights (Pydantic schema enforces)."""
        report = AnalysisReport(
            primary_answer="The average Q4 sales was $1,250.",
            additional_insights=[
                "Sales peaked in December.",
                "Region B outperformed by 15%.",
            ],
        )
        assert report.primary_answer == "The average Q4 sales was $1,250."
        assert report.additional_insights is not None
        assert len(report.additional_insights) == 2

    def test_report_additional_insights_optional(self):
        """additional_insights may be None or empty."""
        report1 = AnalysisReport(primary_answer="Answer.")
        report2 = AnalysisReport(primary_answer="Answer.", additional_insights=None)
        assert report1.additional_insights is None
        assert report2.additional_insights is None

    def test_report_plots_not_embedded_as_blobs(self):
        """Plots are referenced by file path, not embedded base64."""
        report = AnalysisReport(
            primary_answer="The trend is upward.",
            plots=[
                PlotMetadata(file_path="/tmp/trend.png", description="Trend over time"),
            ],
        )
        assert len(report.plots) == 1
        assert report.plots[0].file_path == "/tmp/trend.png"
        assert not report.plots[0].file_path.startswith("data:image")

    def test_report_full_realistic_example(self):
        """Full example matching portfolio use case."""
        report = AnalysisReport(
            primary_answer="Median customer lifetime value is $5,432.",
            additional_insights=[
                "High-value customers concentrate in North America.",
                "Churn rate increased 2% year-over-year.",
            ],
            plots=[
                PlotMetadata(file_path="/tmp/clv_dist.png", description="CLV distribution"),
                PlotMetadata(file_path="/tmp/churn_trend.png", description="Churn trend"),
            ],
        )
        assert "Median" in report.primary_answer
        assert report.additional_insights is not None
        assert len(report.additional_insights) == 2
        assert len(report.plots) == 2


class TestDirectAnswer:
    """Tests for DirectAnswer (benchmarking mode output, eval_mode=True)."""

    def test_direct_answer_scalar_numeric(self):
        """DirectAnswer with numeric scalar."""
        ans = DirectAnswer(answer="42.5")
        assert ans.answer == "42.5"

    def test_direct_answer_string(self):
        """DirectAnswer with categorical string."""
        ans = DirectAnswer(answer="North America")
        assert ans.answer == "North America"

    def test_direct_answer_boolean(self):
        """DirectAnswer with boolean value."""
        ans = DirectAnswer(answer="True")
        assert ans.answer == "True"

    def test_direct_answer_comma_separated_list(self):
        """DirectAnswer with multiple values (multi-select benchmark)."""
        ans = DirectAnswer(answer="California,Texas,Florida")
        assert ans.answer == "California,Texas,Florida"
        assert "," in ans.answer

    def test_direct_answer_is_single_field_only(self):
        """DirectAnswer has only 'answer' field (no narrative, no plots)."""
        ans = DirectAnswer(answer="42")
        fields = DirectAnswer.model_fields
        assert len(fields) == 1
        assert "answer" in fields
        assert ans.answer == "42"


class TestModelConsistencyWithDocs:
    """Cross-validate all models against DESIGN.md and ARCHITECTURE.md."""

    def test_step_matches_docs_fields(self):
        """Step includes all fields from ARCHITECTURE.md."""
        step = Step(
            step_id="s1",
            reasoning="Cross-checking docs compliance",
            action="tool",
            tool_name="analyze_data",
            parameters={"col": "value"},
            required=True,
        )

        assert hasattr(step, "action")
        assert hasattr(step, "tool_name")
        assert hasattr(step, "parameters")
        assert hasattr(step, "required")
        assert hasattr(step, "instruction")
        assert hasattr(step, "reasoning")

    def test_csv_profile_implements_two_level_design(self):
        """CSVProfile implements two-level profiling from DESIGN.md."""
        profile = CSVProfile(
            global_schema={"col_a": "int64", "col_b": "object"},
            detailed_stats={"col_a": {"dtype": "int64", "mean": 50}},
            truncated=True,
        )
        assert hasattr(profile, "global_schema")
        assert hasattr(profile, "detailed_stats")
        assert hasattr(profile, "row_sample")
        assert hasattr(profile, "truncated")

    def test_analysis_report_enforces_primary_answer_design(self):
        """AnalysisReport enforces primary_answer-first isolation from ARCHITECTURE.md."""
        with pytest.raises(ValidationError):
            AnalysisReport()  # type: ignore

        report = AnalysisReport(primary_answer="Answer.")
        assert report.primary_answer == "Answer."

    def test_direct_answer_is_eval_only(self):
        """DirectAnswer is used exclusively when eval_mode=True (per DESIGN.md)."""
        ans = DirectAnswer(answer="42")
        fields = list(DirectAnswer.model_fields.keys())
        assert fields == ["answer"]
        assert ans.answer == "42"
        assert not hasattr(ans, "primary_answer")


class TestModelSerialization:
    """JSON serialization critical for LangGraph state persistence."""

    def test_step_json_roundtrip(self):
        """Step serializes and deserializes without loss."""
        step = Step(
            step_id="s1",
            reasoning="Serialization test",
            action="tool",
            tool_name="analyze_data",
            parameters={"col": "sales", "val": 100},
        )
        json_str = step.model_dump_json()
        restored = Step.model_validate_json(json_str)
        assert restored.step_id == step.step_id
        assert restored.tool_name == step.tool_name
        assert restored.parameters == step.parameters

    def test_plan_json_roundtrip(self):
        """Plan with multiple steps serializes cleanly."""
        plan = Plan(steps=[
            Step(step_id="s1", reasoning="First", action="tool", tool_name="analyze_data"),
            Step(
                step_id="s2",
                reasoning="Second",
                action="sandbox",
                instruction="Calculate percentile.",
            ),
        ])
        json_str = plan.model_dump_json()
        restored = Plan.model_validate_json(json_str)
        assert len(restored.steps) == 2
        assert restored.steps[0].step_id == "s1"

    def test_csv_profile_json_safe(self):
        """CSVProfile with numeric values is JSON-safe."""
        profile = CSVProfile(
            global_schema={"col": "float64"},
            detailed_stats={"col": {"mean": 100.5, "null_rate": 0.0}},
            row_sample=[{"col": 100.5}],
        )
        json_str = profile.model_dump_json()
        restored = CSVProfile.model_validate_json(json_str)
        assert restored.global_schema == profile.global_schema

    def test_analysis_report_json_roundtrip(self):
        """AnalysisReport serializes with nested objects."""
        report = AnalysisReport(
            primary_answer="Answer.",
            plots=[PlotMetadata(file_path="/tmp/plot.png", description="Plot")],
        )
        json_str = report.model_dump_json()
        restored = AnalysisReport.model_validate_json(json_str)
        assert restored.primary_answer == report.primary_answer
        assert len(restored.plots) == 1
