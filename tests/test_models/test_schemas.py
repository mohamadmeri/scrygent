"""Tests for schemas.py: all Param models, output models, and registry."""
import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from src.scrygent.contracts import (
    Aggregation, FilterOperator, NormalizeMethod,
    CorrelationMethod, OutlierMethod,
    PlotType, ToolName, RegressionMethod
)
from scrygent.models.outputs import (
    Metric, FilterCondition, SortCondition,
    AnalyzeDataParams, FilterDatasetParams, NormalizeColumnParams,
    NoParams, CorrelationParams, RegressionParams, OutlierParams,
    ColumnStatsParams, PlotParams, DeriveColumnParams, EvaluateMetricsParams,
    CSVProfile, PlotMetadata, AnalysisReport, DirectAnswer,
    TOOL_PARAM_MODELS,
)
from scrygent.base_model import ScrygentBaseModel


# ── Helpers for property-based testing ──
def enum_strategy(enum_cls):
    """Return Hypothesis strategy that picks a random enum member."""
    return st.sampled_from(list(enum_cls))

@st.composite
def filter_condition_strategy(draw):
    """Strategy to generate a valid FilterCondition."""
    return FilterCondition(
        column=draw(st.text(min_size=1)),
        operator=draw(st.sampled_from(list(FilterOperator))),
        value=draw(st.one_of(st.text(), st.integers(), st.floats(), st.none())),
    )


@st.composite
def regression_params_strategy(draw):
    """Strategy for RegressionParams where target is never in features."""
    target = draw(st.text(min_size=1))
    features = draw(
        st.lists(
            st.text(min_size=1).filter(lambda x, avoid=target: x != avoid),
            min_size=1,
            max_size=5,
        )
    )
    method = draw(st.sampled_from(list(RegressionMethod)))
    return RegressionParams(target=target, features=features, method=method)



# ── Metric ──
class TestMetric:
    @given(st.text(min_size=1), enum_strategy(Aggregation), st.text(min_size=1))
    def test_valid_metric(self, col, agg, alias):
        m = Metric(column=col, aggregation=agg, alias=alias)
        assert m.aggregation == agg

    def test_missing_fields(self):
        with pytest.raises(ValidationError):
            # missing aggregation and alias
            Metric(column="x") # type: ignore  

    def test_invalid_aggregation(self):
        with pytest.raises(ValidationError):
            # should be rejected by enum
            Metric(column="x", aggregation="bogus", alias="a") # type: ignore


# ── FilterCondition ──
class TestFilterCondition:
    @given(st.text(min_size=1), enum_strategy(FilterOperator), st.one_of(st.text(), st.integers(), st.floats(), st.none()))
    def test_valid_filter(self, col, op, val):
        fc = FilterCondition(column=col, operator=op, value=val)
        assert fc.operator == op

    def test_invalid_operator(self):
        with pytest.raises(ValidationError):
            FilterCondition(column="a", operator="~~", value=1) # type: ignore


# ── SortCondition ──
class TestSortCondition:
    def test_valid_asc(self):
        sc = SortCondition(column="x", direction="asc")
        assert sc.direction == "asc"

    def test_valid_desc(self):
        sc = SortCondition(column="x", direction="desc")
        assert sc.direction == "desc"

    def test_invalid_direction(self):
        with pytest.raises(ValidationError):
            SortCondition(column="x", direction="ascending") # type: ignore


# ── AnalyzeDataParams ──
class TestAnalyzeDataParams:
    def test_minimal_valid(self):
        p = AnalyzeDataParams(metrics=[Metric(column="sales", aggregation=Aggregation.SUM, alias="total")])
        assert len(p.metrics) == 1

    def test_empty_metrics_raises(self):
        with pytest.raises(ValidationError):
            AnalyzeDataParams(metrics=[])

    def test_invalid_filter_raises(self):
        with pytest.raises(ValidationError):
            AnalyzeDataParams(
                metrics=[Metric(column="x", aggregation=Aggregation.MEAN, alias="m")],
                filters=[FilterCondition(column="a", operator="??", value=1)], # type: ignore
            )

    def test_limit_ge_1(self):
        with pytest.raises(ValidationError):
            AnalyzeDataParams(metrics=[Metric(column="x", aggregation=Aggregation.COUNT, alias="c")], limit=0)

    def test_sort_valid(self):
        p = AnalyzeDataParams(
            metrics=[Metric(column="x", aggregation=Aggregation.MEAN, alias="m")],
            sort=SortCondition(column="m", direction="desc"),
        )
        assert p.sort.direction == "desc" # type: ignore


# ── Wrangling Params ──
class TestFilterDatasetParams:
    def test_valid(self):
        p = FilterDatasetParams(filters=[FilterCondition(column="region", operator=FilterOperator.EQ, value="West")])
        assert len(p.filters) == 1

    def test_empty_filters_raises(self):
        with pytest.raises(ValidationError):
            FilterDatasetParams(filters=[])

class TestNormalizeColumnParams:
    def test_valid_numeric_method(self):
        p = NormalizeColumnParams(column="age", method=NormalizeMethod.MIN_MAX)
        assert p.method == NormalizeMethod.MIN_MAX

    def test_invalid_method_raises(self):
        with pytest.raises(ValidationError):
            NormalizeColumnParams(column="x", method="invalid") # type: ignore

class TestNoParams:
    def test_empty_params(self):
        p = NoParams()
        assert p.model_dump() == {}

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            NoParams(extra=1) # type: ignore


# ── Statistics Params ──
class TestCorrelationParams:
    def test_minimum_two_columns(self):
        p = CorrelationParams(columns=["a", "b"])
        assert len(p.columns) == 2

    def test_too_few_columns(self):
        with pytest.raises(ValidationError):
            CorrelationParams(columns=["a"])

    def test_default_method_pearson(self):
        p = CorrelationParams(columns=["a", "b"])
        assert p.method == CorrelationMethod.PEARSON

class TestRegressionParams:
    def test_valid(self):
        p = RegressionParams(target="y", features=["x1", "x2"])
        assert p.target == "y"

    def test_target_in_features_raises(self):
        with pytest.raises(ValidationError, match="cannot also appear in features"):
            RegressionParams(target="y", features=["y", "x"])

    def test_empty_features_raises(self):
        with pytest.raises(ValidationError):
            RegressionParams(target="y", features=[])

class TestOutlierParams:
    def test_default_method(self):
        p = OutlierParams(column="val")
        assert p.method == OutlierMethod.IQR

    def test_invalid_method(self):
        with pytest.raises(ValidationError):
            OutlierParams(column="x", method="unknown") # type: ignore

class TestColumnStatsParams:
    def test_single_column(self):
        p = ColumnStatsParams(columns=["a"])
        assert p.columns == ["a"]

    def test_empty_columns_raises(self):
        with pytest.raises(ValidationError):
            ColumnStatsParams(columns=[])


# ── Visualization Params ──
class TestPlotParams:
    def test_valid_bar(self):
        p = PlotParams(plot_type=PlotType.BAR, columns=["cat", "val"])
        assert p.plot_type == PlotType.BAR

    def test_invalid_plot_type(self):
        with pytest.raises(ValidationError):
            PlotParams(plot_type="candlestick", columns=["x"]) # type: ignore

    def test_missing_columns_raises(self):
        with pytest.raises(ValidationError):
            PlotParams(plot_type=PlotType.HISTOGRAM, columns=[])


# ── Arithmetic Params ──
class TestDeriveColumnParams:
    def test_valid(self):
        p = DeriveColumnParams(new_column="ratio", expression="a / b")
        assert p.expression == "a / b"

    def test_empty_expression_raises(self):
        with pytest.raises(ValidationError):
            DeriveColumnParams(new_column="x", expression="")

class TestEvaluateMetricsParams:
    def test_valid(self):
        p = EvaluateMetricsParams(expression="avg * 2", values={"avg": 10.0})
        assert p.values["avg"] == 10.0

    def test_empty_values_raises(self):
        with pytest.raises(ValidationError):
            EvaluateMetricsParams(expression="x", values={})


# ── Registry Integrity ──
class TestTOOL_PARAM_MODELS:
    def test_registry_has_all_tools(self):
        """TOOL_PARAM_MODELS keys exactly match ToolName enum."""
        registry_keys = set(TOOL_PARAM_MODELS.keys())
        enum_keys = set(ToolName)
        assert registry_keys == enum_keys, f"Missing: {enum_keys - registry_keys}, Extra: {registry_keys - enum_keys}"

    def test_registry_values_are_scrygent_models(self):
        for model in TOOL_PARAM_MODELS.values():
            assert issubclass(model, ScrygentBaseModel)


# ── Output Models ──
class TestCSVProfile:
    def test_minimal(self):
        p = CSVProfile(global_schema={"col": "int64"}, row_count= 0)
        assert p.row_count == 0

    def test_full_profile(self):
        p = CSVProfile(
            row_count=100,
            global_schema={"a": "int64", "b": "float64"},
            detailed_stats={"a": {"dtype": "int64", "mean": 50}},
            row_sample=[{"a": 1, "b": 2.0}],
            truncated=True,
            missing_detailed_stats=["b"],
        )
        assert p.truncated is True
        assert "b" in p.missing_detailed_stats

    def test_global_schema_mandatory(self):
        with pytest.raises(ValidationError):
            CSVProfile() # type: ignore

    def test_sanitization_in_stats(self):
        import numpy as np
        raw = {
            "global_schema": {"a": "float64"},
            "detailed_stats": {"a": {"mean": np.float64(3.14), "count": np.int64(10)}},
            "row_count" : 100,
        }
        p = CSVProfile.model_validate(raw)
        assert isinstance(p.detailed_stats["a"]["mean"], float)
        assert isinstance(p.detailed_stats["a"]["count"], int)

class TestPlotMetadata:
    def test_valid(self):
        pm = PlotMetadata(file_path="/tmp/plot.png", description="A plot")
        assert pm.file_path == "/tmp/plot.png"

    def test_no_base64_blob(self):
        pm = PlotMetadata(file_path="/tmp/img.png", description="desc")
        assert not pm.file_path.startswith("data:image")

class TestAnalysisReport:
    def test_primary_answer_required(self):
        with pytest.raises(ValidationError):
            AnalysisReport() # type: ignore

    def test_minimal_report(self):
        report = AnalysisReport(primary_answer="42")
        assert report.primary_answer == "42"
        assert report.additional_insights is None

    def test_full_report(self):
        report = AnalysisReport(
            primary_answer="Yes",
            additional_insights=["insight"],
            plots=[PlotMetadata(file_path="/tmp/p.png", description="plot")],
        )
        assert len(report.additional_insights) == 1 # type: ignore
        assert len(report.plots) == 1

class TestDirectAnswer:
    def test_answer_string(self):
        da = DirectAnswer(answer="hello")
        assert da.answer == "hello"

    def test_only_answer_field(self):
        fields = DirectAnswer.model_fields
        assert list(fields.keys()) == ["answer"]


class TestHypothesisFuzzing:
    """Use Hypothesis to generate valid/invalid payloads and ensure no crashes."""

    @given(st.builds(Metric))
    def test_metric_hypothesis(self, m):
        assert isinstance(m.alias, str)

    @given(filter_condition_strategy())
    def test_filter_condition_hypothesis(self, fc):
        assert fc.column != ""

    @given(st.builds(AnalyzeDataParams))
    def test_analyze_data_params_hypothesis(self, p):
        assert len(p.metrics) >= 1

    @given(regression_params_strategy())
    def test_regression_params_hypothesis(self, p):
        # target not in features enforced by our strategy and validator
        assert p.target not in p.features

    @given(st.builds(PlotParams))
    def test_plot_params_hypothesis(self, p):
        assert len(p.columns) >= 1

    @given(
        st.text(),
        st.dictionaries(
            st.text(), st.floats(allow_nan=False, allow_infinity=False), min_size=1
        ),
    )
    def test_evaluate_metrics_params_hypothesis(self, expr, vals):
        try:
            p = EvaluateMetricsParams(expression=expr, values=vals)
            assert p.expression == expr
        except ValidationError:
            pass  # expected for empty expression

    @given(st.builds(DirectAnswer))
    def test_direct_answer_hypothesis(self, da):
        assert isinstance(da.answer, str)
