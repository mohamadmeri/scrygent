"""Unit tests for the ToolName StrEnum (contract layer only)."""

import pytest
from src.scrygent.contracts.tool_names import ToolName


class TestToolNameMembership:
    EXPECTED = frozenset({
        "ANALYZE_DATA", "FILTER_DATASET", "NORMALIZE_COLUMN",
        "RESET_DATASET", "CORRELATION", "REGRESSION",
        "DETECT_OUTLIERS", "REQUEST_COLUMN_STATS",
        "GENERATE_PLOT", "DERIVE_COLUMN", "EVALUATE_METRICS",
    })
    VALUES = {
        "ANALYZE_DATA": "analyze_data",
        "FILTER_DATASET": "filter_dataset",
        "NORMALIZE_COLUMN": "normalize_column",
        "RESET_DATASET": "reset_dataset",
        "CORRELATION": "correlation",
        "REGRESSION": "regression",
        "DETECT_OUTLIERS": "detect_outliers",
        "REQUEST_COLUMN_STATS": "request_column_stats",
        "GENERATE_PLOT": "generate_plot",
        "DERIVE_COLUMN": "derive_column",
        "EVALUATE_METRICS": "evaluate_metrics",
    }

    def test_member_names(self):
        assert {m.name for m in ToolName} == self.EXPECTED

    def test_no_duplicate_values(self):
        vals = [m.value for m in ToolName]
        assert len(vals) == len(set(vals))

    def test_values_match_spec(self):
        for m in ToolName:
            assert m.value == self.VALUES[m.name], f"Wrong value for {m.name}"

    def test_values_are_lowercase_snake_case(self):
        for m in ToolName:
            assert isinstance(m.value, str)
            assert m.value.islower()
            assert "_" in m.value or m.value.isalpha()  # all have underscore
            assert m.value == m.name.lower()


class TestToolNameCoercion:
    @pytest.mark.parametrize("val", [m.value for m in ToolName])
    def test_valid_construct(self, val):
        m = ToolName(val)
        assert m.value == val
        # name should be the upper version of the value
        assert m.name == val.upper()

    @pytest.mark.parametrize("invalid", [
        "ANALYZE_DATA",          # uppercase
        "Filter_Dataset",        # mixed case
        " analyze_data",         # leading space
        "analyze_data ",         # trailing space
        "analyze  data",         # double space
        "",                      # empty
        "nonexistent_tool",
        "None",
        "eval_metrics",          # misspelling
        "generateplot",          # no underscore
    ])
    def test_invalid_raises(self, invalid):
        with pytest.raises(ValueError):
            ToolName(invalid)

    def test_none_raises(self):
        with pytest.raises(ValueError):
            ToolName(None)  # type: ignore[arg-type]

    def test_number_raises(self):
        with pytest.raises(ValueError):
            ToolName(3)  # type: ignore[arg-type]


class TestToolNameProtocols:
    def test_iteration_order(self):
        expected_order = [
            "ANALYZE_DATA", "FILTER_DATASET", "NORMALIZE_COLUMN",
            "RESET_DATASET", "CORRELATION", "REGRESSION",
            "DETECT_OUTLIERS", "REQUEST_COLUMN_STATS",
            "GENERATE_PLOT", "DERIVE_COLUMN", "EVALUATE_METRICS",
        ]
        assert [m.name for m in ToolName] == expected_order

    def test_hashable(self):
        s = set(ToolName)
        assert len(s) == len(ToolName)

    def test_identity(self):
        for m in ToolName:
            assert ToolName(m.value) is m

    def test_bool(self):
        for m in ToolName:
            assert bool(m)

    def test_repr_and_str(self):
        for m in ToolName:
            assert m.name in repr(m)
            assert str(m) == m.value


class TestToolNameSemantics:
    """Ensure key tool names expected by the architecture are present."""

    def test_core_tools_exist(self):
        assert ToolName.ANALYZE_DATA.value == "analyze_data"
        assert ToolName.FILTER_DATASET.value == "filter_dataset"
        assert ToolName.RESET_DATASET.value == "reset_dataset"
        assert ToolName.REQUEST_COLUMN_STATS.value == "request_column_stats"

    def test_statistical_tools_exist(self):
        assert ToolName.CORRELATION.value == "correlation"
        assert ToolName.REGRESSION.value == "regression"
        assert ToolName.DETECT_OUTLIERS.value == "detect_outliers"

    def test_transformation_tools_exist(self):
        assert ToolName.NORMALIZE_COLUMN.value == "normalize_column"
        assert ToolName.DERIVE_COLUMN.value == "derive_column"

    def test_output_tools_exist(self):
        assert ToolName.GENERATE_PLOT.value == "generate_plot"
        assert ToolName.EVALUATE_METRICS.value == "evaluate_metrics"
