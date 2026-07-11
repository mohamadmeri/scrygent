"""Hypothesis property tests for ToolName StrEnum."""
import pytest

from hypothesis import given, strategies as st
from src.scrygent.contracts.tool_names import ToolName

valid_vals = st.sampled_from([m.value for m in ToolName])
invalid_vals = st.text().filter(lambda x: x not in {m.value for m in ToolName})


class TestToolNameInvariants:
    @given(valid_vals)
    def test_roundtrip(self, value):
        m = ToolName(value)
        assert m.value == value
        assert m.name == value.upper()

    @given(invalid_vals)
    def test_invalid_raises(self, value):
        with pytest.raises(ValueError):
            ToolName(value)

    @given(st.none())
    def test_none_raises(self, none):
        with pytest.raises(ValueError):
            ToolName(none)

    @given(st.integers() | st.floats() | st.booleans())
    def test_non_string_raises(self, non_str):
        with pytest.raises(ValueError):
            ToolName(non_str)

    def test_member_set_is_closed(self):
        all_values = {m.value for m in ToolName}
        expected = {
            "analyze_data", "filter_dataset", "normalize_column",
            "reset_dataset", "correlation", "regression",
            "detect_outliers", "request_column_stats",
            "generate_plot", "derive_column", "evaluate_metrics",
        }
        assert all_values == expected
