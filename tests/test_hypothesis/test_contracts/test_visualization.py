"""Hypothesis property tests for PlotType StrEnum."""
import pytest

from hypothesis import given, strategies as st
from src.scrygent.contracts.visualization import PlotType

valid_vals = st.sampled_from([m.value for m in PlotType])
invalid_vals = st.text().filter(lambda x: x not in {m.value for m in PlotType})


class TestPlotTypeInvariants:
    @given(valid_vals)
    def test_roundtrip(self, value):
        m = PlotType(value)
        assert m.value == value
        assert m.name == value.upper()

    @given(invalid_vals)
    def test_invalid_raises(self, value):
        with pytest.raises(ValueError):
            PlotType(value)

    @given(st.none())
    def test_none_raises(self, none):
        with pytest.raises(ValueError):
            PlotType(none)

    @given(st.integers() | st.floats() | st.booleans())
    def test_non_string_raises(self, non_str):
        with pytest.raises(ValueError):
            PlotType(non_str)

    def test_member_set_is_closed(self):
        all_values = {m.value for m in PlotType}
        expected = {"bar", "line", "scatter", "histogram", "box", "heatmap"}
        assert all_values == expected
