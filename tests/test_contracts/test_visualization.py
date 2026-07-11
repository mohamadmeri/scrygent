"""Unit tests for the PlotType StrEnum (contract layer only)."""

import pytest
from src.scrygent.contracts.visualization import PlotType


class TestPlotTypeMembership:
    EXPECTED = frozenset({"BAR", "LINE", "SCATTER", "HISTOGRAM", "BOX", "HEATMAP"})
    VALUES = {
        "BAR": "bar",
        "LINE": "line",
        "SCATTER": "scatter",
        "HISTOGRAM": "histogram",
        "BOX": "box",
        "HEATMAP": "heatmap",
    }

    def test_member_names(self):
        assert {m.name for m in PlotType} == self.EXPECTED

    def test_no_duplicate_values(self):
        vals = [m.value for m in PlotType]
        assert len(vals) == len(set(vals))

    def test_values_match_spec(self):
        for m in PlotType:
            assert m.value == self.VALUES[m.name]

    def test_values_are_lowercase_strings(self):
        for m in PlotType:
            assert isinstance(m.value, str)
            assert m.value.islower()
            assert m.value == m.name.lower()


class TestPlotTypeCoercion:
    @pytest.mark.parametrize("val", [m.value for m in PlotType])
    def test_valid_construct(self, val):
        m = PlotType(val)
        assert m.value == val
        assert m.name == val.upper()

    @pytest.mark.parametrize("invalid", [
        "BAR",           # uppercase name
        "Line",          # mixed case
        " bar",          # leading space
        "bar ",          # trailing space
        "",              # empty
        "nonexistent",
        "Heatmap",       # wrong case
        "hist",          # abbreviation
        "scatterplot",   # compound
    ])
    def test_invalid_raises(self, invalid):
        with pytest.raises(ValueError):
            PlotType(invalid)

    def test_none_raises(self):
        with pytest.raises(ValueError):
            PlotType(None)  # type: ignore[arg-type]

    def test_number_raises(self):
        with pytest.raises(ValueError):
            PlotType(4)  # type: ignore[arg-type]


class TestPlotTypeProtocols:
    def test_iteration_order(self):
        expected_order = ["BAR", "LINE", "SCATTER", "HISTOGRAM", "BOX", "HEATMAP"]
        assert [m.name for m in PlotType] == expected_order

    def test_hashable(self):
        s = set(PlotType)
        assert len(s) == len(PlotType)

    def test_identity(self):
        for m in PlotType:
            assert PlotType(m.value) is m

    def test_bool(self):
        for m in PlotType:
            assert bool(m)

    def test_repr_and_str(self):
        for m in PlotType:
            assert m.name in repr(m)
            assert str(m) == m.value


class TestPlotTypeSemantics:
    def test_common_plots_exist(self):
        assert PlotType.BAR.value == "bar"
        assert PlotType.LINE.value == "line"
        assert PlotType.SCATTER.value == "scatter"
        assert PlotType.HISTOGRAM.value == "histogram"
        assert PlotType.BOX.value == "box"
        assert PlotType.HEATMAP.value == "heatmap"
