"""Unit tests for statistics contract enums (contract layer only)."""

import pytest
from src.scrygent.contracts.statistics import (
    CorrelationMethod,
    RegressionMethod,
    OutlierMethod,
)


# ---------------------------------------------------------------------------
# CorrelationMethod
# ---------------------------------------------------------------------------
class TestCorrelationMethodMembership:
    EXPECTED = frozenset({"PEARSON", "SPEARMAN", "KENDALL"})
    VALUES = {"PEARSON": "pearson", "SPEARMAN": "spearman", "KENDALL": "kendall"}

    def test_member_names(self):
        assert {m.name for m in CorrelationMethod} == self.EXPECTED

    def test_no_duplicate_values(self):
        vals = [m.value for m in CorrelationMethod]
        assert len(vals) == len(set(vals))

    def test_values_match_spec(self):
        for m in CorrelationMethod:
            assert m.value == self.VALUES[m.name]

    def test_values_are_lowercase_strings(self):
        for m in CorrelationMethod:
            assert isinstance(m.value, str)
            assert m.value.islower()
            assert m.value == m.name.lower()


class TestCorrelationMethodCoercion:
    @pytest.mark.parametrize("val", [m.value for m in CorrelationMethod])
    def test_valid_construct(self, val):
        m = CorrelationMethod(val)
        assert m.value == val
        assert m is getattr(CorrelationMethod, val.upper())

    @pytest.mark.parametrize("invalid", [
        "PEARSON", "Spearman", "kendall ", " kendall", "", "correlation",
        "None", "pearson1",
    ])
    def test_invalid_raises(self, invalid):
        with pytest.raises(ValueError):
            CorrelationMethod(invalid)

    def test_none_raises(self):
        with pytest.raises(ValueError):
            CorrelationMethod(None)  # type: ignore[arg-type]

    def test_number_raises(self):
        with pytest.raises(ValueError):
            CorrelationMethod(0)  # type: ignore[arg-type]


class TestCorrelationMethodProtocols:
    def test_order(self):
        assert [m.name for m in CorrelationMethod] == ["PEARSON", "SPEARMAN", "KENDALL"]

    def test_hashable(self):
        s = set(CorrelationMethod)
        assert len(s) == 3

    def test_identity(self):
        for m in CorrelationMethod:
            assert CorrelationMethod(m.value) is m

    def test_bool(self):
        for m in CorrelationMethod:
            assert bool(m)

    def test_repr_and_str(self):
        for m in CorrelationMethod:
            assert m.name in repr(m)
            assert str(m) == m.value


# ---------------------------------------------------------------------------
# RegressionMethod
# ---------------------------------------------------------------------------
class TestRegressionMethodMembership:
    EXPECTED = frozenset({"LINEAR"})
    VALUES = {"LINEAR": "linear"}

    def test_member_names(self):
        assert {m.name for m in RegressionMethod} == self.EXPECTED

    def test_no_duplicate_values(self):
        vals = [m.value for m in RegressionMethod]
        assert len(vals) == len(set(vals))

    def test_value_matches(self):
        assert RegressionMethod.LINEAR.value == "linear"

    def test_lowercase(self):
        for m in RegressionMethod:
            assert m.value.islower()
            assert m.value == m.name.lower()


class TestRegressionMethodCoercion:
    def test_valid_construct(self):
        m = RegressionMethod("linear")
        assert m is RegressionMethod.LINEAR

    @pytest.mark.parametrize("invalid", [
        "LINEAR", "Linear", " linear", "", "logistic", "None"
    ])
    def test_invalid_raises(self, invalid):
        with pytest.raises(ValueError):
            RegressionMethod(invalid)

    def test_none_raises(self):
        with pytest.raises(ValueError):
            RegressionMethod(None)  # type: ignore[arg-type]

    def test_number_raises(self):
        with pytest.raises(ValueError):
            RegressionMethod(1)  # type: ignore[arg-type]


class TestRegressionMethodProtocols:
    def test_order(self):
        assert [m.name for m in RegressionMethod] == ["LINEAR"]

    def test_hashable(self):
        s = set(RegressionMethod)
        assert len(s) == 1

    def test_identity(self):
        assert RegressionMethod("linear") is RegressionMethod.LINEAR

    def test_bool(self):
        assert bool(RegressionMethod.LINEAR)

    def test_repr_and_str(self):
        r = repr(RegressionMethod.LINEAR)
        s = str(RegressionMethod.LINEAR)
        assert "LINEAR" in r
        assert s == "linear"


# ---------------------------------------------------------------------------
# OutlierMethod
# ---------------------------------------------------------------------------
class TestOutlierMethodMembership:
    EXPECTED = frozenset({"IQR", "Z_SCORE"})
    VALUES = {"IQR": "iqr", "Z_SCORE": "z_score"}

    def test_member_names(self):
        assert {m.name for m in OutlierMethod} == self.EXPECTED

    def test_no_duplicate_values(self):
        vals = [m.value for m in OutlierMethod]
        assert len(vals) == len(set(vals))

    def test_values_match_spec(self):
        for m in OutlierMethod:
            assert m.value == self.VALUES[m.name]

    def test_values_are_lowercase(self):
        for m in OutlierMethod:
            assert m.value.islower()
            # Z_SCORE.name is "Z_SCORE", lower would be "z_score" which is correct,
            # but name.lower() would be "z_score" so equality holds.
            assert m.value == m.name.lower()


class TestOutlierMethodCoercion:
    @pytest.mark.parametrize("val", [m.value for m in OutlierMethod])
    def test_valid_construct(self, val):
        m = OutlierMethod(val)
        assert m.value == val
        # "z_score" -> getattr(OutlierMethod, "Z_SCORE")
        expected_name = val.upper()
        if val == "z_score":
            expected_name = "Z_SCORE"
        elif val == "iqr":
            expected_name = "IQR"
        assert m is getattr(OutlierMethod, expected_name)

    @pytest.mark.parametrize("invalid", [
        "IQR", "Z_SCORE", "Z-Score", "z score", " iqr", "z_score ", "", "outlier", "None"
    ])
    def test_invalid_raises(self, invalid):
        with pytest.raises(ValueError):
            OutlierMethod(invalid)

    def test_none_raises(self):
        with pytest.raises(ValueError):
            OutlierMethod(None)  # type: ignore[arg-type]

    def test_number_raises(self):
        with pytest.raises(ValueError):
            OutlierMethod(2)  # type: ignore[arg-type]


class TestOutlierMethodProtocols:
    def test_order(self):
        assert [m.name for m in OutlierMethod] == ["IQR", "Z_SCORE"]

    def test_hashable(self):
        s = set(OutlierMethod)
        assert len(s) == 2

    def test_identity(self):
        for m in OutlierMethod:
            assert OutlierMethod(m.value) is m

    def test_bool(self):
        for m in OutlierMethod:
            assert bool(m)

    def test_repr_and_str(self):
        for m in OutlierMethod:
            assert m.name in repr(m)
            assert str(m) == m.value
