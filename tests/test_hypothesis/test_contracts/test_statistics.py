"""Hypothesis property tests for statistics contract enums."""
import pytest

from hypothesis import given, strategies as st
from src.scrygent.contracts.statistics import (
    CorrelationMethod,
    RegressionMethod,
    OutlierMethod,
)


class TestCorrelationMethodInvariants:
    valid = st.sampled_from([m.value for m in CorrelationMethod])
    invalid = st.text().filter(lambda x: x not in {m.value for m in CorrelationMethod})

    @given(valid)
    def test_roundtrip(self, value):
        m = CorrelationMethod(value)
        assert m.value == value
        assert m.name == value.upper()

    @given(invalid)
    def test_invalid_raises(self, value):
        with pytest.raises(ValueError):
            CorrelationMethod(value)

    @given(st.none())
    def test_none_raises(self, none):
        with pytest.raises(ValueError):
            CorrelationMethod(none)

    @given(st.integers() | st.floats() | st.booleans())
    def test_non_string_raises(self, non_str):
        with pytest.raises(ValueError):
            CorrelationMethod(non_str)


class TestRegressionMethodInvariants:
    valid = st.sampled_from([m.value for m in RegressionMethod])
    invalid = st.text().filter(lambda x: x != "linear")

    @given(valid)
    def test_roundtrip(self, value):
        m = RegressionMethod(value)
        assert m.value == value
        assert m.name == "LINEAR"

    @given(invalid)
    def test_invalid_raises(self, value):
        with pytest.raises(ValueError):
            RegressionMethod(value)

    @given(st.none())
    def test_none_raises(self, none):
        with pytest.raises(ValueError):
            RegressionMethod(none)

    @given(st.integers() | st.floats() | st.booleans())
    def test_non_string_raises(self, non_str):
        with pytest.raises(ValueError):
            RegressionMethod(non_str)


class TestOutlierMethodInvariants:
    valid = st.sampled_from([m.value for m in OutlierMethod])
    invalid = st.text().filter(lambda x: x not in {m.value for m in OutlierMethod})

    @given(valid)
    def test_roundtrip(self, value):
        m = OutlierMethod(value)
        assert m.value == value
        # handle names correctly
        if value == "iqr":
            assert m.name == "IQR"
        elif value == "z_score":
            assert m.name == "Z_SCORE"

    @given(invalid)
    def test_invalid_raises(self, value):
        with pytest.raises(ValueError):
            OutlierMethod(value)

    @given(st.none())
    def test_none_raises(self, none):
        with pytest.raises(ValueError):
            OutlierMethod(none)

    @given(st.integers() | st.floats() | st.booleans())
    def test_non_string_raises(self, non_str):
        with pytest.raises(ValueError):
            OutlierMethod(non_str)
