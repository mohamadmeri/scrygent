"""Hypothesis property tests for Aggregation StrEnum."""

import pytest

from hypothesis import given, strategies as st
from src.scrygent.contracts.analyze_data import Aggregation

valid_agg_values = st.sampled_from([m.value for m in Aggregation])
invalid_agg_values = st.text().filter(lambda x: x not in [m.value for m in Aggregation])

class TestAggregationInvariants:
    @given(valid_agg_values)
    def test_roundtrip_value_to_member(self, value):
        m = Aggregation(value)
        assert m.value == value
        assert m.name == value.upper()

    @given(invalid_agg_values)
    def test_invalid_strings_raise(self, value):
        with pytest.raises(ValueError):
            Aggregation(value)

    @given(st.none())
    def test_none_always_raises_value_error(self, none):
        with pytest.raises(ValueError):
            Aggregation(none)

    @given(st.integers() | st.floats() | st.booleans())
    def test_non_string_types_raise_value_error(self, non_str):
        with pytest.raises(ValueError):
            Aggregation(non_str)

    def test_member_set_is_closed(self):
        all_values = {m.value for m in Aggregation}
        expected = {"mean", "sum", "count", "nunique", "min", "max", "std", "var", "median"}
        assert all_values == expected
