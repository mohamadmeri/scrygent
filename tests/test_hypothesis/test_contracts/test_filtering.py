"""Hypothesis property tests for FilterOperator StrEnum."""

from hypothesis import given, strategies as st
import pytest

from src.scrygent.contracts.filtering import FilterOperator

valid_vals = st.sampled_from([m.value for m in FilterOperator])
invalid_vals = st.text().filter(lambda x: x not in {m.value for m in FilterOperator})


class TestFilterOperatorInvariants:
    @given(valid_vals)
    def test_roundtrip(self, value):
        m = FilterOperator(value)
        assert m.value == value
        
        # Name must be uppercase value with underscores
        if value == "not in":
            assert m.name == "NOT_IN"
        elif value == "in":
            assert m.name == "IN"
        else:
            # for operators like ">=", name is "GTE", not a simple upper
            # We can't easily derive name from value. Better to just test membership.
            # So skip name check.
            pass

    @given(invalid_vals)
    def test_invalid_raises(self, value):
        with pytest.raises(ValueError):
            FilterOperator(value)

    @given(st.none())
    def test_none_raises(self, none):
        with pytest.raises(ValueError):
            FilterOperator(none)

    @given(st.integers() | st.floats() | st.booleans())
    def test_non_string_raises(self, non_str):
        with pytest.raises(ValueError):
            FilterOperator(non_str)

    def test_member_set_is_closed(self):
        all_values = {m.value for m in FilterOperator}
        expected = {
            "==", "!=", ">", "<", ">=", "<=",
            "in", "not in",
            "contains", "startswith", "endswith",
        }
        assert all_values == expected
