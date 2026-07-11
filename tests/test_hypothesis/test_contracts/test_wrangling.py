"""Hypothesis property tests for NormalizeMethod StrEnum."""
import pytest

from hypothesis import given, strategies as st
from src.scrygent.contracts.wrangling import NormalizeMethod

valid_vals = st.sampled_from([m.value for m in NormalizeMethod])
invalid_vals = st.text().filter(lambda x: x not in {m.value for m in NormalizeMethod})


class TestNormalizeMethodInvariants:
    @given(valid_vals)
    def test_roundtrip(self, value):
        m = NormalizeMethod(value)
        assert m.value == value
        assert m.name == value.upper()

    @given(invalid_vals)
    def test_invalid_raises(self, value):
        with pytest.raises(ValueError):
            NormalizeMethod(value)

    @given(st.none())
    def test_none_raises(self, none):
        with pytest.raises(ValueError):
            NormalizeMethod(none)

    @given(st.integers() | st.floats() | st.booleans())
    def test_non_string_raises(self, non_str):
        with pytest.raises(ValueError):
            NormalizeMethod(non_str)

    def test_member_set_is_closed(self):
        all_values = {m.value for m in NormalizeMethod}
        expected = {"min_max", "z_score", "log", "strip", "lowercase", "uppercase", "title_case"}
        assert all_values == expected
