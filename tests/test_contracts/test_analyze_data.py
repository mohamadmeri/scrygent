"""Unit tests for the Aggregation StrEnum (contract layer only)."""

import pytest
from src.scrygent.contracts.analyze_data import Aggregation


class TestAggregationMembership:
    EXPECTED_MEMBERS = frozenset(
        {"MEAN", "SUM", "COUNT", "NUNIQUE", "MIN", "MAX", "STD", "VAR", "MEDIAN"}
    )

    def test_member_names_exist(self):
        actual = {m.name for m in Aggregation}
        assert actual == self.EXPECTED_MEMBERS, f"Missing members: {self.EXPECTED_MEMBERS - actual}"

    def test_no_duplicate_values(self):
        values = [m.value for m in Aggregation]
        assert len(values) == len(set(values)), f"Duplicate values detected: {values}"

    def test_all_values_are_lowercase_strings(self):
        for member in Aggregation:
            assert isinstance(member.value, str)
            assert member.value.islower(), f"Value {member.value!r} must be lowercase"
            assert member.value == member.name.lower()


class TestStringCoercion:
    @pytest.mark.parametrize("value", [m.value for m in Aggregation])
    def test_construct_from_valid_string(self, value):
        assert Aggregation(value) is getattr(Aggregation, value.upper())

    @pytest.mark.parametrize("invalid", [
        "MEAN",             # uppercase
        " Mean",            # leading whitespace
        "mean ",            # trailing whitespace
        "median\t",         # tab
        "avg",              # completely wrong
        "",                 # empty string
        "NONE",             # None‑like string
    ])
    def test_construct_from_invalid_string_raises(self, invalid):
        with pytest.raises(ValueError):
            Aggregation(invalid)

    def test_construct_from_none_raises_value_error(self):
        """StrEnum.__new__ raises ValueError for None (it becomes 'None' string, not a member)."""
        with pytest.raises(ValueError):
            Aggregation(None)  # type: ignore[arg-type]

    def test_construct_from_number_raises_value_error(self):
        """StrEnum.__new__ raises ValueError for non-string types."""
        with pytest.raises(ValueError):
            Aggregation(1)  # type: ignore[arg-type]


class TestEnumProtocols:
    def test_iteration_is_ordered(self):
        members = list(Aggregation)
        assert [m.name for m in members] == [
            "MEAN", "SUM", "COUNT", "NUNIQUE", "MIN", "MAX", "STD", "VAR", "MEDIAN"
        ]

    def test_members_are_hashable(self):
        s = set(Aggregation)
        assert len(s) == len(Aggregation)
        d = {m: m.value for m in Aggregation}
        assert d[Aggregation.MEAN] == "mean"

    def test_identity_of_constructed_members(self):
        for m in Aggregation:
            assert Aggregation(m.value) is m

    def test_bool_of_members(self):
        for m in Aggregation:
            assert bool(m) is True

    def test_repr_and_str(self):
        for m in Aggregation:
            r = repr(m)
            s = str(m)
            assert m.name in r
            assert s == m.value
