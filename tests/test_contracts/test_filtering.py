"""Unit tests for the FilterOperator StrEnum (contract layer only)."""

import pytest
from src.scrygent.contracts.filtering import FilterOperator


class TestFilterOperatorMembership:
    EXPECTED_MEMBERS = frozenset(
        {
            "EQ",
            "NEQ",
            "GT",
            "LT",
            "GTE",
            "LTE",
            "IN",
            "NOT_IN",
            "CONTAINS",
            "STARTSWITH",
            "ENDSWITH",
        }
    )

    EXPECTED_VALUES = {
        "EQ": "==",
        "NEQ": "!=",
        "GT": ">",
        "LT": "<",
        "GTE": ">=",
        "LTE": "<=",
        "IN": "in",
        "NOT_IN": "not in",
        "CONTAINS": "contains",
        "STARTSWITH": "startswith",
        "ENDSWITH": "endswith",
    }

    def test_member_names_exist(self):
        actual = {m.name for m in FilterOperator}
        assert actual == self.EXPECTED_MEMBERS, (
            f"Missing members: {self.EXPECTED_MEMBERS - actual}"
        )

    def test_no_duplicate_values(self):
        values = [m.value for m in FilterOperator]
        assert len(values) == len(set(values)), f"Duplicate values detected: {values}"

    def test_each_value_matches_spec(self):
        for member in FilterOperator:
            assert (
                member.value == self.EXPECTED_VALUES[member.name]
            ), f"{member.name} value mismatch: got {member.value!r}, expected {self.EXPECTED_VALUES[member.name]!r}"

    def test_all_values_are_strings(self):
        for member in FilterOperator:
            assert isinstance(member.value, str)


class TestStringCoercion:
    @pytest.mark.parametrize("value", [m.value for m in FilterOperator])
    def test_construct_from_valid_string(self, value):
        """Each exact value string constructs the correct member."""
        member = FilterOperator(value)
        assert member.value == value

    @pytest.mark.parametrize(
        "invalid",
        [
            "! =",       # space inside operator
            "! ",        # incomplete
            "eq",        # lowercase name
            "EQ",        # uppercase name
            " not in",   # leading space
            "not in ",   # trailing space
            "NOT_IN",    # underscore version
            "contains ", # trailing space
            "",          # empty string
            "nonexistent",
            "=",         # single equals
            "=<",        # reversed
        ],
    )
    def test_construct_from_invalid_string_raises(self, invalid):
        with pytest.raises(ValueError):
            FilterOperator(invalid)

    def test_construct_from_none_raises_value_error(self):
        with pytest.raises(ValueError):
            FilterOperator(None)  # type: ignore[arg-type]

    def test_construct_from_number_raises_value_error(self):
        with pytest.raises(ValueError):
            FilterOperator(42)  # type: ignore[arg-type]


class TestEnumProtocols:
    def test_iteration_is_ordered(self):
        members = list(FilterOperator)
        assert [m.name for m in members] == [
            "EQ",
            "NEQ",
            "GT",
            "LT",
            "GTE",
            "LTE",
            "IN",
            "NOT_IN",
            "CONTAINS",
            "STARTSWITH",
            "ENDSWITH",
        ]

    def test_members_are_hashable(self):
        s = set(FilterOperator)
        assert len(s) == len(FilterOperator)
        d = {m: m.value for m in FilterOperator}
        assert d[FilterOperator.EQ] == "=="

    def test_identity_of_constructed_members(self):
        for m in FilterOperator:
            assert FilterOperator(m.value) is m

    def test_bool_of_members(self):
        for m in FilterOperator:
            assert bool(m) is True

    def test_repr_and_str(self):
        for m in FilterOperator:
            r = repr(m)
            s = str(m)
            assert m.name in r
            assert s == m.value


class TestOperatorSemantics:
    """Guard that certain operators are present and correctly mapped for tool contracts."""

    def test_comparison_operators_present(self):
        assert FilterOperator.EQ.value == "=="
        assert FilterOperator.NEQ.value == "!="
        assert FilterOperator.GT.value == ">"
        assert FilterOperator.LT.value == "<"
        assert FilterOperator.GTE.value == ">="
        assert FilterOperator.LTE.value == "<="

    def test_collection_operators_present(self):
        assert FilterOperator.IN.value == "in"
        assert FilterOperator.NOT_IN.value == "not in"

    def test_string_operators_present(self):
        assert FilterOperator.CONTAINS.value == "contains"
        assert FilterOperator.STARTSWITH.value == "startswith"
        assert FilterOperator.ENDSWITH.value == "endswith"

    def test_all_values_are_valid_python_identifiers_or_operators(self):
        """Values can be used in pandas query strings; no special validation here, just a sanity."""
        for m in FilterOperator:
            assert isinstance(m.value, str) and len(m.value) > 0
