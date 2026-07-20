"""Destructive test suite for the filter operator contracts.

This module ensures the closed-vocabulary FilterOperator enum strictly
enforces its allowed values and rejects hallucinated or invalid operators
at the boundary.
"""

import pytest

from scrygent.contracts.filtering import FilterOperator


class TestFilterOperatorContract:
    """Tests validating the exact closed vocabulary and type strictness of the filter operator enum."""

    def test_filter_operator_enum_has_exact_closed_vocabulary(self) -> None:
        """Verify the enum contains exactly the expected operators and no others.

        Asserts that the system cannot be extended with new operators without
        explicitly modifying this contract, preventing silent fallbacks or
        hallucinated query logic.
        """
        assert len(FilterOperator) == 11
        assert FilterOperator.EQ == "=="
        assert FilterOperator.NEQ == "!="
        assert FilterOperator.GT == ">"
        assert FilterOperator.LT == "<"
        assert FilterOperator.GTE == ">="
        assert FilterOperator.LTE == "<="
        assert FilterOperator.IN == "in"
        assert FilterOperator.NOT_IN == "not in"
        assert FilterOperator.CONTAINS == "contains"
        assert FilterOperator.STARTSWITH == "startswith"
        assert FilterOperator.ENDSWITH == "endswith"

        members = [member.value for member in FilterOperator]
        assert set(members) == {"==", "!=", ">", "<", ">=", "<=", "in", "not in", "contains", "startswith", "endswith"}

    def test_filter_operator_enum_instances_are_strict_strings(self) -> None:
        """Verify that enum members behave strictly as native strings.

        This guarantees seamless JSON serialization when passing the operator
        to Pandas or downstream tools.
        """
        assert isinstance(FilterOperator.EQ, str)
        assert isinstance(FilterOperator.CONTAINS, str)

    def test_filter_operator_enum_rejects_hallucinated_string_values(self) -> None:
        """Attempt to instantiate the enum with an unsupported operator string.

        The contract must raise a ValueError to immediately halt execution
        if an LLM hallucinates an unsupported operation like 'equals' or '='.
        """
        with pytest.raises(ValueError, match="'equals' is not a valid FilterOperator"):
            FilterOperator("equals")

        with pytest.raises(ValueError, match="'=' is not a valid FilterOperator"):
            FilterOperator("=")

    def test_filter_operator_enum_rejects_non_string_inputs(self) -> None:
        """Attempt to instantiate the enum with non-string garbage types.

        The contract must reject integers, None, or other objects to prevent
        implicit type coercion bugs in the executor layer.
        """
        with pytest.raises(ValueError):
            FilterOperator(123)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            FilterOperator(None)  # type: ignore[arg-type]

    def test_filter_operator_enum_rejects_attribute_access_for_unknown_operators(self) -> None:
        """Attempt to access a non-existent operator via attribute notation.

        Ensures strict fail-fast behavior rather than dynamically returning
        a new enum member or None.
        """
        with pytest.raises(AttributeError):
            _ = FilterOperator.APPROX  # type: ignore[attr-defined]
