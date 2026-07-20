"""Destructive test suite for the analyze_data tool contracts.

This module ensures the closed-vocabulary `Aggregation` enum strictly
enforces its allowed values and rejects hallucinated or invalid aggregation
identifiers at the boundary.
"""

import pytest

from scrygent.contracts.analyze_data import Aggregation


class TestAggregationContract:
    """Tests validating the exact closed vocabulary and type strictness of the aggregation enum."""

    def test_aggregation_enum_has_exact_closed_vocabulary(self) -> None:
        """Verify the enum contains exactly the expected aggregations and no others.

        Asserts that the system cannot be extended with new aggregations without
        explicitly modifying this contract, preventing silent fallbacks or
        hallucinated math operations.
        """
        assert len(Aggregation) == 9
        assert Aggregation.MEAN == "mean"
        assert Aggregation.SUM == "sum"
        assert Aggregation.COUNT == "count"
        assert Aggregation.NUNIQUE == "nunique"
        assert Aggregation.MIN == "min"
        assert Aggregation.MAX == "max"
        assert Aggregation.STD == "std"
        assert Aggregation.VAR == "var"
        assert Aggregation.MEDIAN == "median"

        # Ensure no hallucinated aggregations accidentally slipped into the enum
        members = [member.value for member in Aggregation]
        assert set(members) == {"mean", "sum", "count", "nunique", "min", "max", "std", "var", "median"}

    def test_aggregation_enum_instances_are_strict_strings(self) -> None:
        """Verify that enum members behave strictly as native strings.

        This guarantees seamless JSON serialization when passing the aggregation
        identifier to Pandas or downstream tools.
        """
        assert isinstance(Aggregation.MEAN, str)
        assert isinstance(Aggregation.COUNT, str)

    def test_aggregation_enum_rejects_hallucinated_string_values(self) -> None:
        """Attempt to instantiate the enum with an unsupported aggregation string.

        The contract must raise a `ValueError` to immediately halt execution
        if an LLM hallucinates an unsupported operation like 'mode'.
        """
        with pytest.raises(ValueError) as exc_info:
            Aggregation("mode")

        assert "'mode' is not a valid Aggregation" in str(exc_info.value)

    def test_aggregation_enum_rejects_non_string_inputs(self) -> None:
        """Attempt to instantiate the enum with non-string garbage types.

        The contract must reject integers, None, or other objects to prevent
        implicit type coercion bugs in the executor layer.
        """
        with pytest.raises(ValueError):
            Aggregation(123)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            Aggregation(None)  # type: ignore[arg-type]

    def test_aggregation_enum_rejects_attribute_access_for_unknown_aggregations(self) -> None:
        """Attempt to access a non-existent aggregation via attribute notation.

        Ensures strict fail-fast behavior rather than dynamically returning
        a new enum member or None.
        """
        with pytest.raises(AttributeError):
            _ = Aggregation.MODE  # type: ignore[attr-defined]
