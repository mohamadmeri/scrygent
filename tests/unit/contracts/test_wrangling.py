"""Destructive test suite for the wrangling tool contracts.

This module ensures the closed-vocabulary NormalizeMethod enum strictly
enforces its allowed values and rejects hallucinated or invalid normalization
identifiers at the boundary.
"""

import pytest

from scrygent.contracts.wrangling import NormalizeMethod


class TestNormalizeMethodContract:
    """Validates the exact closed vocabulary and type strictness of the normalize method enum."""

    def test_normalize_method_enum_has_exact_closed_vocabulary(self) -> None:
        """Verify the enum contains exactly the expected methods and no others.

        Asserts that the system cannot be extended with new wrangling algorithms
        without explicitly modifying this contract, preventing silent fallbacks.
        """
        assert len(NormalizeMethod) == 7
        assert NormalizeMethod.MIN_MAX == "min_max"
        assert NormalizeMethod.Z_SCORE == "z_score"
        assert NormalizeMethod.LOG == "log"
        assert NormalizeMethod.STRIP == "strip"
        assert NormalizeMethod.LOWERCASE == "lowercase"
        assert NormalizeMethod.UPPERCASE == "uppercase"
        assert NormalizeMethod.TITLE_CASE == "title_case"

        members = [member.value for member in NormalizeMethod]
        assert set(members) == {"min_max", "z_score", "log", "strip", "lowercase", "uppercase", "title_case"}

    def test_normalize_method_enum_instances_are_strict_strings(self) -> None:
        """Verify that enum members behave strictly as native strings.

        This guarantees seamless JSON serialization when passing the method
        to the deterministic wrangling tool.
        """
        assert isinstance(NormalizeMethod.MIN_MAX, str)
        assert isinstance(NormalizeMethod.LOG, str)

    def test_normalize_method_enum_rejects_hallucinated_string_values(self) -> None:
        """Attempt to instantiate the enum with an unsupported method string.

        The contract must raise a ValueError to immediately halt execution
        if an LLM hallucinates an unsupported operation like 'standardize'.
        """
        with pytest.raises(ValueError, match="'standardize' is not a valid NormalizeMethod"):
            NormalizeMethod("standardize")

    def test_normalize_method_enum_rejects_non_string_inputs(self) -> None:
        """Attempt to instantiate the enum with non-string garbage types.

        The contract must reject integers or None to prevent implicit type
        coercion bugs in the executor layer.
        """
        with pytest.raises(ValueError):
            NormalizeMethod(1)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            NormalizeMethod(None)  # type: ignore[arg-type]

    def test_normalize_method_enum_rejects_attribute_access_for_unknown_methods(self) -> None:
        """Attempt to access a non-existent method via attribute notation.

        Ensures strict fail-fast behavior rather than dynamically returning
        a new enum member or None.
        """
        with pytest.raises(AttributeError):
            _ = NormalizeMethod.L1_NORM  # type: ignore[attr-defined]
