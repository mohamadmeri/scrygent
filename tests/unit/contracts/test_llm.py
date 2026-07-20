"""Destructive test suite for the LLM provider contracts.

This module ensures the closed-vocabulary `LLMProvider` enum strictly
enforces its allowed values and rejects hallucinated or invalid provider
identifiers at the boundary.
"""

import pytest

from scrygent.contracts.llm import LLMProvider


class TestLLMProviderContract:
    """Tests validating the exact closed vocabulary and type strictness of the provider enum."""

    def test_provider_enum_has_exact_closed_vocabulary(self) -> None:
        """Verify the enum contains exactly the expected providers and no others.

        Asserts that the system cannot be extended with new providers without
        explicitly modifying this contract, preventing silent fallbacks.
        """
        assert len(LLMProvider) == 2
        assert LLMProvider.GROQ == "groq"
        assert LLMProvider.OPENROUTER == "openrouter"

        # Ensure no hallucinated providers accidentally slipped into the enum
        members = [member.value for member in LLMProvider]
        assert set(members) == {"groq", "openrouter"}

    def test_provider_enum_instances_are_strict_strings(self) -> None:
        """Verify that enum members behave strictly as native strings.

        This guarantees seamless JSON serialization when passing the provider
        identifier to LLM factory configurations.
        """
        assert isinstance(LLMProvider.GROQ, str)
        assert isinstance(LLMProvider.OPENROUTER, str)

    def test_provider_enum_rejects_hallucinated_string_values(self) -> None:
        """Attempt to instantiate the enum with an unsupported provider string.

        The contract must raise a `ValueError` to immediately halt execution
        if a configuration or payload attempts to route to an unapproved API.
        """
        with pytest.raises(ValueError) as exc_info:
            LLMProvider("anthropic")

        assert "'anthropic' is not a valid LLMProvider" in str(exc_info.value)

    def test_provider_enum_rejects_non_string_inputs(self) -> None:
        """Attempt to instantiate the enum with non-string garbage types.

        The contract must reject integers, None, or other objects to prevent
        implicit type coercion bugs in the factory layer.
        """
        with pytest.raises(ValueError):
            LLMProvider(123)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            LLMProvider(None)  # type: ignore[arg-type]

    def test_provider_enum_rejects_attribute_access_for_unknown_providers(self) -> None:
        """Attempt to access a non-existent provider via attribute notation.

        Ensures strict fail-fast behavior rather than dynamically returning
        a new enum member or None.
        """
        with pytest.raises(AttributeError):
            _ = LLMProvider.CLAUDE  # type: ignore[attr-defined]
