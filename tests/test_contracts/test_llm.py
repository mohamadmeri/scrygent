"""Unit tests for the LLMProvider StrEnum (contract layer only)."""

import pytest
from src.scrygent.contracts.llm import LLMProvider


class TestLLMProviderMembership:
    EXPECTED_MEMBERS = frozenset({"GROQ", "OPENROUTER"})
    EXPECTED_VALUES = {"GROQ": "groq", "OPENROUTER": "openrouter"}

    def test_member_names_exist(self):
        actual = {m.name for m in LLMProvider}
        assert actual == self.EXPECTED_MEMBERS, (
            f"Missing members: {self.EXPECTED_MEMBERS - actual}"
        )

    def test_no_duplicate_values(self):
        values = [m.value for m in LLMProvider]
        assert len(values) == len(set(values)), f"Duplicate values detected: {values}"

    def test_each_value_matches_spec(self):
        for member in LLMProvider:
            expected = self.EXPECTED_VALUES[member.name]
            assert member.value == expected, (
                f"{member.name} value mismatch: got {member.value!r}, expected {expected!r}"
            )

    def test_all_values_are_lowercase_strings(self):
        for member in LLMProvider:
            assert isinstance(member.value, str)
            assert member.value.islower(), f"Value {member.value!r} must be lowercase"
            assert member.value == member.name.lower()


class TestStringCoercion:
    @pytest.mark.parametrize("value", [m.value for m in LLMProvider])
    def test_construct_from_valid_string(self, value):
        member = LLMProvider(value)
        assert member.value == value
        assert member is getattr(LLMProvider, value.upper())

    @pytest.mark.parametrize("invalid", [
        "GROQ",            # uppercase name
        "OpenRouter",      # mixed case
        "open router",     # space
        "openrouter ",     # trailing space
        " openrouter",     # leading space
        "",                # empty
        "grok",            # misspelling
        "nonexistent",
        "None",            # None‑like string
    ])
    def test_construct_from_invalid_string_raises(self, invalid):
        with pytest.raises(ValueError):
            LLMProvider(invalid)

    def test_construct_from_none_raises_value_error(self):
        with pytest.raises(ValueError):
            LLMProvider(None)  # type: ignore[arg-type]

    def test_construct_from_number_raises_value_error(self):
        with pytest.raises(ValueError):
            LLMProvider(42)  # type: ignore[arg-type]


class TestEnumProtocols:
    def test_iteration_is_ordered(self):
        members = list(LLMProvider)
        assert [m.name for m in members] == ["GROQ", "OPENROUTER"]

    def test_members_are_hashable(self):
        s = set(LLMProvider)
        assert len(s) == 2
        d = {m: m.value for m in LLMProvider}
        assert d[LLMProvider.GROQ] == "groq"

    def test_identity_of_constructed_members(self):
        for m in LLMProvider:
            assert LLMProvider(m.value) is m

    def test_bool_of_members(self):
        for m in LLMProvider:
            assert bool(m) is True

    def test_repr_and_str(self):
        for m in LLMProvider:
            r = repr(m)
            s = str(m)
            assert m.name in r
            assert s == m.value
