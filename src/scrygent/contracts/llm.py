"""Enum for supported LLM providers. Consumed by llm_factory.py's
provider-resolution logic. Kept in contracts/ per the Golden Rule --
closed vocabulary, zero dependencies."""

from enum import StrEnum


class LLMProvider(StrEnum):
    GROQ = "groq"
    OPENROUTER = "openrouter"
