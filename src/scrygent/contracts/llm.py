"""Supported Large Language Model provider identifiers.

This module defines the closed vocabulary for LLM routing. It is consumed
by the LLM factory to resolve provider-specific configurations while
maintaining strict dependency isolation at the contract layer.
"""

from enum import StrEnum


class LLMProvider(StrEnum):
    """Supported external LLM API providers."""

    GROQ = "groq"
    OPENROUTER = "openrouter"
