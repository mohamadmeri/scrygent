"""Factory for initializing provider-agnostic, structured LLM clients.

This module abstracts the creation of LangChain chat models, enforcing
strict structured output boundaries and centralizing provider resolution.
It ensures that all LLM interactions are routed through a single,
configurable entry point.
"""

import logging
import os
from typing import Any

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .contracts.llm import LLMProvider

logger = logging.getLogger(__name__)

_DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.GROQ: "llama-3.3-70b-versatile",
    LLMProvider.OPENROUTER: "meta-llama/llama-3.3-70b-instruct",
}


def _resolve_provider(explicit: LLMProvider | str | None) -> LLMProvider:
    """Resolves the target LLM provider from explicit arguments or environment variables."""
    if explicit is not None:
        return LLMProvider(explicit)

    raw = os.getenv("SCRYGENT_LLM_PROVIDER", LLMProvider.GROQ.value).strip().lower()
    try:
        return LLMProvider(raw)
    except ValueError:
        valid = [p.value for p in LLMProvider]
        raise ValueError(f"Invalid SCRYGENT_LLM_PROVIDER='{raw}'. Must be one of: {valid}") from None


def _build_groq_llm(model_name: str) -> ChatGroq:
    """Constructs a ChatGroq client with strict retry constraints."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing.")

    # max_retries=0 disables LangChain's internal Tenacity retry loop.
    # This ensures that core/resilience.py's resilient_call() remains the
    # single source of truth for rate-limit backoff, preventing stacked
    # delays and invisible cooldowns.
    return ChatGroq(
        api_key=SecretStr(api_key),
        model=model_name,  # type: ignore[call-arg]
        temperature=0.0,
        max_retries=0,
    )


def _build_openrouter_llm(model_name: str) -> ChatOpenAI:
    """Constructs a ChatOpenAI client configured for the OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is missing.")

    # max_retries=0 delegates all retry logic to the resilience wrapper.
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=SecretStr(api_key),
        model=model_name,
        temperature=0.0,
        max_retries=0,
    )


_PROVIDER_BUILDERS: dict[LLMProvider, Any] = {
    LLMProvider.GROQ: _build_groq_llm,
    LLMProvider.OPENROUTER: _build_openrouter_llm,
}


def get_structured_llm(
    pydantic_schema: type,
    model_name: str | None = None,
    provider: LLMProvider | str | None = None,
    method: str | None = None,
) -> Any:
    """Initializes a LangChain LLM client bound to a strict Pydantic schema.

    Args:
        pydantic_schema: The Pydantic model class to enforce on the LLM output.
        model_name: Optional specific model identifier. Falls back to provider defaults.
        provider: Optional explicit provider. Falls back to environment configuration.
        method: Optional structured output method (e.g., 'function_calling', 'json_mode').

    Returns:
        A LangChain Runnable configured for structured output generation.
    """
    resolved_provider = _resolve_provider(provider)
    resolved_model = model_name or _DEFAULT_MODELS[resolved_provider]

    llm = _PROVIDER_BUILDERS[resolved_provider](resolved_model)

    logger.info(
        "Structured LLM initialized | provider=%s | model=%s | schema=%s | method=%s",
        resolved_provider.value,
        resolved_model,
        pydantic_schema.__name__,
        method,
    )

    if method is not None:
        return llm.with_structured_output(pydantic_schema, method=method)

    return llm.with_structured_output(pydantic_schema)
