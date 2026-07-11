import os
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from .contracts.llm import LLMProvider

logger = logging.getLogger(__name__)

_DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.GROQ: "llama-3.3-70b-versatile",
    LLMProvider.OPENROUTER: "meta-llama/llama-3.3-70b-instruct",
}


def _resolve_provider(explicit: "LLMProvider | str | None") -> LLMProvider:
    if explicit is not None:
        return LLMProvider(explicit)

    raw = os.getenv("SCRYGENT_LLM_PROVIDER", LLMProvider.GROQ.value).strip().lower()
    try:
        return LLMProvider(raw)
    except ValueError:
        raise ValueError(
            f"Invalid SCRYGENT_LLM_PROVIDER='{raw}'. "
            f"Must be one of: {[p.value for p in LLMProvider]}"
        ) from None


def _build_groq_llm(model_name: str) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing.")
    return ChatGroq(
        api_key=api_key,  # type: ignore
        model=model_name,
        temperature=0.0,
        # max_retries=0: langchain's own tenacity-based retry on 429s runs
        # INSIDE this client, invisibly, before an exception ever reaches
        # core/resilience.py's resilient_call(). That means the UI's cooldown
        # banner would never fire for the first N retries, and Groq's own
        # backoff would stack on top of ours. resilient_call() is the single
        # source of truth for rate-limit retry/backoff -- see planner_node.py.
        max_retries=0,
    )


def _build_openrouter_llm(model_name: str) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is missing.")
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,  # type: ignore
        model=model_name,
        temperature=0.0,
        # See _build_groq_llm: 0 so resilient_call() owns all retry/backoff.
        max_retries=0,
    )


# Same idiom as _TOOL_DISPATCHER, _PLOT_HANDLERS, _NUMERIC_METHODS, etc.
# throughout tools/: StrEnum keys, flat function dispatch. Not a class
# hierarchy -- there's no polymorphic behavior here beyond "which client
# do I construct," so a dispatch dict is the right amount of structure.
_PROVIDER_BUILDERS = {
    LLMProvider.GROQ: _build_groq_llm,
    LLMProvider.OPENROUTER: _build_openrouter_llm,
}


def get_structured_llm(
    pydantic_schema: type,
    model_name: str | None = None,
    provider: LLMProvider | str | None = None,
    method: str | None = None
) -> Any:
    resolved_provider = _resolve_provider(provider)
    resolved_model = model_name or _DEFAULT_MODELS[resolved_provider]

    llm = _PROVIDER_BUILDERS[resolved_provider](resolved_model)

    logger.info(
        "Structured LLM initialized | provider=%s | model=%s | schema=%s | method=%s",
        resolved_provider.value, resolved_model, pydantic_schema.__name__, method
    )

    if method is not None:
        return llm.with_structured_output(pydantic_schema, method=method)

    return llm.with_structured_output(pydantic_schema)
