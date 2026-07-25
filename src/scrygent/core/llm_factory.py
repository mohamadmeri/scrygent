"""Factory for initializing provider-agnostic, structured LLM clients."""

import logging
import threading
import time
from typing import Any

from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from ..contracts.llm import LLMProvider
from .config import settings

logger = logging.getLogger(__name__)

# GLOBAL REQUEST PACER (Proactive Rate Limit Avoidance)
_pacer_lock = threading.Lock()
_last_request_time = 0.0


def _pace_request() -> None:
    """Enforces a minimum interval between LLM requests to prevent 429s."""
    if not settings.pace_requests:
        return

    global _last_request_time
    with _pacer_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < settings.pace_interval:
            sleep_time = settings.pace_interval - elapsed
            logger.debug("Pacing request: sleeping for %.2fs to respect RPM limits.", sleep_time)
            time.sleep(sleep_time)
        _last_request_time = time.time()


# LLM FACTORY DEFAULTS
_DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.GROQ: settings.groq_reasoning_model,
    LLMProvider.OPENROUTER: settings.openrouter_reasoning_model,
}


def _resolve_provider(explicit: LLMProvider | str | None) -> LLMProvider:
    """Resolves the target LLM provider from explicit arguments or environment variables."""
    if explicit is not None:
        return LLMProvider(explicit)

    try:
        return LLMProvider(settings.llm_provider.strip().lower())
    except ValueError:
        valid = [p.value for p in LLMProvider]
        raise ValueError(f"Invalid SCRYGENT_LLM_PROVIDER='{settings.llm_provider}'. Must be one of: {valid}") from None


def _build_groq_llm(model_name: str) -> ChatGroq:
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing.")

    return ChatGroq(
        api_key=settings.groq_api_key,  # type: ignore
        model=model_name,
        temperature=settings.llm_temperature,
        max_retries=0,
        model_kwargs={"seed": settings.llm_seed},
    )


def _build_openrouter_llm(model_name: str) -> ChatOpenAI:
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is missing.")

    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        model=model_name,
        temperature=settings.llm_temperature,
        seed=settings.llm_seed,
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
    """Initializes a LangChain LLM client bound to a strict Pydantic schema."""
    resolved_provider = _resolve_provider(provider)
    resolved_model = model_name or _DEFAULT_MODELS[resolved_provider]

    llm = _PROVIDER_BUILDERS[resolved_provider](resolved_model)

    logger.info(
        "Structured LLM initialized | provider=%s | model=%s | schema=%s | method=%s | pacer=%s",
        resolved_provider.value,
        resolved_model,
        pydantic_schema.__name__,
        method,
        settings.pace_requests,
    )

    if method is not None:
        structured_llm = llm.with_structured_output(pydantic_schema, method=method)
    else:
        structured_llm = llm.with_structured_output(pydantic_schema)

    if settings.pace_requests:

        def _pace_and_pass(x: Any) -> Any:
            _pace_request()
            return x

        return RunnableLambda(_pace_and_pass) | structured_llm

    return structured_llm
