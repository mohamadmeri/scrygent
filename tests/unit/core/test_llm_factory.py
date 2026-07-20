"""Destructive test suite for the LLM factory and request pacer.

This module aggressively tests the provider resolution, client instantiation,
and structured output binding. It ensures that missing API keys, hallucinated
providers, and missing schemas crash immediately, and that the request pacer
correctly injects delays without hitting real network endpoints.
"""

from typing import Any

import pytest
from langchain_core.runnables import Runnable, RunnableLambda, RunnableSequence

from scrygent.base_model import ScrygentBaseModel
from scrygent.contracts.llm import LLMProvider
from scrygent.core import llm_factory
from scrygent.core.llm_factory import _build_groq_llm, _build_openrouter_llm, _resolve_provider, get_structured_llm


class DummySchema(ScrygentBaseModel):
    """Mock schema for structured output binding."""

    answer: str


class MockLLM:
    """Mock LangChain LLM to isolate pacer logic from network calls."""

    def with_structured_output(self, schema: type, **kwargs: Any) -> RunnableLambda:
        return RunnableLambda(lambda x: "success")


class TestProviderResolution:
    """Tests validating the exact closed vocabulary and fallback logic of provider resolution."""

    def test_resolves_explicit_provider_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inject an explicit provider while an env var is set.

        Asserts the explicit argument takes precedence over the environment variable.
        """
        monkeypatch.setenv("SCRYGENT_LLM_PROVIDER", "openrouter")
        assert _resolve_provider(LLMProvider.GROQ) == LLMProvider.GROQ

    def test_resolves_provider_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set the `SCRYGENT_LLM_PROVIDER` environment variable.

        Asserts the factory correctly resolves the provider from the environment.
        """
        monkeypatch.setenv("SCRYGENT_LLM_PROVIDER", "openrouter")
        assert _resolve_provider(None) == LLMProvider.OPENROUTER

    def test_rejects_hallucinated_explicit_provider(self) -> None:
        """Inject an unsupported provider string like 'anthropic'.

        The resolver must raise a ValueError to halt execution immediately.
        """
        with pytest.raises(ValueError, match="'anthropic' is not a valid LLMProvider"):
            _resolve_provider("anthropic")

    def test_rejects_hallucinated_env_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set the env var to an unsupported provider like 'bedrock'.

        The resolver must raise a ValueError guiding the user to valid options.
        """
        monkeypatch.setenv("SCRYGENT_LLM_PROVIDER", "bedrock")
        with pytest.raises(ValueError, match="Invalid SCRYGENT_LLM_PROVIDER='bedrock'"):
            _resolve_provider(None)


class TestLLMBuilders:
    """Tests validating the strict API key enforcement and client configuration."""

    def test_build_groq_llm_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inject a dummy API key for Groq.

        Asserts the client is instantiated with max_retries=0 to delegate
        backoff to the resilience layer.
        """
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        client = _build_groq_llm("llama-3.3-70b-versatile")
        assert client.max_retries == 0
        # LangChain internally maps 0.0 to 1e-08 to avoid division by zero in sampling
        assert client.temperature < 1e-5

    def test_build_groq_llm_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Delete the `GROQ_API_KEY` environment variable.

        The builder must raise a ValueError preventing execution from proceeding.
        """
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GROQ_API_KEY environment variable is missing."):
            _build_groq_llm("llama-3.3-70b-versatile")

    def test_build_openrouter_llm_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inject a dummy API key for OpenRouter.

        Asserts the client is instantiated with max_retries=0.
        """
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        client = _build_openrouter_llm("meta-llama/llama-3.3-70b-instruct:free")
        assert client.max_retries == 0

    def test_build_openrouter_llm_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Delete the `OPENROUTER_API_KEY` environment variable.

        The builder must raise a ValueError.
        """
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY environment variable is missing."):
            _build_openrouter_llm("meta-llama/llama-3.3-70b-instruct:free")


class TestStructuredLLMFactory:
    """Tests validating the top-level factory binding and pacer injection."""

    def test_get_structured_llm_returns_runnable_without_pacer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_api_credentials: None,
    ) -> None:
        """Inject valid credentials with the pacer disabled.

        Asserts the factory returns the raw structured output runnable,
        not a RunnableSequence.
        """
        monkeypatch.setattr(llm_factory, "_pacer_enabled", False)
        monkeypatch.setitem(llm_factory._PROVIDER_BUILDERS, LLMProvider.GROQ, lambda x: MockLLM())

        llm = get_structured_llm(DummySchema, provider=LLMProvider.GROQ)
        assert isinstance(llm, Runnable)
        assert not isinstance(llm, RunnableSequence)

    def test_pacer_prepends_runnable_and_fires_on_invoke(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_api_credentials: None,
    ) -> None:
        """Inject valid credentials with the pacer enabled.

        Asserts the factory returns a RunnableSequence. Mocks `time.sleep`
        and invokes the chain to prove the pacer executes its delay exactly once.
        """
        monkeypatch.setattr(llm_factory, "_pacer_enabled", True)
        monkeypatch.setattr(llm_factory, "_pacer_interval", 5.0)

        # Mock time.time to prevent the elapsed time from being massive (real epoch)
        monkeypatch.setattr(llm_factory.time, "time", lambda: 100.0)
        monkeypatch.setattr(llm_factory, "_last_request_time", 100.0)

        sleep_calls: list[float] = []
        monkeypatch.setattr(llm_factory.time, "sleep", lambda x: sleep_calls.append(x))

        monkeypatch.setitem(llm_factory._PROVIDER_BUILDERS, LLMProvider.GROQ, lambda x: MockLLM())

        llm = get_structured_llm(DummySchema, provider=LLMProvider.GROQ)

        assert isinstance(llm, RunnableSequence)

        # Invoke the chain to trigger the pacer
        result = llm.invoke({"messages": []})

        assert result == "success"
        assert len(sleep_calls) == 1
        assert sleep_calls[0] == 5.0  # Interval(5.0) - elapsed(0.0) = 5.0
