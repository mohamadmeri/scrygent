"""Destructive test suite for the centralized configuration management.

This module aggressively tests the `pydantic-settings` integration. It ensures
that environment variables correctly override defaults, invalid types are
strictly rejected, and dynamic provider-based properties resolve exactly.
"""

import pytest
from pydantic import SecretStr, ValidationError

from scrygent.core.config import Settings


@pytest.fixture
def clean_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Instantiate Settings with a clean environment and no .env file."""
    # Clear any environment variables that might leak from the test runner
    for key in [
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "HF_API_TOKEN",
        "SCRYGENT_PACE_REQUESTS",
        "SCRYGENT_PACE_INTERVAL",
        "SCRYGENT_MAX_RETRIES",
        "SCRYGENT_MEMORY_ENABLED",
        "HF_EMBEDDING_API_URL",
        "SCRYGENT_LLM_PROVIDER",
        "SCRYGENT_LLM_TEMPERATURE",
        "SCRYGENT_LLM_SEED",
        "SCRYGENT_EMISSION_METHOD",
        "SCRYGENT_REPORTER_METHOD",
    ]:
        monkeypatch.delenv(key, raising=False)

    # _env_file=None prevents pydantic-settings from reading the local .env file
    return Settings(_env_file=None)


class TestSettingsDefaults:
    """Tests validating the exact baseline defaults when no env vars are set."""

    def test_defaults_are_set_correctly(self, clean_settings: Settings) -> None:
        """Instantiate settings with no environment variables.

        Asserts the exact default values for pacing, limits, and LLM config.
        """
        s = clean_settings
        assert s.pace_requests is False
        assert s.pace_interval == 2.0
        assert s.max_retries == 3
        assert s.memory_enabled is True
        assert s.max_plot_points == 5000
        assert s.max_detailed_columns == 15
        assert s.llm_provider == "openrouter"
        assert s.llm_temperature == 0.0
        assert s.llm_seed == 42
        assert s.emission_method == "json_mode"

    def test_api_keys_default_to_none(self, clean_settings: Settings) -> None:
        """Instantiate settings with no environment variables.

        Asserts all API credentials default to None to prevent false positives
        in the LLM factory.
        """
        s = clean_settings
        assert s.groq_api_key is None
        assert s.openrouter_api_key is None
        assert s.qdrant_api_key is None
        assert s.hf_api_token is None


class TestSettingsEnvironmentOverrides:
    """Tests validating that environment variables strictly override defaults."""

    def test_env_vars_override_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inject valid environment variables.

        Asserts the Settings object reads the exact values from the environment.
        """
        monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
        monkeypatch.setenv("SCRYGENT_PACE_REQUESTS", "true")
        monkeypatch.setenv("SCRYGENT_MAX_RETRIES", "5")
        monkeypatch.setenv("SCRYGENT_LLM_PROVIDER", "groq")

        s = Settings(_env_file=None)

        assert s.groq_api_key == SecretStr("test-groq-key")
        assert s.pace_requests is True
        assert s.max_retries == 5
        assert s.llm_provider == "groq"

    def test_secret_str_values_are_encapsulated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inject API keys via environment variables.

        Asserts the keys are parsed as SecretStr and their raw values are accessible.
        """
        monkeypatch.setenv("QDRANT_API_KEY", "secret-qdrant")
        monkeypatch.setenv("HF_API_TOKEN", "secret-hf")

        s = Settings(_env_file=None)

        assert isinstance(s.qdrant_api_key, SecretStr)
        assert s.qdrant_api_key.get_secret_value() == "secret-qdrant"
        assert s.hf_api_token.get_secret_value() == "secret-hf"


class TestSettingsDynamicProperties:
    """Tests validating the dynamic model resolution based on the active provider."""

    def test_groq_provider_returns_groq_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set `SCRYGENT_LLM_PROVIDER` to 'groq'.

        Asserts the `reasoning_model` and `formatting_model` properties return
        the Groq-specific model strings.
        """
        monkeypatch.setenv("SCRYGENT_LLM_PROVIDER", "groq")
        s = Settings(_env_file=None)

        assert s.reasoning_model == s.groq_reasoning_model
        assert s.formatting_model == s.groq_formatting_model

    def test_openrouter_provider_returns_openrouter_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set `SCRYGENT_LLM_PROVIDER` to 'openrouter'.

        Asserts the `reasoning_model` and `formatting_model` properties return
        the OpenRouter-specific model strings.
        """
        monkeypatch.setenv("SCRYGENT_LLM_PROVIDER", "openrouter")
        s = Settings(_env_file=None)

        assert s.reasoning_model == s.openrouter_reasoning_model
        assert s.formatting_model == s.openrouter_formatting_model

    def test_emission_method_returns_none_when_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set `SCRYGENT_EMISSION_METHOD` to an empty string.

        Asserts the `get_emission_method` property returns None to support
        LangChain's `method=None` default.
        """
        monkeypatch.setenv("SCRYGENT_EMISSION_METHOD", "")
        s = Settings(_env_file=None)

        assert s.get_emission_method is None

    def test_reporter_method_returns_none_when_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set `SCRYGENT_REPORTER_METHOD` to an empty string.

        Asserts the `get_reporter_method` property returns None.
        """
        monkeypatch.setenv("SCRYGENT_REPORTER_METHOD", "")
        s = Settings(_env_file=None)

        assert s.get_reporter_method is None


class TestSettingsValidationFailures:
    """Tests validating strict type enforcement and rejection of malformed env vars."""

    def test_rejects_non_integer_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inject a non-integer string for `SCRYGENT_MAX_RETRIES`.

        The settings parser must raise a ValidationError.
        """
        monkeypatch.setenv("SCRYGENT_MAX_RETRIES", "not_an_int")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_rejects_non_float_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inject a non-numeric string for `SCRYGENT_LLM_TEMPERATURE`.

        The settings parser must raise a ValidationError.
        """
        monkeypatch.setenv("SCRYGENT_LLM_TEMPERATURE", "hot")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_rejects_non_boolean_pace_requests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inject a non-boolean string for `SCRYGENT_PACE_REQUESTS`.

        The settings parser must raise a ValidationError.
        """
        monkeypatch.setenv("SCRYGENT_PACE_REQUESTS", "not_a_bool")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)
