"""Centralized configuration management via pydantic-settings.

Store all configurations in the environment while providing
strict type validation and IDE autocomplete across the Scrygent codebase.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration object for the Scrygent compiler engine."""

    # Read from .env in the root directory
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API Credentials
    groq_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    hf_api_token: SecretStr | None = None

    # Resilience & Pacing
    pace_requests: bool = Field(default=False, validation_alias="SCRYGENT_PACE_REQUESTS")
    pace_interval: float = Field(default=2.0, validation_alias="SCRYGENT_PACE_INTERVAL")
    max_retries: int = Field(default=3, validation_alias="SCRYGENT_MAX_RETRIES")

    # System Components
    memory_enabled: bool = Field(default=True, validation_alias="SCRYGENT_MEMORY_ENABLED")
    hf_embedding_api_url: str = Field(
        default="https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction",
        validation_alias="HF_EMBEDDING_API_URL",
    )

    # Engine Limits
    max_plot_points: int = Field(default=5000, description="Protects JSON boundary during WebGL plotting.")
    max_detailed_columns: int = Field(default=15, description="Limits prompt bloat during profiling.")

    # LLM Granular Configuration
    llm_provider: str = Field(default="openrouter", validation_alias="SCRYGENT_LLM_PROVIDER")
    llm_temperature: float = Field(default=0.0, validation_alias="SCRYGENT_LLM_TEMPERATURE")
    llm_seed: int = Field(default=42, validation_alias="SCRYGENT_LLM_SEED")

    # Provider-Specific Models
    groq_reasoning_model: str = Field(default="llama-3.3-70b-versatile")
    groq_formatting_model: str = Field(default="llama-3.1-8b-instant")

    openrouter_reasoning_model: str = Field(default="nvidia/nemotron-3-super-120b-a12b:free")
    openrouter_formatting_model: str = Field(default="google/gemma-4-26b-a4b-it:free")

    reporter_reasoning_model: str = Field(default="nvidia/nemotron-3-super-120b-a12b:free")

    # LLM Output Methods
    # Options: "json_mode", "function_calling", or "" (empty string for None)
    emission_method: str = Field(default="json_mode", validation_alias="SCRYGENT_EMISSION_METHOD")
    reporter_method: str = Field(default="", validation_alias="SCRYGENT_REPORTER_METHOD")

    # Dynamic Properties based on the active provider
    @property
    def reasoning_model(self) -> str:
        """Returns the heavy model (Pass 1, Reporter) for the active provider."""
        return self.groq_reasoning_model if self.llm_provider.lower() == "groq" else self.openrouter_reasoning_model

    @property
    def formatting_model(self) -> str:
        """Returns the fast model (Pass 2, Correction Loop) for the active provider."""
        return self.groq_formatting_model if self.llm_provider.lower() == "groq" else self.openrouter_formatting_model

    @property
    def get_emission_method(self) -> str | None:
        return self.emission_method if self.emission_method else None

    @property
    def get_reporter_method(self) -> str | None:
        return self.reporter_method if self.reporter_method else None


# Instantiate the global singleton
settings = Settings()
