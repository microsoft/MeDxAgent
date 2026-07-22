"""Settings and configuration management."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Provider Configuration
    llm_provider: Literal["openai", "gemini", "grok", "deepseek"] = Field(
        default="openai",
        description="LLM provider to use"
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="Model/deployment name. Options: "
                    "openai → gpt-4o (default) | o4-mini; "
                    "gemini → gemini-2.5-pro; "
                    "grok → grok-4-1-fast-non-reasoning; "
                    "deepseek → Deepseek-V3.2"
    )

    # API Keys
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    google_api_key: str | None = Field(default=None, description="Google Gemini API key")
    xai_api_key: str | None = Field(default=None, description="xAI Grok API key")
    deepseek_api_key: str | None = Field(default=None, description="DeepSeek API key")

    # Optional base-URL overrides (e.g. self-hosted vLLM, proxy, regional endpoint).
    # When unset, each client uses its provider's default endpoint.
    openai_base_url: str | None = Field(default=None, description="Override base URL for OpenAI client")
    gemini_base_url: str | None = Field(default=None, description="Override base URL for Gemini client")
    xai_base_url: str | None = Field(default=None, description="Override base URL for Grok client")
    deepseek_base_url: str | None = Field(default=None, description="Override base URL for DeepSeek client")

    # LLM Parameters
    temperature: float = Field(default=0.0, description="LLM temperature for generation")
    max_tokens: int = Field(default=4096, description="Maximum tokens in response")
    
    # Workflow Parameters
    max_dialog_rounds: int = Field(default=20, description="Maximum dialog rounds before exit")
    high_confidence_threshold: int = Field(default=90, description="High confidence threshold for exit")
    
    # Structured Output Mode
    unstructured_mode: bool = Field(default=False, description="If True, use unstructured output mode (LLM generates text, then validate JSON). If False, use native structured output parsing.")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
