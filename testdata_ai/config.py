"""
Configuration management for testdata-ai.
Loads settings from environment variables.
"""

import os
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

__all__ = ["AIProviderConfig", "get_provider_config"]

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}


@dataclass
class AIProviderConfig:
    """Configuration for an AI provider."""
    provider: str
    api_key: str = field(repr=False)
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(f"temperature must be 0.0-1.0, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")


def get_provider_config(provider: Optional[str] = None) -> AIProviderConfig:
    """Get configuration for specified AI provider from environment.

    Args:
        provider: Provider name ('openai', 'anthropic', None=default from env)

    Returns:
        AIProviderConfig with settings

    Raises:
        ValueError: If provider not supported or API key missing
    """
    provider = (provider or os.getenv("AI_PROVIDER", "openai")).lower()

    if provider not in DEFAULT_MODELS:
        supported = ", ".join(DEFAULT_MODELS)
        raise ValueError(
            f"Unsupported AI provider: '{provider}'. Supported: {supported}"
        )

    prefix = provider.upper()

    api_key = os.getenv(f"{prefix}_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            f"{prefix} API key not found! "
            f"Set {prefix}_API_KEY in .env file or environment."
        )

    return AIProviderConfig(
        provider=provider,
        api_key=api_key,
        model=os.getenv(f"{prefix}_MODEL", DEFAULT_MODELS[provider]),
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv(f"{prefix}_MAX_TOKENS", "2000")),
    )
