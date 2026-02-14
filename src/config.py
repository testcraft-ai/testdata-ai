"""
Configuration management for testdata-ai.
Loads settings from environemnt variables.
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file
load_dotenv()

@dataclass
class AIProviderConfig:
    """Configuration for an AI provider"""
    provider: str
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000

class Config:
    """
    Application configuration.

    Reads from environment variables (set in .env file)
    """

    @staticmethod
    def get_provider_config(provider: Optional[str] = None) -> AIProviderConfig:
        """
        Get configuration for specified AI Provider.

        Args:
            provider: Provider name ('openai', 'anthropic', None=default from env)

        Returns: 
            AIProviderConfig with settings

        Raises: ValueError: If provider not supported or API key missing
        """
        # Use specified provider or default from env
        provider = provider or os.getenv("AI_PROVIDER", "openai")
        provider = provider.lower()

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found! "
                    "Set OPENAI_API_KEY in .env file or environment."
                )
            
            return AIProviderConfig(
                provider="openai",
                api_key=api_key,
                model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
            )
        
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "Anthropic API key not found! "
                    "Set ANTHROPIC_API_KEY in .env file or environment."
                )
            
            return AIProviderConfig(
                provider="anthropic",
                api_key=api_key,
                model=os.getenv("ANTHROPIC_MODEL", "gpt-4-turbo-preview"),
                temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "2000"))
            )
        else:
            raise ValueError(
                f"Unsupported AI provider: {provider}. "
                f"Supported: openai, anthropic"
            )
        
    @staticmethod
    def get(key: str, default: str = "") -> str:
        """Get config value from environemnt."""
        return os.getenv(key, default)
