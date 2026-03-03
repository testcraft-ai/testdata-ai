"""
AI provider abstractions.
Supports multiple AI providers (OpenAI, Anthropic, etc.)
"""

from abc import ABC, abstractmethod
import logging
from typing import NoReturn

__all__ = ["AIProvider", "get_provider", "DEFAULT_SYSTEM_PROMPT"]

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a test data generator that returns JSON arrays. "
    "When asked for N items, return an array with exactly N objects, never a single object."
)


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._init_client(api_key)

    @abstractmethod
    def _init_client(self, api_key: str) -> None:
        """Initialize the provider-specific client."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        """Generate response from AI.

        Args:
            prompt: User prompt
            system_prompt: System instructions

        Returns:
            Generated text response
        """

    def _handle_api_error(self, e: Exception) -> NoReturn:
        """Translate provider exceptions into RuntimeError with user-friendly messages."""
        logger.error(f"API error: {e}")
        # Check both exception type hierarchy and message string, since provider SDKs
        # (openai, anthropic) wrap timeouts in their own exception types which may not
        # inherit from Python's built-in TimeoutError.
        is_timeout = isinstance(e, (TimeoutError, OSError)) or any(
            kw in str(e).lower() for kw in ("timed out", "timeout")
        )
        if is_timeout:
            raise RuntimeError(
                f"Request timed out ({self.model}). "
                f"Try reducing --count or using a faster model."
            ) from e
        raise RuntimeError(f"Failed to generate data: {e}") from e


class OpenAIProvider(AIProvider):
    """OpenAI provider (GPT-4o, etc.)"""

    def _init_client(self, api_key: str) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required: pip install openai")
        self.client = OpenAI(api_key=api_key, timeout=120.0, max_retries=3)

    def generate(self, prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            self._handle_api_error(e)

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(
                "OpenAI returned an empty response (possible content filter)"
            )
        return content


class AnthropicProvider(AIProvider):
    """Anthropic provider (Claude)."""

    def _init_client(self, api_key: str) -> None:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package is required: pip install anthropic")
        self.client = Anthropic(api_key=api_key, timeout=120.0, max_retries=3)

    def generate(self, prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            self._handle_api_error(e)

        if not response.content:
            raise RuntimeError("Anthropic returned an empty response")
        return response.content[0].text


_PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(
    provider_name: str, api_key: str, model: str,
    temperature: float, max_tokens: int,
) -> AIProvider:
    """Factory function to create an AI provider instance."""
    cls = _PROVIDERS.get(provider_name)
    if cls is None:
        supported = ", ".join(_PROVIDERS)
        raise ValueError(f"Unsupported provider: '{provider_name}'. Supported: {supported}")
    return cls(api_key, model, temperature, max_tokens)
