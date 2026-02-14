"""
AI provider abstractions.
Supports multiple AI providers (OpenAI, Anthropic, etc.)
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    @abstractmethod
    def generate(self, prompt:str, system_prompt: str = "") -> str:
        """
        Generate response from AI.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions (optional)

        Returns:
            Generated text response
        """
        pass

class OpenAIProvider(AIProvider):
    """OpenAI provider (GPT-4, etc.)"""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        """Initialize OpenAI client."""
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate using OpenAI API."""
        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            responses = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return responses.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"Failed to generate data with OpenAI: {e}") from e
        

class AnthropicProvider(AIProvider):
    """Anthropic provider (Claude)"""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        """Initialize Anthropic client."""
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate using Anthropic API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt if system_prompt else "You are helpful assistant.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            text = response.content[0].text

            # Anthropic doesn't have JSON mode, so we parse manually
            # Remove markdown code blocks if present
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text.strip()

            return text

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise RuntimeError(f"Failed to generate data with Anthropic: {e}") from e
        
def get_provider(provider_name: str, api_key: str, model: str,
                 temperature: float, max_tokens: int) -> AIProvider:
    """
    Factory function to get AI provider instance.

    Args:
        provider_name: 'openai' or 'anthropic'
        api_key: API key for the provider
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
    
    Returns:
        AIProvider instance
    """
    if provider_name == "openai":
        return OpenAIProvider(api_key, model, temperature, max_tokens)
    elif provider_name == "anthropic":
        return AnthropicProvider(api_key, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")
    