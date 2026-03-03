"""
Core test data generator - provider agnostic.
Supports OpenAI, Anthropic, and other AI providers.
"""

__all__ = ["DataGenerator", "generate"]

import json
from typing import Dict, List, Any, Optional
import logging

from testdata_ai.prompts import get_prompt
from testdata_ai.contexts import (
    validate_generated_data,
    ValidationError,
)
from testdata_ai.config import get_provider_config
from testdata_ai.ai_providers import get_provider, AIProvider

logger = logging.getLogger(__name__)


class DataGenerator:
    """AI-powered test data generator.

    Generates realistic, context-aware test data using AI providers
    (OpenAI, Anthropic, or others).

    Example:
        >>> gen = DataGenerator()
        >>> data = gen.generate("ecommerce_customer", count=10)

        >>> gen = DataGenerator(provider="anthropic")
        >>> data = gen.generate("banking_user", count=5)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the generator.

        Args:
            provider: AI provider name ('openai', 'anthropic', or None for default)
            api_key: API key (if None, reads from .env based on provider)
            model: Model name (if None, uses default for provider)
            temperature: Sampling temperature 0.0-1.0 (if None, uses default)
            max_tokens: Maximum tokens for response (if None, uses default)

        Note:
            If arguments are None, values are read from the .env file.
            When passing api_key, provider is required.
        """
        if api_key is not None:
            if not api_key.strip():
                raise ValueError("api_key must not be empty")
            if provider is None:
                raise ValueError("When using custom api_key, you must specify provider")

        # Always load from env as base; explicit args override individual fields.
        # Only forward api_key when the caller explicitly supplied one.
        self.config = (
            get_provider_config(provider, api_key=api_key)
            if api_key is not None
            else get_provider_config(provider)
        )
        if model:
            self.config.model = model
        if temperature is not None:
            if not 0.0 <= temperature <= 1.0:
                raise ValueError(f"temperature must be 0.0-1.0, got {temperature}")
            self.config.temperature = temperature
        if max_tokens is not None:
            self.config.max_tokens = max_tokens

        self.provider: AIProvider = get_provider(
            provider_name=self.config.provider,
            api_key=self.config.api_key,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        logger.info(
            f"Initialized generator with {self.config.provider} provider "
            f"(model: {self.config.model})"
        )

    def set_max_tokens(self, value: int) -> None:
        """Update max_tokens on both the config and the underlying provider."""
        self.config.max_tokens = value
        self.provider.max_tokens = value

    def generate(
        self,
        context: str,
        count: int = 10,
        validate: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generate test data for a given context.

        Args:
            context: Type of data to generate (e.g., "ecommerce_customer", "banking_user")
            count: Number of records to generate
            validate: Whether to validate against schema (default: True)

        Returns:
            List of generated data records as dictionaries

        Raises:
            ValueError: If context is unknown or AI response is not valid JSON
            ValidationError: If generated records are missing required fields
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        if count > 50:
            logger.warning(
                f"count={count} may exceed token limits. "
                f"Consider generating in smaller batches."
            )

        logger.info(f"Generating {count} records for context: {context}")

        prompt = get_prompt(context, count)  # raises ValueError if context unknown
        logger.debug(f"Sending prompt to {self.provider.__class__.__name__}")

        response = _strip_markdown_fences(self.provider.generate(prompt))

        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.debug(f"Response preview: {response[:200]!r}")
            raise ValueError(f"AI response is not valid JSON: {e}") from e

        # Normalize to a list of records. The prompt asks for {"data": [...]};
        # fall back to the first list of dicts found in the response values.
        if isinstance(data, dict):
            if isinstance(data.get("data"), list):
                records = data["data"]
            else:
                for v in data.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        records = v
                        break
                else:
                    # Wrap the entire dict as a single record.
                    records = [data]
        elif isinstance(data, list):
            records = data
        else:
            records = [data]

        logger.info(f"Successfully generated {len(records)} records")

        if len(records) != count:
            logger.warning(
                f"Requested {count} records but received {len(records)}"
            )

        if validate:
            invalid = validate_generated_data(context, records)
            if invalid:
                raise ValidationError(invalid)

        return records


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences that some AI providers wrap JSON in."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        # else: malformed fence with no newline — leave text as-is
    if text.endswith("```"):
        text = text[:text.rfind("```")].rstrip()
    return text.strip()


def generate(context: str, count: int = 10) -> List[Dict[str, Any]]:
    """Convenience function for one-off generation.

    Creates a new DataGenerator each call. For repeated use,
    instantiate DataGenerator directly.

    Example:
        >>> from testdata_ai import generate
        >>> customers = generate('ecommerce_customer', 20)
    """
    gen = DataGenerator()
    return gen.generate(context, count)
