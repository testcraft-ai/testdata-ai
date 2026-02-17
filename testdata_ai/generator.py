"""
Core test data generator - provider agnostic.
Supports OpenAI, Anthropic, and other AI providers.
"""

__all__ = ["TestDataGenerator", "generate"]

import json
import re
from typing import Dict, List, Any, Optional
import logging

from testdata_ai.prompts import get_prompt
from testdata_ai.contexts import (
    get_context_schema,
    validate_generated_data,
    list_contexts,
    ContextSchema,
    ValidationError,
)
from testdata_ai.config import get_provider_config, AIProviderConfig, DEFAULT_MODELS
from testdata_ai.ai_providers import get_provider, AIProvider

logger = logging.getLogger(__name__)


class TestDataGenerator:
    """AI-powered test data generator.

    Generates realistic, context-aware test data using AI providers
    (OpenAI, Anthropic, or others).

    Example:
        >>> gen = TestDataGenerator()
        >>> data = gen.generate("ecommerce_customer", count=10)

        >>> gen = TestDataGenerator(provider="anthropic")
        >>> data = gen.generate("banking_user", count=5)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """Initialize the generator.

        Args:
            provider: AI provider name ('openai', 'anthropic', or None for default)
            api_key: API key (if None, reads from .env based on provider)
            model: Model name (if None, uses default for provider)
            temperature: Sampling temperature 0.0-1.0 (if None, uses default)

        Note:
            If arguments are None, values are read from the .env file.
            When passing api_key, provider is required; model and temperature
            will use provider defaults if not specified.
        """
        if api_key is not None:
            if not api_key.strip():
                raise ValueError("api_key must not be empty")
            if provider is None:
                raise ValueError(
                    "When using custom api_key, you must specify provider"
                )
            if provider not in DEFAULT_MODELS:
                supported = ", ".join(DEFAULT_MODELS)
                raise ValueError(
                    f"Unsupported provider: '{provider}'. Supported: {supported}"
                )
            self.config = AIProviderConfig(
                provider=provider,
                api_key=api_key,
                model=model or DEFAULT_MODELS[provider],
                temperature=temperature if temperature is not None else 0.7,
            )
        else:
            self.config = get_provider_config(provider)

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

        # Normalize to a list of records (JSON object mode wraps arrays
        # under an arbitrary key like "data", "customers", "records", etc.)
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    records = value
                    break
            else:
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

    def list_available_contexts(self) -> List[str]:
        """List all available context identifiers."""
        return list_contexts()

    def get_context_details(self, context: str) -> ContextSchema:
        """Get schema details for a context."""
        return get_context_schema(context)


_MARKDOWN_FENCE_RE = re.compile(r"^(?:```[^\n]*\n)?(.*?)(?:```\s*)?$", re.DOTALL)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences that some AI providers wrap JSON in.

    Handles all common fence variants (```json, ```JSON, ``` json, etc.),
    including missing closing fences.
    """
    text = text.strip()
    # Regex always matches (all groups are optional), so .group(1) is safe.
    return _MARKDOWN_FENCE_RE.match(text).group(1).strip()


def generate(context: str, count: int = 10) -> List[Dict[str, Any]]:
    """Convenience function for one-off generation.

    Creates a new TestDataGenerator each call. For repeated use,
    instantiate TestDataGenerator directly.

    Example:
        >>> from testdata_ai import generate
        >>> customers = generate('ecommerce_customer', 20)
    """
    gen = TestDataGenerator()
    return gen.generate(context, count)
