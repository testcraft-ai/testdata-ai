"""
Core test data generator - provider agnostic.
Supports OpenAI, Anthropic, and other AI providers.
"""

__all__ = ["DataGenerator", "generate"]

import asyncio
import json
import math
import os
import random
from typing import Callable, Dict, Iterator, List, Any, Optional, Union
import logging

from testdata_ai.prompts import get_prompt, _build_prompt
from testdata_ai.contexts import (
    validate_generated_data,
    ValidationError,
    get_context_schema,
)
from testdata_ai.config import get_provider_config
from testdata_ai.ai_providers import get_provider, AIProvider
from testdata_ai.schema_adapter import model_to_context_schema

logger = logging.getLogger(__name__)


def _build_child_prompt(
    schema,
    count: int,
    parent_records: List[Dict[str, Any]],
    fk_field: str,
    parent_pk: str,
    parent_entity_name: str,
    locale: Optional[str] = None,
) -> str:
    """Build a prompt for a child entity, embedding sample parent records for coherence."""
    base = _build_prompt(schema, count, locale)
    parent_json = json.dumps(parent_records, indent=2)
    return (
        f"{base}"
        f"\nPARENT RECORDS — '{parent_entity_name}' (generate semantically consistent"
        f" child records that make sense for these parents):\n"
        f"{parent_json}\n"
        f"\nFor each record, set '{fk_field}' to one of the '{parent_pk}' values"
        f" from the {parent_entity_name} parent records shown above.\n"
    )


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
        locale: Optional[str] = None,
    ):
        """Initialize the generator.

        Args:
            provider: AI provider name ('openai', 'anthropic', or None for default)
            api_key: API key (if None, reads from .env based on provider)
            model: Model name (if None, uses default for provider)
            temperature: Sampling temperature 0.0-1.0 (if None, uses default)
            max_tokens: Maximum tokens for response (if None, uses default)
            locale: BCP 47 locale tag for generated values (e.g. 'pl', 'ja').
                If None, reads from AI_LOCALE env var; if unset, no locale
                instruction is added and the AI produces English data by default.

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

        self.locale: Optional[str] = locale or os.getenv("AI_LOCALE") or None

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

        logger.info(f"Generating {count} records for context: {context}")

        schema = get_context_schema(context)  # raises ValueError if context unknown

        # Retry up to 3 times to fill any shortfall from a low-yield AI response.
        # Raw AI records are accumulated before Faker is applied so that Faker's
        # unique proxy sees the full dataset in a single call.
        raw_records: List[Dict[str, Any]] = []
        for _ in range(3):
            needed = count - len(raw_records)
            if needed <= 0:
                break
            prompt = get_prompt(context, needed, locale=self.locale)
            logger.debug(f"Sending prompt to {self.provider.__class__.__name__} (need {needed})")
            response = _strip_markdown_fences(self.provider.generate(prompt))
            batch = _parse_ai_response(response)
            if not batch:
                break
            raw_records.extend(batch)

        records = raw_records[:count]

        if schema.field_providers:
            from testdata_ai.faker_bridge import apply_faker_fields
            records = apply_faker_fields(
                records,
                schema.field_providers,
                locale=self.locale,
                unique_fields=schema.unique_fields,
            )

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

    def generate_from_model(
        self,
        model_or_schema,
        count: int = 10,
        validate: bool = True,
        field_providers: Optional[Dict[str, str]] = None,
        unique_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate test data from a Pydantic model class or JSON Schema dict.

        Args:
            model_or_schema: A Pydantic model class (v1 or v2) or a JSON Schema dict.
            count: Number of records to generate.
            validate: Whether to validate generated records against the derived schema.

        Returns:
            List of generated data records as dictionaries.

        Raises:
            TypeError: If model_or_schema is not a Pydantic model or dict.
            ValueError: If count < 1 or AI response is not valid JSON.
            ValidationError: If generated records are missing required fields.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")

        schema = model_to_context_schema(model_or_schema)
        logger.info(f"Generating {count} records from schema: {schema.description}")

        raw_records: List[Dict[str, Any]] = []
        for _ in range(3):
            needed = count - len(raw_records)
            if needed <= 0:
                break
            prompt = _build_prompt(schema, needed, locale=self.locale)
            logger.debug(f"Sending prompt to {self.provider.__class__.__name__} (need {needed})")
            response = _strip_markdown_fences(self.provider.generate(prompt))
            batch = _parse_ai_response(response)
            if not batch:
                break
            raw_records.extend(batch)

        records = raw_records[:count]

        if field_providers:
            from testdata_ai.faker_bridge import apply_faker_fields
            records = apply_faker_fields(
                records,
                field_providers,
                locale=self.locale,
                unique_fields=unique_fields,
            )

        logger.info(f"Successfully generated {len(records)} records")

        if len(records) != count:
            logger.warning(f"Requested {count} records but received {len(records)}")

        if validate:
            invalid = [
                {"record_index": i, "missing_fields": schema.missing_fields(r)}
                for i, r in enumerate(records)
                if not schema.validate_record(r)
            ]
            if invalid:
                raise ValidationError(invalid)

        return records

    def generate_batched(
        self,
        context: str,
        count: int,
        batch_size: int = 10,
        validate: bool = True,
    ) -> Iterator[List[Dict[str, Any]]]:
        """Generate records in batches, yielding each completed batch.

        Splits large counts into multiple AI calls of ``batch_size`` records
        each.  Useful for large counts where incremental output is desired.

        Args:
            context: Context name (e.g. "ecommerce_customer")
            count: Total number of records to generate
            batch_size: Records per AI call (default 10)
            validate: Whether to validate each batch against schema

        Yields:
            List[Dict] for each completed batch
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        remaining = count
        total_yielded = 0
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            batch = self.generate(context, current_batch, validate=validate)
            if not batch:
                break
            yield batch
            total_yielded += len(batch)
            remaining -= current_batch
        if total_yielded < count:
            logger.warning(
                f"Requested {count} total records but generated {total_yielded}"
            )

    def generate_with_relationships(
        self,
        graph: Dict[str, Any],
        validate: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate multiple related entity datasets with referential integrity.

        Generates entities in dependency order (parents before children). Child
        prompts include sample parent records so the AI produces semantically
        coherent data (e.g. order amounts that match the parent customer's income).
        FK values are enforced after generation regardless of AI compliance.

        Args:
            graph: Relationship graph dict. Each key is an entity name; each value
                is a dict with required keys ``context`` (str) and ``count`` (int).
                Child nodes additionally require ``parent`` (str), ``fk_field`` (str),
                and ``parent_pk`` (str). Optional ``parent_sample_size`` (int, default 3)
                controls how many parent records are embedded in the child prompt.
                Optional ``batch_size`` (int, default 10) controls records per AI call.

                Example::

                    {
                        "users": {"context": "ecommerce_customer", "count": 5},
                        "orders": {
                            "context": "restaurant_order",
                            "count": 20,
                            "parent": "users",
                            "fk_field": "user_id",
                            "parent_pk": "email",
                        },
                    }

            validate: Whether to validate each entity against its context schema.
            progress_callback: Optional callable invoked with a progress message
                string before each AI call (e.g. ``spinner.update``).

        Returns:
            Dict mapping entity name to list of generated records, in graph key order.

        Raises:
            ValueError: If the graph is malformed, contains cycles, or references
                unknown contexts or undefined parent nodes.
            ValidationError: If generated records fail schema validation and
                ``validate=True``.
        """
        from testdata_ai.relationship_graph import parse_graph, topological_sort, inject_fk

        specs = parse_graph(graph)
        order = topological_sort(specs)

        result: Dict[str, List[Dict[str, Any]]] = {}

        for node_name in order:
            spec = specs[node_name]
            schema = get_context_schema(spec.context)
            total_batches = math.ceil(spec.count / spec.batch_size)

            records = []
            remaining = spec.count
            batch_num = 0

            if spec.parent is None:
                while remaining > 0:
                    batch_num += 1
                    batch_count = min(spec.batch_size, remaining)
                    if progress_callback:
                        progress_callback(
                            f"Generating {node_name} ({batch_num}/{total_batches})…"
                        )
                    batch = self.generate(spec.context, batch_count, validate=validate)
                    if not batch:
                        break
                    records.extend(batch)
                    remaining -= len(batch)
                if len(records) < spec.count:
                    logger.warning(
                        f"Requested {spec.count} {node_name} records but generated {len(records)}"
                    )
            else:
                parent_records = result[spec.parent]
                n_samples = min(spec.parent_sample_size, len(parent_records), 5)
                sample_parents = (
                    random.sample(parent_records, n_samples)
                    if len(parent_records) > n_samples
                    else parent_records
                )

                while remaining > 0:
                    batch_num += 1
                    batch_count = min(spec.batch_size, remaining)
                    if progress_callback:
                        progress_callback(
                            f"Generating {node_name} ({batch_num}/{total_batches})…"
                        )
                    prompt = _build_child_prompt(
                        schema,
                        batch_count,
                        sample_parents,
                        spec.fk_field,
                        spec.parent_pk,
                        spec.parent,
                        self.locale,
                    )
                    raw = _strip_markdown_fences(self.provider.generate(prompt))
                    batch = _parse_ai_response(raw)

                    if not batch:
                        break

                    if schema.field_providers:
                        from testdata_ai.faker_bridge import apply_faker_fields
                        batch = apply_faker_fields(
                            batch,
                            schema.field_providers,
                            locale=self.locale,
                            unique_fields=schema.unique_fields,
                        )

                    batch = inject_fk(batch, parent_records, spec.fk_field, spec.parent_pk)

                    if validate:
                        invalid = [
                            {"record_index": i, "missing_fields": schema.missing_fields(r)}
                            for i, r in enumerate(batch)
                            if not schema.validate_record(r)
                        ]
                        if invalid:
                            raise ValidationError(invalid)

                    records.extend(batch)
                    remaining -= len(batch)

                if len(records) < spec.count:
                    logger.warning(
                        f"Requested {spec.count} {node_name} records but generated {len(records)}"
                    )
                logger.info(
                    f"Generated {len(records)} {node_name} records "
                    f"(parent: {spec.parent}, fk: {spec.fk_field}={spec.parent_pk})"
                )

            result[node_name] = records

        return result

    def generate_as_dataframe(
        self,
        context: str,
        count: int = 10,
        validate: bool = True,
        flatten: bool = True,
    ):
        """Generate test data and return it as a pandas DataFrame.

        Convenience wrapper around ``generate()`` + ``records_to_dataframe()``.

        Args:
            context: Context name (e.g. ``'ecommerce_customer'``).
            count: Number of records to generate.
            validate: Whether to validate against schema (default: True).
            flatten: If ``True`` (default), nested dicts are expanded into
                dot-separated column names via ``pd.json_normalize()``.
                If ``False``, nested objects are kept as object-typed cells.

        Returns:
            ``pandas.DataFrame`` with one row per generated record.

        Raises:
            ImportError: If ``pandas`` is not installed.
            ValueError: If context is unknown or AI response is not valid JSON.
            ValidationError: If generated records fail schema validation.
        """
        from testdata_ai.pandas_bridge import records_to_dataframe
        records = self.generate(context, count=count, validate=validate)
        return records_to_dataframe(records, flatten=flatten)


def _parse_ai_response(raw: str) -> List[Dict[str, Any]]:
    """Parse and normalize an AI JSON response to a list of records."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to parse AI response as JSON: {e}")
        logger.debug(f"Response preview: {raw[:200]!r}")
        raise ValueError(f"AI response is not valid JSON: {e}") from e

    # Normalize to a list of records. The prompt asks for {"data": [...]};
    # fall back to the first list of dicts found in the response values.
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return data["data"]
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return [data]
    if isinstance(data, list):
        return data
    return [data]


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


def generate(
    input: Union[str, type, Dict[str, Any], List[Any]],
    count: int = 10,
    *,
    validate: bool = True,
    locale: Optional[str] = None,
    batch_size: int = 10,
    field_providers: Optional[Dict[str, str]] = None,
    unique_fields: Optional[List[str]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    global_unique_fields: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    """Generate test data with automatic type dispatch.

    Args:
        input: What to generate from. Accepts:
            - ``str``: context name (e.g. ``"ecommerce_customer"``)
            - ``type`` or ``dict`` without ``"nodes"`` key: Pydantic model class
              or JSON Schema dict
            - ``dict`` with ``"nodes"`` key: relationship graph
            - ``list`` of :class:`GenerateSpec`: parallel generation tasks
        count: Number of records (ignored for graph and list inputs).
        validate: Validate generated records against schema (default True).
        locale: BCP 47 locale tag (e.g. ``"pl"``, ``"ja"``).
        batch_size: Records per AI call when using context-name dispatch.
        field_providers: Faker overrides for specific fields, e.g.
            ``{"email": "faker:email"}``. Only used for context and model dispatch.
        unique_fields: Subset of field_providers keys that must be unique within
            a batch. Only used for context and model dispatch.
        provider: AI provider name; None reads from AI_PROVIDER env var.
        model: Model name; None uses provider default.
        temperature: Sampling temperature 0.0–1.0.
        max_tokens: Maximum tokens for AI response.
        api_key: API key (requires provider to be set).
        global_unique_fields: Fields deduplicated across all parallel tasks.
            Requires ``pip install testdata-ai[faker]``. Only used for list dispatch.
        progress_callback: Called with a progress string before each AI call.
            Only used for graph dispatch.

    Returns:
        :class:`GenerateResult` for str/type/dict-schema inputs,
        :class:`RelationshipResult` for graph-dict and list-of-specs inputs.

    Raises:
        TypeError: If input type is not supported.
        ValueError: If arguments are invalid (e.g. count < 1, bad context).
        ValidationError: If generated records fail schema validation.

    Examples::

        result = generate("ecommerce_customer", count=20)
        result = generate(MyModel, count=5)
        result = generate({"properties": {...}}, count=3)
        result = generate({"nodes": {"users": {"context": "ecommerce_customer", "count": 3}}})
        result = generate([GenerateSpec("ecommerce_customer", 100), ...])
    """
    from testdata_ai.result_types import GenerateResult, RelationshipResult

    # str → context name
    if isinstance(input, str):
        gen = DataGenerator(
            provider=provider, model=model, temperature=temperature,
            max_tokens=max_tokens, api_key=api_key, locale=locale,
        )
        records: List[Dict[str, Any]] = []
        for batch in gen.generate_batched(input, count, batch_size, validate=validate):
            records.extend(batch)
        return GenerateResult(records)

    # type (Pydantic model) or dict without "nodes" → generate_from_model
    if isinstance(input, type) or (isinstance(input, dict) and "nodes" not in input):
        gen = DataGenerator(
            provider=provider, model=model, temperature=temperature,
            max_tokens=max_tokens, api_key=api_key, locale=locale,
        )
        records = gen.generate_from_model(
            input, count, validate,
            field_providers=field_providers,
            unique_fields=unique_fields,
        )
        return GenerateResult(records)

    # dict with "nodes" → relationship graph
    if isinstance(input, dict) and "nodes" in input:
        gen = DataGenerator(
            provider=provider, model=model, temperature=temperature,
            max_tokens=max_tokens, api_key=api_key, locale=locale,
        )
        result = gen.generate_with_relationships(
            input["nodes"], validate=validate, progress_callback=progress_callback
        )
        return RelationshipResult(result)

    # list[GenerateSpec] → parallel generation (sync wrapper)
    if isinstance(input, list):
        from testdata_ai.async_generator import generate_parallel

        generator_kwargs: Dict[str, Any] = {}
        if model is not None:
            generator_kwargs["model"] = model
        if temperature is not None:
            generator_kwargs["temperature"] = temperature
        if max_tokens is not None:
            generator_kwargs["max_tokens"] = max_tokens
        if api_key is not None:
            generator_kwargs["api_key"] = api_key

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            raise RuntimeError(
                "generate() with a list of GenerateSpec cannot be called from an async "
                "context (event loop already running). "
                "Use: result = await async_generate([...]) instead."
            )

        result = asyncio.run(
            generate_parallel(
                input,
                global_unique_fields=global_unique_fields,
                provider=provider,
                **generator_kwargs,
            )
        )
        return RelationshipResult(result)

    raise TypeError(
        f"Unsupported input type: {type(input).__name__}. "
        "Expected: str (context name), type or dict (model/schema), "
        "dict with 'nodes' key (graph), or list of GenerateSpec (parallel)."
    )
