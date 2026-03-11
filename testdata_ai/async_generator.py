"""
Async parallel test data generation for testdata-ai.

Allows multiple contexts to be generated concurrently via asyncio.
Each synchronous DataGenerator / AI provider call runs in a thread pool
via asyncio.to_thread() (available since Python 3.9).
"""

__all__ = ["GenerateSpec", "async_generate"]

import asyncio
import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field name → Faker method name mapping (used by _UniqueFieldManager)
# ---------------------------------------------------------------------------

FIELD_FAKER_MAP: Dict[str, str] = {
    "email": "email",
    "id": "uuid4",
    "user_id": "uuid4",
    "customer_id": "uuid4",
    "order_id": "uuid4",
    "phone": "phone_number",
    "phone_number": "phone_number",
    "name": "name",
    "full_name": "name",
    "username": "user_name",
    "address": "address",
    "city": "city",
    "zip": "zipcode",
    "postcode": "zipcode",
}


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class GenerateSpec:
    """Specification for one parallel generation task.

    Attributes:
        context:  Context identifier (e.g. 'ecommerce_customer').
        count:    Number of records to generate.
        locale:   BCP 47 locale tag; overrides AI_LOCALE env var when set.
        validate: Whether to run schema validation on results.
        label:    Custom key in the results dict. Defaults to ``context``.
                  Use label to differentiate two specs with the same context
                  (e.g. label='buyers' and label='sellers').
                  When label is None, results from the same context are merged.
    """

    context: str
    count: int
    locale: Optional[str] = None
    validate: bool = False
    label: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal: cross-call uniqueness manager
# ---------------------------------------------------------------------------


class _UniqueFieldManager:
    """Cross-call deduplication for specified fields using Faker.

    After all parallel tasks complete, iterates every result list and
    replaces duplicate values in ``fields`` with fresh Faker-generated
    values.  Uniqueness is guaranteed across the entire results dict.

    Faker is checked at construction time so ImportError is raised before
    any async work starts.

    Args:
        fields: Field names to deduplicate across all results.
        locale: BCP 47 locale tag passed to Faker(); None → default locale.

    Raises:
        ImportError: If the ``faker`` package is not installed.
    """

    _MAX_RETRIES = 1000

    def __init__(self, fields: List[str], locale: Optional[str] = None) -> None:
        try:
            from faker import Faker as _Faker
        except ImportError:
            raise ImportError(
                "The 'faker' package is required for global_unique_fields. "
                "Install it with: pip install 'testdata-ai[faker]'"
            )
        self._fields = list(fields)
        self._faker = _Faker(locale) if locale else _Faker()
        self._seen: Dict[str, set] = {f: set() for f in self._fields}

    def _faker_value(self, field: str) -> Any:
        """Generate a value for ``field`` using the mapped Faker method.

        Unknown field names fall back to ``uuid4`` (globally unique, no
        exhaustion risk).
        """
        method_name = FIELD_FAKER_MAP.get(field, "uuid4")
        method = getattr(self._faker, method_name, None)
        if method is None:
            method = self._faker.uuid4
        return method()

    def deduplicate(
        self, results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Replace duplicate field values across all result lists.

        Iterates in insertion order; the first occurrence of a value wins
        and all later occurrences are replaced with Faker-generated values.
        Returns a new dict with shallow-copied records — the input is not
        mutated.

        Retries up to _MAX_RETRIES times per replacement; logs a warning
        and leaves the value as-is if exhausted (e.g. low-cardinality fields).
        """
        clean: Dict[str, List[Dict[str, Any]]] = {}
        for label, records in results.items():
            clean[label] = [{**r} for r in records]

        for label, records in clean.items():
            for record in records:
                for field in self._fields:
                    val = record.get(field)
                    if val is None:
                        continue
                    if val in self._seen[field]:
                        replaced = False
                        for _ in range(self._MAX_RETRIES):
                            new_val = self._faker_value(field)
                            if new_val not in self._seen[field]:
                                record[field] = new_val
                                self._seen[field].add(new_val)
                                replaced = True
                                break
                        if not replaced:
                            logger.warning(
                                "Could not find unique value for field '%s' "
                                "after %d retries; leaving as-is.",
                                field,
                                self._MAX_RETRIES,
                            )
                    else:
                        self._seen[field].add(val)

        return clean


# ---------------------------------------------------------------------------
# Internal: single async task
# ---------------------------------------------------------------------------


async def _generate_one(
    spec: GenerateSpec,
    provider_name: Optional[str],
    batch_id: str,
    **generator_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Run one generation task in a thread pool.

    Follows the same pattern as ``generate_with_relationships`` child nodes
    in generator.py: instantiates a fresh DataGenerator, builds the prompt
    directly (with batch_id injected), and calls gen.provider.generate()
    without going through DataGenerator.generate() — so we can pass batch_id
    without modifying DataGenerator.

    Args:
        spec: GenerateSpec describing what to generate.
        provider_name: AI provider name; None → reads from AI_PROVIDER env var.
        batch_id: 8-char hex UUID prefix injected into the prompt (Layer 1).
        **generator_kwargs: Forwarded to DataGenerator() constructor.

    Returns:
        List of generated record dicts.
    """
    from testdata_ai.generator import DataGenerator, _strip_markdown_fences, _parse_ai_response
    from testdata_ai.contexts import get_context_schema, validate_generated_data, ValidationError
    from testdata_ai.prompts import get_prompt

    def _sync_generate() -> List[Dict[str, Any]]:
        # spec.locale takes precedence; fall back to locale in generator_kwargs
        # so that passing locale= via **kwargs doesn't cause a collision.
        kw = dict(generator_kwargs)
        effective_locale = spec.locale if spec.locale is not None else kw.pop("locale", None)

        gen = DataGenerator(
            provider=provider_name,
            locale=effective_locale,
            **kw,
        )

        prompt = get_prompt(
            spec.context,
            spec.count,
            locale=effective_locale,
            batch_id=batch_id,
        )

        raw = _strip_markdown_fences(gen.provider.generate(prompt))
        records = _parse_ai_response(raw)

        schema = get_context_schema(spec.context)
        if schema.field_providers:
            from testdata_ai.faker_bridge import apply_faker_fields
            records = apply_faker_fields(
                records,
                schema.field_providers,
                locale=spec.locale,
                unique_fields=schema.unique_fields,
            )

        if len(records) != spec.count:
            logger.warning(
                "[batch_id=%s] Requested %d %s records but received %d",
                batch_id,
                spec.count,
                spec.context,
                len(records),
            )

        if spec.validate:
            invalid = validate_generated_data(spec.context, records)
            if invalid:
                raise ValidationError(invalid)

        logger.info(
            "[batch_id=%s] Generated %d %s records",
            batch_id,
            len(records),
            spec.context,
        )
        return records

    return await asyncio.to_thread(_sync_generate)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def generate_parallel(
    specs: List[GenerateSpec],
    global_unique_fields: Optional[List[str]] = None,
    provider: Optional[str] = None,
    **generator_kwargs: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate multiple contexts in parallel using asyncio.

    Each spec runs as an independent asyncio task. Blocking AI provider
    calls are offloaded to a thread pool via asyncio.to_thread (Python 3.9+).

    **Result keying rules:**

    - If ``spec.label`` is set → result is stored under that label.
    - If ``spec.label`` is None and multiple specs share the same context →
      their results are **merged** into one list under the context name.
      This enables single-context parallel generation::

          results = await generate_parallel([
              GenerateSpec("ecommerce_customer", 1000),
              GenerateSpec("ecommerce_customer", 1000),
              GenerateSpec("ecommerce_customer", 1000),
          ])
          # results["ecommerce_customer"] has ~3000 records

    **Uniqueness:**

    - *Layer 1* (statistical): each task receives a unique ``batch_id`` injected
      into its prompt. Reduces duplicates but does not guarantee uniqueness.
    - *Layer 2* (guaranteed): when ``global_unique_fields`` is set, a
      synchronous dedup pass replaces confirmed cross-context duplicates using
      Faker. Requires ``pip install testdata-ai[faker]``.

    Args:
        specs: List of GenerateSpec, one per parallel generation task.
        global_unique_fields: Field names that must be unique across all
            results. Requires Faker. None → no cross-call dedup.
        provider: AI provider name ('openai', 'anthropic', 'ollama').
            None → reads from AI_PROVIDER env var.
        **generator_kwargs: Extra kwargs forwarded to DataGenerator()
            (e.g. model='gpt-4o', temperature=0.5).

    Returns:
        Dict mapping label (or context name) → List[Dict] of records.

    Raises:
        ValueError: If specs is empty.
        ImportError: If global_unique_fields set and faker not installed.
        RuntimeError / ValueError / ValidationError: Propagated from any
            failed task (asyncio.gather raises the first exception).
    """
    if not specs:
        raise ValueError("specs must be a non-empty list of GenerateSpec")

    # Warn when a label value matches a different spec's context name — likely accidental.
    keys_to_contexts: dict = {}
    for spec in specs:
        key = spec.label if spec.label is not None else spec.context
        if key in keys_to_contexts and keys_to_contexts[key] != spec.context:
            logger.warning(
                "Result key '%s' is shared by specs with different contexts ('%s' and '%s'). "
                "Their records will be merged. Use distinct labels to keep them separate.",
                key,
                keys_to_contexts[key],
                spec.context,
            )
        else:
            keys_to_contexts[key] = spec.context

    # Validate Faker availability early — before spawning async tasks.
    manager: Optional[_UniqueFieldManager] = None
    if global_unique_fields:
        locale_for_manager = next(
            (s.locale for s in specs if s.locale is not None), None
        )
        manager = _UniqueFieldManager(global_unique_fields, locale=locale_for_manager)

    batch_ids = [uuid.uuid4().hex[:8] for _ in specs]

    logger.info(
        "Starting parallel generation: %d tasks [provider=%s]",
        len(specs),
        provider or "env-default",
    )

    tasks = [
        _generate_one(spec, provider, bid, **generator_kwargs)
        for spec, bid in zip(specs, batch_ids)
    ]

    results_list: List[List[Dict[str, Any]]] = await asyncio.gather(*tasks)

    # Build results dict; specs without label are merged by context name.
    results: Dict[str, List[Dict[str, Any]]] = {}
    for spec, records in zip(specs, results_list):
        key = spec.label if spec.label is not None else spec.context
        if key in results:
            results[key].extend(records)
        else:
            results[key] = list(records)

    logger.info(
        "Parallel generation complete: %s",
        ", ".join(f"{k}={len(v)}" for k, v in results.items()),
    )

    if manager is not None:
        results = manager.deduplicate(results)
        logger.info("Cross-call deduplication complete")

    return results


async def async_generate(
    input: Any,
    count: int = 10,
    *,
    parallelism: int = 3,
    batch_size: Optional[int] = None,
    validate: bool = True,
    locale: Optional[str] = None,
    global_unique_fields: Optional[List[str]] = None,
    provider: Optional[str] = None,
    field_providers: Optional[Dict[str, Any]] = None,
    unique_fields: Optional[List[str]] = None,
    progress_callback: Optional[Any] = None,
    **generator_kwargs: Any,
):
    """Generate test data asynchronously with automatic type dispatch.

    Args:
        input: What to generate from. Accepts:
            - ``str``: context name (e.g. ``"ecommerce_customer"``)
            - ``type`` or ``dict`` without ``"nodes"`` key: Pydantic model class
              or JSON Schema dict
            - ``dict`` with ``"nodes"`` key: relationship graph
            - ``list`` of :class:`GenerateSpec`: parallel generation tasks
        count: Number of records (ignored for graph and list inputs).
        parallelism: Max concurrent AI calls when using context-name dispatch.
        batch_size: Records per AI call for context-name dispatch.
        validate: Validate generated records against schema.
        locale: BCP 47 locale tag (e.g. ``"pl"``, ``"ja"``).
        global_unique_fields: Fields deduplicated across parallel tasks.
            Requires ``pip install testdata-ai[faker]``.
        provider: AI provider name; None reads from AI_PROVIDER env var.
        field_providers: Faker overrides for specific fields. Only used for
            context and model dispatch.
        unique_fields: Subset of field_providers keys requiring uniqueness.
            Only used for context and model dispatch.
        progress_callback: Called with progress string before each AI call.
            Only used for graph dispatch.
        **generator_kwargs: Extra kwargs forwarded to DataGenerator()
            (e.g. model, temperature, max_tokens).
            Do NOT pass ``locale=`` here; use the explicit ``locale`` param.

    Returns:
        :class:`GenerateResult` for str/type/dict-schema inputs,
        :class:`RelationshipResult` for graph-dict and list-of-specs inputs.

    Raises:
        TypeError: If input type is not supported.
        ValueError: If count < 1 or parallelism < 1.
        ImportError: If global_unique_fields set and faker not installed.

    Examples::

        result = await async_generate("ecommerce_customer", 3000, parallelism=3)
        result = await async_generate(MyModel, count=5)
        result = await async_generate({"nodes": {...}})
        result = await async_generate([GenerateSpec("ecommerce_customer", 100), ...])
    """
    from testdata_ai.result_types import GenerateResult, RelationshipResult
    from testdata_ai.generator import DataGenerator

    # str → parallel batch generation (original async_generate behavior)
    if isinstance(input, str):
        context = input
        if count < 1:
            raise ValueError("count must be >= 1")
        if parallelism < 1:
            raise ValueError("parallelism must be >= 1")

        effective_batch = batch_size or math.ceil(count / parallelism)

        batch_counts: List[int] = []
        remaining = count
        while remaining > 0:
            batch_counts.append(min(effective_batch, remaining))
            remaining -= effective_batch

        manager: Optional[_UniqueFieldManager] = None
        if global_unique_fields:
            manager = _UniqueFieldManager(global_unique_fields, locale=locale)

        sem = asyncio.Semaphore(parallelism)
        batch_ids = [uuid.uuid4().hex[:8] for _ in batch_counts]
        specs = [GenerateSpec(context, n, locale=locale) for n in batch_counts]

        logger.info(
            "async_generate: %d records for '%s' → %d batches, max %d concurrent",
            count,
            context,
            len(batch_counts),
            parallelism,
        )

        async def _limited(spec: GenerateSpec, bid: str) -> List[Dict[str, Any]]:
            async with sem:
                return await _generate_one(spec, provider, bid, **generator_kwargs)

        results_list: List[List[Dict[str, Any]]] = await asyncio.gather(
            *[_limited(s, bid) for s, bid in zip(specs, batch_ids)]
        )

        all_records = [record for batch in results_list for record in batch]

        if manager is not None:
            all_records = manager.deduplicate({"_": all_records})["_"]

        return GenerateResult(all_records)

    # type (Pydantic model) or dict without "nodes" → from_model in thread
    if isinstance(input, type) or (isinstance(input, dict) and "nodes" not in input):
        def _sync_from_model():
            gen = DataGenerator(provider=provider, locale=locale, **generator_kwargs)
            return gen.generate_from_model(
                input, count, validate,
                field_providers=field_providers,
                unique_fields=unique_fields,
            )

        records = await asyncio.to_thread(_sync_from_model)
        return GenerateResult(records)

    # dict with "nodes" → relationship graph in thread
    if isinstance(input, dict) and "nodes" in input:
        def _sync_relationships():
            gen = DataGenerator(provider=provider, locale=locale, **generator_kwargs)
            return gen.generate_with_relationships(
                input["nodes"], validate=validate, progress_callback=progress_callback
            )

        result = await asyncio.to_thread(_sync_relationships)
        return RelationshipResult(result)

    # list[GenerateSpec] → generate_parallel
    if isinstance(input, list):
        result = await generate_parallel(
            input,
            global_unique_fields=global_unique_fields,
            provider=provider,
            **generator_kwargs,
        )
        return RelationshipResult(result)

    raise TypeError(
        f"Unsupported input type: {type(input).__name__}. "
        "Expected: str (context name), type or dict (model/schema), "
        "dict with 'nodes' key (graph), or list of GenerateSpec (parallel)."
    )
