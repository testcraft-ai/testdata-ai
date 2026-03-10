"""
Prompt builder for AI test data generation.

Builds prompts dynamically from context schemas (fields, sample, hints)
so there is a single source of truth in contexts.py.
"""

__all__ = ["get_prompt"]

import json
from typing import Optional

from testdata_ai.contexts import ContextSchema, get_context_schema


def _build_prompt(
    schema: ContextSchema,
    count: int,
    locale: Optional[str] = None,
    batch_id: Optional[str] = None,
) -> str:
    """Build a prompt from a ContextSchema object."""
    hints = "\n".join(f"- {hint}" for hint in schema.prompt_hints)
    sample_json = json.dumps(schema.sample, indent=2)

    locale_instruction = (
        f"Generate all text values in the '{locale}' locale/language. "
        f"Use culturally appropriate names, addresses, phone formats, and other locale-specific data. "
        f"Keep all JSON field names in English.\n\n"
        if locale else ""
    )

    batch_instruction = (
        f"Use batch identifier '{batch_id}' to ensure uniqueness. "
        f"Generate values distinct from any other batch with a different identifier.\n\n"
        if batch_id else ""
    )

    return (
        f"Generate exactly {count} realistic {schema.description}.\n"
        f"\n"
        f"{locale_instruction}"
        f"{batch_instruction}"
        f"Return a JSON object with a \"data\" key containing an array "
        f"of exactly {count} objects. Example: {{\"data\": [...]}}\n"
        f"\n"
        f"Requirements for realistic data:\n"
        f"{hints}\n"
        f"\n"
        f"Each object in the array must follow this structure:\n"
        f"{sample_json}\n"
    )


def get_prompt(
    context: str,
    count: int,
    locale: Optional[str] = None,
    batch_id: Optional[str] = None,
) -> str:
    """Build a prompt for the given context and record count.

    Args:
        context: Context identifier (e.g., 'ecommerce_customer')
        count: Number of records to generate (must be >= 1)
        locale: BCP 47 locale tag for generated values (e.g. 'pl', 'ja').
            When None, no locale instruction is added and the AI produces
            English data by default.
        batch_id: Optional short identifier (e.g. 8-char hex UUID prefix)
            injected into the prompt to statistically encourage the AI to
            generate values distinct from other batches. Has no effect
            when None (default single-call behaviour is unchanged).

    Returns:
        Formatted prompt string ready to send to AI

    Raises:
        ValueError: If context is unknown
    """
    schema = get_context_schema(context)
    return _build_prompt(schema, count, locale, batch_id=batch_id)
