"""
Prompt builder for AI test data generation.

Builds prompts dynamically from context schemas (fields, sample, hints)
so there is a single source of truth in contexts.py.
"""

__all__ = ["get_prompt"]

import json
from typing import Optional

from testdata_ai.contexts import get_context_schema


def get_prompt(context: str, count: int, locale: Optional[str] = None) -> str:
    """Build a prompt for the given context and record count.

    Args:
        context: Context identifier (e.g., 'ecommerce_customer')
        count: Number of records to generate (must be >= 1)
        locale: BCP 47 locale tag for generated values (e.g. 'pl', 'ja').
            When None, no locale instruction is added and the AI produces
            English data by default.

    Returns:
        Formatted prompt string ready to send to AI

    Raises:
        ValueError: If context is unknown
    """
    schema = get_context_schema(context)

    hints = "\n".join(f"- {hint}" for hint in schema.prompt_hints)
    sample_json = json.dumps(schema.sample, indent=2)

    locale_instruction = (
        f"Generate all text values in the '{locale}' locale/language. "
        f"Use culturally appropriate names, addresses, phone formats, and other locale-specific data. "
        f"Keep all JSON field names in English.\n\n"
        if locale else ""
    )

    return (
        f"Generate exactly {count} realistic {schema.description}.\n"
        f"\n"
        f"{locale_instruction}"
        f"Return a JSON object with a \"data\" key containing an array "
        f"of exactly {count} objects. Example: {{\"data\": [...]}}\n"
        f"\n"
        f"Requirements for realistic data:\n"
        f"{hints}\n"
        f"\n"
        f"Each object in the array must follow this structure:\n"
        f"{sample_json}\n"
    )
