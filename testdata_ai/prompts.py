"""
Prompt builder for AI test data generation.

Builds prompts dynamically from context schemas (fields, sample, hints)
so there is a single source of truth in contexts.py.
"""

__all__ = ["get_prompt"]

import json

from testdata_ai.contexts import get_context_schema


def get_prompt(context: str, count: int) -> str:
    """Build a prompt for the given context and record count.

    Args:
        context: Context identifier (e.g., 'ecommerce_customer')
        count: Number of records to generate (must be >= 1)

    Returns:
        Formatted prompt string ready to send to AI

    Raises:
        ValueError: If context is unknown
    """
    schema = get_context_schema(context)

    hints = "\n".join(f"- {hint}" for hint in schema.prompt_hints)
    sample_json = json.dumps(schema.sample, indent=2)

    return (
        f"Generate exactly {count} realistic {schema.description}.\n"
        f"\n"
        f"Return a JSON object with a \"data\" key containing an array "
        f"of exactly {count} objects. Example: {{\"data\": [...]}}\n"
        f"\n"
        f"Requirements for realistic data:\n"
        f"{hints}\n"
        f"\n"
        f"Each object in the array must follow this structure:\n"
        f"{sample_json}\n"
    )
