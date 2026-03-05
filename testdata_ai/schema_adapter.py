"""
Convert Pydantic models or JSON Schema dicts to ContextSchema.

Allows developers to generate test data directly from existing models
without writing ContextSchema by hand.
"""

__all__ = ["model_to_context_schema"]

import json
from typing import Any, Dict, List, Optional, Union

from testdata_ai.contexts import ContextSchema


def model_to_context_schema(model_or_schema: Union[type, dict]) -> ContextSchema:
    """Convert a Pydantic model class or JSON Schema dict to a ContextSchema.

    Args:
        model_or_schema: A Pydantic model class (v1 or v2) or a JSON Schema dict.

    Returns:
        ContextSchema derived from the model/schema structure.

    Raises:
        TypeError: If model_or_schema is neither a Pydantic model class nor a dict.
        ValueError: If the schema has no properties to derive a sample from.
    """
    if isinstance(model_or_schema, dict):
        return _json_schema_to_context_schema(model_or_schema)
    if isinstance(model_or_schema, type) and (
        hasattr(model_or_schema, "model_json_schema") or hasattr(model_or_schema, "schema")
    ):
        schema = _get_json_schema(model_or_schema)
        return _json_schema_to_context_schema(schema, name=model_or_schema.__name__)
    raise TypeError(
        f"Expected a Pydantic model class or a JSON Schema dict, got {type(model_or_schema).__name__!r}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_json_schema(model_class: type) -> dict:
    """Extract JSON Schema from a Pydantic model (v1 or v2)."""
    if hasattr(model_class, "model_json_schema"):
        # Pydantic v2
        return model_class.model_json_schema()
    # Pydantic v1
    return model_class.schema()


def _json_schema_to_context_schema(schema: dict, name: Optional[str] = None) -> ContextSchema:
    """Build a ContextSchema from a JSON Schema dict."""
    title = name or schema.get("title") or "GeneratedData"
    description = schema.get("description") or f"Auto-generated from {title} schema"

    # Collect $defs / definitions for $ref resolution
    defs: dict = {**schema.get("$defs", {}), **schema.get("definitions", {})}

    properties = schema.get("properties", {})
    if not properties:
        raise ValueError(
            f"JSON Schema for '{title}' has no 'properties' — cannot derive sample fields."
        )

    sample = _build_sample(properties, defs)
    hints = _build_hints(properties, defs)

    return ContextSchema(
        description=description,
        sample=sample,
        prompt_hints=hints,
        category="custom",
    )


def _build_sample(properties: dict, defs: dict) -> dict:
    """Recursively build a sample dict from JSON Schema properties."""
    return {field: _sample_value(prop, defs, field) for field, prop in properties.items()}


def _sample_value(prop: dict, defs: dict, field_name: str = "") -> Any:
    """Return a representative default value for a JSON Schema property."""
    # Resolve $ref
    if "$ref" in prop:
        ref_key = prop["$ref"].split("/")[-1]
        resolved = defs.get(ref_key, {})
        return _sample_value(resolved, defs, field_name)

    # anyOf / oneOf — use the first non-null option
    for combinator in ("anyOf", "oneOf"):
        if combinator in prop:
            for sub in prop[combinator]:
                if sub.get("type") != "null" and sub != {"type": "null"}:
                    return _sample_value(sub, defs, field_name)
            return None

    # Enum — use first value
    if "enum" in prop:
        return prop["enum"][0] if prop["enum"] else None

    # const
    if "const" in prop:
        return prop["const"]

    # Type-based default
    t = prop.get("type")

    if t == "string":
        # Use format hints when available
        fmt = prop.get("format", "")
        if fmt == "email":
            return "user@example.com"
        if fmt == "date":
            return "2024-01-01"
        if fmt == "date-time":
            return "2024-01-01T00:00:00Z"
        if fmt == "uri":
            return "https://example.com"
        return prop.get("default", f"example_{field_name}" if field_name else "example")

    if t == "integer":
        minimum = prop.get("minimum", prop.get("exclusiveMinimum"))
        return prop.get("default", max(1, minimum + 1) if isinstance(minimum, (int, float)) else 1)

    if t == "number":
        minimum = prop.get("minimum", prop.get("exclusiveMinimum"))
        return prop.get("default", max(1.0, float(minimum) + 0.1) if isinstance(minimum, (int, float)) else 1.0)

    if t == "boolean":
        return prop.get("default", True)

    if t == "null":
        return None

    if t == "array":
        items_schema = prop.get("items", {})
        item_val = _sample_value(items_schema, defs) if items_schema else "example"
        return [item_val]

    if t == "object":
        nested_props = prop.get("properties", {})
        if nested_props:
            return _build_sample(nested_props, defs)
        return {}

    # Unknown / no type: try to derive from default or return empty string
    return prop.get("default", "")


def _build_hints(properties: dict, defs: dict) -> List[str]:
    """Extract prompt hints from property descriptions, enums, and constraints."""
    hints: List[str] = []
    for field_name, prop in properties.items():
        # Resolve $ref before reading metadata
        resolved = prop
        if "$ref" in prop:
            ref_key = prop["$ref"].split("/")[-1]
            resolved = defs.get(ref_key, prop)

        if "description" in resolved:
            hints.append(f"{field_name}: {resolved['description']}")

        if "enum" in resolved and resolved["enum"]:
            vals = ", ".join(json.dumps(v) for v in resolved["enum"])
            hints.append(f"{field_name} must be one of: {vals}")

        constraints = []
        for key, label in (
            ("minimum", "min"), ("maximum", "max"),
            ("exclusiveMinimum", "exclusiveMin"), ("exclusiveMaximum", "exclusiveMax"),
            ("minLength", "minLength"), ("maxLength", "maxLength"),
            ("minItems", "minItems"), ("maxItems", "maxItems"),
        ):
            if key in resolved:
                constraints.append(f"{label}={resolved[key]}")
        if constraints:
            hints.append(f"{field_name}: {', '.join(constraints)}")

    return hints
