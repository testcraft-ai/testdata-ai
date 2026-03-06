"""Tests for schema_adapter — no AI calls needed."""

import pytest

from testdata_ai.contexts import ContextSchema
from testdata_ai.schema_adapter import model_to_context_schema


# ---------------------------------------------------------------------------
# Helpers: fake Pydantic-like classes (no pydantic dependency in tests)
# ---------------------------------------------------------------------------

class _FakeModelV2:
    """Mimics a Pydantic v2 model class."""
    __name__ = "FakeModelV2"

    @classmethod
    def model_json_schema(cls):
        return {
            "title": "FakeModelV2",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }


class _FakeModelV1:
    """Mimics a Pydantic v1 model class (has .schema(), no .model_json_schema())."""
    __name__ = "FakeModelV1"

    @classmethod
    def schema(cls):
        return {
            "title": "FakeModelV1",
            "properties": {
                "username": {"type": "string"},
                "score": {"type": "number"},
            },
        }


# ---------------------------------------------------------------------------
# Tests: Pydantic v2 model
# ---------------------------------------------------------------------------

def test_pydantic_v2_simple_model():
    result = model_to_context_schema(_FakeModelV2)
    assert isinstance(result, ContextSchema)
    assert "name" in result.sample
    assert "age" in result.sample
    assert isinstance(result.sample["name"], str)
    assert isinstance(result.sample["age"], int)


def test_pydantic_v2_description_from_title():
    result = model_to_context_schema(_FakeModelV2)
    assert "FakeModelV2" in result.description


# ---------------------------------------------------------------------------
# Tests: Pydantic v1 model
# ---------------------------------------------------------------------------

def test_pydantic_v1_model():
    result = model_to_context_schema(_FakeModelV1)
    assert isinstance(result, ContextSchema)
    assert "username" in result.sample
    assert "score" in result.sample
    assert isinstance(result.sample["score"], float)


# ---------------------------------------------------------------------------
# Tests: JSON Schema dict
# ---------------------------------------------------------------------------

def test_json_schema_dict():
    schema = {
        "title": "Widget",
        "properties": {
            "name": {"type": "string"},
            "qty": {"type": "integer"},
        },
    }
    result = model_to_context_schema(schema)
    assert isinstance(result, ContextSchema)
    assert result.sample["name"] == "example_name"
    assert result.sample["qty"] == 1


def test_json_schema_uses_title_as_description():
    schema = {"title": "MyThing", "properties": {"x": {"type": "integer"}}}
    result = model_to_context_schema(schema)
    assert "MyThing" in result.description


def test_json_schema_uses_description_field():
    schema = {
        "title": "T",
        "description": "A custom description",
        "properties": {"x": {"type": "integer"}},
    }
    result = model_to_context_schema(schema)
    assert result.description == "A custom description"


# ---------------------------------------------------------------------------
# Tests: field type defaults
# ---------------------------------------------------------------------------

def test_string_field_default():
    schema = {"properties": {"label": {"type": "string"}}}
    result = model_to_context_schema(schema)
    assert result.sample["label"] == "example_label"


def test_boolean_field_default():
    schema = {"properties": {"active": {"type": "boolean"}}}
    result = model_to_context_schema(schema)
    assert result.sample["active"] is True


def test_array_field():
    schema = {"properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    result = model_to_context_schema(schema)
    assert isinstance(result.sample["tags"], list)
    assert len(result.sample["tags"]) == 1
    assert isinstance(result.sample["tags"][0], str)


def test_nested_object():
    schema = {
        "properties": {
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "zip": {"type": "string"},
                },
            }
        }
    }
    result = model_to_context_schema(schema)
    assert isinstance(result.sample["address"], dict)
    assert "street" in result.sample["address"]
    assert "zip" in result.sample["address"]


def test_enum_field():
    schema = {"properties": {"status": {"enum": ["active", "inactive", "pending"]}}}
    result = model_to_context_schema(schema)
    assert result.sample["status"] == "active"


def test_optional_field_anyof_with_null():
    """anyOf: [string, null] → should use the string default."""
    schema = {
        "properties": {
            "nickname": {"anyOf": [{"type": "string"}, {"type": "null"}]}
        }
    }
    result = model_to_context_schema(schema)
    assert isinstance(result.sample["nickname"], str)


def test_ref_resolution():
    schema = {
        "properties": {
            "address": {"$ref": "#/$defs/Address"}
        },
        "$defs": {
            "Address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
            }
        },
    }
    result = model_to_context_schema(schema)
    assert isinstance(result.sample["address"], dict)
    assert "city" in result.sample["address"]


def test_string_format_email():
    schema = {"properties": {"email": {"type": "string", "format": "email"}}}
    result = model_to_context_schema(schema)
    assert "@" in result.sample["email"]


def test_string_format_date():
    schema = {"properties": {"dob": {"type": "string", "format": "date"}}}
    result = model_to_context_schema(schema)
    assert result.sample["dob"] == "2024-01-01"


# ---------------------------------------------------------------------------
# Tests: prompt hints
# ---------------------------------------------------------------------------

def test_description_extracted_as_hint():
    schema = {
        "properties": {
            "age": {"type": "integer", "description": "Age of the user in years"}
        }
    }
    result = model_to_context_schema(schema)
    assert any("Age of the user" in h for h in result.prompt_hints)


def test_enum_hint_generated():
    schema = {"properties": {"role": {"enum": ["admin", "user"]}}}
    result = model_to_context_schema(schema)
    assert any("must be one of" in h for h in result.prompt_hints)


def test_min_max_hint_generated():
    schema = {"properties": {"score": {"type": "integer", "minimum": 0, "maximum": 100}}}
    result = model_to_context_schema(schema)
    assert any("min=0" in h and "max=100" in h for h in result.prompt_hints)


# ---------------------------------------------------------------------------
# Tests: error cases
# ---------------------------------------------------------------------------

def test_unknown_input_raises_typeerror():
    with pytest.raises(TypeError, match="Pydantic model class or a JSON Schema dict"):
        model_to_context_schema("not_a_model")


def test_unknown_input_int_raises_typeerror():
    with pytest.raises(TypeError):
        model_to_context_schema(42)


def test_empty_properties_raises_valueerror():
    with pytest.raises(ValueError, match="no 'properties'"):
        model_to_context_schema({"title": "Empty"})


def test_category_is_custom():
    schema = {"properties": {"x": {"type": "string"}}}
    result = model_to_context_schema(schema)
    assert result.category == "custom"
