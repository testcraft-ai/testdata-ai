"""Tests for testdata_ai.contexts — schemas, listing, and validation."""

import pytest

from testdata_ai.contexts import (
    CONTEXTS,
    ContextSchema,
    ValidationError,
    get_context_schema,
    list_contexts,
    validate_generated_data,
)


class TestContextSchema:
    """Unit tests for the ContextSchema dataclass."""

    def test_fields_derived_from_sample_keys(self):
        schema = ContextSchema(
            description="test",
            sample={"a": 1, "b": 2, "c": 3},
            prompt_hints=[],
        )
        assert schema.fields == ["a", "b", "c"]

    def test_fields_empty_sample(self):
        schema = ContextSchema(description="empty", sample={}, prompt_hints=[])
        assert schema.fields == []

    def test_validate_record_passes_when_all_fields_present(self):
        schema = ContextSchema(
            description="t", sample={"x": 1, "y": 2}, prompt_hints=[]
        )
        assert schema.validate_record({"x": 10, "y": 20}) is True

    def test_validate_record_passes_with_extra_fields(self):
        schema = ContextSchema(
            description="t", sample={"x": 1}, prompt_hints=[]
        )
        assert schema.validate_record({"x": 10, "extra": 99}) is True

    def test_validate_record_fails_when_field_missing(self):
        schema = ContextSchema(
            description="t", sample={"x": 1, "y": 2}, prompt_hints=[]
        )
        assert schema.validate_record({"x": 10}) is False

    def test_validate_record_empty_record(self):
        schema = ContextSchema(
            description="t", sample={"x": 1}, prompt_hints=[]
        )
        assert schema.validate_record({}) is False

    def test_missing_fields_returns_correct_list(self):
        schema = ContextSchema(
            description="t", sample={"a": 1, "b": 2, "c": 3}, prompt_hints=[]
        )
        assert schema.missing_fields({"b": 2}) == ["a", "c"]

    def test_missing_fields_none_missing(self):
        schema = ContextSchema(
            description="t", sample={"a": 1}, prompt_hints=[]
        )
        assert schema.missing_fields({"a": 1}) == []

    def test_default_category_is_general(self):
        schema = ContextSchema(description="t", sample={}, prompt_hints=[])
        assert schema.category == "general"

    def test_validate_record_non_dict_returns_false(self):
        """Non-dict records (e.g. integers) fail validation when fields are expected."""
        schema = ContextSchema(
            description="t", sample={"x": 1}, prompt_hints=[]
        )
        assert schema.validate_record(42) is False

    def test_missing_fields_non_dict_returns_all(self):
        """Non-dict records report all fields as missing."""
        schema = ContextSchema(
            description="t", sample={"x": 1, "y": 2}, prompt_hints=[]
        )
        assert schema.missing_fields(42) == ["x", "y"]



class TestBuiltinContexts:
    """Verify every built-in context is well-formed."""

    ALL_CONTEXT_NAMES = list(CONTEXTS.keys())

    @pytest.mark.parametrize("name", ALL_CONTEXT_NAMES)
    def test_schema_has_nonempty_description(self, name):
        assert CONTEXTS[name].description

    @pytest.mark.parametrize("name", ALL_CONTEXT_NAMES)
    def test_schema_has_nonempty_sample(self, name):
        assert CONTEXTS[name].sample

    @pytest.mark.parametrize("name", ALL_CONTEXT_NAMES)
    def test_schema_has_prompt_hints(self, name):
        assert CONTEXTS[name].prompt_hints

    @pytest.mark.parametrize("name", ALL_CONTEXT_NAMES)
    def test_schema_has_category(self, name):
        assert CONTEXTS[name].category

    @pytest.mark.parametrize("name", ALL_CONTEXT_NAMES)
    def test_sample_validates_against_own_schema(self, name):
        schema = CONTEXTS[name]
        assert schema.validate_record(schema.sample) is True



class TestGetContextSchema:

    def test_returns_schema_for_valid_context(self):
        schema = get_context_schema("ecommerce_customer")
        assert isinstance(schema, ContextSchema)
        assert schema.category == "ecommerce"

    def test_unknown_context_error_message(self):
        with pytest.raises(ValueError) as excinfo:
            get_context_schema("nope")

        assert "Unknown context: 'nope'" in str(excinfo.value)
        assert "Available contexts:" in str(excinfo.value)



class TestListContexts:

    def test_returns_all_when_no_category(self):
        result = list_contexts()
        assert set(result) == set(CONTEXTS.keys())

    def test_filter_by_category(self):
        result = list_contexts(category="finance")
        assert "banking_user" in result
        for name in result:
            assert CONTEXTS[name].category == "finance"

    def test_nonexistent_category_returns_empty(self):
        assert list_contexts(category="nonexistent") == []



class TestValidateGeneratedData:

    def test_all_valid_returns_empty_list(self):
        sample = CONTEXTS["banking_user"].sample
        assert validate_generated_data("banking_user", [sample, sample]) == []

    def test_detects_missing_fields(self):
        incomplete = {"name": "Test"}  # missing many fields
        result = validate_generated_data("banking_user", [incomplete])
        assert len(result) == 1
        assert result[0]["record_index"] == 0
        assert "email" in result[0]["missing_fields"]

    def test_mixed_valid_and_invalid(self):
        sample = CONTEXTS["saas_trial"].sample
        bad = {"name": "X"}
        result = validate_generated_data("saas_trial", [sample, bad, sample])
        assert len(result) == 1
        assert result[0]["record_index"] == 1

    def test_empty_records_list(self):
        assert validate_generated_data("ecommerce_customer", []) == []

    def test_raises_for_unknown_context(self):
        with pytest.raises(ValueError):
            validate_generated_data("fake_context", [{}])

    def test_non_dict_record_reports_all_fields_missing(self):
        """A non-dict record (e.g. string) should report all fields missing."""
        result = validate_generated_data("banking_user", ["not a dict"])
        assert len(result) == 1
        assert result[0]["record_index"] == 0
        assert len(result[0]["missing_fields"]) == len(CONTEXTS["banking_user"].fields)


class TestValidationError:

    def test_stores_invalid_records(self):
        records = [
            {"record_index": 0, "missing_fields": ["email", "age"]},
            {"record_index": 2, "missing_fields": ["name"]},
        ]
        err = ValidationError(records)
        assert err.invalid_records is records
        assert len(err.invalid_records) == 2

    def test_message_includes_count(self):
        records = [{"record_index": 0, "missing_fields": ["x"]}]
        err = ValidationError(records)
        assert "1 record(s) failed validation" in str(err)

    def test_message_includes_details(self):
        records = [
            {"record_index": 0, "missing_fields": ["email"]},
            {"record_index": 3, "missing_fields": ["name", "age"]},
        ]
        err = ValidationError(records)
        msg = str(err)
        assert "record 0" in msg
        assert "record 3" in msg
        assert "email" in msg
        assert "name" in msg

    def test_is_subclass_of_value_error(self):
        err = ValidationError([])
        assert isinstance(err, ValueError)
