"""Tests for testdata_ai.contexts — ContextSchema dataclass and ValidationError."""

import pytest

from testdata_ai.contexts import ContextSchema, ValidationError


class TestContextSchema:

    def test_fields_derived_from_sample_keys(self):
        schema = ContextSchema(
            description="test",
            sample={"a": 1, "b": 2, "c": 3},
            prompt_hints=[],
        )
        assert schema.fields == ["a", "b", "c"]

    def test_empty_sample_raises(self):
        with pytest.raises(ValueError, match="non-empty dict"):
            ContextSchema(description="empty", sample={}, prompt_hints=[])

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

    def test_validate_record_passes_nested_dict_with_all_keys(self):
        schema = ContextSchema(
            description="t", sample={"stats": {"str": 10, "dex": 12}}, prompt_hints=[]
        )
        assert schema.validate_record({"stats": {"str": 14, "dex": 18, "extra": 99}}) is True

    def test_validate_record_fails_nested_dict_missing_key(self):
        schema = ContextSchema(
            description="t", sample={"stats": {"str": 10, "dex": 12}}, prompt_hints=[]
        )
        assert schema.validate_record({"stats": {"str": 14}}) is False

    def test_validate_record_fails_when_nested_dict_is_scalar(self):
        schema = ContextSchema(
            description="t", sample={"stats": {"str": 10}}, prompt_hints=[]
        )
        assert schema.validate_record({"stats": "oops"}) is False

    def test_validate_record_passes_nested_list(self):
        schema = ContextSchema(
            description="t", sample={"items": ["a", "b"]}, prompt_hints=[]
        )
        assert schema.validate_record({"items": ["x", "y", "z"]}) is True

    def test_validate_record_fails_when_list_field_is_scalar(self):
        schema = ContextSchema(
            description="t", sample={"items": ["a"]}, prompt_hints=[]
        )
        assert schema.validate_record({"items": "not a list"}) is False

    def test_missing_fields_includes_nested_dotted_paths(self):
        schema = ContextSchema(
            description="t", sample={"stats": {"str": 10, "dex": 12}}, prompt_hints=[]
        )
        result = schema.missing_fields({"stats": {"str": 14}})
        assert result == ["stats.dex"]

    def test_missing_fields_reports_field_when_nested_dict_is_wrong_type(self):
        schema = ContextSchema(
            description="t", sample={"stats": {"str": 10}}, prompt_hints=[]
        )
        assert schema.missing_fields({"stats": "bad"}) == ["stats"]

    def test_missing_fields_reports_field_when_list_is_wrong_type(self):
        schema = ContextSchema(
            description="t", sample={"items": ["a"]}, prompt_hints=[]
        )
        assert schema.missing_fields({"items": "bad"}) == ["items"]

    def test_default_category_is_general(self):
        schema = ContextSchema(description="t", sample={"x": 1}, prompt_hints=[])
        assert schema.category == "general"

    def test_validate_record_non_dict_returns_false(self):
        schema = ContextSchema(
            description="t", sample={"x": 1}, prompt_hints=[]
        )
        assert schema.validate_record(42) is False

    def test_missing_fields_non_dict_returns_all(self):
        schema = ContextSchema(
            description="t", sample={"x": 1, "y": 2}, prompt_hints=[]
        )
        assert schema.missing_fields(42) == ["x", "y"]

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            ContextSchema(description="", sample={"x": 1}, prompt_hints=[])

    def test_non_string_description_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            ContextSchema(description=42, sample={"x": 1}, prompt_hints=[])  # type: ignore[arg-type]

    def test_non_dict_sample_raises(self):
        with pytest.raises(ValueError, match="non-empty dict"):
            ContextSchema(description="t", sample=["a", "b"], prompt_hints=[])  # type: ignore[arg-type]

    def test_non_list_prompt_hints_raises(self):
        with pytest.raises(ValueError, match="'prompt_hints' must be a list"):
            ContextSchema(description="t", sample={"x": 1}, prompt_hints="hint")  # type: ignore[arg-type]

    def test_field_providers_none_is_valid(self):
        schema = ContextSchema(description="t", sample={"email": "x"}, prompt_hints=[], field_providers=None)
        assert schema.field_providers is None

    def test_field_providers_valid_spec(self):
        schema = ContextSchema(
            description="t",
            sample={"email": "x", "phone": "y"},
            prompt_hints=[],
            field_providers={"email": "faker:email", "phone": "faker:phone_number"},
        )
        assert schema.field_providers == {"email": "faker:email", "phone": "faker:phone_number"}

    def test_field_providers_bare_faker_raises(self):
        with pytest.raises(ValueError, match="faker:method_name"):
            ContextSchema(
                description="t", sample={"email": "x"}, prompt_hints=[],
                field_providers={"email": "faker"},
            )

    def test_field_providers_wrong_prefix_raises(self):
        with pytest.raises(ValueError, match="faker:method_name"):
            ContextSchema(
                description="t", sample={"email": "x"}, prompt_hints=[],
                field_providers={"email": "random:email"},
            )

    def test_field_providers_not_dict_raises(self):
        with pytest.raises(ValueError, match="'field_providers' must be a dict"):
            ContextSchema(
                description="t", sample={"email": "x"}, prompt_hints=[],
                field_providers="faker:email",  # type: ignore[arg-type]
            )


class TestContextSchemaUniqueFields:

    def test_unique_fields_none_is_default(self):
        schema = ContextSchema(
            description="t",
            sample={"email": "x"},
            prompt_hints=[],
            field_providers={"email": "faker:email"},
        )
        assert schema.unique_fields is None

    def test_unique_fields_valid_subset_of_field_providers(self):
        schema = ContextSchema(
            description="t",
            sample={"email": "x", "phone": "y"},
            prompt_hints=[],
            field_providers={"email": "faker:email", "phone": "faker:phone_number"},
            unique_fields=["email"],
        )
        assert schema.unique_fields == ["email"]

    def test_unique_fields_empty_list_is_valid(self):
        schema = ContextSchema(
            description="t",
            sample={"email": "x"},
            prompt_hints=[],
            field_providers={"email": "faker:email"},
            unique_fields=[],
        )
        assert schema.unique_fields == []

    def test_unique_fields_all_fp_keys_is_valid(self):
        schema = ContextSchema(
            description="t",
            sample={"email": "x", "phone": "y"},
            prompt_hints=[],
            field_providers={"email": "faker:email", "phone": "faker:phone_number"},
            unique_fields=["email", "phone"],
        )
        assert set(schema.unique_fields) == {"email", "phone"}

    def test_unique_fields_not_list_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            ContextSchema(
                description="t",
                sample={"email": "x"},
                prompt_hints=[],
                field_providers={"email": "faker:email"},
                unique_fields="email",  # type: ignore[arg-type]
            )

    def test_unique_fields_without_field_providers_raises(self):
        with pytest.raises(ValueError, match="requires 'field_providers'"):
            ContextSchema(
                description="t",
                sample={"email": "x"},
                prompt_hints=[],
                unique_fields=["email"],
            )

    def test_unique_fields_field_not_in_field_providers_raises(self):
        with pytest.raises(ValueError, match="not covered by 'field_providers'"):
            ContextSchema(
                description="t",
                sample={"email": "x", "name": "y"},
                prompt_hints=[],
                field_providers={"email": "faker:email"},
                unique_fields=["name"],
            )

    def test_unique_fields_error_names_the_bad_field(self):
        with pytest.raises(ValueError, match="name"):
            ContextSchema(
                description="t",
                sample={"email": "x", "name": "y"},
                prompt_hints=[],
                field_providers={"email": "faker:email"},
                unique_fields=["name"],
            )

    def test_unique_fields_partial_invalid_raises(self):
        with pytest.raises(ValueError, match="not covered by 'field_providers'"):
            ContextSchema(
                description="t",
                sample={"email": "x", "phone": "y", "name": "z"},
                prompt_hints=[],
                field_providers={"email": "faker:email", "phone": "faker:phone_number"},
                unique_fields=["email", "name"],
            )


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
