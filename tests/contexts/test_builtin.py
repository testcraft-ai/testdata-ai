"""Tests for testdata_ai.contexts — built-in contexts, lookup, and validation."""

import pytest

from testdata_ai.contexts import (
    CONTEXTS,
    ContextSchema,
    get_context_schema,
    list_contexts,
    validate_generated_data,
)


class TestBuiltinContexts:

    ALL_CONTEXT_NAMES = list(CONTEXTS.keys())

    def test_expected_builtin_context_count(self):
        assert len(CONTEXTS) == 13

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
        incomplete = {"name": "Test"}
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
        result = validate_generated_data("banking_user", ["not a dict"])
        assert len(result) == 1
        assert result[0]["record_index"] == 0
        assert len(result[0]["missing_fields"]) == len(CONTEXTS["banking_user"].fields)
