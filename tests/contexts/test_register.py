"""Tests for testdata_ai.contexts — register_context."""

import warnings
import pytest

from testdata_ai.contexts import (
    ContextSchema,
    get_context_schema,
    list_contexts,
    register_context,
)


@pytest.mark.usefixtures("clean_contexts")
class TestRegisterContext:

    def test_register_with_schema_instance(self):
        schema = ContextSchema(
            description="widgets", category="custom",
            sample={"id": "W-1", "name": "Widget"}, prompt_hints=["realistic names"],
        )
        register_context("_test_widget", schema)
        assert get_context_schema("_test_widget") is schema

    def test_register_with_dict(self):
        register_context("_test_dict_ctx", {
            "description": "dict-based context",
            "category": "test",
            "sample": {"foo": 1, "bar": "x"},
            "prompt_hints": ["hint"],
        })
        result = get_context_schema("_test_dict_ctx")
        assert result.description == "dict-based context"
        assert result.fields == ["foo", "bar"]

    def test_register_dict_defaults_category_to_custom(self):
        register_context("_test_cat", {
            "description": "no cat",
            "sample": {"a": 1},
            "prompt_hints": ["hint"],
        })
        assert get_context_schema("_test_cat").category == "custom"

    def test_duplicate_raises_without_overwrite(self):
        schema = ContextSchema(description="d", sample={"x": 1}, prompt_hints=["hint"])
        register_context("_test_dup", schema)
        with pytest.raises(ValueError, match="already registered"):
            register_context("_test_dup", schema)

    def test_duplicate_succeeds_with_overwrite(self):
        s1 = ContextSchema(description="v1", sample={"x": 1}, prompt_hints=["hint"])
        s2 = ContextSchema(description="v2", sample={"y": 2}, prompt_hints=["hint"])
        register_context("_test_over", s1)
        register_context("_test_over", s2, overwrite=True)
        assert get_context_schema("_test_over").description == "v2"

    def test_dict_missing_description_raises(self):
        with pytest.raises(ValueError, match="missing required keys"):
            register_context("_bad", {"sample": {"a": 1}, "prompt_hints": []})

    def test_dict_missing_sample_raises(self):
        with pytest.raises(ValueError, match="missing required keys"):
            register_context("_bad", {"description": "d", "prompt_hints": []})

    def test_dict_missing_prompt_hints_raises(self):
        with pytest.raises(ValueError, match="missing required keys"):
            register_context("_bad", {"description": "d", "sample": {"a": 1}})

    def test_dict_sample_not_dict_raises(self):
        with pytest.raises(ValueError, match="non-empty dict"):
            register_context("_bad", {
                "description": "d", "sample": ["a", "b"], "prompt_hints": [],
            })

    def test_dict_prompt_hints_not_list_raises(self):
        with pytest.raises(ValueError, match="'prompt_hints' must be a list"):
            register_context("_bad", {
                "description": "d", "sample": {"a": 1}, "prompt_hints": "hint",
            })

    def test_registered_context_appears_in_list_contexts(self):
        register_context("_test_list", ContextSchema(
            description="d", sample={"a": 1}, prompt_hints=["hint"],
        ))
        assert "_test_list" in list_contexts()

    def test_invalid_name_raises(self):
        schema = ContextSchema(description="d", sample={"a": 1}, prompt_hints=[])
        with pytest.raises(ValueError, match="Invalid context name"):
            register_context("", schema)

    def test_name_with_spaces_raises(self):
        schema = ContextSchema(description="d", sample={"a": 1}, prompt_hints=[])
        with pytest.raises(ValueError, match="Invalid context name"):
            register_context("bad name", schema)

    def test_name_starting_with_digit_raises(self):
        schema = ContextSchema(description="d", sample={"a": 1}, prompt_hints=[])
        with pytest.raises(ValueError, match="Invalid context name"):
            register_context("1bad", schema)

    def test_name_starting_with_underscore_accepted(self):
        schema = ContextSchema(description="d", sample={"a": 1}, prompt_hints=["hint"])
        register_context("_internal", schema)
        assert "_internal" in list_contexts()

    def test_name_with_hyphens_raises(self):
        schema = ContextSchema(description="d", sample={"a": 1}, prompt_hints=[])
        with pytest.raises(ValueError, match="Invalid context name"):
            register_context("my-context_v2", schema)

    def test_unknown_schema_type_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a ContextSchema or dict"):
            register_context("_bad_type", "not_a_schema")  # type: ignore[arg-type]

    def test_dict_empty_description_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            register_context("_bad", {"description": "", "sample": {"a": 1}, "prompt_hints": []})

    def test_dict_empty_sample_raises(self):
        with pytest.raises(ValueError, match="non-empty dict"):
            register_context("_bad", {"description": "d", "sample": {}, "prompt_hints": []})

    def test_empty_prompt_hints_warns(self):
        schema = ContextSchema(description="d", sample={"a": 1}, prompt_hints=[])
        with pytest.warns(UserWarning, match="prompt_hints.*empty"):
            register_context("_test_warn_hints", schema)

    def test_nested_sample_no_longer_warns(self):
        schema = ContextSchema(description="d", sample={"a": {"nested": 1}}, prompt_hints=["hint"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            register_context("_test_warn_nested", schema)  # must not raise

    def test_overwrite_builtin_warns(self):
        schema = ContextSchema(description="d", sample={"x": 1}, prompt_hints=["hint"])
        with pytest.warns(UserWarning, match="shadows a built-in"):
            register_context("ecommerce_customer", schema, overwrite=True)
