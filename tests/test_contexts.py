"""Tests for testdata_ai.contexts — schemas, listing, and validation."""

import json

import pytest

from testdata_ai.contexts import (
    CONTEXTS,
    ContextSchema,
    ValidationError,
    get_context_schema,
    list_contexts,
    load_contexts_from_file,
    register_context,
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

    # --- deep (nested) validation ---

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



class TestBuiltinContexts:
    """Verify every built-in context is well-formed."""

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
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            register_context("_test_warn_nested", schema)  # must not raise

    def test_overwrite_builtin_warns(self):
        schema = ContextSchema(description="d", sample={"x": 1}, prompt_hints=["hint"])
        with pytest.warns(UserWarning, match="shadows a built-in"):
            register_context("ecommerce_customer", schema, overwrite=True)


@pytest.mark.usefixtures("clean_contexts")
class TestLoadContextsFromFile:

    @pytest.fixture()
    def _require_yaml(self):
        pytest.importorskip("yaml")

    _VALID_DATA = {
        "my_widget": {
            "description": "widget records",
            "category": "test",
            "sample": {"widget_id": "W-1", "name": "Widget Alpha", "price": 9.99},
            "prompt_hints": ["realistic names"],
        }
    }

    def test_load_yaml(self, tmp_path, _require_yaml):
        import yaml
        f = tmp_path / "ctx.yaml"
        f.write_text(yaml.dump(self._VALID_DATA))
        names = load_contexts_from_file(f)
        assert names == ["my_widget"]
        schema = get_context_schema("my_widget")
        assert schema.description == "widget records"
        assert set(schema.fields) == {"widget_id", "name", "price"}

    def test_load_yml_extension(self, tmp_path, _require_yaml):
        import yaml
        f = tmp_path / "ctx.yml"
        f.write_text(yaml.dump(self._VALID_DATA))
        names = load_contexts_from_file(f)
        assert "my_widget" in names

    def test_load_json(self, tmp_path):
        f = tmp_path / "ctx.json"
        f.write_text(json.dumps(self._VALID_DATA))
        names = load_contexts_from_file(f)
        assert names == ["my_widget"]

    def test_returns_list_of_registered_names(self, tmp_path, _require_yaml):
        import yaml
        data = {
            "ctx_a": {"description": "a", "sample": {"x": 1}, "prompt_hints": ["hint"]},
            "ctx_b": {"description": "b", "sample": {"y": 2}, "prompt_hints": ["hint"]},
        }
        f = tmp_path / "two.yaml"
        f.write_text(yaml.dump(data))
        names = load_contexts_from_file(f)
        assert set(names) == {"ctx_a", "ctx_b"}

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "ctx.toml"
        f.write_text("")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_contexts_from_file(f)

    def test_non_mapping_top_level_raises(self, tmp_path, _require_yaml):
        import yaml
        f = tmp_path / "bad.yaml"
        f.write_text(yaml.dump(["item1", "item2"]))
        with pytest.raises(ValueError, match="top-level mapping"):
            load_contexts_from_file(f)

    def test_entry_not_dict_raises(self, tmp_path, _require_yaml):
        import yaml
        f = tmp_path / "bad.yaml"
        f.write_text(yaml.dump({"my_ctx": "not_a_dict"}))
        with pytest.raises(ValueError, match="must be a mapping"):
            load_contexts_from_file(f)

    def test_duplicate_raises_without_overwrite(self, tmp_path, _require_yaml):
        import yaml
        f = tmp_path / "ctx.yaml"
        f.write_text(yaml.dump(self._VALID_DATA))
        load_contexts_from_file(f)
        with pytest.raises(ValueError, match="already registered"):
            load_contexts_from_file(f)

    def test_overwrite_replaces_existing(self, tmp_path, _require_yaml):
        import yaml
        f = tmp_path / "ctx.yaml"
        f.write_text(yaml.dump(self._VALID_DATA))
        load_contexts_from_file(f)
        updated = {
            "my_widget": {**self._VALID_DATA["my_widget"], "description": "updated"}
        }
        f.write_text(yaml.dump(updated))
        load_contexts_from_file(f, overwrite=True)
        assert get_context_schema("my_widget").description == "updated"

    def test_string_path_accepted(self, tmp_path, _require_yaml):
        import yaml
        f = tmp_path / "ctx.yaml"
        f.write_text(yaml.dump(self._VALID_DATA))
        names = load_contexts_from_file(str(f))
        assert "my_widget" in names

    def test_atomic_rollback_on_partial_failure(self, tmp_path, _require_yaml):
        """If one entry is invalid, no contexts from the file should be registered."""
        import yaml
        data = {
            "ctx_good": {"description": "ok", "sample": {"x": 1}, "prompt_hints": []},
            "ctx_bad": {"description": "missing sample and prompt_hints"},
        }
        f = tmp_path / "mixed.yaml"
        f.write_text(yaml.dump(data))
        with pytest.raises(ValueError):
            load_contexts_from_file(f)
        assert "ctx_good" not in list_contexts()

    def test_empty_sample_in_file_raises(self, tmp_path, _require_yaml):
        import yaml
        data = {"ctx_a": {"description": "d", "sample": {}, "prompt_hints": []}}
        f = tmp_path / "bad.yaml"
        f.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match="non-empty dict"):
            load_contexts_from_file(f)

    def test_invalid_context_name_in_file_raises(self, tmp_path, _require_yaml):
        import yaml
        data = {"bad name": {"description": "d", "sample": {"a": 1}, "prompt_hints": []}}
        f = tmp_path / "bad.yaml"
        f.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match="Invalid context name"):
            load_contexts_from_file(f)

    def test_malformed_yaml_raises(self, tmp_path):
        pytest.importorskip("yaml")
        f = tmp_path / "bad.yaml"
        f.write_text("ctx_a: [unclosed")
        with pytest.raises(ValueError, match="Malformed YAML"):
            load_contexts_from_file(f)

    def test_malformed_json_raises(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text('{"ctx_a": {broken}')
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_contexts_from_file(f)

    def test_duplicate_yaml_keys_raises(self, tmp_path):
        pytest.importorskip("yaml")
        f = tmp_path / "dup.yaml"
        # yaml.dump would deduplicate; write raw text to preserve duplicate keys
        f.write_text("ctx_a:\n  description: first\nctx_a:\n  description: second\n")
        with pytest.raises(ValueError, match="duplicate key"):
            load_contexts_from_file(f)

    def test_duplicate_json_keys_raises(self, tmp_path):
        f = tmp_path / "dup.json"
        f.write_text('{"ctx_a": {"description": "d"}, "ctx_a": {"description": "d2"}}')
        with pytest.raises(ValueError, match="duplicate key"):
            load_contexts_from_file(f)


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
