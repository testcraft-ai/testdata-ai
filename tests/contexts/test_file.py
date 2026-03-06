"""Tests for testdata_ai.contexts — load_contexts_from_file."""

import json
import pytest

from testdata_ai.contexts import (
    get_context_schema,
    list_contexts,
    load_contexts_from_file,
)


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
        f.write_text("ctx_a:\n  description: first\nctx_a:\n  description: second\n")
        with pytest.raises(ValueError, match="duplicate key"):
            load_contexts_from_file(f)

    def test_duplicate_json_keys_raises(self, tmp_path):
        f = tmp_path / "dup.json"
        f.write_text('{"ctx_a": {"description": "d"}, "ctx_a": {"description": "d2"}}')
        with pytest.raises(ValueError, match="duplicate key"):
            load_contexts_from_file(f)

    def test_overwrite_builtin_warns(self, tmp_path, clean_contexts):
        pytest.importorskip("yaml")
        import yaml
        data = {
            "ecommerce_customer": {
                "description": "custom override",
                "sample": {"id": "X"},
                "prompt_hints": ["test"],
            }
        }
        f = tmp_path / "override.yaml"
        f.write_text(yaml.dump(data))
        with pytest.warns(UserWarning, match="shadows a built-in"):
            load_contexts_from_file(f, overwrite=True)
