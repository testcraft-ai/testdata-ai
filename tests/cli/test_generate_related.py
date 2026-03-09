"""Tests for the 'generate-related' CLI command."""

import json

import pytest
import yaml
from unittest.mock import MagicMock, patch

from testdata_ai.cli import cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USERS = [{"email": "alice@x.com", "name": "Alice", "age": 30}]
_ORDERS = [{"order_id": "O1", "amount": 99.99, "status": "pending", "user_id": "alice@x.com"}]

_SIMPLE_RESULT = {"users": _USERS, "orders": _ORDERS}

_SIMPLE_GRAPH = {
    "users": {"context": "ecommerce_customer", "count": 1},
    "orders": {
        "context": "restaurant_order",
        "count": 1,
        "parent": "users",
        "fk_field": "user_id",
        "parent_pk": "email",
    },
}


def _patch_related_generator(result=None, *, side_effect=None):
    """Patch DataGenerator so generate_with_relationships returns result."""
    mock_gen = MagicMock()
    if side_effect is not None:
        mock_gen.generate_with_relationships.side_effect = side_effect
    else:
        mock_gen.generate_with_relationships.return_value = result or {}
    mock_gen.config = MagicMock(provider="openai", model="test-model", max_tokens=4096)
    mock_gen.provider = MagicMock(max_tokens=4096)
    return patch("testdata_ai.cli.DataGenerator", return_value=mock_gen)


def _graph_yaml_file(tmp_path, graph=None):
    f = tmp_path / "graph.yaml"
    f.write_text(yaml.dump(graph or _SIMPLE_GRAPH))
    return str(f)


def _graph_json_file(tmp_path, graph=None):
    f = tmp_path / "graph.json"
    f.write_text(json.dumps(graph or _SIMPLE_GRAPH))
    return str(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateRelatedCmd:

    def test_requires_graph_file(self, runner):
        result = runner.invoke(cli, ["generate-related"])
        assert result.exit_code != 0
        assert "graph-file" in result.output or "Missing option" in result.output

    def test_nonexistent_graph_file_errors(self, runner, tmp_path):
        result = runner.invoke(
            cli, ["generate-related", "--graph-file", str(tmp_path / "nope.yaml"), "-q"]
        )
        assert result.exit_code != 0

    def test_valid_yaml_produces_json_output(self, runner, tmp_path):
        graph_file = _graph_yaml_file(tmp_path)
        with _patch_related_generator(_SIMPLE_RESULT):
            result = runner.invoke(cli, ["generate-related", "--graph-file", graph_file, "-q"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert set(data.keys()) == {"users", "orders"}

    def test_valid_json_produces_json_output(self, runner, tmp_path):
        graph_file = _graph_json_file(tmp_path)
        with _patch_related_generator(_SIMPLE_RESULT):
            result = runner.invoke(cli, ["generate-related", "--graph-file", graph_file, "-q"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "users" in data and "orders" in data

    def test_default_output_format_is_json(self, runner, tmp_path):
        graph_file = _graph_yaml_file(tmp_path)
        with _patch_related_generator(_SIMPLE_RESULT):
            result = runner.invoke(cli, ["generate-related", "--graph-file", graph_file, "-q"])
        assert result.exit_code == 0
        # Must be parseable as a single JSON object
        data = json.loads(result.output)
        assert isinstance(data, dict)

    def test_jsonl_format(self, runner, tmp_path):
        graph_file = _graph_yaml_file(tmp_path)
        with _patch_related_generator(_SIMPLE_RESULT):
            result = runner.invoke(
                cli,
                ["generate-related", "--graph-file", graph_file, "-o", "jsonl", "-q"],
            )
        assert result.exit_code == 0
        lines = [l for l in result.output.strip().splitlines() if l]
        assert len(lines) == 2
        parsed = [json.loads(l) for l in lines]
        entity_names = {p["entity"] for p in parsed}
        assert entity_names == {"users", "orders"}
        for p in parsed:
            assert "records" in p
            assert isinstance(p["records"], list)

    def test_yaml_format(self, runner, tmp_path):
        graph_file = _graph_yaml_file(tmp_path)
        with _patch_related_generator(_SIMPLE_RESULT):
            result = runner.invoke(
                cli,
                ["generate-related", "--graph-file", graph_file, "-o", "yaml", "-q"],
            )
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert set(data.keys()) == {"users", "orders"}

    def test_csv_format(self, runner, tmp_path):
        graph_file = _graph_yaml_file(tmp_path)
        with _patch_related_generator(_SIMPLE_RESULT):
            result = runner.invoke(
                cli,
                ["generate-related", "--graph-file", graph_file, "-o", "csv", "-q"],
            )
        assert result.exit_code == 0
        assert "# entity: users" in result.output
        assert "# entity: orders" in result.output
        assert "email" in result.output
        assert "order_id" in result.output

    def test_sql_format(self, runner, tmp_path):
        graph_file = _graph_yaml_file(tmp_path)
        with _patch_related_generator(_SIMPLE_RESULT):
            result = runner.invoke(
                cli,
                ["generate-related", "--graph-file", graph_file, "-o", "sql", "-q"],
            )
        assert result.exit_code == 0
        assert "users" in result.output
        assert "orders" in result.output
        assert "INSERT INTO" in result.output

    def test_quiet_output_is_only_json(self, runner, tmp_path):
        """With -q, output should be valid JSON with no extra status lines."""
        graph_file = _graph_yaml_file(tmp_path)
        with _patch_related_generator(_SIMPLE_RESULT):
            result = runner.invoke(
                cli, ["generate-related", "--graph-file", graph_file, "-q"],
            )
        assert result.exit_code == 0
        # Output must be parseable as JSON (no status text mixed in)
        data = json.loads(result.output)
        assert isinstance(data, dict)

    def test_non_quiet_shows_status_messages(self, runner, tmp_path):
        """Without -q, status messages (entity names or counts) appear in output."""
        graph_file = _graph_yaml_file(tmp_path)
        with _patch_related_generator(_SIMPLE_RESULT):
            result = runner.invoke(
                cli, ["generate-related", "--graph-file", graph_file],
            )
        assert result.exit_code == 0
        # Status messages include entity names
        assert "users" in result.output or "Generated" in result.output

    def test_value_error_shows_error(self, runner, tmp_path):
        """ValueError from generate_with_relationships shows as click error."""
        graph_file = _graph_yaml_file(tmp_path)
        with _patch_related_generator(side_effect=ValueError("Cycle detected")):
            result = runner.invoke(cli, ["generate-related", "--graph-file", graph_file, "-q"])
        assert result.exit_code != 0
        assert "Cycle detected" in result.output

    def test_malformed_graph_file_shows_error(self, runner, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("- this is a list\n- not a dict\n")
        result = runner.invoke(cli, ["generate-related", "--graph-file", str(bad_file), "-q"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_provider_flag_forwarded(self, runner, tmp_path):
        graph_file = _graph_yaml_file(tmp_path)
        with _patch_related_generator(_SIMPLE_RESULT) as mock_cls:
            runner.invoke(
                cli,
                ["generate-related", "--graph-file", graph_file, "--provider", "anthropic", "-q"],
            )
        _, kwargs = mock_cls.call_args
        assert kwargs.get("provider") == "anthropic"

    def test_locale_flag_forwarded(self, runner, tmp_path):
        graph_file = _graph_yaml_file(tmp_path)
        with _patch_related_generator(_SIMPLE_RESULT) as mock_cls:
            runner.invoke(
                cli,
                ["generate-related", "--graph-file", graph_file, "--locale", "pl", "-q"],
            )
        _, kwargs = mock_cls.call_args
        assert kwargs.get("locale") == "pl"

    def test_no_validate_flag_forwarded(self, runner, tmp_path):
        graph_file = _graph_yaml_file(tmp_path)
        with _patch_related_generator(_SIMPLE_RESULT) as mock_cls:
            runner.invoke(
                cli,
                ["generate-related", "--graph-file", graph_file, "--no-validate", "-q"],
            )
        mock_gen = mock_cls.return_value
        _, kwargs = mock_gen.generate_with_relationships.call_args
        assert kwargs.get("validate") is False

    def test_graph_passed_to_generator(self, runner, tmp_path):
        """The parsed graph dict is forwarded to generate_with_relationships."""
        graph_file = _graph_yaml_file(tmp_path)
        with _patch_related_generator(_SIMPLE_RESULT) as mock_cls:
            runner.invoke(cli, ["generate-related", "--graph-file", graph_file, "-q"])
        mock_gen = mock_cls.return_value
        call_args = mock_gen.generate_with_relationships.call_args
        graph_arg = call_args[0][0]
        assert "users" in graph_arg
        assert "orders" in graph_arg

    def test_batch_size_injected_into_graph_nodes(self, runner, tmp_path):
        """--batch-size is applied as default batch_size to all graph nodes."""
        graph_file = _graph_yaml_file(tmp_path)
        with _patch_related_generator(_SIMPLE_RESULT) as mock_cls:
            runner.invoke(
                cli,
                ["generate-related", "--graph-file", graph_file, "--batch-size", "3", "-q"],
            )
        graph_arg = mock_cls.return_value.generate_with_relationships.call_args[0][0]
        assert graph_arg["users"]["batch_size"] == 3
        assert graph_arg["orders"]["batch_size"] == 3
