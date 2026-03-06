"""Tests for testdata_ai.cli — generate command and --context-file option."""

import json
import pytest
from unittest.mock import patch, MagicMock

from testdata_ai.cli import cli
from testdata_ai.contexts import CONTEXTS, ValidationError


def _patch_generator(records=None, *, side_effect=None):
    """Return a context manager that patches DataGenerator.

    Args:
        records: If given, ``gen.stream_generate()`` yields ``[records]`` (one batch).
        side_effect: If given, both ``generate`` and ``stream_generate`` raise/call this.
    """
    mock_gen = MagicMock()
    if side_effect is not None:
        mock_gen.generate.side_effect = side_effect
        mock_gen.generate_batched.side_effect = side_effect
    else:
        mock_gen.generate.return_value = records or []
        mock_gen.generate_batched.return_value = iter([records or []])
    mock_gen.config = MagicMock(
        provider="openai", model="test-model", max_tokens=4096
    )
    mock_gen.provider = MagicMock(max_tokens=4096)

    return patch(
        "testdata_ai.cli.DataGenerator", return_value=mock_gen
    )


class TestGenerateCmd:

    def test_generate_json_to_stdout(self, runner):
        sample = CONTEXTS["ecommerce_customer"].sample
        with _patch_generator([sample]):
            result = runner.invoke(
                cli, ["generate", "--context", "ecommerce_customer", "--count", "1", "-q"]
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_generate_csv_to_stdout(self, runner):
        import csv, io
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample]):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "-o", "csv", "-q"],
            )
        assert result.exit_code == 0
        reader = csv.reader(io.StringIO(result.output.strip()))
        rows = list(reader)
        assert len(rows) == 2  # header + 1 data row

    def test_generate_sql_to_stdout(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample]):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "-o", "sql", "-q"],
            )
        assert result.exit_code == 0
        assert "CREATE TABLE IF NOT EXISTS banking_user" in result.output
        assert "INSERT INTO banking_user VALUES" in result.output

    def test_generate_sql_custom_table(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample]):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "-o", "sql", "--table", "users", "-q"],
            )
        assert result.exit_code == 0
        assert "CREATE TABLE IF NOT EXISTS users" in result.output
        assert "INSERT INTO users VALUES" in result.output

    def test_generate_jsonl_to_stdout(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample, sample]):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "2", "-o", "jsonl", "-q"],
            )
        assert result.exit_code == 0
        lines = result.output.strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            assert isinstance(json.loads(line), dict)

    def test_generate_yaml_to_stdout(self, runner):
        import yaml
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample]):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "-o", "yaml", "-q"],
            )
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_generate_with_locale(self, runner):
        sample = CONTEXTS["ecommerce_customer"].sample
        with patch("testdata_ai.cli.DataGenerator") as mock_cls:
            mock_gen = MagicMock()
            mock_gen.generate_batched.return_value = iter([[sample]])
            mock_gen.config = MagicMock(provider="openai", model="test-model", max_tokens=4096)
            mock_gen.provider = MagicMock(max_tokens=4096)
            mock_cls.return_value = mock_gen
            result = runner.invoke(
                cli,
                ["generate", "--context", "ecommerce_customer", "--count", "1",
                 "--locale", "pl", "-q"],
            )
        assert result.exit_code == 0
        _, kwargs = mock_cls.call_args
        assert kwargs.get("locale") == "pl"

    def test_generate_unknown_context_errors(self, runner):
        result = runner.invoke(
            cli, ["generate", "--context", "nonexistent", "-q"]
        )
        assert result.exit_code != 0
        assert "Unknown context" in result.output

    def test_generate_requires_context_or_schema_file(self, runner):
        result = runner.invoke(cli, ["generate"])
        assert result.exit_code != 0
        assert "--context or --schema-file" in result.output

    def test_generate_context_and_schema_file_are_mutually_exclusive(self, runner, tmp_path):
        schema_file = tmp_path / "s.json"
        schema_file.write_text('{"properties": {"x": {"type": "string"}}}')
        result = runner.invoke(
            cli, ["generate", "--context", "ecommerce_customer", "--schema-file", str(schema_file)]
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_generate_no_validate_flag(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample]) as mock_cls:
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "--no-validate", "-q"],
            )
        assert result.exit_code == 0
        mock_cls.return_value.generate_batched.assert_called_once_with(
            "banking_user", 1, 10, validate=False
        )

    def test_generate_quiet_suppresses_status(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample]):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "-q"],
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_generate_api_runtime_error(self, runner):
        with _patch_generator(side_effect=RuntimeError("API connection refused")):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "-q"],
            )
        assert result.exit_code != 0
        assert "API error" in result.output

    def test_generate_import_error_missing_provider(self, runner):
        with patch(
            "testdata_ai.cli.DataGenerator",
            side_effect=ImportError("openai package is required"),
        ):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "-q"],
            )
        assert result.exit_code != 0
        assert "openai package is required" in result.output

    def test_generate_fewer_records_warns(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample]):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "5"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert "Warning: Requested 5 records but received 1" in result.output

    def test_generate_invalid_json_from_ai(self, runner):
        with _patch_generator(side_effect=ValueError("AI response is not valid JSON: oops")):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "-q"],
            )
        assert result.exit_code != 0
        assert "not valid JSON" in result.output

    def test_generate_non_quiet_shows_success(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample]):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert "Generated 1 records" in result.output

    def test_generate_validation_error_from_generator(self, runner):
        invalid = [{"record_index": 0, "missing_fields": ["email", "balance"]}]
        with _patch_generator(side_effect=ValidationError(invalid)):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "-q"],
            )
        assert result.exit_code != 0
        assert "failed validation" in result.output

    def test_generate_with_provider_and_model_flags(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample]):
            result = runner.invoke(
                cli,
                [
                    "generate", "--context", "banking_user", "--count", "1",
                    "--provider", "anthropic", "--model", "claude-sonnet",
                    "-q",
                ],
            )
        assert result.exit_code == 0

    def test_generate_with_max_tokens_flag(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample]):
            result = runner.invoke(
                cli,
                [
                    "generate", "--context", "banking_user", "--count", "1",
                    "--max-tokens", "8192", "-q",
                ],
            )
        assert result.exit_code == 0

    def test_generate_with_temperature_flag(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample]):
            result = runner.invoke(
                cli,
                [
                    "generate", "--context", "banking_user", "--count", "1",
                    "--temperature", "0.3", "-q",
                ],
            )
        assert result.exit_code == 0


class TestStreaming:

    def test_jsonl_emits_one_line_per_record(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample, sample]):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "2",
                 "--batch-size", "10", "-o", "jsonl", "-q"],
            )
        assert result.exit_code == 0
        lines = [l for l in result.output.strip().splitlines() if l]
        assert len(lines) == 2
        for line in lines:
            assert isinstance(json.loads(line), dict)

    def test_jsonl_multi_batch_emits_all_records(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample, sample]) as mock_cls:
            mock_cls.return_value.generate_batched.return_value = iter(
                [[sample, sample], [sample, sample]]
            )
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "4",
                 "--batch-size", "2", "-o", "jsonl", "-q"],
            )
        assert result.exit_code == 0
        lines = [l for l in result.output.strip().splitlines() if l]
        assert len(lines) == 4

    def test_json_format_accumulates_and_outputs_at_end(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample, sample]):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "2",
                 "--batch-size", "10", "-o", "json", "-q"],
            )
        assert result.exit_code == 0
        records = json.loads(result.output)
        assert isinstance(records, list)
        assert len(records) == 2

    def test_progress_shown_for_multiple_batches(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample]) as mock_cls:
            mock_cls.return_value.generate_batched.return_value = iter(
                [[sample] * 10, [sample] * 5]
            )
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "15",
                 "--batch-size", "10", "-o", "json"],
            )
        assert result.exit_code == 0
        assert "Batch 1/2" in result.output

    def test_run_streaming_returns_all_records(self):
        from testdata_ai.cli import _run_streaming
        sample = CONTEXTS["banking_user"].sample
        mock_gen = MagicMock()
        mock_gen.generate_batched.return_value = iter([[sample, sample], [sample]])
        mock_gen.config = MagicMock(provider="openai", model="test-model")

        records = _run_streaming(mock_gen, "banking_user", 3, 2, "json", False, True)
        assert len(records) == 3

    def test_batch_size_default_is_10(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample]) as mock_cls:
            runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "-q"],
            )
        mock_cls.return_value.generate_batched.assert_called_once_with(
            "banking_user", 1, 10, validate=True
        )


@pytest.mark.usefixtures("clean_contexts")
class TestContextFileCLI:
    """Tests for --context-file option across generate, list-contexts, show-context."""

    _CTX_DATA = {
        "cli_test_ctx": {
            "description": "CLI test context",
            "category": "test",
            "sample": {"id": "T-1", "label": "alpha"},
            "prompt_hints": ["realistic labels"],
        }
    }

    @pytest.fixture()
    def yaml_ctx_file(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        f = tmp_path / "ctx.yaml"
        f.write_text(yaml.dump(self._CTX_DATA))
        return str(f)

    def test_list_contexts_shows_custom_context(self, runner, yaml_ctx_file):
        result = runner.invoke(cli, ["list-contexts", "--context-file", yaml_ctx_file])
        assert result.exit_code == 0, result.output
        assert "cli_test_ctx" in result.output

    def test_show_context_with_context_file(self, runner, yaml_ctx_file):
        result = runner.invoke(cli, ["show-context", "cli_test_ctx", "--context-file", yaml_ctx_file])
        assert result.exit_code == 0, result.output
        assert "cli_test_ctx" in result.output
        assert "Fields:" in result.output

    def test_generate_with_context_file(self, runner, yaml_ctx_file):
        sample = self._CTX_DATA["cli_test_ctx"]["sample"]
        with _patch_generator([sample]):
            result = runner.invoke(
                cli,
                ["generate", "--context", "cli_test_ctx", "--context-file", yaml_ctx_file, "--count", "1", "-q"],
            )
        assert result.exit_code == 0, result.output

    def test_context_file_not_found_errors(self, runner):
        result = runner.invoke(cli, ["list-contexts", "--context-file", "/no/such/file.yaml"])
        assert result.exit_code != 0
        assert "--context-file" in result.output

    def test_context_file_invalid_content_errors(self, runner, tmp_path):
        yaml = pytest.importorskip("yaml")
        f = tmp_path / "bad.yaml"
        f.write_text(yaml.dump(["not", "a", "mapping"]))
        result = runner.invoke(cli, ["list-contexts", "--context-file", str(f)])
        assert result.exit_code != 0
        assert "--context-file" in result.output

    def test_context_file_yaml_missing_pyyaml_shows_friendly_error(self, runner, tmp_path):
        f = tmp_path / "ctx.yaml"
        f.write_text("ctx_a:\n  description: d\n  sample:\n    x: 1\n  prompt_hints: []\n")
        with patch.dict("sys.modules", {"yaml": None}):
            result = runner.invoke(cli, ["list-contexts", "--context-file", str(f)])
        assert result.exit_code != 0
        assert "--context-file" in result.output

    def test_malformed_yaml_shows_cli_error(self, runner, tmp_path):
        pytest.importorskip("yaml")
        f = tmp_path / "bad.yaml"
        f.write_text("ctx_a: [unclosed")
        result = runner.invoke(cli, ["list-contexts", "--context-file", str(f)])
        assert result.exit_code != 0
        assert "--context-file" in result.output

    def test_duplicate_yaml_keys_shows_cli_error(self, runner, tmp_path):
        pytest.importorskip("yaml")
        f = tmp_path / "dup.yaml"
        f.write_text("ctx_a:\n  description: first\nctx_a:\n  description: second\n")
        result = runner.invoke(cli, ["list-contexts", "--context-file", str(f)])
        assert result.exit_code != 0
        assert "--context-file" in result.output

    def test_oserror_shows_cli_error(self, runner, tmp_path):
        f = tmp_path / "ctx.json"
        f.write_text('{"ctx_x": {"description": "d", "sample": {"a": 1}, "prompt_hints": []}}')
        with patch("testdata_ai.cli.load_contexts_from_file", side_effect=OSError("Permission denied")):
            result = runner.invoke(cli, ["list-contexts", "--context-file", str(f)])
        assert result.exit_code != 0
        assert "--context-file" in result.output
