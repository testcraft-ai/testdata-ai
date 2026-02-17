"""Tests for testdata_ai.cli — Click CLI commands via CliRunner."""

import csv
import io
import json
import os
from unittest.mock import patch, MagicMock

import click
import pytest

from testdata_ai.cli import cli, _flatten_dict, _records_to_csv, _apply_overrides, _adjust_max_tokens, _Spinner
from testdata_ai.contexts import CONTEXTS, ValidationError


class TestListContextsCmd:

    def test_lists_all_contexts(self, runner):
        result = runner.invoke(cli, ["list-contexts"])
        assert result.exit_code == 0
        assert "ecommerce_customer" in result.output
        assert "banking_user" in result.output

    def test_filter_by_category(self, runner):
        result = runner.invoke(cli, ["list-contexts", "--category", "finance"])
        assert result.exit_code == 0
        assert "banking_user" in result.output
        assert "ecommerce_customer" not in result.output

    def test_nonexistent_category_shows_empty(self, runner):
        result = runner.invoke(cli, ["list-contexts", "--category", "nope"])
        assert result.exit_code == 0
        assert "No contexts found" in result.output



class TestShowContextCmd:

    def test_shows_context_details(self, runner):
        result = runner.invoke(cli, ["show-context", "banking_user"])
        assert result.exit_code == 0
        assert "banking_user" in result.output
        assert "finance" in result.output
        assert "Fields:" in result.output
        assert "Sample record:" in result.output
        assert "Prompt hints:" in result.output

    def test_shows_all_fields(self, runner):
        result = runner.invoke(cli, ["show-context", "saas_trial"])
        assert result.exit_code == 0
        for field in CONTEXTS["saas_trial"].fields:
            assert field in result.output

    def test_unknown_context_errors(self, runner):
        result = runner.invoke(cli, ["show-context", "nonexistent"])
        assert result.exit_code != 0
        assert "Unknown context" in result.output



def _patch_generator(records=None, *, side_effect=None):
    """Return a context manager that patches TestDataGenerator.

    Args:
        records: If given, ``gen.generate()`` returns this list.
        side_effect: If given, ``gen.generate()`` raises or calls this instead.
    """
    mock_gen = MagicMock()
    if side_effect is not None:
        mock_gen.generate.side_effect = side_effect
    else:
        mock_gen.generate.return_value = records
    mock_gen.config = MagicMock(
        provider="openai", model="test-model", max_tokens=4096
    )
    mock_gen.provider = MagicMock(max_tokens=4096)

    return patch(
        "testdata_ai.cli.TestDataGenerator", return_value=mock_gen
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
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample]):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "-f", "csv", "-q"],
            )
        assert result.exit_code == 0
        reader = csv.reader(io.StringIO(result.output.strip()))
        rows = list(reader)
        assert len(rows) == 2  # header + 1 data row

    def test_generate_writes_to_file(self, runner, tmp_path):
        sample = CONTEXTS["saas_trial"].sample
        outfile = str(tmp_path / "out.json")
        with _patch_generator([sample]):
            result = runner.invoke(
                cli,
                ["generate", "--context", "saas_trial", "--count", "1", "-o", outfile, "-q"],
            )
        assert result.exit_code == 0
        with open(outfile) as f:
            data = json.load(f)
        assert len(data) == 1

    def test_generate_unknown_context_errors(self, runner):
        result = runner.invoke(
            cli, ["generate", "--context", "nonexistent", "-q"]
        )
        assert result.exit_code != 0
        assert "Unknown context" in result.output

    def test_generate_requires_context(self, runner):
        result = runner.invoke(cli, ["generate"])
        assert result.exit_code != 0
        assert "Missing option" in result.output
        assert "'--context'" in result.output

    def test_generate_no_validate_flag(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample]) as mock_cls:
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "--no-validate", "-q"],
            )
        assert result.exit_code == 0
        mock_cls.return_value.generate.assert_called_once_with(
            "banking_user", count=1, validate=False
        )

    def test_generate_quiet_suppresses_status(self, runner):
        sample = CONTEXTS["banking_user"].sample
        with _patch_generator([sample]):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "-q"],
            )
        assert result.exit_code == 0
        # In quiet mode the only stdout should be the JSON data, no status text
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
            "testdata_ai.cli.TestDataGenerator",
            side_effect=ImportError("openai package is required"),
        ):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "-q"],
            )
        assert result.exit_code != 0
        assert "openai package is required" in result.output

    def test_generate_csv_to_file(self, runner, tmp_path):
        sample = CONTEXTS["banking_user"].sample
        outfile = str(tmp_path / "out.csv")
        with _patch_generator([sample]):
            result = runner.invoke(
                cli,
                [
                    "generate", "--context", "banking_user", "--count", "1",
                    "-f", "csv", "-o", outfile, "-q",
                ],
            )
        assert result.exit_code == 0
        with open(outfile) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 2  # header + 1 data row

    def test_generate_fewer_records_warns(self, runner):
        """Non-quiet mode: warning when fewer records returned than requested."""
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
        """ValueError from gen.generate (e.g. invalid JSON) is caught by _run_generation."""
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

    def test_generate_non_quiet_file_shows_saved(self, runner, tmp_path):
        sample = CONTEXTS["banking_user"].sample
        outfile = str(tmp_path / "out.json")
        with _patch_generator([sample]):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "-o", outfile],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert "Saved to" in result.output

    def test_generate_validation_error_from_generator(self, runner):
        """ValidationError (subclass of ValueError) is caught by _run_generation."""
        invalid = [{"record_index": 0, "missing_fields": ["email", "balance"]}]
        with _patch_generator(side_effect=ValidationError(invalid)):
            result = runner.invoke(
                cli,
                ["generate", "--context", "banking_user", "--count", "1", "-q"],
            )
        assert result.exit_code != 0
        assert "failed validation" in result.output

    def test_generate_with_provider_and_model_flags(self, runner, monkeypatch):
        """CLI --provider and --model flags are propagated to env vars."""
        monkeypatch.delenv("AI_PROVIDER", raising=False)
        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
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

    def test_generate_with_max_tokens_flag(self, runner, monkeypatch):
        """CLI --max-tokens flag is propagated to env vars."""
        monkeypatch.delenv("OPENAI_MAX_TOKENS", raising=False)
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

    def test_generate_with_temperature_flag(self, runner, monkeypatch):
        """CLI --temperature flag is propagated to env vars."""
        monkeypatch.delenv("OPENAI_TEMPERATURE", raising=False)
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


class TestVersion:

    def test_version_flag(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()



class TestFlattenDict:

    def test_flat_dict_unchanged(self):
        assert _flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested_dict(self):
        assert _flatten_dict({"a": {"b": 1}}) == {"a.b": 1}

    def test_deeply_nested(self):
        assert _flatten_dict({"a": {"b": {"c": 3}}}) == {"a.b.c": 3}

    def test_lists_become_json_strings(self):
        result = _flatten_dict({"tags": ["x", "y"]})
        assert result == {"tags": '["x", "y"]'}

    def test_mixed_nesting(self):
        d = {"name": "Test", "loc": {"city": "NYC", "zip": "10001"}, "tags": [1]}
        result = _flatten_dict(d)
        assert result == {
            "name": "Test",
            "loc.city": "NYC",
            "loc.zip": "10001",
            "tags": "[1]",
        }

    def test_empty_dict(self):
        assert _flatten_dict({}) == {}

    def test_none_value_preserved(self):
        assert _flatten_dict({"a": None}) == {"a": None}


class TestRecordsToCsv:

    def test_empty_records(self):
        assert _records_to_csv([]) == ""

    def test_single_flat_record(self):
        csv_text = _records_to_csv([{"a": 1, "b": 2}])
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["a"] == "1"
        assert rows[0]["b"] == "2"

    def test_nested_records_flattened(self):
        records = [{"name": "A", "loc": {"city": "NYC"}}]
        csv_text = _records_to_csv(records)
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        assert rows[0]["loc.city"] == "NYC"

    def test_superset_fieldnames_across_records(self):
        records = [{"a": 1}, {"a": 2, "b": 3}]
        csv_text = _records_to_csv(records)
        reader = csv.DictReader(io.StringIO(csv_text))
        fieldnames = reader.fieldnames
        assert "a" in fieldnames
        assert "b" in fieldnames

    def test_disjoint_keys_across_records(self):
        records = [{"a": 1}, {"b": 2}]
        csv_text = _records_to_csv(records)
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == 2
        assert set(reader.fieldnames) == {"a", "b"}
        assert rows[0]["a"] == "1"
        assert rows[0]["b"] == ""
        assert rows[1]["a"] == ""
        assert rows[1]["b"] == "2"



@pytest.mark.usefixtures("clean_ai_env_fixture")
class TestApplyOverrides:

    def _run(self, provider, model, max_tokens, temperature, monkeypatch):
        """
        Apply overrides for the provider/model and ensure any newly added 
        environment variables are registered for cleanup.
        """
        # Snapshot existing environment variable keys
        existing_env_keys = set(os.environ)

        # Apply overrides (may add new environment variables)
        _apply_overrides(provider, model, max_tokens, temperature)

        # Find new environment variables and register them with monkeypatch
        new_env_keys = set(os.environ) - existing_env_keys
        for key in new_env_keys:
            monkeypatch.setenv(key, os.environ[key])


    def test_sets_provider_env(self, monkeypatch):
        self._run("anthropic", None, None, None, monkeypatch)
        assert os.environ["AI_PROVIDER"] == "anthropic"

    def test_sets_model_for_default_provider(self, monkeypatch):
        self._run(None, "gpt-4o", None, None, monkeypatch)
        assert os.environ["OPENAI_MODEL"] == "gpt-4o"

    def test_sets_model_for_explicit_provider(self, monkeypatch):
        self._run("anthropic", "claude-sonnet", None, None, monkeypatch)
        assert os.environ["ANTHROPIC_MODEL"] == "claude-sonnet"

    def test_sets_max_tokens(self, monkeypatch):
        self._run(None, None, 8192, None, monkeypatch)
        assert os.environ["OPENAI_MAX_TOKENS"] == "8192"

    def test_sets_temperature(self, monkeypatch):
        self._run(None, None, None, 0.5, monkeypatch)
        assert os.environ["OPENAI_TEMPERATURE"] == "0.5"

    def test_skips_none_values(self, monkeypatch):
        self._run(None, None, None, None, monkeypatch)
        assert "AI_PROVIDER" not in os.environ
        assert "OPENAI_MODEL" not in os.environ
        assert "OPENAI_MAX_TOKENS" not in os.environ
        assert "OPENAI_TEMPERATURE" not in os.environ

    def test_resolves_provider_from_existing_env(self, monkeypatch):
        """When provider arg is None, resolves prefix from AI_PROVIDER env var."""
        monkeypatch.setenv("AI_PROVIDER", "anthropic")
        self._run(None, "claude-sonnet", None, None, monkeypatch)
        assert os.environ["ANTHROPIC_MODEL"] == "claude-sonnet"


class TestAdjustMaxTokens:

    def test_no_adjustment_when_within_limit(self, mock_generator, mock_context_schema):
        gen = mock_generator
        gen.config.max_tokens = 4096
        gen.provider.max_tokens = 4096
        _adjust_max_tokens(gen, mock_context_schema, count=1, quiet=True)
        assert gen.config.max_tokens == 4096

    def test_quiet_mode_auto_increases(self, mock_generator, mock_context_schema):
        gen = mock_generator
        gen.config.max_tokens = 100
        gen.provider.max_tokens = 100
        _adjust_max_tokens(gen, mock_context_schema, count=500, quiet=True)
        assert gen.config.max_tokens > 100
        assert gen.provider.max_tokens == gen.config.max_tokens

    def test_interactive_increase(self, mock_generator, mock_context_schema):
        with patch("testdata_ai.cli.click.prompt", return_value="increase"), \
             patch("testdata_ai.cli.click.echo"):
            gen = mock_generator
            gen.config.max_tokens = 100
            gen.provider.max_tokens = 100
            _adjust_max_tokens(gen, mock_context_schema, count=500, quiet=False)
            assert gen.config.max_tokens > 100
            assert gen.provider.max_tokens == gen.config.max_tokens

    def test_interactive_continue_keeps_original(self, mock_generator, mock_context_schema):
        with patch("testdata_ai.cli.click.prompt", return_value="continue"), \
             patch("testdata_ai.cli.click.echo"):
            gen = mock_generator
            gen.config.max_tokens = 100
            gen.provider.max_tokens = 100
            _adjust_max_tokens(gen, mock_context_schema, count=500, quiet=False)
            assert gen.config.max_tokens == 100

    def test_interactive_cancel_aborts(self, mock_generator, mock_context_schema):
        with patch("testdata_ai.cli.click.prompt", return_value="cancel"), \
             patch("testdata_ai.cli.click.echo"):
            gen = mock_generator
            gen.config.max_tokens = 100
            gen.provider.max_tokens = 100
            with pytest.raises(click.Abort):
                _adjust_max_tokens(gen, mock_context_schema, count=500, quiet=False)


class TestSpinner:

    def test_silent_mode_no_thread(self):
        spinner = _Spinner("testing", silent=True)
        with spinner:
            assert spinner._thread is None

    def test_non_silent_starts_and_stops_thread(self):
        spinner = _Spinner("testing", silent=False)
        with spinner:
            assert spinner._thread is not None
            assert spinner._thread.is_alive()
        assert not spinner._thread.is_alive()

    def test_spin_writes_to_stderr(self, capsys):
        """Exercise the _spin loop body: wait for at least one frame to render."""
        import time
        spinner = _Spinner("working", silent=False)
        with spinner:
            time.sleep(0.25)  # let at least one animation frame render
        captured = capsys.readouterr()
        assert "working" in captured.err
