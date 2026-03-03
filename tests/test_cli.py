"""Tests for testdata_ai.cli — Click CLI commands via CliRunner."""

import csv
import io
import json
from unittest.mock import patch, MagicMock

from testdata_ai.cli import cli, _flatten_dict, _records_to_csv, _adjust_max_tokens, _Spinner
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
    """Return a context manager that patches DataGenerator.

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

    def test_generate_with_provider_and_model_flags(self, runner):
        """CLI --provider and --model flags are passed directly to DataGenerator."""
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
        """CLI --max-tokens flag is passed directly to DataGenerator."""
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
        """CLI --temperature flag is passed directly to DataGenerator."""
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




class TestAdjustMaxTokens:

    def test_no_adjustment_when_within_limit(self, mock_generator, mock_context_schema):
        gen = mock_generator
        gen.config.max_tokens = 4096
        gen.provider.max_tokens = 4096
        _adjust_max_tokens(gen, mock_context_schema, count=1, quiet=True, user_set=False)
        assert gen.config.max_tokens == 4096

    def test_quiet_mode_auto_increases(self, mock_generator, mock_context_schema):
        gen = mock_generator
        gen.config.max_tokens = 100
        _adjust_max_tokens(gen, mock_context_schema, count=500, quiet=True, user_set=False)
        gen.set_max_tokens.assert_called_once()
        called_value = gen.set_max_tokens.call_args[0][0]
        assert called_value > 100

    def test_non_quiet_auto_increases_and_echoes(self, mock_generator, mock_context_schema):
        gen = mock_generator
        gen.provider.max_tokens = 100
        with patch("testdata_ai.cli.click.echo") as mock_echo:
            _adjust_max_tokens(gen, mock_context_schema, count=500, quiet=False, user_set=False)
        gen.set_max_tokens.assert_called_once()
        assert gen.set_max_tokens.call_args[0][0] > 100
        mock_echo.assert_called_once()


class TestSpinner:

    def test_silent_mode_no_output(self, capsys):
        with _Spinner("testing", silent=True):
            pass
        assert capsys.readouterr().err == ""

    def test_non_silent_writes_start_and_done(self, capsys):
        with _Spinner("working", silent=False):
            pass
        err = capsys.readouterr().err
        assert "working" in err
        assert "Done" in err

    def test_elapsed_time_shown(self, capsys):
        with _Spinner("task", silent=False):
            pass
        assert "s)" in capsys.readouterr().err
