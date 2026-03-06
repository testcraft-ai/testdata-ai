"""Tests for testdata_ai.cli — listing and inspection commands."""

import pytest
from unittest.mock import patch, MagicMock

from testdata_ai.cli import cli
from testdata_ai.contexts import CONTEXTS


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


class TestVersion:

    def test_version_flag(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()


class TestListModelsCmd:

    def _patch_gen_with_ollama(self, models):
        from testdata_ai.ai_providers import OllamaProvider
        mock_gen = MagicMock()
        mock_gen.config.provider = "ollama"
        mock_gen.provider = MagicMock(spec=OllamaProvider)
        mock_gen.provider.list_models.return_value = models
        return patch("testdata_ai.cli.DataGenerator", return_value=mock_gen)

    def _patch_gen_non_ollama(self):
        mock_gen = MagicMock()
        mock_gen.config.provider = "openai"
        mock_gen.provider = MagicMock()
        mock_gen.provider.__class__.__name__ = "OpenAIProvider"
        return patch("testdata_ai.cli.DataGenerator", return_value=mock_gen)

    def test_lists_available_models(self, runner):
        with self._patch_gen_with_ollama(["llama3:8b", "qwen2.5:14b"]):
            result = runner.invoke(cli, ["list-models"])
        assert result.exit_code == 0
        assert "llama3:8b" in result.output
        assert "qwen2.5:14b" in result.output

    def test_no_models_shows_pull_hint(self, runner):
        with self._patch_gen_with_ollama([]):
            result = runner.invoke(cli, ["list-models"])
        assert result.exit_code == 0
        assert "ollama pull" in result.output

    def test_non_ollama_provider_shows_error(self, runner):
        with self._patch_gen_non_ollama():
            result = runner.invoke(cli, ["list-models"])
        assert result.exit_code != 0
        assert "list-models is only supported for Ollama" in result.output

    def test_list_models_runtime_error(self, runner):
        with self._patch_gen_with_ollama(None) as mock_cls:
            mock_cls.return_value.provider.list_models.side_effect = RuntimeError("timeout")
            result = runner.invoke(cli, ["list-models"])
        assert result.exit_code != 0
        assert "timeout" in result.output

    def test_generator_init_error_shows_cli_error(self, runner):
        with patch("testdata_ai.cli.DataGenerator", side_effect=ValueError("bad provider")):
            result = runner.invoke(cli, ["list-models"])
        assert result.exit_code != 0
        assert "bad provider" in result.output
