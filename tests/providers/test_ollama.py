"""Tests for testdata_ai.ai_providers — Ollama provider."""

import json
import pytest
import urllib.error
from unittest.mock import patch, MagicMock

from testdata_ai.ai_providers import OllamaProvider, DEFAULT_SYSTEM_PROMPT


def _set_ollama_response(mock_urllib, content):
    response_data = json.dumps({"message": {"content": content}}).encode()
    mock_urllib.urlopen.return_value.__enter__.return_value.read.return_value = response_data


def _set_ollama_tags_response(mock_urllib, models):
    response_data = json.dumps({"models": [{"name": m} for m in models]}).encode()
    mock_urllib.urlopen.return_value.__enter__.return_value.read.return_value = response_data


def _make_ctx_manager_mock(data: bytes):
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=data)))
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


class TestOllamaProvider:

    def test_init_client_reads_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://myhost:11434")
        provider = OllamaProvider.__new__(OllamaProvider)
        provider.model = "qwen2.5:14b"
        provider.temperature = 0.7
        provider.max_tokens = 4096
        provider._init_client("ollama")
        assert provider.base_url == "http://myhost:11434"

    def test_init_client_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://myhost:11434/")
        provider = OllamaProvider.__new__(OllamaProvider)
        provider.model = "qwen2.5:14b"
        provider.temperature = 0.7
        provider.max_tokens = 4096
        provider._init_client("ollama")
        assert provider.base_url == "http://myhost:11434"

    def test_init_client_defaults_to_localhost(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        provider = OllamaProvider.__new__(OllamaProvider)
        provider.model = "qwen2.5:14b"
        provider.temperature = 0.7
        provider.max_tokens = 4096
        provider._init_client("ollama")
        assert provider.base_url == "http://localhost:11434"

    def test_generate_success(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        _set_ollama_response(mock_urllib, '{"data": [{"name": "Alice"}]}')

        result = provider.generate("test prompt")
        assert result == '{"data": [{"name": "Alice"}]}'
        mock_urllib.urlopen.assert_called_once()

    def test_generate_posts_to_correct_url(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        _set_ollama_response(mock_urllib, "{}")

        provider.generate("test prompt")
        req = mock_urllib.Request.call_args[0][0]
        assert req == "http://localhost:11434/api/chat"

    def test_generate_includes_system_prompt(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        _set_ollama_response(mock_urllib, "{}")

        provider.generate("test prompt", system_prompt="custom system")
        payload = json.loads(mock_urllib.Request.call_args[1]["data"])
        assert payload["messages"][0] == {"role": "system", "content": "custom system"}
        assert payload["messages"][1] == {"role": "user", "content": "test prompt"}

    def test_generate_skips_system_prompt_when_empty(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        _set_ollama_response(mock_urllib, "{}")

        provider.generate("test prompt", system_prompt="")
        payload = json.loads(mock_urllib.Request.call_args[1]["data"])
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    def test_generate_uses_default_system_prompt(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        _set_ollama_response(mock_urllib, "{}")

        provider.generate("test prompt")
        payload = json.loads(mock_urllib.Request.call_args[1]["data"])
        assert payload["messages"][0]["content"] == DEFAULT_SYSTEM_PROMPT

    def test_generate_forwards_model_and_params(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        _set_ollama_response(mock_urllib, "{}")

        provider.generate("test")
        payload = json.loads(mock_urllib.Request.call_args[1]["data"])
        assert payload["model"] == "qwen2.5:14b"
        assert payload["options"]["temperature"] == 0.7
        assert payload["options"]["num_predict"] == 4096

    def test_generate_requests_json_format(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        _set_ollama_response(mock_urllib, "{}")

        provider.generate("test")
        payload = json.loads(mock_urllib.Request.call_args[1]["data"])
        assert payload["format"] == "json"
        assert payload["stream"] is False

    def test_generate_raises_on_empty_response(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        empty_data = json.dumps({"message": {"content": ""}}).encode()
        mock_urllib.urlopen.return_value.__enter__.return_value.read.return_value = empty_data

        with pytest.raises(RuntimeError, match="empty response"):
            provider.generate("test prompt")

    def test_generate_raises_on_timeout(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        mock_urllib.urlopen.side_effect = Exception("timed out")

        with pytest.raises(RuntimeError, match="timed out"):
            provider.generate("test prompt")

    def test_generate_raises_model_not_found_on_http_404(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        mock_urllib.urlopen.side_effect = urllib.error.HTTPError(
            url=None, code=404, msg="Not Found", hdrs={}, fp=None
        )

        with pytest.raises(RuntimeError, match="not found in Ollama"):
            provider.generate("test prompt")

    def test_generate_model_not_found_includes_pull_command(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        mock_urllib.urlopen.side_effect = urllib.error.HTTPError(
            url=None, code=404, msg="Not Found", hdrs={}, fp=None
        )

        with pytest.raises(RuntimeError, match="ollama pull qwen2.5:14b"):
            provider.generate("test prompt")

    def test_generate_raises_on_connection_refused(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        mock_urllib.urlopen.side_effect = urllib.error.URLError("Connection refused")

        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            provider.generate("test prompt")

    def test_generate_raises_on_ollama_not_running(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        mock_urllib.urlopen.side_effect = urllib.error.URLError("Connection refused")

        with pytest.raises(RuntimeError, match="ollama serve"):
            provider.generate("test prompt")

    def test_generate_raises_on_other_http_error(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        mock_urllib.urlopen.side_effect = urllib.error.HTTPError(
            url=None, code=500, msg="Internal Server Error", hdrs={}, fp=None
        )

        with pytest.raises(RuntimeError, match="Ollama HTTP error 500"):
            provider.generate("test prompt")


class TestOllamaProviderValidateModel:

    def test_validate_model_success_sets_flag(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        provider._model_validated = False

        provider._validate_model()

        assert provider._model_validated is True
        mock_urllib.urlopen.assert_called_once()

    def test_validate_model_posts_to_api_show(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        provider._model_validated = False

        provider._validate_model()

        url = mock_urllib.Request.call_args[0][0]
        assert url == "http://localhost:11434/api/show"

    def test_validate_model_raises_on_404(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        provider._model_validated = False
        mock_urllib.urlopen.side_effect = urllib.error.HTTPError(
            url=None, code=404, msg="Not Found", hdrs={}, fp=None
        )

        with pytest.raises(RuntimeError, match="ollama pull qwen2.5:14b"):
            provider._validate_model()

    def test_generate_calls_validate_model_once(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        provider._model_validated = False
        success_data = json.dumps({"message": {"content": '{"data": []}'}}).encode()
        mock_urllib.urlopen.return_value.__enter__.return_value.read.return_value = (
            success_data
        )

        provider.generate("prompt")
        provider.generate("prompt")

        assert mock_urllib.urlopen.call_count == 3


class TestOllamaProviderRetry:

    def test_retries_on_5xx_and_succeeds(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        provider._max_retries = 2
        success_data = json.dumps({"message": {"content": '{"ok": true}'}}).encode()
        mock_urllib.urlopen.side_effect = [
            urllib.error.HTTPError(
                url=None, code=503, msg="Service Unavailable", hdrs={}, fp=None
            ),
            _make_ctx_manager_mock(success_data),
        ]

        result = provider.generate("prompt")

        assert result == '{"ok": true}'
        assert mock_urllib.urlopen.call_count == 2

    def test_retries_on_url_error_and_succeeds(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        provider._max_retries = 2
        success_data = json.dumps({"message": {"content": '{"ok": true}'}}).encode()
        mock_urllib.urlopen.side_effect = [
            urllib.error.URLError("temporary network error"),
            _make_ctx_manager_mock(success_data),
        ]

        result = provider.generate("prompt")

        assert result == '{"ok": true}'
        assert mock_urllib.urlopen.call_count == 2

    def test_no_retry_on_404(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        provider._max_retries = 3
        mock_urllib.urlopen.side_effect = urllib.error.HTTPError(
            url=None, code=404, msg="Not Found", hdrs={}, fp=None
        )

        with pytest.raises(RuntimeError, match="not found in Ollama"):
            provider.generate("prompt")

        assert mock_urllib.urlopen.call_count == 1


class TestOllamaProviderJsonAndFormat:

    def test_generate_raises_on_invalid_json(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        mock_urllib.urlopen.return_value.__enter__.return_value.read.return_value = (
            b"not valid json"
        )

        with pytest.raises(RuntimeError, match="invalid JSON"):
            provider.generate("prompt")

    def test_generate_without_json_format(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        provider._use_json_format = False
        _set_ollama_response(mock_urllib, '{"data": []}')

        provider.generate("prompt")

        payload = json.loads(mock_urllib.Request.call_args[1]["data"])
        assert "format" not in payload

    def test_generate_with_json_format_enabled(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        provider._use_json_format = True
        _set_ollama_response(mock_urllib, '{"data": []}')

        provider.generate("prompt")

        payload = json.loads(mock_urllib.Request.call_args[1]["data"])
        assert payload["format"] == "json"


class TestOllamaProviderListModels:

    def test_list_models_returns_model_names(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        _set_ollama_tags_response(mock_urllib, ["qwen2.5:14b", "mistral:latest"])

        result = provider.list_models()

        assert result == ["qwen2.5:14b", "mistral:latest"]

    def test_list_models_calls_api_tags(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        _set_ollama_tags_response(mock_urllib, [])

        provider.list_models()

        url = mock_urllib.Request.call_args[0][0]
        assert url == "http://localhost:11434/api/tags"

    def test_list_models_returns_empty_list_when_no_models(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        _set_ollama_tags_response(mock_urllib, [])

        result = provider.list_models()

        assert result == []

    def test_list_models_raises_on_request_error(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        mock_urllib.urlopen.side_effect = Exception("network timeout")

        with pytest.raises(RuntimeError):
            provider.list_models()


class TestOllamaProviderValidateModelErrors:

    def test_validate_model_non_404_http_error_raises(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        provider._model_validated = False
        mock_urllib.urlopen.side_effect = urllib.error.HTTPError(
            url=None, code=500, msg="Internal Server Error", hdrs={}, fp=None
        )

        with pytest.raises(RuntimeError, match="Ollama HTTP error 500"):
            provider._validate_model()

    def test_validate_model_generic_exception_raises(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        provider._model_validated = False
        mock_urllib.urlopen.side_effect = Exception("connection reset")

        with pytest.raises(RuntimeError):
            provider._validate_model()


class TestOllamaRetryExhausted:

    def test_else_branch_fires_when_range_is_empty(self, ollama_provider_mock):
        provider, mock_urllib = ollama_provider_mock
        provider._max_retries = -1  # range(0) → loop body never executes

        with pytest.raises(RuntimeError):
            provider.generate("prompt")
