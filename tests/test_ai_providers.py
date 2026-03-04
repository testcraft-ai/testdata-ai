"""Tests for testdata_ai.ai_providers — provider factory, OpenAI, Anthropic."""

from unittest.mock import patch, MagicMock

import pytest

import json

from testdata_ai.ai_providers import (
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    get_provider,
    DEFAULT_SYSTEM_PROMPT,
)



def _set_openai_response(mock_client, content):
    """Configure mock_client to return a chat completion with given content."""
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=content))]
    )

def _set_anthropic_response(mock_client, text):
    """Configure mock_client to return a message with given text."""
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=text)]
    )


class TestGetProvider:

    def test_returns_openai_provider(self):
        with patch.object(OpenAIProvider, "_init_client"):
            provider = get_provider("openai", "sk-key", "gpt-4o-mini", 0.7, 4096)
        assert isinstance(provider, OpenAIProvider)

    def test_returns_anthropic_provider(self):
        with patch.object(AnthropicProvider, "_init_client"):
            provider = get_provider("anthropic", "ant-key", "claude-haiku", 0.7, 4096)
        assert isinstance(provider, AnthropicProvider)

    def test_returns_ollama_provider(self):
        with patch.object(OllamaProvider, "_init_client"):
            provider = get_provider("ollama", "ollama", "qwen2.5:14b", 0.7, 4096)
        assert isinstance(provider, OllamaProvider)

    def test_raises_for_unknown_provider(self):
        with pytest.raises(ValueError, match="Unsupported provider: 'mistral'"):
            get_provider("mistral", "key", "model", 0.7, 4096)

    def test_sets_model_and_params(self):
        with patch.object(OpenAIProvider, "_init_client"):
            provider = get_provider("openai", "sk-key", "gpt-4o", 0.3, 2048)
        assert provider.model == "gpt-4o"
        assert provider.temperature == 0.3
        assert provider.max_tokens == 2048


class TestOpenAIProvider:

    def test_init_client_imports_openai(self):
        mock_openai_cls = MagicMock()
        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
            provider = OpenAIProvider.__new__(OpenAIProvider)
            provider.model = "gpt-4o-mini"
            provider.temperature = 0.7
            provider.max_tokens = 4096
            provider._init_client("sk-test")
        mock_openai_cls.assert_called_once_with(
            api_key="sk-test", timeout=120.0, max_retries=3
        )

    def test_init_client_raises_import_error(self):
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai package is required"):
                OpenAIProvider("sk-fake", "gpt-4o-mini", 0.7, 4096)

    def test_generate_success(self, openai_provider_mock):
        provider, mock_client = openai_provider_mock
        _set_openai_response(mock_client, '{"data": [{"name": "Test"}]}')

        result = provider.generate("test prompt")
        assert result == '{"data": [{"name": "Test"}]}'
        mock_client.chat.completions.create.assert_called_once()

    def test_generate_includes_system_prompt(self, openai_provider_mock):
        provider, mock_client = openai_provider_mock
        _set_openai_response(mock_client, "{}")

        provider.generate("test prompt", system_prompt="custom system")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "custom system"}
        assert messages[1] == {"role": "user", "content": "test prompt"}

    def test_generate_uses_default_system_prompt(self, openai_provider_mock):
        provider, mock_client = openai_provider_mock
        _set_openai_response(mock_client, "{}")

        provider.generate("test prompt")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}

    def test_generate_skips_system_prompt_when_empty(self, openai_provider_mock):
        provider, mock_client = openai_provider_mock
        _set_openai_response(mock_client, "{}")

        provider.generate("test prompt", system_prompt="")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_generate_raises_on_timeout(self, openai_provider_mock):
        provider, mock_client = openai_provider_mock
        mock_client.chat.completions.create.side_effect = Exception(
            "Request timed out"
        )

        with pytest.raises(RuntimeError, match="timed out"):
            provider.generate("test prompt")

    def test_generate_raises_on_generic_api_error(self, openai_provider_mock):
        provider, mock_client = openai_provider_mock
        mock_client.chat.completions.create.side_effect = Exception("rate limit")

        with pytest.raises(RuntimeError, match="Failed to generate data"):
            provider.generate("test prompt")

    def test_generate_raises_on_empty_response(self, openai_provider_mock):
        provider, mock_client = openai_provider_mock
        _set_openai_response(mock_client, None)

        with pytest.raises(RuntimeError, match="empty response"):
            provider.generate("test prompt")

    def test_generate_uses_json_response_format(self, openai_provider_mock):
        provider, mock_client = openai_provider_mock
        _set_openai_response(mock_client, "{}")

        provider.generate("test")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_generate_forwards_model_and_params(self, openai_provider_mock):
        provider, mock_client = openai_provider_mock
        _set_openai_response(mock_client, "{}")

        provider.generate("test")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 4096


class TestAnthropicProvider:

    def test_init_client_imports_anthropic(self):
        mock_anthropic_cls = MagicMock()
        with patch.dict(
            "sys.modules", {"anthropic": MagicMock(Anthropic=mock_anthropic_cls)}
        ):
            provider = AnthropicProvider.__new__(AnthropicProvider)
            provider.model = "claude-haiku"
            provider.temperature = 0.7
            provider.max_tokens = 4096
            provider._init_client("ant-test")
        mock_anthropic_cls.assert_called_once_with(
            api_key="ant-test", timeout=120.0, max_retries=3
        )

    def test_init_client_raises_import_error(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic package is required"):
                AnthropicProvider("ant-fake", "claude-haiku", 0.7, 4096)

    def test_generate_success(self, anthropic_provider_mock):
        provider, mock_client = anthropic_provider_mock
        _set_anthropic_response(mock_client, '{"data": [{"name": "Test"}]}')

        result = provider.generate("test prompt")
        assert result == '{"data": [{"name": "Test"}]}'
        mock_client.messages.create.assert_called_once()

    def test_generate_passes_system_prompt(self, anthropic_provider_mock):
        provider, mock_client = anthropic_provider_mock
        _set_anthropic_response(mock_client, "{}")

        provider.generate("test prompt", system_prompt="custom")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "custom"
        assert call_kwargs["messages"] == [{"role": "user", "content": "test prompt"}]

    def test_generate_uses_default_system_prompt(self, anthropic_provider_mock):
        provider, mock_client = anthropic_provider_mock
        _set_anthropic_response(mock_client, "{}")

        provider.generate("test prompt")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == DEFAULT_SYSTEM_PROMPT

    def test_generate_passes_empty_system_prompt(self, anthropic_provider_mock):
        provider, mock_client = anthropic_provider_mock
        _set_anthropic_response(mock_client, "{}")

        provider.generate("test prompt", system_prompt="")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == ""

    def test_generate_raises_on_timeout(self, anthropic_provider_mock):
        provider, mock_client = anthropic_provider_mock
        mock_client.messages.create.side_effect = Exception("Request timed out")

        with pytest.raises(RuntimeError, match="timed out"):
            provider.generate("test prompt")

    def test_generate_raises_on_generic_api_error(self, anthropic_provider_mock):
        provider, mock_client = anthropic_provider_mock
        mock_client.messages.create.side_effect = Exception("server error")

        with pytest.raises(RuntimeError, match="Failed to generate data"):
            provider.generate("test prompt")

    def test_generate_raises_on_empty_response(self, anthropic_provider_mock):
        provider, mock_client = anthropic_provider_mock
        mock_client.messages.create.return_value = MagicMock(content=[])

        with pytest.raises(RuntimeError, match="empty response"):
            provider.generate("test prompt")

    def test_generate_forwards_model_and_params(self, anthropic_provider_mock):
        provider, mock_client = anthropic_provider_mock
        _set_anthropic_response(mock_client, "{}")

        provider.generate("test")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 4096


def _set_ollama_response(mock_urllib, content):
    """Configure mock_urllib to return an Ollama chat response."""
    response_data = json.dumps({"message": {"content": content}}).encode()
    mock_urllib.urlopen.return_value.__enter__.return_value.read.return_value = response_data


def _set_ollama_tags_response(mock_urllib, models):
    """Configure mock_urllib to return an Ollama /api/tags response."""
    response_data = json.dumps({"models": [{"name": m} for m in models]}).encode()
    mock_urllib.urlopen.return_value.__enter__.return_value.read.return_value = response_data


def _make_ctx_manager_mock(data: bytes):
    """Return a MagicMock that behaves as a context manager returning data."""
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
        import urllib.error
        provider, mock_urllib = ollama_provider_mock
        mock_urllib.urlopen.side_effect = urllib.error.HTTPError(
            url=None, code=404, msg="Not Found", hdrs={}, fp=None
        )

        with pytest.raises(RuntimeError, match="not found in Ollama"):
            provider.generate("test prompt")

    def test_generate_model_not_found_includes_pull_command(self, ollama_provider_mock):
        import urllib.error
        provider, mock_urllib = ollama_provider_mock
        mock_urllib.urlopen.side_effect = urllib.error.HTTPError(
            url=None, code=404, msg="Not Found", hdrs={}, fp=None
        )

        with pytest.raises(RuntimeError, match="ollama pull qwen2.5:14b"):
            provider.generate("test prompt")

    def test_generate_raises_on_connection_refused(self, ollama_provider_mock):
        import urllib.error
        provider, mock_urllib = ollama_provider_mock
        mock_urllib.urlopen.side_effect = urllib.error.URLError("Connection refused")

        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            provider.generate("test prompt")

    def test_generate_raises_on_ollama_not_running(self, ollama_provider_mock):
        import urllib.error
        provider, mock_urllib = ollama_provider_mock
        mock_urllib.urlopen.side_effect = urllib.error.URLError("Connection refused")

        with pytest.raises(RuntimeError, match="ollama serve"):
            provider.generate("test prompt")

    def test_generate_raises_on_other_http_error(self, ollama_provider_mock):
        import urllib.error
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
        import urllib.error
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

        provider.generate("prompt")   # urlopen #1 = validate, #2 = generate
        provider.generate("prompt")   # urlopen #3 = generate only (no re-validate)

        assert mock_urllib.urlopen.call_count == 3


class TestOllamaProviderRetry:

    def test_retries_on_5xx_and_succeeds(self, ollama_provider_mock):
        import urllib.error
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
        import urllib.error
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
        import urllib.error
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
        """Exception from urlopen in list_models() → _handle_api_error."""
        import urllib.error
        provider, mock_urllib = ollama_provider_mock
        mock_urllib.urlopen.side_effect = Exception("network timeout")

        with pytest.raises(RuntimeError):
            provider.list_models()


class TestOllamaProviderValidateModelErrors:
    """Cover non-404 HTTP error and generic exception in _validate_model."""

    def test_validate_model_non_404_http_error_raises(self, ollama_provider_mock):
        """A 500 error during model validation should raise RuntimeError."""
        import urllib.error
        provider, mock_urllib = ollama_provider_mock
        provider._model_validated = False
        mock_urllib.urlopen.side_effect = urllib.error.HTTPError(
            url=None, code=500, msg="Internal Server Error", hdrs={}, fp=None
        )

        with pytest.raises(RuntimeError, match="Ollama HTTP error 500"):
            provider._validate_model()

    def test_validate_model_generic_exception_raises(self, ollama_provider_mock):
        """A generic exception during model validation should propagate."""
        provider, mock_urllib = ollama_provider_mock
        provider._model_validated = False
        mock_urllib.urlopen.side_effect = Exception("connection reset")

        with pytest.raises(RuntimeError):
            provider._validate_model()


class TestOllamaRetryExhausted:
    """Cover the for-else branch when the retry loop completes without break."""

    def test_else_branch_fires_when_range_is_empty(self, ollama_provider_mock):
        """With _max_retries=-1, range(0) is empty, so the else clause runs."""
        provider, mock_urllib = ollama_provider_mock
        provider._max_retries = -1  # range(0) → loop body never executes

        with pytest.raises(RuntimeError):
            provider.generate("prompt")
