"""Tests for testdata_ai.ai_providers — get_provider factory and OpenAI provider."""

import pytest
from unittest.mock import patch, MagicMock

from testdata_ai.ai_providers import (
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    get_provider,
    DEFAULT_SYSTEM_PROMPT,
)


def _set_openai_response(mock_client, content):
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=content))]
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
        mock_client.chat.completions.create.side_effect = Exception("Request timed out")

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
