"""Tests for testdata_ai.ai_providers — Anthropic provider."""

import pytest
from unittest.mock import patch, MagicMock

from testdata_ai.ai_providers import AnthropicProvider, DEFAULT_SYSTEM_PROMPT


def _set_anthropic_response(mock_client, text):
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=text)]
    )


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
