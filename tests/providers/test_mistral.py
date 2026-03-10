"""Tests for testdata_ai.ai_providers — MistralProvider."""

import pytest
from unittest.mock import patch, MagicMock

from testdata_ai.ai_providers import MistralProvider, DEFAULT_SYSTEM_PROMPT


def _set_mistral_response(mock_client, content):
    mock_client.chat.complete.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=content))]
    )


class TestMistralProvider:

    def test_init_client_imports_mistralai(self):
        mock_mistral_cls = MagicMock()
        with patch.dict("sys.modules", {"mistralai": MagicMock(Mistral=mock_mistral_cls)}):
            provider = MistralProvider.__new__(MistralProvider)
            provider.model = "mistral-small-latest"
            provider.temperature = 0.7
            provider.max_tokens = 4096
            provider._init_client("mst-test")
        mock_mistral_cls.assert_called_once_with(api_key="mst-test")

    def test_init_client_raises_import_error(self):
        with patch.dict("sys.modules", {"mistralai": None}):
            with pytest.raises(ImportError, match="mistralai package is required"):
                MistralProvider("mst-fake", "mistral-small-latest", 0.7, 4096)

    def test_generate_success(self, mistral_provider_mock):
        provider, mock_client = mistral_provider_mock
        _set_mistral_response(mock_client, '[{"name": "Alice"}]')

        result = provider.generate("test prompt")

        assert result == '[{"name": "Alice"}]'
        mock_client.chat.complete.assert_called_once()

    def test_generate_includes_system_prompt(self, mistral_provider_mock):
        provider, mock_client = mistral_provider_mock
        _set_mistral_response(mock_client, "{}")

        provider.generate("test prompt", system_prompt="custom system")

        call_kwargs = mock_client.chat.complete.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "custom system"}
        assert messages[1] == {"role": "user", "content": "test prompt"}

    def test_generate_uses_default_system_prompt(self, mistral_provider_mock):
        provider, mock_client = mistral_provider_mock
        _set_mistral_response(mock_client, "{}")

        provider.generate("test prompt")

        call_kwargs = mock_client.chat.complete.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}

    def test_generate_skips_system_prompt_when_empty(self, mistral_provider_mock):
        provider, mock_client = mistral_provider_mock
        _set_mistral_response(mock_client, "{}")

        provider.generate("test prompt", system_prompt="")

        call_kwargs = mock_client.chat.complete.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_generate_uses_json_response_format(self, mistral_provider_mock):
        provider, mock_client = mistral_provider_mock
        _set_mistral_response(mock_client, "{}")

        provider.generate("test prompt")

        call_kwargs = mock_client.chat.complete.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_generate_forwards_model_and_params(self, mistral_provider_mock):
        provider, mock_client = mistral_provider_mock
        _set_mistral_response(mock_client, "{}")

        provider.generate("test prompt")

        call_kwargs = mock_client.chat.complete.call_args[1]
        assert call_kwargs["model"] == "mistral-small-latest"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 4096

    def test_generate_raises_on_timeout(self, mistral_provider_mock):
        provider, mock_client = mistral_provider_mock
        mock_client.chat.complete.side_effect = Exception("Request timed out")

        with pytest.raises(RuntimeError, match="timed out"):
            provider.generate("test prompt")

    def test_generate_raises_on_generic_error(self, mistral_provider_mock):
        provider, mock_client = mistral_provider_mock
        mock_client.chat.complete.side_effect = Exception("rate limit exceeded")

        with pytest.raises(RuntimeError, match="Failed to generate data"):
            provider.generate("test prompt")

    def test_generate_raises_on_empty_response(self, mistral_provider_mock):
        provider, mock_client = mistral_provider_mock
        _set_mistral_response(mock_client, None)

        with pytest.raises(RuntimeError, match="empty response"):
            provider.generate("test prompt")
