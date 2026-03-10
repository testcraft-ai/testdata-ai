"""Tests for testdata_ai.ai_providers — CohereProvider."""

import pytest
from unittest.mock import patch, MagicMock

from testdata_ai.ai_providers import CohereProvider, DEFAULT_SYSTEM_PROMPT


def _set_cohere_response(mock_client, text):
    content_item = MagicMock()
    content_item.text = text
    mock_client.chat.return_value = MagicMock(
        message=MagicMock(content=[content_item] if text is not None else [])
    )


class TestCohereProvider:

    def test_init_client_imports_cohere(self):
        mock_cohere_module = MagicMock()
        mock_client_v2_cls = MagicMock()
        mock_cohere_module.ClientV2 = mock_client_v2_cls
        with patch.dict("sys.modules", {"cohere": mock_cohere_module}):
            provider = CohereProvider.__new__(CohereProvider)
            provider.model = "command-r"
            provider.temperature = 0.7
            provider.max_tokens = 4096
            provider._init_client("co-test")
        mock_client_v2_cls.assert_called_once_with(api_key="co-test")

    def test_init_client_raises_import_error(self):
        with patch.dict("sys.modules", {"cohere": None}):
            with pytest.raises(ImportError, match="cohere package is required"):
                CohereProvider("co-fake", "command-r", 0.7, 4096)

    def test_generate_success(self, cohere_provider_mock):
        provider, mock_client = cohere_provider_mock
        _set_cohere_response(mock_client, '[{"name": "Alice"}]')

        result = provider.generate("test prompt")

        assert result == '[{"name": "Alice"}]'
        mock_client.chat.assert_called_once()

    def test_generate_includes_system_prompt(self, cohere_provider_mock):
        provider, mock_client = cohere_provider_mock
        _set_cohere_response(mock_client, "{}")

        provider.generate("test prompt", system_prompt="custom system")

        call_kwargs = mock_client.chat.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "custom system"}
        assert messages[1] == {"role": "user", "content": "test prompt"}

    def test_generate_uses_default_system_prompt(self, cohere_provider_mock):
        provider, mock_client = cohere_provider_mock
        _set_cohere_response(mock_client, "{}")

        provider.generate("test prompt")

        call_kwargs = mock_client.chat.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}

    def test_generate_skips_system_prompt_when_empty(self, cohere_provider_mock):
        provider, mock_client = cohere_provider_mock
        _set_cohere_response(mock_client, "{}")

        provider.generate("test prompt", system_prompt="")

        call_kwargs = mock_client.chat.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_generate_forwards_model_and_params(self, cohere_provider_mock):
        provider, mock_client = cohere_provider_mock
        _set_cohere_response(mock_client, "{}")

        provider.generate("test prompt")

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["model"] == "command-r"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 4096

    def test_generate_raises_on_timeout(self, cohere_provider_mock):
        provider, mock_client = cohere_provider_mock
        mock_client.chat.side_effect = Exception("Request timed out")

        with pytest.raises(RuntimeError, match="timed out"):
            provider.generate("test prompt")

    def test_generate_raises_on_generic_error(self, cohere_provider_mock):
        provider, mock_client = cohere_provider_mock
        mock_client.chat.side_effect = Exception("rate limit exceeded")

        with pytest.raises(RuntimeError, match="Failed to generate data"):
            provider.generate("test prompt")

    def test_generate_raises_on_empty_content_list(self, cohere_provider_mock):
        provider, mock_client = cohere_provider_mock
        mock_client.chat.return_value = MagicMock(message=MagicMock(content=[]))

        with pytest.raises(RuntimeError, match="empty response"):
            provider.generate("test prompt")
