"""Tests for testdata_ai.ai_providers — GeminiProvider."""

import pytest
from unittest.mock import patch, MagicMock

from testdata_ai.ai_providers import GeminiProvider, DEFAULT_SYSTEM_PROMPT


def _set_gemini_response(mock_client, text):
    mock_client.models.generate_content.return_value = MagicMock(text=text)


class TestGeminiProvider:

    def test_init_client_imports_google_genai(self):
        mock_types = MagicMock()
        mock_genai = MagicMock()
        mock_genai_pkg = MagicMock(Client=mock_genai.Client, genai=mock_genai)
        with patch.dict("sys.modules", {
            "google": MagicMock(genai=mock_genai),
            "google.genai": MagicMock(Client=mock_genai.Client, types=mock_types),
        }):
            provider = GeminiProvider.__new__(GeminiProvider)
            provider.model = "gemini-2.0-flash"
            provider.temperature = 0.7
            provider.max_tokens = 4096
            provider._init_client("fake-api-key")
        mock_genai.Client.assert_called_once_with(api_key="fake-api-key")

    def test_init_client_raises_import_error(self):
        with patch.dict("sys.modules", {"google": None, "google.genai": None}):
            with pytest.raises(ImportError, match="google-genai package is required"):
                GeminiProvider("fake-api-key", "gemini-2.0-flash", 0.7, 4096)

    def test_generate_success(self, gemini_provider_mock):
        provider, mock_client, mock_types = gemini_provider_mock
        _set_gemini_response(mock_client, '[{"name": "Alice"}]')

        result = provider.generate("test prompt")

        assert result == '[{"name": "Alice"}]'
        mock_client.models.generate_content.assert_called_once()

    def test_generate_includes_system_prompt(self, gemini_provider_mock):
        provider, mock_client, mock_types = gemini_provider_mock
        _set_gemini_response(mock_client, "{}")

        provider.generate("test prompt", system_prompt="custom system")

        _, config_kwargs = mock_types.GenerateContentConfig.call_args
        assert config_kwargs["system_instruction"] == "custom system"

    def test_generate_uses_default_system_prompt(self, gemini_provider_mock):
        provider, mock_client, mock_types = gemini_provider_mock
        _set_gemini_response(mock_client, "{}")

        provider.generate("test prompt")

        _, config_kwargs = mock_types.GenerateContentConfig.call_args
        assert config_kwargs["system_instruction"] == DEFAULT_SYSTEM_PROMPT

    def test_generate_skips_system_prompt_when_empty(self, gemini_provider_mock):
        provider, mock_client, mock_types = gemini_provider_mock
        _set_gemini_response(mock_client, "{}")

        provider.generate("test prompt", system_prompt="")

        _, config_kwargs = mock_types.GenerateContentConfig.call_args
        assert config_kwargs["system_instruction"] is None

    def test_generate_uses_json_mime_type(self, gemini_provider_mock):
        provider, mock_client, mock_types = gemini_provider_mock
        _set_gemini_response(mock_client, "{}")

        provider.generate("test prompt")

        _, config_kwargs = mock_types.GenerateContentConfig.call_args
        assert config_kwargs["response_mime_type"] == "application/json"

    def test_generate_forwards_model_and_params(self, gemini_provider_mock):
        provider, mock_client, mock_types = gemini_provider_mock
        _set_gemini_response(mock_client, "{}")

        provider.generate("test prompt")

        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-2.0-flash"
        assert call_kwargs["contents"] == "test prompt"
        _, config_kwargs = mock_types.GenerateContentConfig.call_args
        assert config_kwargs["temperature"] == 0.7
        assert config_kwargs["max_output_tokens"] == 4096

    def test_generate_raises_on_timeout(self, gemini_provider_mock):
        provider, mock_client, mock_types = gemini_provider_mock
        mock_client.models.generate_content.side_effect = Exception("Request timed out")

        with pytest.raises(RuntimeError, match="timed out"):
            provider.generate("test prompt")

    def test_generate_raises_on_generic_error(self, gemini_provider_mock):
        provider, mock_client, mock_types = gemini_provider_mock
        mock_client.models.generate_content.side_effect = Exception("quota exceeded")

        with pytest.raises(RuntimeError, match="Failed to generate data"):
            provider.generate("test prompt")

    def test_generate_raises_on_empty_response(self, gemini_provider_mock):
        provider, mock_client, mock_types = gemini_provider_mock
        _set_gemini_response(mock_client, "")

        with pytest.raises(RuntimeError, match="empty response"):
            provider.generate("test prompt")

    def test_generate_raises_on_none_response(self, gemini_provider_mock):
        provider, mock_client, mock_types = gemini_provider_mock
        _set_gemini_response(mock_client, None)

        with pytest.raises(RuntimeError, match="empty response"):
            provider.generate("test prompt")
