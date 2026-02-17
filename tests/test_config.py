"""Tests for testdata_ai.config — provider config and env loading."""

import os

import pytest

from testdata_ai.config import AIProviderConfig, get_provider_config


class TestAIProviderConfig:

    def test_valid_config(self):
        cfg = AIProviderConfig(
            provider="openai", api_key="sk-test", model="gpt-4o-mini"
        )
        assert cfg.provider == "openai"
        assert cfg.api_key == "sk-test"
        assert cfg.model == "gpt-4o-mini"
        assert cfg.temperature == 0.7  # default
        assert cfg.max_tokens == 4096  # default

    def test_custom_temperature_and_tokens(self):
        cfg = AIProviderConfig(
            provider="openai", api_key="sk-test", model="m", temperature=0.3, max_tokens=1024
        )
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 1024

    @pytest.mark.parametrize("temp", [0.0, 0.5, 1.0])
    def test_valid_temperature_accepted(self, temp):
        cfg = AIProviderConfig(
            provider="openai", api_key="k", model="m", temperature=temp
        )
        assert cfg.temperature == temp

    @pytest.mark.parametrize("temp", [-0.1, 1.1, 2.0, -1.0])
    def test_invalid_temperature_raises(self, temp):
        with pytest.raises(ValueError, match="temperature"):
            AIProviderConfig(
                provider="openai", api_key="k", model="m", temperature=temp
            )

    @pytest.mark.parametrize("tokens", [1, 1024, 4096])
    def test_valid_max_tokens(self, tokens):
        cfg = AIProviderConfig(
            provider="openai", api_key="k", model="m", max_tokens=tokens
        )
        assert cfg.max_tokens == tokens

    @pytest.mark.parametrize("tokens", [0, -1, -100])
    def test_invalid_max_tokens_raises(self, tokens):
        with pytest.raises(ValueError, match="max_tokens"):
            AIProviderConfig(
                provider="openai", api_key="k", model="m", max_tokens=tokens
            )

    def test_api_key_hidden_in_repr(self):
        cfg = AIProviderConfig(
            provider="openai", api_key="super-secret-key", model="m"
        )
        assert "super-secret-key" not in repr(cfg)



@pytest.mark.usefixtures("clean_ai_env_fixture")
class TestGetProviderConfig:

    def test_loads_openai_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        cfg = get_provider_config("openai")
        assert cfg.provider == "openai"
        assert cfg.api_key == "sk-test-123"
        assert cfg.model == "gpt-4o-mini"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096

    def test_loads_anthropic_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        cfg = get_provider_config("anthropic")
        assert cfg.provider == "anthropic"
        assert cfg.api_key == "ant-key"
        assert cfg.model == "claude-haiku-4-5-20251001"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096

    def test_defaults_to_openai_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-default")
        cfg = get_provider_config(None)
        assert cfg.provider == "openai"

    def test_respects_ai_provider_env(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        cfg = get_provider_config(None)
        assert cfg.provider == "anthropic"

    def test_custom_model_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        cfg = get_provider_config("openai")
        assert cfg.model == "gpt-4o"

    def test_custom_temperature_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_TEMPERATURE", "0.2")
        cfg = get_provider_config("openai")
        assert cfg.temperature == 0.2

    def test_custom_max_tokens_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "8192")
        cfg = get_provider_config("openai")
        assert cfg.max_tokens == 8192

    @pytest.mark.usefixtures("clean_ai_env_fixture")
    def test_raises_for_unsupported_provider(self):
        with pytest.raises(ValueError, match="Unsupported AI provider"):
            get_provider_config("mistral")

    @pytest.mark.usefixtures("clean_ai_env_fixture")
    def test_raises_when_api_key_missing(self):
        with pytest.raises(ValueError, match="API key not found"):
            get_provider_config("openai")

    def test_raises_when_api_key_whitespace_only(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "   ")
        with pytest.raises(ValueError, match="API key not found"):
            get_provider_config("openai")

    def test_provider_name_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = get_provider_config("OpenAI")
        assert cfg.provider == "openai"

    def test_invalid_temperature_string_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_TEMPERATURE", "not-a-number")
        with pytest.raises(ValueError):
            get_provider_config("openai")

    def test_invalid_max_tokens_string_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "xyz")
        with pytest.raises(ValueError):
            get_provider_config("openai")
