"""Shared fixtures for testdata-ai tests."""

from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from testdata_ai.ai_providers import OpenAIProvider, AnthropicProvider
from testdata_ai.contexts import ContextSchema


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def clean_ai_env_fixture(monkeypatch):
    """Internal fixture that cleans AI-related environment variables for isolation."""
    for var in [
        "AI_PROVIDER",
        "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_TEMPERATURE", "OPENAI_MAX_TOKENS",
        "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_TEMPERATURE", "ANTHROPIC_MAX_TOKENS",
    ]:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def openai_provider_mock():
    """Create an OpenAIProvider with a mocked client."""
    with patch.object(OpenAIProvider, "_init_client"):
        provider = OpenAIProvider("sk-fake", "gpt-4o-mini", 0.7, 4096)
    mock_client = MagicMock()
    provider.client = mock_client
    return provider, mock_client


@pytest.fixture
def anthropic_provider_mock():
    """Create an AnthropicProvider with a mocked client."""
    with patch.object(AnthropicProvider, "_init_client"):
        provider = AnthropicProvider("ant-fake", "claude-haiku", 0.7, 4096)
    mock_client = MagicMock()
    provider.client = mock_client
    return provider, mock_client


@pytest.fixture
def mock_generator(max_tokens=4096):
    """Create a mocked TestDataGenerator for CLI tests."""
    gen = MagicMock()
    gen.config.max_tokens = max_tokens
    gen.provider.max_tokens = max_tokens
    return gen


@pytest.fixture
def mock_context_schema():
    """Create a test ContextSchema for CLI tests."""
    return ContextSchema(
        description="test",
        sample={"name": "Test", "email": "test@test.com"},
        prompt_hints=["hint"],
    )
