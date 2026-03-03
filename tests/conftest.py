"""Shared fixtures for testdata-ai tests."""

from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from testdata_ai.ai_providers import OpenAIProvider, AnthropicProvider, OllamaProvider
from testdata_ai.contexts import ContextSchema


pytest_plugins = ["testdata_ai.pytest_plugin", "pytester"]


@pytest.fixture(autouse=True)
def _attach_caplog_to_testdata_logger(caplog):
    """Attach caplog handler directly to testdata_ai logger.

    The testdata_ai logger sets propagate=False to prevent double-logging
    in user projects that configure the root logger. This fixture re-attaches
    caplog's handler so that caplog still captures testdata_ai log messages.
    """
    import logging
    logger = logging.getLogger("testdata_ai")
    logger.addHandler(caplog.handler)
    yield
    logger.removeHandler(caplog.handler)


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
        "OLLAMA_API_KEY", "OLLAMA_MODEL", "OLLAMA_TEMPERATURE", "OLLAMA_MAX_TOKENS",
        "OLLAMA_BASE_URL",
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
def ollama_provider_mock():
    """Create an OllamaProvider with a mocked urllib.request."""
    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider("ollama", "qwen2.5:14b", 0.7, 4096)
    mock_urllib = MagicMock()
    provider._urllib = mock_urllib
    provider.base_url = "http://localhost:11434"
    provider._timeout = 600
    provider._model_validated = True    # skip validation in existing tests
    provider._max_retries = 0           # no retry loop in existing error tests
    provider._use_json_format = True    # keep format assertions passing
    provider._sleep = lambda s: None    # no real sleep in tests
    return provider, mock_urllib


@pytest.fixture
def mock_generator(max_tokens=4096):
    """Create a mocked DataGenerator for CLI tests."""
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
