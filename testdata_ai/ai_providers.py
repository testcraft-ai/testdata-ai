"""
AI provider abstractions.
Supports multiple AI providers (OpenAI, Anthropic, Ollama, etc.)
"""

from abc import ABC, abstractmethod
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import NoReturn

__all__ = ["AIProvider", "get_provider", "DEFAULT_SYSTEM_PROMPT"]

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a test data generator that returns JSON arrays. "
    "When asked for N items, return an array with exactly N objects, never a single object."
)


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._init_client(api_key)

    @abstractmethod
    def _init_client(self, api_key: str) -> None:
        """Initialize the provider-specific client."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        """Generate response from AI.

        Args:
            prompt: User prompt
            system_prompt: System instructions

        Returns:
            Generated text response
        """

    def _handle_api_error(self, e: Exception) -> NoReturn:
        """Translate provider exceptions into RuntimeError with user-friendly messages."""
        logger.error(f"API error: {e}")
        # Check both exception type hierarchy and message string, since provider SDKs
        # (openai, anthropic) wrap timeouts in their own exception types which may not
        # inherit from Python's built-in TimeoutError.
        is_timeout = isinstance(e, (TimeoutError, OSError)) or any(
            kw in str(e).lower() for kw in ("timed out", "timeout")
        )
        if is_timeout:
            raise RuntimeError(
                f"Request timed out ({self.model}). "
                f"Try reducing --count or using a faster model."
            ) from e
        raise RuntimeError(f"Failed to generate data: {e}") from e


class OpenAIProvider(AIProvider):
    """OpenAI provider (GPT-4o, etc.)"""

    def _init_client(self, api_key: str) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required: pip install openai")
        self.client = OpenAI(api_key=api_key, timeout=120.0, max_retries=3)

    def generate(self, prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            self._handle_api_error(e)

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(
                "OpenAI returned an empty response (possible content filter)"
            )
        return content


class AnthropicProvider(AIProvider):
    """Anthropic provider (Claude)."""

    def _init_client(self, api_key: str) -> None:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package is required: pip install anthropic")
        self.client = Anthropic(api_key=api_key, timeout=120.0, max_retries=3)

    def generate(self, prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            self._handle_api_error(e)

        if not response.content:
            raise RuntimeError("Anthropic returned an empty response")
        return response.content[0].text


class OllamaProvider(AIProvider):
    """Ollama provider for local LLMs (llama3.2, mistral, phi3, etc.)

    Requires a running Ollama instance (https://ollama.com).
    No API key needed. Configure base URL via OLLAMA_BASE_URL env var
    (default: http://localhost:11434).
    """

    def _handle_api_error(self, e: Exception) -> NoReturn:
        if isinstance(e, urllib.error.HTTPError):
            if e.code == 404:
                raise RuntimeError(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Run: ollama pull {self.model}"
                ) from e
            raise RuntimeError(f"Ollama HTTP error {e.code}: {e.reason}") from e
        if isinstance(e, urllib.error.URLError):
            if "connection refused" in str(e.reason).lower():
                raise RuntimeError(
                    "Cannot connect to Ollama. Is it running? Try: ollama serve"
                ) from e
        super()._handle_api_error(e)

    def _init_client(self, _api_key: str) -> None:
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self._timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))
        self._max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
        self._use_json_format = os.getenv("OLLAMA_JSON_FORMAT", "true").lower() != "false"
        self._model_validated = False
        # Store urllib.request and time.sleep on the instance so tests can mock them easily.
        self._urllib = urllib.request
        self._sleep = time.sleep

    def _validate_model(self) -> None:
        """Check that the model exists in Ollama before the first generate call.

        Calls POST /api/show. On success sets _model_validated = True so the
        check is skipped on subsequent calls.
        """
        payload = json.dumps({"name": self.model}).encode()
        req = self._urllib.Request(
            f"{self.base_url}/api/show",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._urllib.urlopen(req, timeout=self._timeout):
                pass  # 200 OK is sufficient
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise RuntimeError(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Run: ollama pull {self.model}"
                ) from e
            self._handle_api_error(e)
        except Exception as e:
            self._handle_api_error(e)
        self._model_validated = True

    def generate(self, prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        if not self._model_validated:
            self._validate_model()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if self._use_json_format:
            body["format"] = "json"

        payload = json.dumps(body).encode()
        req = self._urllib.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_exc: Exception = RuntimeError("No attempts made")
        raw = b""
        for attempt in range(self._max_retries + 1):
            try:
                with self._urllib.urlopen(req, timeout=self._timeout) as resp:
                    raw = resp.read()
                break  # success
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    self._handle_api_error(e)  # never retry 404
                if e.code >= 500:
                    last_exc = e
                    if attempt < self._max_retries:
                        self._sleep(2 ** attempt)
                        continue
                self._handle_api_error(e)  # 4xx other than 404
            except urllib.error.URLError as e:
                last_exc = e
                if attempt < self._max_retries:
                    self._sleep(2 ** attempt)
                    continue
                self._handle_api_error(e)
            except Exception as e:
                self._handle_api_error(e)
        else:
            self._handle_api_error(last_exc)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                "Ollama returned invalid JSON. Some models may not support JSON format "
                "mode — try a different model or set OLLAMA_JSON_FORMAT=false"
            ) from e

        content = data.get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("Ollama returned an empty response")
        return content

    def list_models(self) -> list:
        """Return list of model names available in this Ollama instance.

        Calls GET /api/tags.

        Raises:
            RuntimeError: If Ollama is not running or an HTTP error occurs.
        """
        req = self._urllib.Request(
            f"{self.base_url}/api/tags",
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        try:
            with self._urllib.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            self._handle_api_error(e)
        return [m["name"] for m in data.get("models", [])]


class GeminiProvider(AIProvider):
    """Google Gemini provider (gemini-2.0-flash, etc.)"""

    def _init_client(self, api_key: str) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai package is required: pip install google-genai")
        self.client = genai.Client(api_key=api_key)
        # Store types on instance so tests can mock it without the package installed.
        self._types = types

    def generate(self, prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        config = self._types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            self._handle_api_error(e)
        if not response.text:
            raise RuntimeError("Gemini returned an empty response")
        return response.text


class MistralProvider(AIProvider):
    """Mistral AI provider (mistral-small, mistral-large, etc.)"""

    def _init_client(self, api_key: str) -> None:
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError("mistralai package is required: pip install mistralai")
        self.client = Mistral(api_key=api_key)

    def generate(self, prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            self._handle_api_error(e)
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Mistral returned an empty response")
        return content


class CohereProvider(AIProvider):
    """Cohere provider (command-r, command-r-plus, etc.)"""

    def _init_client(self, api_key: str) -> None:
        try:
            import cohere
        except ImportError:
            raise ImportError("cohere package is required: pip install cohere")
        self.client = cohere.ClientV2(api_key=api_key)

    def generate(self, prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            self._handle_api_error(e)
        if not response.message.content:
            raise RuntimeError("Cohere returned an empty response")
        return response.message.content[0].text


_PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
    "mistral": MistralProvider,
    "cohere": CohereProvider,
}


def get_provider(
    provider_name: str, api_key: str, model: str,
    temperature: float, max_tokens: int,
) -> AIProvider:
    """Factory function to create an AI provider instance."""
    cls = _PROVIDERS.get(provider_name)
    if cls is None:
        supported = ", ".join(_PROVIDERS)
        raise ValueError(f"Unsupported provider: '{provider_name}'. Supported: {supported}")
    return cls(api_key, model, temperature, max_tokens)
