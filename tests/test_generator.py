"""Tests for testdata_ai.generator — TestDataGenerator and helpers."""

import json
import logging
from unittest.mock import patch, MagicMock

import pytest

from testdata_ai.contexts import CONTEXTS, ValidationError
from testdata_ai.generator import TestDataGenerator, _strip_markdown_fences, generate


class TestStripMarkdownFences:

    def test_plain_json_unchanged(self):
        text = '{"data": [1, 2, 3]}'
        assert _strip_markdown_fences(text) == text

    def test_strips_json_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert _strip_markdown_fences(text) == '{"a": 1}'

    def test_strips_uppercase_json_fence(self):
        text = '```JSON\n{"a": 1}\n```'
        assert _strip_markdown_fences(text) == '{"a": 1}'

    def test_strips_bare_fence(self):
        text = '```\n[1, 2, 3]\n```'
        assert _strip_markdown_fences(text) == '[1, 2, 3]'

    def test_strips_fence_with_trailing_whitespace(self):
        text = '```json\n{"x": 1}\n```  \n'
        assert _strip_markdown_fences(text) == '{"x": 1}'

    def test_preserves_whitespace_inside_json(self):
        inner = '{\n  "a": 1,\n  "b": 2\n}'
        text = f"```json\n{inner}\n```"
        assert _strip_markdown_fences(text) == inner

    def test_strips_missing_closing_fence(self):
        text = '```json\n{"a": 1}'  # missing closing fence
        assert _strip_markdown_fences(text) == '{"a": 1}'

    def test_strips_missing_opening_fence(self):
        text = '[1, 2, 3]\n```'  # missing opening fence
        assert _strip_markdown_fences(text) == '[1, 2, 3]'

    def test_empty_string(self):
        assert _strip_markdown_fences("") == ""

    def test_whitespace_only(self):
        assert _strip_markdown_fences("   \n  ") == ""


@pytest.fixture
def make_generator():
    """Create a TestDataGenerator with a mocked AI provider.

    The patches are only active during __init__; the returned generator keeps
    references to the mock provider, so calls to gen.generate() still use
    the mock even after the patch context exits.
    """
    def _make(provider_response):
        with patch("testdata_ai.generator.get_provider_config") as mock_config, \
             patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_config.return_value = MagicMock(
                provider="openai",
                api_key="sk-fake",
                model="test-model",
                temperature=0.7,
                max_tokens=4096,
            )
            mock_prov = MagicMock()
            mock_prov.generate.return_value = provider_response
            mock_get_prov.return_value = mock_prov

            gen = TestDataGenerator()
            return gen
    return _make


class TestTestDataGenerator:

    def test_generate_returns_list_of_dicts(self, make_generator):
        sample = CONTEXTS["banking_user"].sample
        response = json.dumps({"data": [sample]})
        gen = make_generator(response)
        result = gen.generate("banking_user", count=1)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == sample["name"]

    def test_generate_normalizes_dict_with_data_key(self, make_generator):
        records = [{"name": "A"}, {"name": "B"}]
        gen = make_generator(json.dumps({"data": records}))
        result = gen.generate("ecommerce_customer", count=2, validate=False)
        assert len(result) == 2

    def test_generate_normalizes_dict_with_arbitrary_key(self, make_generator):
        records = [{"name": "A"}]
        gen = make_generator(json.dumps({"customers": records}))
        result = gen.generate("ecommerce_customer", count=1, validate=False)
        assert len(result) == 1

    def test_generate_normalizes_bare_list(self, make_generator):
        records = [{"name": "A"}, {"name": "B"}]
        gen = make_generator(json.dumps(records))
        result = gen.generate("ecommerce_customer", count=2, validate=False)
        assert len(result) == 2

    def test_generate_wraps_single_object(self, make_generator):
        gen = make_generator(json.dumps({"name": "Solo"}))
        result = gen.generate("ecommerce_customer", count=1, validate=False)
        assert len(result) == 1
        assert result[0]["name"] == "Solo"

    def test_generate_wraps_non_dict_non_list_as_single_record(self, make_generator):
        gen = make_generator(json.dumps("hello"))
        result = gen.generate("ecommerce_customer", count=1, validate=False)
        assert result == ["hello"]

    def test_generate_strips_markdown_fences(self, make_generator):
        sample = CONTEXTS["saas_trial"].sample
        response = f'```json\n{json.dumps({"data": [sample]})}\n```'
        gen = make_generator(response)
        result = gen.generate("saas_trial", count=1)
        assert len(result) == 1

    def test_generate_raises_on_invalid_json(self, make_generator):
        gen = make_generator("this is not json at all")
        with pytest.raises(ValueError, match="not valid JSON"):
            gen.generate("ecommerce_customer", count=1)

    def test_generate_raises_on_unknown_context(self, make_generator):
        gen = make_generator("{}")
        with pytest.raises(ValueError, match="Unknown context"):
            gen.generate("nonexistent", count=1)

    @pytest.mark.parametrize("count", [0, -1, -100])
    def test_generate_raises_when_count_less_than_1(self, make_generator, count):
        gen = make_generator("{}")
        with pytest.raises(ValueError, match="count must be >= 1"):
            gen.generate("ecommerce_customer", count=count)

    def test_generate_raises_on_validation_failure(self, make_generator):
        incomplete = {"name": "Test"}
        gen = make_generator(json.dumps({"data": [incomplete]}))
        with pytest.raises(ValidationError, match="missing") as exc_info:
            gen.generate("banking_user", count=1)
        assert len(exc_info.value.invalid_records) == 1
        assert "balance" in exc_info.value.invalid_records[0]["missing_fields"]

    def test_generate_skips_validation_when_disabled(self, make_generator):
        incomplete = {"name": "Test"}
        gen = make_generator(json.dumps({"data": [incomplete]}))
        result = gen.generate("banking_user", count=1, validate=False)
        assert len(result) == 1

    def test_generate_warns_when_count_exceeds_50(self, make_generator, caplog):
        sample = CONTEXTS["banking_user"].sample
        gen = make_generator(json.dumps({"data": [sample]}))
        with caplog.at_level(logging.WARNING, logger="testdata_ai.generator"):
            gen.generate("banking_user", count=51, validate=False)
        assert "may exceed token limits" in caplog.text

    def test_generate_warns_on_count_mismatch(self, make_generator, caplog):
        records = [CONTEXTS["banking_user"].sample]
        gen = make_generator(json.dumps({"data": records}))
        with caplog.at_level(logging.WARNING, logger="testdata_ai.generator"):
            gen.generate("banking_user", count=5, validate=False)
        assert "Requested 5 records but received 1" in caplog.text

    def test_generate_dict_with_multiple_lists_uses_first(self, make_generator):
        """When dict has multiple list values, the first one is used."""
        response = json.dumps({"data": [{"a": 1}], "extra": [{"b": 2}]})
        gen = make_generator(response)
        result = gen.generate("ecommerce_customer", count=1, validate=False)
        assert len(result) == 1
        assert result[0] == {"a": 1}

    def test_list_available_contexts(self, make_generator):
        gen = make_generator("{}")
        result = gen.list_available_contexts()
        assert "ecommerce_customer" in result
        assert "banking_user" in result

    def test_get_context_details(self, make_generator):
        gen = make_generator("{}")
        schema = gen.get_context_details("saas_trial")
        assert schema.category == "saas"


class TestGeneratorInit:

    def test_raises_when_api_key_given_without_provider(self):
        with pytest.raises(ValueError, match="must specify provider"):
            TestDataGenerator(api_key="sk-test")

    def test_raises_when_api_key_is_empty(self):
        with pytest.raises(ValueError, match="api_key must not be empty"):
            TestDataGenerator(api_key="   ", provider="openai")

    def test_raises_for_unsupported_provider_with_api_key(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            TestDataGenerator(api_key="sk-test", provider="mistral")

    def test_init_with_explicit_api_key_and_model(self):
        with patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_get_prov.return_value = MagicMock()
            gen = TestDataGenerator(
                api_key="sk-test", provider="openai", model="gpt-4o", temperature=0.5
            )
        assert gen.config.provider == "openai"
        assert gen.config.model == "gpt-4o"
        assert gen.config.temperature == 0.5

    def test_init_with_api_key_uses_default_model_and_temperature(self):
        with patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_get_prov.return_value = MagicMock()
            gen = TestDataGenerator(api_key="sk-test", provider="openai")
        assert gen.config.model == "gpt-4o-mini"
        assert gen.config.temperature == 0.7

    def test_init_with_provider_only_uses_env_config(self):
        """provider= without api_key falls through to get_provider_config."""
        with patch("testdata_ai.generator.get_provider_config") as mock_config, \
             patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_config.return_value = MagicMock(
                provider="anthropic",
                api_key="ant-from-env",
                model="claude-haiku",
                temperature=0.7,
                max_tokens=4096,
            )
            mock_get_prov.return_value = MagicMock()
            gen = TestDataGenerator(provider="anthropic")
        mock_config.assert_called_once_with("anthropic")
        assert gen.config.provider == "anthropic"


class TestGenerateConvenienceFunction:

    def test_calls_generator_and_returns_records(self):
        sample = CONTEXTS["banking_user"].sample
        with patch("testdata_ai.generator.TestDataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = [sample]
            mock_cls.return_value = mock_instance

            result = generate("banking_user", count=1)

        assert result == [sample]
        mock_cls.assert_called_once_with()
        mock_instance.generate.assert_called_once_with("banking_user", 1)

    def test_uses_default_count_of_10(self):
        with patch("testdata_ai.generator.TestDataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = []
            mock_cls.return_value = mock_instance

            generate("ecommerce_customer")

        mock_instance.generate.assert_called_once_with("ecommerce_customer", 10)
