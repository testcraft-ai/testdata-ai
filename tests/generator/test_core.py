"""Tests for testdata_ai.generator — DataGenerator core generate() and __init__."""

import json
import logging
import pytest
from unittest.mock import patch, MagicMock

from testdata_ai.contexts import CONTEXTS, ValidationError
from testdata_ai.generator import DataGenerator


class TestDataGenerator:

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

    def test_generate_warns_on_count_mismatch(self, make_generator, caplog):
        records = [CONTEXTS["banking_user"].sample]
        gen = make_generator(json.dumps({"data": records}))
        with caplog.at_level(logging.WARNING, logger="testdata_ai.generator"):
            gen.generate("banking_user", count=5, validate=False)
        assert "Requested 5 records but received 1" in caplog.text

    def test_generate_dict_with_multiple_lists_uses_first(self, make_generator):
        response = json.dumps({"data": [{"a": 1}], "extra": [{"b": 2}]})
        gen = make_generator(response)
        result = gen.generate("ecommerce_customer", count=1, validate=False)
        assert len(result) == 1
        assert result[0] == {"a": 1}

    def test_generate_passes_locale_to_prompt(self, make_generator):
        sample = CONTEXTS["ecommerce_customer"].sample
        gen = make_generator(json.dumps({"data": [sample]}))
        gen.locale = "pl"
        with patch("testdata_ai.generator.get_prompt", wraps=__import__("testdata_ai.prompts", fromlist=["get_prompt"]).get_prompt) as mock_prompt:
            gen.generate("ecommerce_customer", count=1, validate=False)
        mock_prompt.assert_called_once_with("ecommerce_customer", 1, locale="pl")


class TestGeneratorInit:

    def test_raises_when_api_key_given_without_provider(self):
        with pytest.raises(ValueError, match="must specify provider"):
            DataGenerator(api_key="sk-test")

    def test_raises_when_api_key_is_empty(self):
        with pytest.raises(ValueError, match="api_key must not be empty"):
            DataGenerator(api_key="   ", provider="openai")

    def test_raises_for_unsupported_provider_with_api_key(self):
        with pytest.raises(ValueError, match="Unsupported AI provider"):
            DataGenerator(api_key="sk-test", provider="fakeai")

    def test_init_with_explicit_api_key_and_model(self):
        with patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_get_prov.return_value = MagicMock()
            gen = DataGenerator(
                api_key="sk-test", provider="openai", model="gpt-4o", temperature=0.5
            )
        assert gen.config.provider == "openai"
        assert gen.config.model == "gpt-4o"
        assert gen.config.temperature == 0.5

    def test_init_with_api_key_uses_default_model_and_temperature(self, clean_ai_env_fixture):
        with patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_get_prov.return_value = MagicMock()
            gen = DataGenerator(api_key="sk-test", provider="openai")
        assert gen.config.model == "gpt-4o-mini"
        assert gen.config.temperature == 0.7

    def test_init_with_provider_only_uses_env_config(self):
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
            gen = DataGenerator(provider="anthropic")
        mock_config.assert_called_once_with("anthropic")
        assert gen.config.provider == "anthropic"

    def test_init_raises_on_temperature_out_of_range(self):
        with patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_get_prov.return_value = MagicMock()
            with pytest.raises(ValueError, match="temperature must be 0.0-1.0"):
                DataGenerator(api_key="sk-test", provider="openai", temperature=1.5)

    def test_init_accepts_explicit_max_tokens(self):
        with patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_get_prov.return_value = MagicMock()
            gen = DataGenerator(api_key="sk-test", provider="openai", max_tokens=2048)
        assert gen.config.max_tokens == 2048

    def test_set_max_tokens_updates_config_and_provider(self):
        with patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_prov = MagicMock()
            mock_get_prov.return_value = mock_prov
            gen = DataGenerator(api_key="sk-test", provider="openai")
        gen.set_max_tokens(8192)
        assert gen.config.max_tokens == 8192
        assert mock_prov.max_tokens == 8192

    def test_locale_stored_on_instance(self):
        with patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_get_prov.return_value = MagicMock()
            gen = DataGenerator(api_key="sk-test", provider="openai", locale="pl")
        assert gen.locale == "pl"

    def test_locale_defaults_to_none_without_env(self, monkeypatch):
        monkeypatch.delenv("AI_LOCALE", raising=False)
        with patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_get_prov.return_value = MagicMock()
            gen = DataGenerator(api_key="sk-test", provider="openai")
        assert gen.locale is None

    def test_locale_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("AI_LOCALE", "ja")
        with patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_get_prov.return_value = MagicMock()
            gen = DataGenerator(api_key="sk-test", provider="openai")
        assert gen.locale == "ja"

    def test_explicit_locale_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AI_LOCALE", "de")
        with patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_get_prov.return_value = MagicMock()
            gen = DataGenerator(api_key="sk-test", provider="openai", locale="pl")
        assert gen.locale == "pl"
