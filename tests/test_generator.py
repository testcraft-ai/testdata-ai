"""Tests for testdata_ai.generator — DataGenerator and helpers."""

import json
import logging
from unittest.mock import patch, MagicMock

import pytest

from testdata_ai.contexts import CONTEXTS, ValidationError
from testdata_ai.generator import (
    DataGenerator, _strip_markdown_fences, generate, generate_batched, generate_from_model,
)


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
    """Create a DataGenerator with a mocked AI provider.

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

            gen = DataGenerator()
            return gen
    return _make


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
        """When dict has multiple list values, the first one is used."""
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
            DataGenerator(api_key="sk-test", provider="mistral")

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


class TestGenerateConvenienceFunction:

    def test_calls_generator_and_returns_records(self):
        sample = CONTEXTS["banking_user"].sample
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate_batched.return_value = iter([[sample]])
            mock_cls.return_value = mock_instance

            result = generate("banking_user", count=1)

        assert result == [sample]
        mock_cls.assert_called_once_with(locale=None)
        mock_instance.generate_batched.assert_called_once_with("banking_user", 1, 10, validate=True)

    def test_uses_default_count_of_10(self):
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate_batched.return_value = iter([[]])
            mock_cls.return_value = mock_instance

            generate("ecommerce_customer")

        mock_instance.generate_batched.assert_called_once_with("ecommerce_customer", 10, 10, validate=True)

    def test_auto_batches_when_count_exceeds_batch_size(self):
        """generate() uses generate_batched when count > batch_size."""
        sample = CONTEXTS["banking_user"].sample
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate_batched.return_value = iter([[sample] * 10, [sample] * 5])
            mock_cls.return_value = mock_instance

            result = generate("banking_user", count=15, batch_size=10)

        assert len(result) == 15
        mock_instance.generate_batched.assert_called_once_with("banking_user", 15, 10, validate=True)
        mock_instance.generate.assert_not_called()

    def test_validate_false_forwarded_to_generate(self):
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate_batched.return_value = iter([[]])
            mock_cls.return_value = mock_instance

            generate("ecommerce_customer", count=5, validate=False)

        mock_instance.generate_batched.assert_called_once_with("ecommerce_customer", 5, 10, validate=False)

    def test_validate_false_forwarded_to_generate_batched(self):
        sample = CONTEXTS["banking_user"].sample
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate_batched.return_value = iter([[sample] * 10, [sample] * 5])
            mock_cls.return_value = mock_instance

            generate("banking_user", count=15, batch_size=10, validate=False)

        mock_instance.generate_batched.assert_called_once_with("banking_user", 15, 10, validate=False)


class TestGenerateBatched:

    def test_yields_single_batch_when_count_lte_batch_size(self, make_generator):
        sample = CONTEXTS["banking_user"].sample
        gen = make_generator(json.dumps({"data": [sample] * 5}))
        batches = list(gen.generate_batched("banking_user", count=5, batch_size=10, validate=False))
        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_yields_multiple_batches(self):
        sample = CONTEXTS["banking_user"].sample
        with patch("testdata_ai.generator.get_provider_config") as mock_config, \
             patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_config.return_value = MagicMock(
                provider="openai", api_key="sk-fake",
                model="test-model", temperature=0.7, max_tokens=4096,
            )
            mock_prov = MagicMock()
            mock_prov.generate.side_effect = [
                json.dumps({"data": [sample] * 10}),
                json.dumps({"data": [sample] * 10}),
                json.dumps({"data": [sample] * 5}),
            ]
            mock_get_prov.return_value = mock_prov
            gen = DataGenerator()

        with patch("testdata_ai.generator.get_prompt") as mock_prompt:
            mock_prompt.side_effect = lambda _, n, locale=None: f"generate {n}"
            batches = list(gen.generate_batched("banking_user", count=25, batch_size=10, validate=False))

        assert len(batches) == 3
        assert [len(b) for b in batches] == [10, 10, 5]
        counts_requested = [call.args[1] for call in mock_prompt.call_args_list]
        assert counts_requested == [10, 10, 5]

    def test_last_batch_requests_remaining_count(self, make_generator):
        sample = CONTEXTS["banking_user"].sample
        responses = [
            json.dumps({"data": [sample] * 10}),
            json.dumps({"data": [sample] * 5}),
        ]
        with patch("testdata_ai.generator.get_provider_config") as mock_config, \
             patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_config.return_value = MagicMock(
                provider="openai", api_key="sk-fake",
                model="test-model", temperature=0.7, max_tokens=4096,
            )
            mock_prov = MagicMock()
            mock_prov.generate.side_effect = responses
            mock_get_prov.return_value = mock_prov
            gen = DataGenerator()

        batches = list(gen.generate_batched("banking_user", count=15, batch_size=10, validate=False))
        assert len(batches) == 2
        assert mock_prov.generate.call_count == 2

    def test_no_extra_calls_when_ai_underdelivers(self):
        """AI returning fewer records than requested must not cause extra API calls."""
        sample = CONTEXTS["banking_user"].sample
        with patch("testdata_ai.generator.get_provider_config") as mock_config, \
             patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_config.return_value = MagicMock(
                provider="openai", api_key="sk-fake",
                model="test-model", temperature=0.7, max_tokens=4096,
            )
            mock_prov = MagicMock()
            # AI returns only 3 records when asked for 10
            mock_prov.generate.side_effect = [
                json.dumps({"data": [sample] * 3}),
                json.dumps({"data": [sample] * 3}),
            ]
            mock_get_prov.return_value = mock_prov
            gen = DataGenerator()

        with patch("testdata_ai.generator.get_prompt") as mock_prompt:
            mock_prompt.side_effect = lambda _, n, locale=None: f"generate {n}"
            batches = list(gen.generate_batched("banking_user", count=20, batch_size=10, validate=False))

        assert mock_prov.generate.call_count == 2
        assert all(len(b) > 0 for b in batches)

    def test_empty_batch_stops_iteration_without_yielding(self, make_generator):
        """An empty AI response stops iteration; no empty batch is yielded."""
        gen = make_generator(json.dumps({"data": []}))
        batches = list(gen.generate_batched("banking_user", count=5, batch_size=10, validate=False))
        assert batches == []

    @pytest.mark.parametrize("count", [0, -1, -100])
    def test_raises_when_count_less_than_1(self, make_generator, count):
        gen = make_generator("{}")
        with pytest.raises(ValueError, match="count must be >= 1"):
            list(gen.generate_batched("banking_user", count=count, batch_size=10))

    def test_raises_on_invalid_batch_size(self, make_generator):
        gen = make_generator("{}")
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            list(gen.generate_batched("banking_user", count=5, batch_size=0))

    def test_total_records_equal_sum_of_batches(self, make_generator):
        sample = CONTEXTS["banking_user"].sample
        gen = make_generator(json.dumps({"data": [sample] * 10}))
        batches = list(gen.generate_batched("banking_user", count=30, batch_size=10, validate=False))
        assert sum(len(b) for b in batches) == 30

    def test_warns_when_total_yielded_less_than_count(self, caplog):
        """Summary warning is emitted when AI underdelivers across all batches."""
        sample = CONTEXTS["banking_user"].sample
        with patch("testdata_ai.generator.get_provider_config") as mock_config, \
             patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_config.return_value = MagicMock(
                provider="openai", api_key="sk-fake",
                model="test-model", temperature=0.7, max_tokens=4096,
            )
            mock_prov = MagicMock()
            mock_prov.generate.side_effect = [
                json.dumps({"data": [sample] * 3}),
                json.dumps({"data": [sample] * 3}),
            ]
            mock_get_prov.return_value = mock_prov
            gen = DataGenerator()

        with caplog.at_level(logging.WARNING, logger="testdata_ai.generator"):
            batches = list(gen.generate_batched("banking_user", count=20, batch_size=10, validate=False))

        total = sum(len(b) for b in batches)
        assert total == 6
        assert "Requested 20 total records but generated 6" in caplog.text

    def test_module_level_generate_batched(self):
        sample = CONTEXTS["banking_user"].sample
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate_batched.return_value = iter([[sample]])
            mock_cls.return_value = mock_instance

            batches = list(generate_batched("banking_user", count=1, batch_size=10))

        assert batches == [[sample]]
        mock_cls.assert_called_once_with(locale=None)
        mock_instance.generate_batched.assert_called_once_with("banking_user", 1, 10, validate=True)

    def test_module_level_generate_batched_validate_false(self):
        sample = CONTEXTS["banking_user"].sample
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate_batched.return_value = iter([[sample]])
            mock_cls.return_value = mock_instance

            list(generate_batched("banking_user", count=1, batch_size=10, validate=False))

        mock_instance.generate_batched.assert_called_once_with("banking_user", 1, 10, validate=False)


# ---------------------------------------------------------------------------
# Fake Pydantic-like model for generate_from_model tests
# ---------------------------------------------------------------------------

class _FakeModel:
    __name__ = "FakeModel"

    @classmethod
    def model_json_schema(cls):
        return {
            "title": "FakeModel",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }


class TestGenerateFromModel:

    def test_returns_list_of_dicts_from_pydantic_model(self, make_generator):
        records = [{"name": "Alice", "age": 30}]
        gen = make_generator(json.dumps({"data": records}))
        result = gen.generate_from_model(_FakeModel, count=1)
        assert result == records

    def test_returns_list_of_dicts_from_json_schema_dict(self, make_generator):
        schema = {
            "title": "Widget",
            "properties": {"name": {"type": "string"}, "qty": {"type": "integer"}},
        }
        records = [{"name": "Bolt", "qty": 5}]
        gen = make_generator(json.dumps({"data": records}))
        result = gen.generate_from_model(schema, count=1)
        assert result == records

    def test_validate_false_skips_validation(self, make_generator):
        gen = make_generator(json.dumps({"data": [{"name": "X"}]}))  # missing age
        result = gen.generate_from_model(_FakeModel, count=1, validate=False)
        assert len(result) == 1

    def test_validation_error_on_missing_fields(self, make_generator):
        gen = make_generator(json.dumps({"data": [{"name": "X"}]}))  # missing age
        with pytest.raises(ValidationError):
            gen.generate_from_model(_FakeModel, count=1, validate=True)

    def test_raises_for_count_less_than_1(self, make_generator):
        gen = make_generator("{}")
        with pytest.raises(ValueError, match="count must be >= 1"):
            gen.generate_from_model(_FakeModel, count=0)

    def test_raises_for_invalid_input_type(self, make_generator):
        gen = make_generator("{}")
        with pytest.raises(TypeError):
            gen.generate_from_model("not_a_model", count=1)

    def test_raises_on_invalid_json_response(self, make_generator):
        gen = make_generator("not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            gen.generate_from_model(_FakeModel, count=1)

    def test_module_level_generate_from_model(self):
        records = [{"name": "Bob", "age": 25}]
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate_from_model.return_value = records
            mock_cls.return_value = mock_instance

            result = generate_from_model(_FakeModel, count=1)

        assert result == records
        mock_cls.assert_called_once_with(locale=None)
        mock_instance.generate_from_model.assert_called_once_with(
            _FakeModel, 1, True, field_providers=None, unique_fields=None
        )

    def test_module_level_passes_locale(self):
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate_from_model.return_value = []
            mock_cls.return_value = mock_instance

            generate_from_model(_FakeModel, count=1, locale="pl")

        mock_cls.assert_called_once_with(locale="pl")

    def test_generate_from_model_applies_field_providers(self, make_generator):
        """field_providers overwrites specified fields after AI parsing."""
        records = [{"name": "Alice", "age": 30, "email": "ai@gen.com"}]
        gen = make_generator(json.dumps({"data": records}))

        fake_instance = MagicMock()
        fake_instance.email.return_value = "faker@example.com"
        fake_cls = MagicMock(return_value=fake_instance)

        with patch("testdata_ai.faker_bridge.Faker", fake_cls):
            result = gen.generate_from_model(
                _FakeModel,
                count=1,
                validate=False,
                field_providers={"email": "faker:email"},
            )

        assert result[0]["email"] == "faker@example.com"
        assert result[0]["name"] == "Alice"  # AI value preserved

    def test_module_level_generate_from_model_passes_field_providers(self):
        records = [{"name": "Bob", "age": 25}]
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate_from_model.return_value = records
            mock_cls.return_value = mock_instance

            fp = {"email": "faker:email"}
            result = generate_from_model(_FakeModel, count=1, field_providers=fp)

        assert result == records
        mock_instance.generate_from_model.assert_called_once_with(
            _FakeModel, 1, True, field_providers=fp, unique_fields=None
        )


class TestFakerHybridGenerate:
    """Integration tests for Faker hybrid mode in DataGenerator.generate()."""

    def test_generate_applies_field_providers_from_schema(self, make_generator):
        """generate() overwrites fields declared in ContextSchema.field_providers."""
        from testdata_ai.contexts import register_context, ContextSchema

        sample = {"name": "Jan", "email": "jan@x.com"}
        register_context(
            "test_faker_ctx",
            ContextSchema(
                description="test context",
                sample=sample,
                prompt_hints=[],
                field_providers={"email": "faker:email"},
            ),
            overwrite=True,
        )

        ai_response = json.dumps({"data": [{"name": "AI Name", "email": "ai@ai.com"}]})
        gen = make_generator(ai_response)

        fake_instance = MagicMock()
        fake_instance.email.return_value = "faker@example.com"
        fake_cls = MagicMock(return_value=fake_instance)

        with patch("testdata_ai.faker_bridge.Faker", fake_cls):
            result = gen.generate("test_faker_ctx", count=1, validate=False)

        assert result[0]["email"] == "faker@example.com"
        assert result[0]["name"] == "AI Name"

    def test_generate_no_field_providers_unchanged(self, make_generator):
        """generate() without field_providers leaves AI output untouched."""
        sample = CONTEXTS["banking_user"].sample
        gen = make_generator(json.dumps({"data": [sample]}))
        result = gen.generate("banking_user", count=1)
        assert result[0]["name"] == sample["name"]

    def test_generate_passes_locale_to_faker(self, make_generator):
        """Faker is initialized with DataGenerator.locale."""
        from testdata_ai.contexts import register_context, ContextSchema

        register_context(
            "test_faker_locale_ctx",
            ContextSchema(
                description="locale test",
                sample={"phone": "000"},
                prompt_hints=[],
                field_providers={"phone": "faker:phone_number"},
            ),
            overwrite=True,
        )

        ai_response = json.dumps({"data": [{"phone": "000"}]})
        gen = make_generator(ai_response)
        gen.locale = "pl_PL"

        fake_instance = MagicMock()
        fake_instance.phone_number.return_value = "+48 123 456 789"
        fake_cls = MagicMock(return_value=fake_instance)

        with patch("testdata_ai.faker_bridge.Faker", fake_cls):
            gen.generate("test_faker_locale_ctx", count=1, validate=False)

        fake_cls.assert_called_once_with("pl_PL")


class TestUniqueFieldsIntegration:
    """unique_fields flows through DataGenerator.generate() and generate_from_model()."""

    def test_generate_passes_unique_fields_to_apply_faker_fields(self, make_generator):
        from testdata_ai.contexts import register_context, ContextSchema

        register_context(
            "test_unique_ctx",
            ContextSchema(
                description="unique test",
                sample={"email": "x@x.com"},
                prompt_hints=[],
                field_providers={"email": "faker:email"},
                unique_fields=["email"],
            ),
            overwrite=True,
        )

        ai_response = json.dumps({"data": [{"email": "ai@ai.com"}]})
        gen = make_generator(ai_response)

        with patch("testdata_ai.faker_bridge.apply_faker_fields") as mock_apply:
            mock_apply.return_value = [{"email": "faker@example.com"}]
            gen.generate("test_unique_ctx", count=1, validate=False)

        mock_apply.assert_called_once_with(
            [{"email": "ai@ai.com"}],
            {"email": "faker:email"},
            locale=None,
            unique_fields=["email"],
        )

    def test_generate_from_model_passes_unique_fields(self, make_generator):
        records = [{"name": "Alice", "age": 30}]
        gen = make_generator(json.dumps({"data": records}))

        with patch("testdata_ai.faker_bridge.apply_faker_fields") as mock_apply:
            mock_apply.return_value = records
            gen.generate_from_model(
                _FakeModel,
                count=1,
                validate=False,
                field_providers={"name": "faker:name"},
                unique_fields=["name"],
            )

        mock_apply.assert_called_once_with(
            records,
            {"name": "faker:name"},
            locale=None,
            unique_fields=["name"],
        )

    def test_module_level_generate_from_model_passes_unique_fields(self):
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate_from_model.return_value = []
            mock_cls.return_value = mock_instance

            generate_from_model(
                _FakeModel,
                count=1,
                field_providers={"name": "faker:name"},
                unique_fields=["name"],
            )

        mock_instance.generate_from_model.assert_called_once_with(
            _FakeModel, 1, True,
            field_providers={"name": "faker:name"},
            unique_fields=["name"],
        )
