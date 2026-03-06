"""Tests for testdata_ai.generator — batched generation and module-level convenience functions."""

import json
import logging
import pytest
from unittest.mock import patch, MagicMock

from testdata_ai.contexts import CONTEXTS
from testdata_ai.generator import DataGenerator, generate, generate_batched


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
        with patch("testdata_ai.generator.get_provider_config") as mock_config, \
             patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_config.return_value = MagicMock(
                provider="openai", api_key="sk-fake",
                model="test-model", temperature=0.7, max_tokens=4096,
            )
            mock_prov = MagicMock()
            mock_prov.generate.side_effect = [
                json.dumps({"data": [sample] * 10}),
                json.dumps({"data": [sample] * 5}),
            ]
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
