"""Tests for testdata_ai.generator — batched generation and generate() convenience function."""

import json
import logging
import pytest
from unittest.mock import patch, MagicMock

from testdata_ai.contexts import CONTEXTS
from testdata_ai.generator import DataGenerator, generate
from testdata_ai.result_types import GenerateResult


def _make_gen_with_responses(responses):
    """Create a DataGenerator whose provider returns each response in turn."""
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
    gen._mock_prov = mock_prov
    return gen

_DG_DEFAULTS = dict(provider=None, model=None, temperature=None, max_tokens=None, api_key=None, locale=None)


class TestGenerateConvenienceFunction:

    def test_calls_generator_and_returns_generate_result(self):
        sample = CONTEXTS["banking_user"].sample
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate_batched.return_value = iter([[sample]])
            mock_cls.return_value = mock_instance

            result = generate("banking_user", count=1)

        assert isinstance(result, GenerateResult)
        assert result == [sample]
        mock_cls.assert_called_once_with(**_DG_DEFAULTS)
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

    def test_advances_to_next_batch_even_when_ai_underdelivers(self):
        """generate_batched() yields each batch and advances even when generate() returns short."""
        sample = CONTEXTS["banking_user"].sample
        # generate() now retries up to 3× per batch; provide enough responses for 2 batches.
        with patch("testdata_ai.generator.get_provider_config") as mock_config, \
             patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_config.return_value = MagicMock(
                provider="openai", api_key="sk-fake",
                model="test-model", temperature=0.7, max_tokens=4096,
            )
            mock_prov = MagicMock()
            mock_prov.generate.side_effect = [json.dumps({"data": [sample] * 3})] * 6
            mock_get_prov.return_value = mock_prov
            gen = DataGenerator()

        with patch("testdata_ai.generator.get_prompt") as mock_prompt:
            mock_prompt.side_effect = lambda _, n, locale=None: f"generate {n}"
            batches = list(gen.generate_batched("banking_user", count=20, batch_size=10, validate=False))

        assert len(batches) == 2
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
        # generate() now retries up to 3× per batch; 2 batches × 3 retries = 6 responses needed.
        # With 3 records per response, each batch yields 9, total = 18 < 20 → warning.
        with patch("testdata_ai.generator.get_provider_config") as mock_config, \
             patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_config.return_value = MagicMock(
                provider="openai", api_key="sk-fake",
                model="test-model", temperature=0.7, max_tokens=4096,
            )
            mock_prov = MagicMock()
            mock_prov.generate.side_effect = [json.dumps({"data": [sample] * 3})] * 6
            mock_get_prov.return_value = mock_prov
            gen = DataGenerator()

        with caplog.at_level(logging.WARNING, logger="testdata_ai.generator"):
            batches = list(gen.generate_batched("banking_user", count=20, batch_size=10, validate=False))

        total = sum(len(b) for b in batches)
        assert total == 18
        assert "Requested 20 total records but generated 18" in caplog.text


class TestGenerateRetry:
    """DataGenerator.generate() retries when AI returns fewer records than requested."""

    def test_retries_on_short_response_and_returns_full_count(self):
        sample = CONTEXTS["banking_user"].sample
        gen = _make_gen_with_responses([
            json.dumps({"data": [sample] * 9}),  # 9 instead of 10
            json.dumps({"data": [sample] * 5}),  # top-up (trimmed to 1)
        ])
        result = gen.generate("banking_user", count=10, validate=False)
        assert len(result) == 10
        assert gen._mock_prov.generate.call_count == 2

    def test_retry_requests_only_missing_count(self):
        sample = CONTEXTS["banking_user"].sample
        gen = _make_gen_with_responses([
            json.dumps({"data": [sample] * 7}),
            json.dumps({"data": [sample] * 3}),
        ])
        with patch("testdata_ai.generator.get_prompt") as mock_prompt:
            mock_prompt.side_effect = lambda ctx, n, locale=None: f"generate {n}"
            gen.generate("banking_user", count=10, validate=False)

        counts = [call.args[1] for call in mock_prompt.call_args_list]
        assert counts == [10, 3]

    def test_stops_after_three_attempts(self):
        sample = CONTEXTS["banking_user"].sample
        gen = _make_gen_with_responses([
            json.dumps({"data": [sample] * 3}),
            json.dumps({"data": [sample] * 3}),
            json.dumps({"data": [sample] * 3}),
        ])
        result = gen.generate("banking_user", count=15, validate=False)
        assert gen._mock_prov.generate.call_count == 3
        assert len(result) == 9

    def test_stops_on_empty_response(self):
        sample = CONTEXTS["banking_user"].sample
        gen = _make_gen_with_responses([
            json.dumps({"data": [sample] * 5}),
            json.dumps({"data": []}),
        ])
        result = gen.generate("banking_user", count=10, validate=False)
        assert gen._mock_prov.generate.call_count == 2
        assert len(result) == 5

    def test_trims_excess_records_from_retry(self):
        sample = CONTEXTS["banking_user"].sample
        gen = _make_gen_with_responses([
            json.dumps({"data": [sample] * 8}),
            json.dumps({"data": [sample] * 10}),  # more than needed
        ])
        result = gen.generate("banking_user", count=10, validate=False)
        assert len(result) == 10

    def test_warns_when_still_short_after_all_retries(self, caplog):
        sample = CONTEXTS["banking_user"].sample
        gen = _make_gen_with_responses([
            json.dumps({"data": [sample] * 3}),
            json.dumps({"data": [sample] * 3}),
            json.dumps({"data": [sample] * 3}),
        ])
        with caplog.at_level(logging.WARNING, logger="testdata_ai.generator"):
            gen.generate("banking_user", count=15, validate=False)
        assert "Requested 15 records but received 9" in caplog.text

    def test_no_retry_when_exact_count_on_first_call(self):
        sample = CONTEXTS["banking_user"].sample
        gen = _make_gen_with_responses([
            json.dumps({"data": [sample] * 10}),
        ])
        result = gen.generate("banking_user", count=10, validate=False)
        assert gen._mock_prov.generate.call_count == 1
        assert len(result) == 10

