"""Tests for generate() type dispatch in testdata_ai.generator."""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from testdata_ai.generator import generate
from testdata_ai.result_types import GenerateResult, RelationshipResult
from testdata_ai.async_generator import GenerateSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RECORDS = [{"name": "Alice", "email": "alice@example.com"}]

_DG_DEFAULTS = dict(
    provider=None, model=None, temperature=None, max_tokens=None, api_key=None, locale=None
)


def _make_mock_gen(batched_records=None, from_model_records=None, relationships_result=None):
    mock_gen = MagicMock()
    mock_gen.generate_batched.return_value = iter([[r] for r in (batched_records or _RECORDS)])
    mock_gen.generate_from_model.return_value = from_model_records or _RECORDS
    mock_gen.generate_with_relationships.return_value = relationships_result or {"users": _RECORDS}
    mock_gen.config = MagicMock(provider="openai", model="test", max_tokens=4096)
    mock_gen.provider = MagicMock(max_tokens=4096)
    return mock_gen


class _FakeModel:
    __name__ = "FakeModel"

    @classmethod
    def model_json_schema(cls):
        return {
            "title": "FakeModel",
            "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
        }


# ---------------------------------------------------------------------------
# str dispatch → context name
# ---------------------------------------------------------------------------


class TestDispatchStr:

    def test_returns_generate_result(self):
        with patch("testdata_ai.generator.DataGenerator", return_value=_make_mock_gen()):
            result = generate("ecommerce_customer", count=1)
        assert isinstance(result, GenerateResult)

    def test_calls_generate_batched(self):
        mock_gen = _make_mock_gen()
        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen):
            generate("ecommerce_customer", count=5, batch_size=3)
        mock_gen.generate_batched.assert_called_once_with("ecommerce_customer", 5, 3, validate=True)

    def test_records_flattened_from_batches(self):
        mock_gen = MagicMock()
        mock_gen.generate_batched.return_value = iter([[{"a": 1}], [{"a": 2}]])
        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen):
            result = generate("ecommerce_customer", count=2)
        assert len(result) == 2

    def test_locale_forwarded_to_datagenerator(self):
        mock_gen = _make_mock_gen()
        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen) as mock_cls:
            generate("ecommerce_customer", locale="pl")
        mock_cls.assert_called_once_with(**{**_DG_DEFAULTS, "locale": "pl"})

    def test_provider_forwarded(self):
        mock_gen = _make_mock_gen()
        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen) as mock_cls:
            generate("ecommerce_customer", provider="anthropic")
        mock_cls.assert_called_once_with(**{**_DG_DEFAULTS, "provider": "anthropic"})

    def test_validate_false_forwarded(self):
        mock_gen = _make_mock_gen()
        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen):
            generate("ecommerce_customer", validate=False)
        mock_gen.generate_batched.assert_called_once_with("ecommerce_customer", 10, 10, validate=False)


# ---------------------------------------------------------------------------
# type dispatch → Pydantic model / JSON Schema
# ---------------------------------------------------------------------------


class TestDispatchType:

    def test_pydantic_model_returns_generate_result(self):
        mock_gen = _make_mock_gen()
        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen):
            result = generate(_FakeModel, count=1)
        assert isinstance(result, GenerateResult)

    def test_calls_generate_from_model(self):
        mock_gen = _make_mock_gen()
        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen):
            generate(_FakeModel, count=3)
        mock_gen.generate_from_model.assert_called_once_with(
            _FakeModel, 3, True, field_providers=None, unique_fields=None
        )

    def test_json_schema_dict_dispatched_to_from_model(self):
        schema = {"properties": {"name": {"type": "string"}}}
        mock_gen = _make_mock_gen()
        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen):
            result = generate(schema, count=2)
        assert isinstance(result, GenerateResult)
        mock_gen.generate_from_model.assert_called_once_with(
            schema, 2, True, field_providers=None, unique_fields=None
        )

    def test_field_providers_forwarded(self):
        mock_gen = _make_mock_gen()
        fp = {"email": "faker:email"}
        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen):
            generate(_FakeModel, count=1, field_providers=fp)
        mock_gen.generate_from_model.assert_called_once_with(
            _FakeModel, 1, True, field_providers=fp, unique_fields=None
        )

    def test_unique_fields_forwarded(self):
        mock_gen = _make_mock_gen()
        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen):
            generate(_FakeModel, count=1, field_providers={"e": "faker:email"}, unique_fields=["e"])
        _, call_kwargs = mock_gen.generate_from_model.call_args
        assert call_kwargs["unique_fields"] == ["e"]


# ---------------------------------------------------------------------------
# dict with "nodes" dispatch → relationships
# ---------------------------------------------------------------------------


class TestDispatchGraph:

    def test_graph_dict_returns_relationship_result(self):
        graph = {"nodes": {"users": {"context": "ecommerce_customer", "count": 2}}}
        mock_gen = _make_mock_gen(relationships_result={"users": _RECORDS})
        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen):
            result = generate(graph)
        assert isinstance(result, RelationshipResult)

    def test_calls_generate_with_relationships(self):
        graph = {"nodes": {"users": {"context": "ecommerce_customer", "count": 2}}}
        mock_gen = _make_mock_gen(relationships_result={"users": _RECORDS})
        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen):
            generate(graph)
        mock_gen.generate_with_relationships.assert_called_once_with(
            graph["nodes"], validate=True, progress_callback=None
        )

    def test_validate_false_forwarded(self):
        graph = {"nodes": {"users": {"context": "ecommerce_customer", "count": 2}}}
        mock_gen = _make_mock_gen(relationships_result={"users": _RECORDS})
        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen):
            generate(graph, validate=False)
        mock_gen.generate_with_relationships.assert_called_once_with(
            graph["nodes"], validate=False, progress_callback=None
        )

    def test_dict_without_nodes_goes_to_from_model(self):
        schema = {"properties": {"name": {"type": "string"}}}
        mock_gen = _make_mock_gen()
        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen):
            generate(schema, count=1)
        mock_gen.generate_from_model.assert_called_once()
        mock_gen.generate_with_relationships.assert_not_called()


# ---------------------------------------------------------------------------
# list[GenerateSpec] dispatch → parallel
# ---------------------------------------------------------------------------


class TestDispatchList:

    def test_list_of_specs_returns_relationship_result(self):
        specs = [GenerateSpec("ecommerce_customer", 1)]
        parallel_result = {"ecommerce_customer": _RECORDS}
        with patch("testdata_ai.async_generator.generate_parallel", new=AsyncMock(return_value=parallel_result)), \
             patch("testdata_ai.generator.asyncio.get_running_loop", side_effect=RuntimeError):
            result = generate(specs)
        assert isinstance(result, RelationshipResult)

    def test_runtime_error_in_async_context(self):
        specs = [GenerateSpec("ecommerce_customer", 1)]
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True
        with patch("testdata_ai.generator.asyncio.get_running_loop", return_value=mock_loop):
            with pytest.raises(RuntimeError, match="async_generate"):
                generate(specs)


# ---------------------------------------------------------------------------
# TypeError for unsupported types
# ---------------------------------------------------------------------------


class TestDispatchTypeError:

    @pytest.mark.parametrize("bad_input", [42, 3.14, None, object()])
    def test_unsupported_type_raises_type_error(self, bad_input):
        with pytest.raises(TypeError, match="Unsupported input type"):
            generate(bad_input)
