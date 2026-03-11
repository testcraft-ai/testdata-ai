"""Tests for testdata_ai.generator — generate_from_model (DataGenerator method and dispatch)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from testdata_ai.contexts import ValidationError
from testdata_ai.generator import generate
from testdata_ai.result_types import GenerateResult

_DG_DEFAULTS = dict(provider=None, model=None, temperature=None, max_tokens=None, api_key=None, locale=None)


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
        gen = make_generator(json.dumps({"data": [{"name": "X"}]}))
        result = gen.generate_from_model(_FakeModel, count=1, validate=False)
        assert len(result) == 1

    def test_validation_error_on_missing_fields(self, make_generator):
        gen = make_generator(json.dumps({"data": [{"name": "X"}]}))
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

    def test_dispatch_type_calls_generate_from_model(self):
        records = [{"name": "Bob", "age": 25}]
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate_from_model.return_value = records
            mock_cls.return_value = mock_instance

            result = generate(_FakeModel, count=1)

        assert isinstance(result, GenerateResult)
        assert result == records
        mock_cls.assert_called_once_with(**_DG_DEFAULTS)
        mock_instance.generate_from_model.assert_called_once_with(
            _FakeModel, 1, True, field_providers=None, unique_fields=None
        )

    def test_dispatch_type_passes_locale(self):
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate_from_model.return_value = []
            mock_cls.return_value = mock_instance

            generate(_FakeModel, count=1, locale="pl")

        mock_cls.assert_called_once_with(**{**_DG_DEFAULTS, "locale": "pl"})

    def test_dispatch_dict_schema_calls_generate_from_model(self):
        schema = {
            "title": "Widget",
            "properties": {"name": {"type": "string"}, "qty": {"type": "integer"}},
        }
        records = [{"name": "Bolt", "qty": 5}]
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate_from_model.return_value = records
            mock_cls.return_value = mock_instance

            result = generate(schema, count=1)

        assert isinstance(result, GenerateResult)
        mock_instance.generate_from_model.assert_called_once()

    def test_generate_from_model_applies_field_providers(self, make_generator):
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
        assert result[0]["name"] == "Alice"

    def test_dispatch_passes_field_providers(self):
        records = [{"name": "Bob", "age": 25}]
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.generate_from_model.return_value = records
            mock_cls.return_value = mock_instance

            fp = {"email": "faker:email"}
            result = generate(_FakeModel, count=1, field_providers=fp)

        assert isinstance(result, GenerateResult)
        mock_instance.generate_from_model.assert_called_once_with(
            _FakeModel, 1, True, field_providers=fp, unique_fields=None
        )
