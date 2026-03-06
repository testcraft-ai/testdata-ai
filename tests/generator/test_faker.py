"""Tests for testdata_ai.generator — Faker hybrid mode and unique_fields integration."""

import json
import pytest
from unittest.mock import patch, MagicMock

from testdata_ai.contexts import CONTEXTS, register_context, ContextSchema
from testdata_ai.generator import generate_from_model


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


@pytest.mark.usefixtures("clean_contexts")
class TestFakerHybridGenerate:
    """Integration tests for Faker hybrid mode in DataGenerator.generate()."""

    def test_generate_applies_field_providers_from_schema(self, make_generator):
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
        sample = CONTEXTS["banking_user"].sample
        gen = make_generator(json.dumps({"data": [sample]}))
        result = gen.generate("banking_user", count=1)
        assert result[0]["name"] == sample["name"]

    def test_generate_passes_locale_to_faker(self, make_generator):
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


@pytest.mark.usefixtures("clean_contexts")
class TestUniqueFieldsIntegration:
    """unique_fields flows through DataGenerator.generate() and generate_from_model()."""

    def test_generate_passes_unique_fields_to_apply_faker_fields(self, make_generator):
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
