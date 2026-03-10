"""Tests for testdata_ai.pandas_bridge — records_to_dataframe, relationships_to_dataframes."""
from unittest.mock import MagicMock, patch

import pytest

from testdata_ai.pandas_bridge import records_to_dataframe, relationships_to_dataframes


@pytest.fixture
def mock_pd():
    """Return a mock pandas module with json_normalize and DataFrame."""
    m = MagicMock()
    mock_df = MagicMock()
    m.json_normalize.return_value = mock_df
    m.DataFrame.return_value = mock_df
    return m, mock_df


SAMPLE_RECORDS = [
    {"name": "Alice", "email": "alice@example.com"},
    {"name": "Bob", "email": "bob@example.com"},
]

NESTED_RECORDS = [
    {"name": "Alice", "address": {"city": "Warsaw", "zip": "00-001"}},
    {"name": "Bob", "address": {"city": "Krakow", "zip": "30-001"}},
]


class TestRecordsToDataframe:

    def test_flatten_true_calls_json_normalize(self, mock_pd):
        m, mock_df = mock_pd
        with patch("testdata_ai.pandas_bridge.pd", m):
            result = records_to_dataframe(SAMPLE_RECORDS, flatten=True)
        m.json_normalize.assert_called_once_with(SAMPLE_RECORDS)
        m.DataFrame.assert_not_called()
        assert result is mock_df

    def test_flatten_false_calls_dataframe_constructor(self, mock_pd):
        m, mock_df = mock_pd
        with patch("testdata_ai.pandas_bridge.pd", m):
            result = records_to_dataframe(SAMPLE_RECORDS, flatten=False)
        m.DataFrame.assert_called_once_with(SAMPLE_RECORDS)
        m.json_normalize.assert_not_called()
        assert result is mock_df

    def test_default_flatten_is_true(self, mock_pd):
        m, mock_df = mock_pd
        with patch("testdata_ai.pandas_bridge.pd", m):
            records_to_dataframe(SAMPLE_RECORDS)
        m.json_normalize.assert_called_once()
        m.DataFrame.assert_not_called()

    def test_returns_dataframe_object(self, mock_pd):
        m, mock_df = mock_pd
        with patch("testdata_ai.pandas_bridge.pd", m):
            result = records_to_dataframe(SAMPLE_RECORDS)
        assert result is mock_df

    def test_empty_records_list(self, mock_pd):
        m, mock_df = mock_pd
        with patch("testdata_ai.pandas_bridge.pd", m):
            result = records_to_dataframe([], flatten=True)
        m.json_normalize.assert_called_once_with([])
        assert result is mock_df

    def test_nested_records_flatten_true_uses_json_normalize(self, mock_pd):
        m, mock_df = mock_pd
        with patch("testdata_ai.pandas_bridge.pd", m):
            records_to_dataframe(NESTED_RECORDS, flatten=True)
        m.json_normalize.assert_called_once_with(NESTED_RECORDS)

    def test_original_records_list_not_mutated(self, mock_pd):
        m, _ = mock_pd
        original = [{"name": "Alice", "score": 42}]
        copy = [dict(r) for r in original]
        with patch("testdata_ai.pandas_bridge.pd", m):
            records_to_dataframe(original)
        assert original == copy

    def test_pandas_not_installed_raises_import_error(self):
        with patch("testdata_ai.pandas_bridge.pd", None):
            with pytest.raises(ImportError, match="pip install"):
                records_to_dataframe(SAMPLE_RECORDS)

    def test_import_error_message_mentions_extra(self):
        with patch("testdata_ai.pandas_bridge.pd", None):
            with pytest.raises(ImportError, match="testdata-ai\\[pandas\\]"):
                records_to_dataframe(SAMPLE_RECORDS)


class TestRelationshipsToDataframes:

    def test_returns_dict_of_dataframes(self, mock_pd):
        m, mock_df = mock_pd
        result_data = {
            "users": SAMPLE_RECORDS,
            "orders": [{"order_id": 1}, {"order_id": 2}],
        }
        with patch("testdata_ai.pandas_bridge.pd", m):
            result = relationships_to_dataframes(result_data)
        assert set(result.keys()) == {"users", "orders"}

    def test_keys_match_entity_names(self, mock_pd):
        m, _ = mock_pd
        result_data = {"customers": SAMPLE_RECORDS, "products": [{"name": "Widget"}]}
        with patch("testdata_ai.pandas_bridge.pd", m):
            result = relationships_to_dataframes(result_data)
        assert list(result.keys()) == ["customers", "products"]

    def test_json_normalize_called_per_entity_flatten_true(self, mock_pd):
        m, _ = mock_pd
        result_data = {"a": [{"x": 1}], "b": [{"y": 2}]}
        with patch("testdata_ai.pandas_bridge.pd", m):
            relationships_to_dataframes(result_data, flatten=True)
        assert m.json_normalize.call_count == 2

    def test_dataframe_called_per_entity_flatten_false(self, mock_pd):
        m, _ = mock_pd
        result_data = {"a": [{"x": 1}], "b": [{"y": 2}]}
        with patch("testdata_ai.pandas_bridge.pd", m):
            relationships_to_dataframes(result_data, flatten=False)
        assert m.DataFrame.call_count == 2
        m.json_normalize.assert_not_called()

    def test_empty_result_dict(self, mock_pd):
        m, _ = mock_pd
        with patch("testdata_ai.pandas_bridge.pd", m):
            result = relationships_to_dataframes({})
        assert result == {}
        m.json_normalize.assert_not_called()

    def test_pandas_not_installed_raises_import_error(self):
        with patch("testdata_ai.pandas_bridge.pd", None):
            with pytest.raises(ImportError, match="pip install"):
                relationships_to_dataframes({"users": SAMPLE_RECORDS})


class TestDataGeneratorGenerateAsDataframe:

    def test_returns_dataframe(self, make_generator, mock_pd):
        m, mock_df = mock_pd
        gen = make_generator('[{"name": "Alice", "email": "a@a.com"}]')
        with patch("testdata_ai.pandas_bridge.pd", m):
            result = gen.generate_as_dataframe("ecommerce_customer", count=1, validate=False)
        assert result is mock_df

    def test_flatten_true_default_uses_json_normalize(self, make_generator, mock_pd):
        m, _ = mock_pd
        gen = make_generator('[{"name": "Alice", "email": "a@a.com"}]')
        with patch("testdata_ai.pandas_bridge.pd", m):
            gen.generate_as_dataframe("ecommerce_customer", count=1, validate=False)
        m.json_normalize.assert_called_once()
        m.DataFrame.assert_not_called()

    def test_flatten_false_uses_dataframe_constructor(self, make_generator, mock_pd):
        m, _ = mock_pd
        gen = make_generator('[{"name": "Alice", "email": "a@a.com"}]')
        with patch("testdata_ai.pandas_bridge.pd", m):
            gen.generate_as_dataframe("ecommerce_customer", count=1, validate=False, flatten=False)
        m.DataFrame.assert_called_once()
        m.json_normalize.assert_not_called()

    def test_delegates_to_generate(self, make_generator, mock_pd):
        m, _ = mock_pd
        response = '[{"name": "Alice", "email": "a@a.com"}]'
        gen = make_generator(response)
        with patch("testdata_ai.pandas_bridge.pd", m):
            gen.generate_as_dataframe("ecommerce_customer", count=1, validate=False)
        gen.provider.generate.assert_called_once()

    def test_pandas_not_installed_raises_import_error(self, make_generator):
        gen = make_generator('[{"name": "Alice", "email": "a@a.com"}]')
        with patch("testdata_ai.pandas_bridge.pd", None):
            with pytest.raises(ImportError, match="pip install"):
                gen.generate_as_dataframe("ecommerce_customer", count=1, validate=False)


class TestModuleLevelGenerateAsDataframe:

    def test_returns_dataframe(self, mock_pd):
        m, mock_df = mock_pd
        response = '[{"name": "Alice", "email": "a@a.com"}]'
        with patch("testdata_ai.generator.get_provider_config") as mock_cfg, \
             patch("testdata_ai.generator.get_provider") as mock_prov, \
             patch("testdata_ai.pandas_bridge.pd", m):
            mock_cfg.return_value = MagicMock(
                provider="openai", api_key="sk-fake",
                model="test-model", temperature=0.7, max_tokens=4096,
            )
            mock_prov.return_value = MagicMock()
            mock_prov.return_value.generate.return_value = response

            from testdata_ai.generator import generate_as_dataframe
            result = generate_as_dataframe("ecommerce_customer", count=1, validate=False)

        assert result is mock_df

    def test_pandas_not_installed_raises_import_error(self):
        with patch("testdata_ai.generator.DataGenerator") as mock_gen_cls, \
             patch("testdata_ai.pandas_bridge.pd", None):
            mock_instance = MagicMock()
            mock_gen_cls.return_value = mock_instance
            mock_instance.generate.return_value = [{"name": "Alice", "email": "a@a.com"}]

            from testdata_ai.generator import generate_as_dataframe
            with pytest.raises(ImportError, match="pip install"):
                generate_as_dataframe("ecommerce_customer", count=1, validate=False)

    def test_locale_forwarded_to_generator(self, mock_pd):
        m, _ = mock_pd
        with patch("testdata_ai.generator.DataGenerator") as mock_gen_cls, \
             patch("testdata_ai.pandas_bridge.pd", m):
            mock_instance = MagicMock()
            mock_gen_cls.return_value = mock_instance
            mock_instance.generate.return_value = [{"name": "Alice", "email": "a@a.com"}]

            from testdata_ai.generator import generate_as_dataframe
            generate_as_dataframe("ecommerce_customer", count=1, validate=False, locale="pl")

        mock_gen_cls.assert_called_once_with(locale="pl")
