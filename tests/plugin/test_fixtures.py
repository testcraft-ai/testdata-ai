"""Tests for testdata_ai pytest plugin — testdata fixture, context fixtures, lazy generator."""

import pytest
from unittest.mock import patch, MagicMock

import testdata_ai.pytest_plugin as plugin_mod
from testdata_ai.pytest_plugin import (
    _make_context_fixture,
    _LazyGenerator,
    _PluginConfigError,
    DEFAULT_COUNT,
    _get_cache_manager,
)


class TestTestdataFixture:
    """Test the testdata fixture by calling the underlying function directly."""

    @staticmethod
    def _call_testdata(request_mock):
        fn = plugin_mod.testdata._fixture_function
        return fn(request_mock)

    def test_fails_when_marker_is_missing(self):
        request = MagicMock()
        request.node.get_closest_marker.return_value = None
        request.config._testdata_cache_manager = MagicMock()

        with pytest.raises(pytest.fail.Exception, match="requires @pytest.mark.testdata"):
            self._call_testdata(request)

    def test_fails_when_context_missing_from_marker(self):
        marker = MagicMock()
        marker.kwargs = {}
        request = MagicMock()
        request.node.get_closest_marker.return_value = marker
        request.config._testdata_cache_manager = MagicMock()

        with pytest.raises(pytest.fail.Exception, match="requires 'context' argument"):
            self._call_testdata(request)

    def test_returns_data_from_cache_manager(self):
        marker = MagicMock()
        marker.kwargs = {"context": "ecommerce_customer", "count": 3}
        request = MagicMock()
        request.node.get_closest_marker.return_value = marker

        cm = MagicMock()
        expected = [{"id": 1}, {"id": 2}, {"id": 3}]
        cm.get_data.return_value = expected
        request.config._testdata_cache_manager = cm

        result = self._call_testdata(request)

        cm.get_data.assert_called_once_with("ecommerce_customer", 3, locale=None)
        assert result == expected

    def test_defaults_count_to_1(self):
        marker = MagicMock()
        marker.kwargs = {"context": "banking_user"}
        request = MagicMock()
        request.node.get_closest_marker.return_value = marker

        cm = MagicMock()
        cm.get_data.return_value = [{"name": "User"}]
        request.config._testdata_cache_manager = cm

        self._call_testdata(request)

        cm.get_data.assert_called_once_with("banking_user", 1, locale=None)

    def test_fails_with_actionable_message_when_plugin_setup_missing(self):
        marker = MagicMock()
        marker.kwargs = {"context": "banking_user", "count": 1}
        request = MagicMock()
        request.node.get_closest_marker.return_value = marker

        cm = MagicMock()
        cm.get_data.side_effect = _PluginConfigError("missing OPENAI_API_KEY")
        request.config._testdata_cache_manager = cm

        with pytest.raises(pytest.fail.Exception, match="OPENAI_API_KEY"):
            self._call_testdata(request)


class TestMakeContextFixture:

    @staticmethod
    def _call_fixture(fixture_func, request_mock):
        fn = fixture_func._fixture_function
        return fn(request_mock)

    def test_singular_returns_first_record(self):
        fixture_func = _make_context_fixture("ecommerce_customer", singular=True)
        request = MagicMock()
        cm = MagicMock()
        cm.get_data.return_value = [{"name": "A"}, {"name": "B"}]
        request.config._testdata_cache_manager = cm

        result = self._call_fixture(fixture_func, request)

        assert result == {"name": "A"}
        cm.get_data.assert_called_once_with("ecommerce_customer", 1)

    def test_plural_returns_list(self):
        fixture_func = _make_context_fixture("ecommerce_customer", singular=False)
        request = MagicMock()
        cm = MagicMock()
        data = [{"name": "A"}, {"name": "B"}]
        cm.get_data.return_value = data
        request.config._testdata_cache_manager = cm

        result = self._call_fixture(fixture_func, request)

        assert result == data
        cm.get_data.assert_called_once_with("ecommerce_customer", DEFAULT_COUNT)

    def test_fixture_scope_is_session(self):
        fixture_func = _make_context_fixture("banking_user", singular=True)
        assert fixture_func._fixture_function_marker.scope == "session"

    def test_fixture_fails_with_actionable_message_on_plugin_setup_error(self):
        fixture_func = _make_context_fixture("banking_user", singular=True)
        request = MagicMock()
        cm = MagicMock()
        cm.get_data.side_effect = _PluginConfigError("install testdata-ai[all]")
        request.config._testdata_cache_manager = cm

        with pytest.raises(pytest.fail.Exception, match="testdata-ai\\[all\\]"):
            self._call_fixture(fixture_func, request)

    def test_singular_fails_cleanly_when_no_records_returned(self):
        fixture_func = _make_context_fixture("banking_user", singular=True)
        request = MagicMock()
        cm = MagicMock()
        cm.get_data.return_value = []
        request.config._testdata_cache_manager = cm

        with pytest.raises(
            pytest.fail.Exception,
            match="returned no records for context 'banking_user'",
        ):
            self._call_fixture(fixture_func, request)


class TestLazyGenerator:

    def test_initialization_is_deferred_until_generate_call(self):
        lazy = _LazyGenerator()
        with patch("testdata_ai.DataGenerator") as mock_gen_cls:
            assert lazy._generator is None
            mock_gen_cls.assert_not_called()

    def test_generate_initializes_once_and_reuses_generator(self):
        lazy = _LazyGenerator()
        real_gen = MagicMock()
        real_gen.generate_batched.return_value = [[{"id": 1}]]
        with patch("testdata_ai.DataGenerator", return_value=real_gen) as mock_gen_cls:
            first = lazy.generate("ecommerce_customer", 1)
            second = lazy.generate("ecommerce_customer", 1)

        assert first == [{"id": 1}]
        assert second == [{"id": 1}]
        mock_gen_cls.assert_called_once()
        assert real_gen.generate_batched.call_count == 2

    def test_generate_raises_plugin_config_error_with_helpful_message(self):
        lazy = _LazyGenerator()
        with patch("testdata_ai.DataGenerator", side_effect=ValueError("bad env")):
            with pytest.raises(_PluginConfigError, match="OPENAI_API_KEY"):
                lazy.generate("ecommerce_customer", 1)
