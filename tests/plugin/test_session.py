"""Tests for testdata_ai pytest plugin — session lifecycle hooks."""

import logging
import pytest
from unittest.mock import MagicMock

from testdata_ai.pytest_plugin import (
    pytest_sessionfinish,
    pytest_unconfigure,
    _get_cache_manager,
)


class TestPytestSessionfinish:

    def test_deletes_temp_seed_on_session_finish(self):
        session = MagicMock()
        cm = MagicMock()
        cm.seed = "TEMP-abc123"
        session.config._testdata_cache_manager = cm

        pytest_sessionfinish(session, 0)

        cm.delete_seed.assert_called_once_with("TEMP-abc123")

    def test_does_not_delete_named_seed(self):
        session = MagicMock()
        cm = MagicMock()
        cm.seed = "my-seed"
        session.config._testdata_cache_manager = cm

        pytest_sessionfinish(session, 0)

        cm.delete_seed.assert_not_called()

    def test_handles_missing_cache_manager(self):
        session = MagicMock()
        delattr(session.config, "_testdata_cache_manager")

        pytest_sessionfinish(session, 0)

    def test_io_error_during_delete_does_not_propagate(self):
        session = MagicMock()
        cm = MagicMock()
        cm.seed = "TEMP-xyz789"
        cm.delete_seed.side_effect = OSError("disk full")
        session.config._testdata_cache_manager = cm

        pytest_sessionfinish(session, 0)

        cm.delete_seed.assert_called_once_with("TEMP-xyz789")


class TestPytestSessionfinishEarlyReturn:

    def test_skips_delete_when_seed_path_does_not_exist(self):
        session = MagicMock()
        cm = MagicMock()
        cm.seed = "TEMP-gone"
        cm.seed_path.return_value = MagicMock(**{"exists.return_value": False})
        session.config._testdata_cache_manager = cm

        pytest_sessionfinish(session, 0)

        cm.delete_seed.assert_not_called()


class TestPytestUnconfigure:

    def test_finalizes_named_seed(self):
        config = MagicMock()
        cm = MagicMock()
        cm.seed = "my-named-seed"
        config._testdata_cache_manager = cm

        pytest_unconfigure(config)

        cm.finalize.assert_called_once()

    def test_does_not_finalize_temp_seed(self):
        config = MagicMock()
        cm = MagicMock()
        cm.seed = "TEMP-abc"
        config._testdata_cache_manager = cm

        pytest_unconfigure(config)

        cm.finalize.assert_not_called()

    def test_does_not_finalize_when_no_seed(self):
        config = MagicMock()
        cm = MagicMock()
        cm.seed = None
        config._testdata_cache_manager = cm

        pytest_unconfigure(config)

        cm.finalize.assert_not_called()

    def test_finalize_error_is_logged_not_raised(self, caplog):
        config = MagicMock()
        cm = MagicMock()
        cm.seed = "named-seed"
        cm.finalize.side_effect = OSError("disk full")
        config._testdata_cache_manager = cm

        with caplog.at_level(logging.WARNING, logger="testdata_ai"):
            pytest_unconfigure(config)

        assert "could not finalize" in caplog.text

    def test_no_cache_manager_is_noop(self):
        config = MagicMock(spec=[])
        pytest_unconfigure(config)


class TestGetCacheManager:

    def test_raises_fail_when_cache_manager_not_set(self):
        request = MagicMock()
        request.config._testdata_cache_manager = None

        with pytest.raises(pytest.fail.Exception, match="not initialized"):
            _get_cache_manager(request)

    def test_returns_cache_manager_when_set(self):
        request = MagicMock()
        cm = MagicMock()
        request.config._testdata_cache_manager = cm

        result = _get_cache_manager(request)
        assert result is cm
