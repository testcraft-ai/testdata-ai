"""Tests for testdata_ai pytest plugin — configuration and admin options."""

import logging
import pytest
from unittest.mock import patch, MagicMock

import testdata_ai.pytest_plugin as plugin_mod
from testdata_ai.pytest_plugin import (
    pytest_addoption,
    pytest_configure,
    _LazyGenerator,
)


def _make_config(**overrides):
    """Build a mock pytest config with option defaults."""
    defaults = {
        "--testdata-seed": None,
        "--testdata-last-seed": False,
        "--testdata-delete-seed": None,
        "--testdata-delete-last": False,
        "--testdata-clear-cache": False,
        "--testdata-show-cache": None,
        "--testdata-list-seeds": False,
    }
    defaults.update(overrides)
    config = MagicMock()
    config.getoption = MagicMock(side_effect=lambda opt: defaults[opt])
    return config


class TestPytestAddoption:

    def test_registers_all_options(self):
        parser = MagicMock()
        pytest_addoption(parser)

        registered = str(parser.addoption.call_args_list)
        for option in [
            "--testdata-seed",
            "--testdata-last-seed",
            "--testdata-delete-seed",
            "--testdata-delete-last",
            "--testdata-clear-cache",
            "--testdata-show-cache",
            "--testdata-list-seeds",
        ]:
            assert option in registered, f"{option} not registered"

    def test_registers_exactly_seven_options(self):
        parser = MagicMock()
        pytest_addoption(parser)
        assert parser.addoption.call_count == 7


class TestPytestConfigure:

    def test_registers_testdata_marker(self):
        config = _make_config()
        with patch("testdata_ai.cache_manager.CacheManager"):
            pytest_configure(config)

        config.addinivalue_line.assert_called_once_with(
            "markers",
            "testdata(context, count=1, locale=None): generate AI test data",
        )

    def test_creates_cache_manager_with_temp_seed_when_no_seed_given(self):
        config = _make_config()
        with patch("testdata_ai.cache_manager.CacheManager") as mock_cm:
            pytest_configure(config)

        seed = mock_cm.call_args[1]["seed"]
        assert seed.startswith("TEMP-")
        assert isinstance(mock_cm.call_args[1]["generator"], _LazyGenerator)

    def test_creates_cache_manager_with_provided_seed(self):
        config = _make_config(**{"--testdata-seed": "my-seed"})
        with patch("testdata_ai.cache_manager.CacheManager") as mock_cm:
            pytest_configure(config)

        assert mock_cm.call_args[1]["seed"] == "my-seed"

    def test_does_not_instantiate_generator_during_configure(self):
        config = _make_config()
        with patch("testdata_ai.DataGenerator") as mock_gen_cls, \
             patch("testdata_ai.cache_manager.CacheManager"):
            pytest_configure(config)

        mock_gen_cls.assert_not_called()

    def test_last_seed_switches_to_most_recent_seed(self):
        config = _make_config(**{"--testdata-last-seed": True})
        cm = MagicMock()
        cm.load_last_seed.return_value = "recent-seed"

        with patch("testdata_ai.cache_manager.CacheManager", return_value=cm):
            pytest_configure(config)

        cm.load_last_seed.assert_called_once()

    def test_last_seed_with_empty_queue_keeps_temp_seed(self):
        config = _make_config(**{"--testdata-last-seed": True})
        cm = MagicMock()
        cm.load_last_seed.return_value = None

        with patch("testdata_ai.cache_manager.CacheManager", return_value=cm):
            pytest_configure(config)

        cm.load_last_seed.assert_called_once()

    def test_delete_seed_calls_cache_manager(self):
        config = _make_config(**{"--testdata-delete-seed": "old-seed"})
        cm = MagicMock()

        with patch("testdata_ai.cache_manager.CacheManager", return_value=cm), \
             patch.object(plugin_mod.pytest, "exit") as mock_exit:
            pytest_configure(config)

        cm.delete_seed.assert_called_once_with("old-seed")
        mock_exit.assert_called_once_with("testdata-ai admin action completed", returncode=0)

    def test_delete_last_calls_cache_manager(self):
        config = _make_config(**{"--testdata-delete-last": True})
        cm = MagicMock()

        with patch("testdata_ai.cache_manager.CacheManager", return_value=cm), \
             patch.object(plugin_mod.pytest, "exit") as mock_exit:
            pytest_configure(config)

        cm.delete_last_seed.assert_called_once()
        mock_exit.assert_called_once_with("testdata-ai admin action completed", returncode=0)

    def test_clear_cache_calls_cache_manager(self):
        config = _make_config(**{"--testdata-clear-cache": True})
        cm = MagicMock()

        with patch("testdata_ai.cache_manager.CacheManager", return_value=cm), \
             patch.object(plugin_mod.pytest, "exit") as mock_exit:
            pytest_configure(config)

        cm.clear_cache.assert_called_once()
        mock_exit.assert_called_once_with("testdata-ai admin action completed", returncode=0)

    def test_show_cache_with_explicit_seed(self):
        config = _make_config(**{"--testdata-show-cache": "seed-123"})
        cm = MagicMock()

        with patch("testdata_ai.cache_manager.CacheManager", return_value=cm), \
             patch.object(plugin_mod.pytest, "exit") as mock_exit:
            pytest_configure(config)

        cm.show_cache.assert_called_once_with("seed-123")
        mock_exit.assert_called_once_with("testdata-ai admin action completed", returncode=0)

    def test_show_cache_with_current_resolves_to_active_seed(self):
        config = _make_config(**{"--testdata-show-cache": "current"})
        cm = MagicMock()
        cm.seed = "active-seed"

        with patch("testdata_ai.cache_manager.CacheManager", return_value=cm), \
             patch.object(plugin_mod.pytest, "exit") as mock_exit:
            pytest_configure(config)

        cm.show_cache.assert_called_once_with("active-seed")
        mock_exit.assert_called_once_with("testdata-ai admin action completed", returncode=0)

    def test_list_seeds_calls_cache_manager(self):
        config = _make_config(**{"--testdata-list-seeds": True})
        cm = MagicMock()
        cm.list_seeds.return_value = ["s1", "s2"]

        with patch("testdata_ai.cache_manager.CacheManager", return_value=cm), \
             patch.object(plugin_mod.pytest, "exit") as mock_exit:
            pytest_configure(config)

        cm.list_seeds.assert_called_once()
        mock_exit.assert_called_once_with("testdata-ai admin action completed", returncode=0)

    def test_no_action_options_only_initializes(self):
        config = _make_config()
        cm = MagicMock()

        with patch("testdata_ai.cache_manager.CacheManager", return_value=cm), \
             patch.object(plugin_mod.pytest, "exit") as mock_exit:
            pytest_configure(config)

        cm.delete_seed.assert_not_called()
        cm.delete_last_seed.assert_not_called()
        cm.clear_cache.assert_not_called()
        cm.show_cache.assert_not_called()
        cm.list_seeds.assert_not_called()
        mock_exit.assert_not_called()

    def test_named_seed_added_to_last_seeds(self):
        config = _make_config(**{"--testdata-seed": "my-seed"})
        cm = MagicMock()
        cm.seed = "my-seed"

        with patch("testdata_ai.cache_manager.CacheManager", return_value=cm):
            pytest_configure(config)

        cm.add_to_last_seeds.assert_called_once_with("my-seed")

    def test_temp_seed_not_added_to_last_seeds(self):
        config = _make_config()
        cm = MagicMock()
        cm.seed = "TEMP-abc123"

        with patch("testdata_ai.cache_manager.CacheManager", return_value=cm):
            pytest_configure(config)

        cm.add_to_last_seeds.assert_not_called()

    def test_last_seed_not_re_added_when_using_last_seed(self):
        config = _make_config(**{"--testdata-last-seed": True})
        cm = MagicMock()
        cm.load_last_seed.return_value = "old-seed"
        cm.seed = "old-seed"

        with patch("testdata_ai.cache_manager.CacheManager", return_value=cm):
            pytest_configure(config)

        cm.add_to_last_seeds.assert_not_called()


class TestXdistSupport:

    def test_temp_seed_includes_worker_id(self):
        config = _make_config()
        with patch.object(plugin_mod, "WORKER_ID", "gw3"), \
             patch("testdata_ai.cache_manager.CacheManager") as mock_cm:
            pytest_configure(config)

        seed = mock_cm.call_args[1]["seed"]
        assert seed.startswith("TEMP-gw3-")

    def test_warns_when_worker_has_no_named_seed(self, caplog):
        config = _make_config()
        with patch.object(plugin_mod, "WORKER_ID", "gw0"), \
             patch("testdata_ai.cache_manager.CacheManager"), \
             caplog.at_level(logging.WARNING, logger="testdata_ai"):
            pytest_configure(config)

        assert any("xdist worker" in r.message for r in caplog.records)

    def test_no_warning_when_worker_has_named_seed(self, caplog):
        config = _make_config(**{"--testdata-seed": "shared-seed"})
        cm = MagicMock()
        cm.seed = "shared-seed"
        with patch.object(plugin_mod, "WORKER_ID", "gw1"), \
             patch("testdata_ai.cache_manager.CacheManager", return_value=cm), \
             caplog.at_level(logging.WARNING, logger="testdata_ai"):
            pytest_configure(config)

        assert not any("xdist worker" in r.message for r in caplog.records)

    def test_master_process_never_warns(self, caplog):
        config = _make_config()
        with patch.object(plugin_mod, "WORKER_ID", "master"), \
             patch("testdata_ai.cache_manager.CacheManager"), \
             caplog.at_level(logging.WARNING, logger="testdata_ai"):
            pytest_configure(config)

        assert not any("xdist worker" in r.message for r in caplog.records)


class TestShowCacheOption:

    from _pytest.outcomes import Exit as _PytestExit

    def _configure_with_show_cache(self, cm, show_cache_value):
        from _pytest.outcomes import Exit
        config = _make_config(**{"--testdata-show-cache": show_cache_value})
        with patch("testdata_ai.cache_manager.CacheManager", return_value=cm):
            try:
                pytest_configure(config)
            except Exit:
                pass

    def test_show_cache_current_with_no_active_seed_logs_message(self, caplog):
        cm = MagicMock()
        cm.seed = None
        cm.show_cache.return_value = None

        with caplog.at_level(logging.INFO, logger="testdata_ai"):
            self._configure_with_show_cache(cm, "current")

        assert any("No active seed" in r.message for r in caplog.records)

    def test_show_cache_missing_file_logs_message(self, caplog):
        cm = MagicMock()
        cm.seed = "my-seed"
        cm.show_cache.return_value = None
        cm.seed_path.return_value = MagicMock()

        with caplog.at_level(logging.INFO, logger="testdata_ai"):
            self._configure_with_show_cache(cm, "my-seed")

        assert any("No cache file found" in r.message for r in caplog.records)

    def test_show_cache_empty_file_logs_message(self, caplog):
        cm = MagicMock()
        cm.seed = "my-seed"
        cm.show_cache.return_value = {}
        cm.seed_path.return_value = MagicMock()

        with caplog.at_level(logging.INFO, logger="testdata_ai"):
            self._configure_with_show_cache(cm, "my-seed")

        assert any("empty" in r.message for r in caplog.records)

    def test_show_cache_with_data_logs_context_counts(self, caplog):
        cm = MagicMock()
        cm.seed = "my-seed"
        cm.show_cache.return_value = {"ecommerce_customer": 5, "banking_user": 3}
        cm.seed_path.return_value = MagicMock()

        with caplog.at_level(logging.INFO, logger="testdata_ai"):
            self._configure_with_show_cache(cm, "my-seed")

        log_text = " ".join(r.message for r in caplog.records)
        assert "ecommerce_customer" in log_text
        assert "5" in log_text


class TestLogFileSetupFailure:

    def test_warns_when_log_file_cannot_be_created(self):
        from testdata_ai.pytest_plugin import _setup_logging

        original_handlers = list(plugin_mod.logger.handlers)
        plugin_mod.logger.handlers.clear()
        try:
            with patch.object(plugin_mod.logger, "warning") as mock_warn, \
                 patch("testdata_ai.pytest_plugin.RotatingFileHandler",
                       side_effect=OSError("permission denied")):
                _setup_logging()
        finally:
            plugin_mod.logger.handlers.clear()
            plugin_mod.logger.handlers.extend(original_handlers)

        assert mock_warn.called
        assert "could not set up log file" in mock_warn.call_args[0][0]
