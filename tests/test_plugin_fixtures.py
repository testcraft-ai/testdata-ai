"""Tests for testdata_ai pytest plugin fixtures and hooks."""

import logging
from unittest.mock import patch, MagicMock

import pytest

import testdata_ai.pytest_plugin as plugin_mod
from testdata_ai.pytest_plugin import (
    pytest_addoption,
    pytest_configure,
    _make_context_fixture,
    _LazyGenerator,
    _PluginConfigError,
    pytest_sessionfinish,
    DEFAULT_COUNT,
    _get_cache_manager,
    pytest_unconfigure,
)


def _make_config(**overrides):
    """Build a mock pytest config with option defaults.

    Every --testdata-* option defaults to the "off" value (None / False).
    Pass keyword arguments to override specific options, e.g.
        _make_config(**{"--testdata-seed": "my-seed"})
    """
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
            "testdata(context, count=1): generate AI test data",
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
        """When --testdata-last-seed is used but no seeds exist, keep temp."""
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
        """--testdata-show-cache without a value uses const='current',
        which should resolve to the active seed."""
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
        """When no action flags are set, only marker + CacheManager init happen."""
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
        """Named seeds should be tracked in the last_seeds queue."""
        config = _make_config(**{"--testdata-seed": "my-seed"})
        cm = MagicMock()
        cm.seed = "my-seed"

        with patch("testdata_ai.cache_manager.CacheManager", return_value=cm):
            pytest_configure(config)

        cm.add_to_last_seeds.assert_called_once_with("my-seed")

    def test_temp_seed_not_added_to_last_seeds(self):
        """Temporary seeds should not be tracked in the last_seeds queue."""
        config = _make_config()
        cm = MagicMock()
        cm.seed = "TEMP-abc123"

        with patch("testdata_ai.cache_manager.CacheManager", return_value=cm):
            pytest_configure(config)

        cm.add_to_last_seeds.assert_not_called()

    def test_last_seed_not_re_added_when_using_last_seed(self):
        """--testdata-last-seed must NOT re-add the loaded seed to the queue.

        Re-adding would bump it to the front on every subsequent run, causing
        self-loads that lock the queue to a single seed indefinitely.
        """
        config = _make_config(**{"--testdata-last-seed": True})
        cm = MagicMock()
        cm.load_last_seed.return_value = "old-seed"
        cm.seed = "old-seed"

        with patch("testdata_ai.cache_manager.CacheManager", return_value=cm):
            pytest_configure(config)

        cm.add_to_last_seeds.assert_not_called()


class TestTestdataFixture:
    """Test the testdata fixture by calling the underlying function directly.

    pytest >=8 blocks direct fixture calls, so we reach through to the
    wrapped callable stored by the @pytest.fixture decorator.
    """

    @staticmethod
    def _call_testdata(request_mock):
        """Call the real testdata fixture function, bypassing the decorator guard."""
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

        cm.get_data.assert_called_once_with("ecommerce_customer", 3)
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

        cm.get_data.assert_called_once_with("banking_user", 1)

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


class TestXdistSupport:
    """Unit tests for xdist worker behaviour in pytest_configure."""

    def test_temp_seed_includes_worker_id(self):
        """TEMP seed label embeds the worker ID so each xdist worker gets
        its own isolated cache namespace."""
        config = _make_config()
        with patch.object(plugin_mod, "WORKER_ID", "gw3"), \
             patch("testdata_ai.cache_manager.CacheManager") as mock_cm:
            pytest_configure(config)

        seed = mock_cm.call_args[1]["seed"]
        assert seed.startswith("TEMP-gw3-")

    def test_warns_when_worker_has_no_named_seed(self, caplog):
        """A warning is issued when a worker starts without --testdata-seed,
        because each worker maintains its own isolated cache and makes its
        own AI calls."""
        config = _make_config()
        with patch.object(plugin_mod, "WORKER_ID", "gw0"), \
             patch("testdata_ai.cache_manager.CacheManager"), \
             caplog.at_level(logging.WARNING, logger="testdata_ai"):
            pytest_configure(config)

        assert any("xdist worker" in r.message for r in caplog.records)

    def test_no_warning_when_worker_has_named_seed(self, caplog):
        """No isolation warning when workers share a named seed — they
        coordinate via the FileLock-protected cache file."""
        config = _make_config(**{"--testdata-seed": "shared-seed"})
        cm = MagicMock()
        cm.seed = "shared-seed"
        with patch.object(plugin_mod, "WORKER_ID", "gw1"), \
             patch("testdata_ai.cache_manager.CacheManager", return_value=cm), \
             caplog.at_level(logging.WARNING, logger="testdata_ai"):
            pytest_configure(config)

        assert not any("xdist worker" in r.message for r in caplog.records)

    def test_master_process_never_warns(self, caplog):
        """The controller process (WORKER_ID == 'master') must never emit
        the xdist-isolation warning, even without a named seed."""
        config = _make_config()
        with patch.object(plugin_mod, "WORKER_ID", "master"), \
             patch("testdata_ai.cache_manager.CacheManager"), \
             caplog.at_level(logging.WARNING, logger="testdata_ai"):
            pytest_configure(config)

        assert not any("xdist worker" in r.message for r in caplog.records)


class TestPytestSessionfinish:

    def test_deletes_temp_seed_on_session_finish(self):
        """Temporary seeds should be cleaned up after the test session."""
        session = MagicMock()
        cm = MagicMock()
        cm.seed = "TEMP-abc123"
        session.config._testdata_cache_manager = cm

        pytest_sessionfinish(session, 0)

        cm.delete_seed.assert_called_once_with("TEMP-abc123")

    def test_does_not_delete_named_seed(self):
        """Named seeds should not be auto-deleted."""
        session = MagicMock()
        cm = MagicMock()
        cm.seed = "my-seed"
        session.config._testdata_cache_manager = cm

        pytest_sessionfinish(session, 0)

        cm.delete_seed.assert_not_called()

    def test_handles_missing_cache_manager(self):
        """Should handle case where cache manager is not set."""
        session = MagicMock()
        delattr(session.config, "_testdata_cache_manager")

        # Should not raise an exception
        pytest_sessionfinish(session, 0)

    def test_io_error_during_delete_does_not_propagate(self):
        """An I/O error while deleting a temporary seed must be caught and
        logged, not allowed to surface as an unhandled exception."""
        session = MagicMock()
        cm = MagicMock()
        cm.seed = "TEMP-xyz789"
        cm.delete_seed.side_effect = OSError("disk full")
        session.config._testdata_cache_manager = cm

        # Should not raise
        pytest_sessionfinish(session, 0)

        cm.delete_seed.assert_called_once_with("TEMP-xyz789")


# ---------------------------------------------------------------------------
# pytest_sessionfinish — early return when seed_path doesn't exist (line 315)
# ---------------------------------------------------------------------------

class TestPytestSessionfinishEarlyReturn:

    def test_skips_delete_when_seed_path_does_not_exist(self):
        """If the TEMP seed file has already been removed, skip delete."""
        session = MagicMock()
        cm = MagicMock()
        cm.seed = "TEMP-gone"
        cm.seed_path.return_value = MagicMock(**{"exists.return_value": False})
        session.config._testdata_cache_manager = cm

        pytest_sessionfinish(session, 0)

        cm.delete_seed.assert_not_called()


# ---------------------------------------------------------------------------
# pytest_unconfigure (lines 328-334)
# ---------------------------------------------------------------------------

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
        config = MagicMock(spec=[])  # no _testdata_cache_manager attribute
        pytest_unconfigure(config)  # should not raise


# ---------------------------------------------------------------------------
# _get_cache_manager — pytest.fail when cm is None (line 243)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Log file setup failure (lines 87-88)
# ---------------------------------------------------------------------------

class TestLogFileSetupFailure:

    def test_warns_when_log_file_cannot_be_created(self):
        """_setup_logging() warns when the log file cannot be created.

        The guard ``if logger.handlers: return`` prevents re-entry, so we
        call _setup_logging() with a temporarily empty logger and capture
        warnings via patch.object instead of caplog.
        """
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


# ---------------------------------------------------------------------------
# show-cache option coverage (lines 218-222, 228, 230, 234)
# ---------------------------------------------------------------------------

class TestShowCacheOption:

    # pytest_configure() calls pytest.exit() when admin options are active,
    # which raises _pytest.outcomes.Exit (not SystemExit).
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
        cm.show_cache.return_value = None  # None = file not found
        cm.seed_path.return_value = MagicMock()

        with caplog.at_level(logging.INFO, logger="testdata_ai"):
            self._configure_with_show_cache(cm, "my-seed")

        assert any("No cache file found" in r.message for r in caplog.records)

    def test_show_cache_empty_file_logs_message(self, caplog):
        cm = MagicMock()
        cm.seed = "my-seed"
        cm.show_cache.return_value = {}  # empty = file exists but no data
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
