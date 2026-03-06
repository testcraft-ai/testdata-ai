"""Tests for testdata_ai.cli — spinner, adjust_max_tokens, JSONL pretty-print."""

import json
import pytest
from unittest.mock import patch, MagicMock

from testdata_ai.cli import _adjust_max_tokens, _run_streaming, _Spinner
from testdata_ai.contexts import CONTEXTS


class TestAdjustMaxTokens:

    def test_no_adjustment_when_within_limit(self, mock_generator, mock_context_schema):
        gen = mock_generator
        gen.config.max_tokens = 4096
        gen.provider.max_tokens = 4096
        _adjust_max_tokens(gen, mock_context_schema, count=1, quiet=True, user_set=False)
        assert gen.config.max_tokens == 4096

    def test_quiet_mode_auto_increases(self, mock_generator, mock_context_schema):
        gen = mock_generator
        gen.config.max_tokens = 100
        _adjust_max_tokens(gen, mock_context_schema, count=500, quiet=True, user_set=False)
        gen.set_max_tokens.assert_called_once()
        called_value = gen.set_max_tokens.call_args[0][0]
        assert called_value > 100

    def test_non_quiet_auto_increases_and_echoes(self, mock_generator, mock_context_schema):
        gen = mock_generator
        gen.provider.max_tokens = 100
        with patch("testdata_ai.cli.click.echo") as mock_echo:
            _adjust_max_tokens(gen, mock_context_schema, count=500, quiet=False, user_set=False)
        gen.set_max_tokens.assert_called_once()
        assert gen.set_max_tokens.call_args[0][0] > 100
        mock_echo.assert_called_once()


class TestSpinner:

    def test_silent_mode_no_output(self, capsys):
        with _Spinner("testing", silent=True):
            pass
        assert capsys.readouterr().err == ""

    def test_non_silent_writes_start_and_done(self, capsys):
        with _Spinner("working", silent=False):
            pass
        err = capsys.readouterr().err
        assert "working" in err
        assert "Done" in err

    def test_elapsed_time_shown(self, capsys):
        with _Spinner("task", silent=False):
            pass
        assert "s)" in capsys.readouterr().err


class TestSpinnerTTY:
    """Cover the TTY (animated) code paths of _Spinner."""

    def _tty_spinner(self, msg="test"):
        spinner = _Spinner(msg, silent=False)
        spinner._is_tty = True
        return spinner

    def test_tty_enter_starts_thread(self):
        import time
        spinner = self._tty_spinner()
        with spinner:
            assert spinner._thread is not None
            assert spinner._thread.is_alive()
            time.sleep(0.15)

    def test_tty_exit_joins_thread_and_writes_done(self, capsys):
        import time
        spinner = self._tty_spinner("working")
        with spinner:
            time.sleep(0.15)
        err = capsys.readouterr().err
        assert "Done" in err

    def test_tty_hidden_clears_line(self, capsys):
        import time
        spinner = self._tty_spinner("working")
        flag = []
        with spinner:
            time.sleep(0.05)
            with spinner.hidden():
                flag.append("inside")
        assert flag == ["inside"]

    def test_tty_update_writes_non_tty_fallback_when_not_is_tty(self, capsys):
        spinner = _Spinner("msg", silent=False)
        with spinner:
            spinner.update("new message")
        err = capsys.readouterr().err
        assert "new message" in err


class TestJsonlPrettyPrint:

    def test_jsonl_pretty_indents_records(self):
        import io
        sample = CONTEXTS["banking_user"].sample
        mock_gen = MagicMock()
        mock_gen.generate_batched.return_value = iter([[sample]])
        mock_gen.config = MagicMock(provider="openai", model="test-model")

        buf = io.StringIO()
        buf.isatty = lambda: True
        with patch("sys.stdout", buf):
            _run_streaming(mock_gen, "banking_user", 1, 10, "jsonl", False, True)

        output = buf.getvalue()
        assert output.count("\n") > 1
