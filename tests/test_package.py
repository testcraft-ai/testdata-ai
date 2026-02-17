"""Tests for testdata_ai package-level attributes and __main__ entry point."""

import subprocess
import sys
from unittest.mock import patch
from importlib.metadata import PackageNotFoundError


class TestPackageVersion:

    def test_version_is_a_string(self):
        import testdata_ai
        assert isinstance(testdata_ai.__version__, str)

    def test_version_fallback_when_not_installed(self):
        import importlib
        import testdata_ai
        try:
            with patch(
                "importlib.metadata.version", side_effect=PackageNotFoundError
            ):
                importlib.reload(testdata_ai)
                assert testdata_ai.__version__ == "0.0.0-dev"
        finally:
            importlib.reload(testdata_ai)


class TestMainModule:

    def test_main_invokes_cli(self, runner):
        """Smoke test: `python -m testdata_ai --help` should work."""
        from testdata_ai.cli import cli

        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "AI-powered test data generator" in result.output

    def test_main_module_runs_via_subprocess(self):
        """Cover __main__.py by actually running it as a subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "testdata_ai", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "AI-powered test data generator" in result.stdout


class TestPublicApi:

    def test_exports_expected_names(self):
        import testdata_ai
        assert hasattr(testdata_ai, "TestDataGenerator")
        assert hasattr(testdata_ai, "generate")
        assert hasattr(testdata_ai, "list_contexts")
        assert hasattr(testdata_ai, "get_context_schema")
