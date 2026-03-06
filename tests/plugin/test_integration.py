"""Integration-style tests for pytest plugin behavior."""

from pathlib import Path

import pytest


def _enable_plugin(pytester):
    repo_root = str(Path(__file__).resolve().parents[1])
    pytester.makeconftest(
        f"""
import sys
import os

# Keep subprocess runs deterministic, regardless of outer shell env.
for _env_var in ("AI_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(_env_var, None)

# Force provider config failure path so plugin surfaces actionable guidance.
os.environ["AI_PROVIDER"] = "invalid-provider"

sys.path.insert(0, {repo_root!r})
# Plugin is auto-loaded via pytest11 entry point; no explicit registration needed.
"""
    )


def test_autoload_without_provider_config_fails_with_actionable_message(pytester, monkeypatch):
    _enable_plugin(pytester)
    for env_var in ("AI_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)

    pytester.makepyfile(
        """
import pytest

@pytest.mark.testdata(context="ecommerce_customer", count=1)
def test_needs_provider_config(testdata):
    assert testdata
"""
    )

    result = pytester.runpytest_subprocess("-q")

    assert result.ret != 0
    output = result.stdout.str() + result.stderr.str()
    assert "could not initialize AI provider" in output
    assert "OPENAI_API_KEY" in output


def test_user_facing_context_fixture_names_are_exposed(pytester):
    _enable_plugin(pytester)
    result = pytester.runpytest_subprocess("--fixtures", "-q")

    assert result.ret == 0
    stdout = result.stdout.str()
    assert "ecommerce_customer" in stdout
    assert "ecommerce_customers" in stdout
    assert "banking_user" in stdout
    assert "banking_users" in stdout


def test_admin_options_short_circuit_test_execution(pytester):
    _enable_plugin(pytester)
    pytester.makepyfile(
        """
def test_should_not_run():
    assert False, "admin options must exit before this runs"
"""
    )

    result = pytester.runpytest_subprocess("--testdata-list-seeds", "-q")

    assert result.ret == 0
    output = result.stdout.str() + result.stderr.str()
    assert "admin options must exit before this runs" not in output
    assert "FAILED" not in output


# ---------------------------------------------------------------------------
# xdist integration tests
# ---------------------------------------------------------------------------


def test_xdist_workers_without_named_seed_warn_about_isolation(pytester):
    """Each xdist worker should log a warning when no --testdata-seed is given,
    because every worker maintains its own isolated cache and makes separate
    AI calls."""
    pytest.importorskip("xdist")
    _enable_plugin(pytester)
    pytester.makepyfile(
        """
def test_placeholder():
    pass
"""
    )

    result = pytester.runpytest_subprocess("-n", "2", "-q", "-s")

    output = result.stdout.str() + result.stderr.str()
    assert "xdist worker" in output


def test_xdist_workers_with_named_seed_do_not_warn(pytester):
    """No isolation warning when workers share a named seed — they coordinate
    through the FileLock-protected cache file."""
    pytest.importorskip("xdist")
    _enable_plugin(pytester)
    pytester.makepyfile(
        """
def test_placeholder():
    pass
"""
    )

    result = pytester.runpytest_subprocess(
        "-n", "2", "--testdata-seed", "shared", "-q", "-s"
    )

    output = result.stdout.str() + result.stderr.str()
    assert "xdist worker" not in output


def test_xdist_admin_options_exit_before_tests(pytester):
    """Admin CLI flags (e.g. --testdata-list-seeds) must still short-circuit
    test collection even when -n is passed."""
    pytest.importorskip("xdist")
    _enable_plugin(pytester)
    pytester.makepyfile(
        """
def test_should_not_run():
    assert False, "admin options must exit before this runs"
"""
    )

    result = pytester.runpytest_subprocess(
        "-n", "2", "--testdata-list-seeds", "-q"
    )

    assert result.ret == 0
    output = result.stdout.str() + result.stderr.str()
    assert "admin options must exit before this runs" not in output
