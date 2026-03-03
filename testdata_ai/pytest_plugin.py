"""
Pytest plugin for testdata-ai.
"""

import os
import logging
import sys
import uuid
from pathlib import Path
from logging.handlers import RotatingFileHandler

import pytest

from testdata_ai import DataGenerator
from testdata_ai.contexts import list_contexts
from testdata_ai.cache_manager import CacheManager


DEFAULT_COUNT = 10
_log_file_path = Path(".testdata_ai.log")

WORKER_ID = os.environ.get("PYTEST_XDIST_WORKER", "master")

logger = logging.getLogger("testdata_ai")

_CONTEXT_FIXTURES_PLUGIN = "_testdata_context_fixtures"


class _PluginConfigError(RuntimeError):
    """Raised when testdata plugin cannot initialize AI generation."""


class _LazyGenerator:
    """Lazily instantiate DataGenerator on first use."""

    _BATCH_SIZE = 10

    def __init__(self):
        self._generator = None

    def generate(self, context, count):
        if self._generator is None:
            try:
                self._generator = DataGenerator()
            except Exception as exc:
                raise _PluginConfigError(
                    "testdata-ai plugin could not initialize AI provider. "
                    "Set provider env vars (for example OPENAI_API_KEY) and install "
                    "provider dependencies (for example pip install testdata-ai[all])."
                ) from exc
        results = []
        for batch in self._generator.generate_batched(context, count, self._BATCH_SIZE):
            results.extend(batch)
        return results


def _setup_logging():
    """Configure plugin logging. Called from pytest_configure to avoid
    interfering with caplog and other test-time log capture.

    Idempotent: uses logger.handlers as the sentinel so the guard works
    correctly in each process, including xdist workers (which import the
    module independently and therefore start with an empty logger).
    File-handler errors (e.g. permission denied) are swallowed and reported
    as a warning so they never prevent the rest of the plugin from starting.
    """
    if logger.handlers:
        return

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(f"%(asctime)s [%(levelname)s] [{WORKER_ID}] %(message)s")

    term_handler = logging.StreamHandler(sys.stderr)
    term_handler.setFormatter(formatter)
    logger.addHandler(term_handler)

    try:
        file_handler = RotatingFileHandler(
            filename=str(_log_file_path),
            maxBytes=5_000_000,
            backupCount=3,
            mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as exc:
        logger.warning(
            f"testdata-ai: could not set up log file '{_log_file_path}': {exc}; "
            "falling back to console-only logging"
        )

def _build_context_fixtures_class():
    """Build a class with all context fixtures as class attributes.

    Fixtures must be *class* attributes (not instance attributes) because
    pytest 9's parsefactories inspects type(plugin) when the plugin is a
    plain object instance. Using type() avoids introducing a top-level class
    that would need to be manually updated whenever contexts change.
    """
    attrs = {}
    for ctx_name in list_contexts():
        attrs[ctx_name] = _make_context_fixture(ctx_name, singular=True)
        attrs[ctx_name + "s"] = _make_context_fixture(ctx_name, singular=False)
    return type("_ContextFixtures", (), attrs)


def pytest_addoption(parser):
    parser.addoption(
        "--testdata-seed",
        action="store",
        default=None,
        help="Seed name for persistent testdata cache"
    )
    parser.addoption(
        "--testdata-last-seed", 
        action="store_true",
        default=False,
        help="Use the last seed from the cache queue"
    )
    parser.addoption(
        "--testdata-delete-seed",
        action="store",
        default=None,
        help="Delete a specific seed from cache"
    )
    parser.addoption(
        "--testdata-delete-last",
        action="store_true",
        default=False,
        help="Delete the last seed from cache queue"
    )
    parser.addoption(
        "--testdata-clear-cache", 
        action="store_true",
        default=False,
        help="Clear all seeds and reset last seeds queue"
    )
    parser.addoption(
        "--testdata-show-cache",
        nargs="?",
        const="current",
        default=None,
        help="Show cache contents for a given seed (or current if not specified)"
    )
    parser.addoption(
        "--testdata-list-seeds", 
        action="store_true",
        default=False,
        help="List all available seeds in cache"
    )


def pytest_configure(config):
    _setup_logging()

    config.addinivalue_line(
        "markers",
        "testdata(context, count=1): generate AI test data"
    )

    is_admin_run = (
        config.getoption("--testdata-delete-seed") is not None
        or config.getoption("--testdata-delete-last")
        or config.getoption("--testdata-clear-cache")
        or config.getoption("--testdata-show-cache") is not None
        or config.getoption("--testdata-list-seeds")
    )

    # Always register even on admin runs: pytest.exit() prevents tests from
    # executing, and skipping registration would silently cause "fixture not
    # found" if --testdata-seed is combined with an admin flag.
    if not config.pluginmanager.has_plugin(_CONTEXT_FIXTURES_PLUGIN):
        ctx_cls = _build_context_fixtures_class()
        config.pluginmanager.register(ctx_cls(), _CONTEXT_FIXTURES_PLUGIN)

    seed = config.getoption("--testdata-seed")

    # For admin-only runs (no --testdata-seed given) skip generating a TEMP
    # seed — no test data will be produced, so there is no cache to track.
    if is_admin_run:
        seed_label = seed  # may be None
    else:
        seed_label = seed or f"TEMP-{WORKER_ID}-{uuid.uuid4().hex[:6]}"
        if not seed and WORKER_ID != "master":
            logger.warning(
                "testdata-ai: running in xdist worker without --testdata-seed; "
                "each worker maintains an isolated cache and will make its own AI calls. "
                "Pass --testdata-seed=<name> to share a single cache across all workers."
            )

    config._testdata_cache_manager = CacheManager(
        generator=_LazyGenerator(),
        seed=seed_label
    )

    cm = config._testdata_cache_manager

    if config.getoption("--testdata-last-seed"):
        loaded = cm.load_last_seed()
        if loaded is None:
            logger.info("No last seed found, using temporary cache")
    elif cm.seed and not cm.seed.startswith("TEMP-"):
        # Track named seeds in the last_seeds queue for reuse with --testdata-last-seed.
        # Skipped when --testdata-last-seed is active: the loaded seed is already in
        # the queue and re-promoting it would cause self-loads on subsequent runs.
        cm.add_to_last_seeds(cm.seed)

    if is_admin_run:
        if delete_seed := config.getoption("--testdata-delete-seed"):
            cm.delete_seed(delete_seed)
        if config.getoption("--testdata-delete-last"):
            cm.delete_last_seed()
        if config.getoption("--testdata-clear-cache"):
            cm.clear_cache()
        if (show_cache := config.getoption("--testdata-show-cache")) is not None:
            if show_cache == "current" and cm.seed is None:
                logger.info(
                    "No active seed. Specify a name with --testdata-show-cache <name> "
                    "or use --testdata-seed."
                )
                show_cache = None
            if show_cache is not None:
                seed_to_show = show_cache if show_cache != "current" else cm.seed
                contents = cm.show_cache(seed_to_show)
                cache_path = cm.seed_path(seed_to_show)
                if contents is None:
                    logger.info(f"No cache file found for seed '{seed_to_show}' ({cache_path})")
                elif not contents:
                    logger.info(f"Cache file exists but is empty for seed '{seed_to_show}' ({cache_path})")
                else:
                    logger.info(f"Cache for seed '{seed_to_show}' ({cache_path}):")
                    for ctx, count in contents.items():
                        logger.info(f"  Context '{ctx}': {count} items")
        if config.getoption("--testdata-list-seeds"):
            seeds = cm.list_seeds()
            logger.info(f"Available seeds: {seeds}")
        pytest.exit("testdata-ai admin action completed", returncode=0)

def _get_cache_manager(request):
    cm = getattr(request.config, "_testdata_cache_manager", None)
    if cm is None:
        pytest.fail(
            "testdata-ai plugin is not initialized. "
            "pytest_configure may not have run (e.g. missing entry-point or xdist misconfiguration)."
        )
    return cm


@pytest.fixture
def testdata(request):
    """
    Fixture that generates AI test data based on marker with caching.

    Usage:
        @pytest.mark.testdata(context="ecommerce_customer", count=5)
        def test_example(testdata):
            assert len(testdata) == 5
    """
    cm = _get_cache_manager(request)

    marker = request.node.get_closest_marker("testdata")
    if marker is None:
        pytest.fail("testdata fixture requires @pytest.mark.testdata")

    context = marker.kwargs.get("context")
    if context is None:
        pytest.fail(
            "@pytest.mark.testdata requires 'context' argument"
        )

    count = marker.kwargs.get("count", 1)
    try:
        return cm.get_data(context, count)
    except _PluginConfigError as exc:
        pytest.fail(str(exc))



def _make_context_fixture(context_name, singular):
    """Create a session-scoped fixture that returns AI-generated test data."""
    fixture_name = context_name if singular else f"{context_name}s"

    @pytest.fixture(scope="session")
    def _context_fixture(request):
        cm = _get_cache_manager(request)
        requested_count = 1 if singular else DEFAULT_COUNT
        try:
            data = cm.get_data(context_name, requested_count)
        except _PluginConfigError as exc:
            pytest.fail(str(exc))
        if singular:
            if not data:
                pytest.fail(
                    f"testdata-ai returned no records for context '{context_name}'. "
                    "Expected at least 1 record."
                )
            return data[0]
        return data

    _context_fixture.__name__ = fixture_name
    _context_fixture.__qualname__ = fixture_name
    _context_fixture.__doc__ = (
        f"AI-generated {'single record' if singular else 'list of records'} "
        f"for the '{context_name}' context."
    )
    return _context_fixture


def pytest_sessionfinish(session, exitstatus):
    """Clean up temporary seeds after test session ends."""
    cm = getattr(session.config, "_testdata_cache_manager", None)
    if cm is not None and cm.seed and cm.seed.startswith("TEMP-"):
        if not cm.seed_path().exists():
            return
        try:
            cm.delete_seed(cm.seed)
            logger.info(f"Cleaned up temporary seed '{cm.seed}' after session end")
        except Exception as exc:
            logger.warning(f"Could not clean up temporary seed '{cm.seed}': {exc}")

def pytest_unconfigure(config):
    """Final cleanup before pytest exits. Ensures named seeds are persisted.

    TEMP-* seeds are cleaned up in pytest_sessionfinish; skipping finalize
    here prevents them from being re-created on disk after deletion.
    """
    cm = getattr(config, "_testdata_cache_manager", None)
    if cm is not None and cm.seed and not cm.seed.startswith("TEMP-"):
        try:
            cm.finalize()
        except Exception as exc:
            logger.warning(f"testdata-ai: could not finalize cache for seed '{cm.seed}': {exc}")
        logger.info("Finalized testdata cache state")
