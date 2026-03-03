"""Test to verify temporary seed cleanup."""
from unittest.mock import MagicMock

import testdata_ai.pytest_plugin as plugin_mod


def _call_testdata(request_mock):
    fn = plugin_mod.testdata._fixture_function
    return fn(request_mock)


def test_with_temp_seed():
    """Verifies testdata fixture returns data via cache manager (no --testdata-seed).

    Uses a mock cache manager so no real AI call is made.
    TEMP seed creation/cleanup logic is covered in test_plugin_fixtures.py.
    """
    marker = MagicMock()
    marker.kwargs = {"context": "ecommerce_customer", "count": 1}
    request = MagicMock()
    request.node.get_closest_marker.return_value = marker

    cm = MagicMock()
    cm.get_data.return_value = [{"customer_id": "c-001", "name": "Test User"}]
    request.config._testdata_cache_manager = cm

    result = _call_testdata(request)

    assert len(result) == 1
    assert result[0] is not None
    cm.get_data.assert_called_once_with("ecommerce_customer", 1)
