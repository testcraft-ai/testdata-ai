"""Tests for GenerateResult and RelationshipResult."""

import json
import pytest
from unittest.mock import patch, MagicMock

from testdata_ai.result_types import GenerateResult, RelationshipResult


_RECORDS = [
    {"name": "Alice", "email": "alice@example.com", "age": 30},
    {"name": "Bob", "email": "bob@example.com", "age": 25},
]

_GRAPH = {
    "users": [{"email": "alice@example.com", "name": "Alice"}],
    "orders": [{"order_id": "O1", "user_id": "alice@example.com"}],
}


# ---------------------------------------------------------------------------
# GenerateResult — list-like interface
# ---------------------------------------------------------------------------


class TestGenerateResultInterface:

    def test_len(self):
        r = GenerateResult(_RECORDS)
        assert len(r) == 2

    def test_getitem(self):
        r = GenerateResult(_RECORDS)
        assert r[0] == _RECORDS[0]

    def test_getitem_slice(self):
        r = GenerateResult(_RECORDS)
        assert r[0:1] == _RECORDS[0:1]

    def test_iter(self):
        r = GenerateResult(_RECORDS)
        assert list(r) == _RECORDS

    def test_eq_with_list(self):
        r = GenerateResult(_RECORDS)
        assert r == _RECORDS

    def test_eq_with_generate_result(self):
        r1 = GenerateResult(_RECORDS)
        r2 = GenerateResult(_RECORDS)
        assert r1 == r2

    def test_eq_returns_not_implemented_for_unknown(self):
        r = GenerateResult(_RECORDS)
        assert r.__eq__(42) == NotImplemented

    def test_repr(self):
        r = GenerateResult([{"a": 1}])
        assert "GenerateResult" in repr(r)

    def test_empty(self):
        r = GenerateResult([])
        assert len(r) == 0
        assert list(r) == []

    def test_to_records_returns_plain_list(self):
        r = GenerateResult(_RECORDS)
        plain = r.to_records()
        assert isinstance(plain, list)
        assert plain == _RECORDS


# ---------------------------------------------------------------------------
# GenerateResult — conversion methods
# ---------------------------------------------------------------------------


class TestGenerateResultConversions:

    def test_to_json_returns_string(self):
        r = GenerateResult(_RECORDS)
        text = r.to_json()
        assert isinstance(text, str)
        assert json.loads(text) == _RECORDS

    def test_to_json_writes_file(self, tmp_path):
        r = GenerateResult(_RECORDS)
        path = str(tmp_path / "out.json")
        result = r.to_json(path)
        assert result is None
        with open(path) as f:
            assert json.load(f) == _RECORDS

    def test_to_csv_returns_string(self):
        r = GenerateResult(_RECORDS)
        text = r.to_csv()
        assert isinstance(text, str)
        assert "alice@example.com" in text
        assert "name" in text  # header

    def test_to_csv_writes_file(self, tmp_path):
        r = GenerateResult(_RECORDS)
        path = str(tmp_path / "out.csv")
        result = r.to_csv(path)
        assert result is None
        with open(path) as f:
            content = f.read()
        assert "alice@example.com" in content

    def test_to_csv_empty_records(self):
        r = GenerateResult([])
        assert r.to_csv() == ""

    def test_to_yaml_returns_string(self):
        r = GenerateResult(_RECORDS)
        text = r.to_yaml()
        assert isinstance(text, str)
        assert "alice@example.com" in text

    def test_to_yaml_writes_file(self, tmp_path):
        r = GenerateResult(_RECORDS)
        path = str(tmp_path / "out.yaml")
        result = r.to_yaml(path)
        assert result is None
        with open(path) as f:
            content = f.read()
        assert "alice@example.com" in content

    def test_to_yaml_raises_import_error_when_no_pyyaml(self):
        r = GenerateResult(_RECORDS)
        with patch("builtins.__import__", side_effect=lambda n, *a, **kw: (_ for _ in ()).throw(ImportError) if n == "yaml" else __import__(n, *a, **kw)):
            # use a simpler approach
            pass
        # Test via patching yaml directly
        import sys
        import unittest.mock
        with unittest.mock.patch.dict(sys.modules, {"yaml": None}):
            with pytest.raises((ImportError, TypeError)):
                r.to_yaml()

    def test_to_dataframe_calls_pandas_bridge(self):
        r = GenerateResult(_RECORDS)
        mock_df = MagicMock()
        with patch("testdata_ai.pandas_bridge.records_to_dataframe", return_value=mock_df) as mock_fn:
            result = r.to_dataframe()
        mock_fn.assert_called_once_with(_RECORDS, flatten=True)
        assert result is mock_df

    def test_to_dataframe_passes_flatten_false(self):
        r = GenerateResult(_RECORDS)
        mock_df = MagicMock()
        with patch("testdata_ai.pandas_bridge.records_to_dataframe", return_value=mock_df) as mock_fn:
            r.to_dataframe(flatten=False)
        mock_fn.assert_called_once_with(_RECORDS, flatten=False)

    def test_to_dataframe_raises_import_error_without_pandas(self):
        r = GenerateResult(_RECORDS)
        with patch("testdata_ai.pandas_bridge.pd", None):
            with pytest.raises(ImportError, match="pandas"):
                r.to_dataframe()

    def test_to_batches_yields_correct_chunks(self):
        r = GenerateResult(list(range(10)))
        batches = list(r.to_batches(batch_size=3))
        assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_to_batches_single_batch(self):
        r = GenerateResult(_RECORDS)
        batches = list(r.to_batches(batch_size=100))
        assert len(batches) == 1
        assert batches[0] == _RECORDS

    def test_to_batches_empty(self):
        r = GenerateResult([])
        batches = list(r.to_batches())
        assert batches == []


# ---------------------------------------------------------------------------
# RelationshipResult
# ---------------------------------------------------------------------------


class TestRelationshipResult:

    def test_is_dict_subclass(self):
        r = RelationshipResult(_GRAPH)
        assert isinstance(r, dict)

    def test_access_by_key(self):
        r = RelationshipResult(_GRAPH)
        assert r["users"] == _GRAPH["users"]

    def test_to_json_returns_string(self):
        r = RelationshipResult(_GRAPH)
        text = r.to_json()
        assert isinstance(text, str)
        data = json.loads(text)
        assert "users" in data and "orders" in data

    def test_to_json_writes_file(self, tmp_path):
        r = RelationshipResult(_GRAPH)
        path = str(tmp_path / "out.json")
        result = r.to_json(path)
        assert result is None
        with open(path) as f:
            data = json.load(f)
        assert "users" in data

    def test_to_yaml_returns_string(self):
        r = RelationshipResult(_GRAPH)
        text = r.to_yaml()
        assert isinstance(text, str)
        assert "users" in text

    def test_to_yaml_writes_file(self, tmp_path):
        r = RelationshipResult(_GRAPH)
        path = str(tmp_path / "out.yaml")
        result = r.to_yaml(path)
        assert result is None
        with open(path) as f:
            content = f.read()
        assert "users" in content

    def test_to_dataframes_calls_pandas_bridge(self):
        r = RelationshipResult(_GRAPH)
        mock_dfs = {"users": MagicMock(), "orders": MagicMock()}
        with patch("testdata_ai.pandas_bridge.relationships_to_dataframes", return_value=mock_dfs) as mock_fn:
            result = r.to_dataframes()
        mock_fn.assert_called_once_with(dict(_GRAPH), flatten=True)
        assert result is mock_dfs

    def test_to_dataframes_flatten_false(self):
        r = RelationshipResult(_GRAPH)
        mock_dfs = {}
        with patch("testdata_ai.pandas_bridge.relationships_to_dataframes", return_value=mock_dfs) as mock_fn:
            r.to_dataframes(flatten=False)
        mock_fn.assert_called_once_with(dict(_GRAPH), flatten=False)

    def test_to_dataframes_raises_import_error_without_pandas(self):
        r = RelationshipResult(_GRAPH)
        with patch("testdata_ai.pandas_bridge.pd", None):
            with pytest.raises(ImportError, match="pandas"):
                r.to_dataframes()
