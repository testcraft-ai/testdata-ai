"""Tests for testdata_ai.cli — output formatting helpers."""

import csv
import io

from testdata_ai.cli import _flatten_dict, _records_to_csv, _records_to_sql


class TestFlattenDict:

    def test_flat_dict_unchanged(self):
        assert _flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested_dict(self):
        assert _flatten_dict({"a": {"b": 1}}) == {"a.b": 1}

    def test_deeply_nested(self):
        assert _flatten_dict({"a": {"b": {"c": 3}}}) == {"a.b.c": 3}

    def test_lists_become_json_strings(self):
        result = _flatten_dict({"tags": ["x", "y"]})
        assert result == {"tags": '["x", "y"]'}

    def test_mixed_nesting(self):
        d = {"name": "Test", "loc": {"city": "NYC", "zip": "10001"}, "tags": [1]}
        result = _flatten_dict(d)
        assert result == {
            "name": "Test",
            "loc.city": "NYC",
            "loc.zip": "10001",
            "tags": "[1]",
        }

    def test_empty_dict(self):
        assert _flatten_dict({}) == {}

    def test_none_value_preserved(self):
        assert _flatten_dict({"a": None}) == {"a": None}


class TestRecordsToCsv:

    def test_empty_records(self):
        assert _records_to_csv([]) == ""

    def test_single_flat_record(self):
        csv_text = _records_to_csv([{"a": 1, "b": 2}])
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["a"] == "1"
        assert rows[0]["b"] == "2"

    def test_nested_records_flattened(self):
        records = [{"name": "A", "loc": {"city": "NYC"}}]
        csv_text = _records_to_csv(records)
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        assert rows[0]["loc.city"] == "NYC"

    def test_superset_fieldnames_across_records(self):
        records = [{"a": 1}, {"a": 2, "b": 3}]
        csv_text = _records_to_csv(records)
        reader = csv.DictReader(io.StringIO(csv_text))
        fieldnames = reader.fieldnames
        assert "a" in fieldnames
        assert "b" in fieldnames

    def test_disjoint_keys_across_records(self):
        records = [{"a": 1}, {"b": 2}]
        csv_text = _records_to_csv(records)
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == 2
        assert set(reader.fieldnames) == {"a", "b"}
        assert rows[0]["a"] == "1"
        assert rows[0]["b"] == ""
        assert rows[1]["a"] == ""
        assert rows[1]["b"] == "2"


class TestRecordsToSql:

    def test_empty_records(self):
        assert _records_to_sql([]) == ""

    def test_single_flat_record(self):
        sql = _records_to_sql([{"id": 1, "name": "Alice"}])
        assert "CREATE TABLE IF NOT EXISTS records" in sql
        assert '"id" INTEGER' in sql
        assert '"name" TEXT' in sql
        assert "INSERT INTO records VALUES (1, 'Alice');" in sql

    def test_custom_table_name(self):
        sql = _records_to_sql([{"x": 1}], table="users")
        assert "CREATE TABLE IF NOT EXISTS users" in sql
        assert "INSERT INTO users VALUES" in sql

    def test_nested_dict_column_names_use_underscore(self):
        sql = _records_to_sql([{"addr": {"city": "NYC"}}])
        assert '"addr_city"' in sql
        assert "'NYC'" in sql

    def test_type_inference_integer(self):
        sql = _records_to_sql([{"n": 42}])
        assert '"n" INTEGER' in sql

    def test_type_inference_real(self):
        sql = _records_to_sql([{"x": 3.14}])
        assert '"x" REAL' in sql

    def test_type_inference_text(self):
        sql = _records_to_sql([{"s": "hello"}])
        assert '"s" TEXT' in sql

    def test_boolean_as_integer(self):
        sql = _records_to_sql([{"flag": True}])
        assert '"flag" INTEGER' in sql
        assert "VALUES (1);" in sql

    def test_none_as_null(self):
        sql = _records_to_sql([{"a": None}])
        assert "VALUES (NULL);" in sql

    def test_string_escaping(self):
        sql = _records_to_sql([{"name": "O'Brien"}])
        assert "'O''Brien'" in sql

    def test_multiple_records(self):
        sql = _records_to_sql([{"id": 1}, {"id": 2}])
        assert sql.count("INSERT INTO") == 2

    def test_missing_field_is_null(self):
        sql = _records_to_sql([{"a": 1, "b": 2}, {"a": 3}])
        lines = [l for l in sql.splitlines() if l.startswith("INSERT")]
        assert lines[1].endswith("(3, NULL);")
