"""Tests for testdata_ai.cache_manager — data access, xdist merge, edge cases."""

import json
import pytest
from unittest.mock import patch

from testdata_ai.cache_manager import CacheManager


class _GeneratorStub:
    def generate(self, context, count, locale=None):
        return []


class _GeneratorStubWithData:
    def __init__(self, records):
        self.records = records
        self.call_count = 0

    def generate(self, context, count, locale=None):
        self.call_count += 1
        return self.records[:count]


# ---------------------------------------------------------------------------
# show_cache
# ---------------------------------------------------------------------------

def test_show_cache_defaults_to_active_seed(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)

    cm = CacheManager(generator=_GeneratorStub(), seed="active-seed")
    seed_file = tmp_path / "seed_active-seed.json"
    seed_file.write_text(json.dumps({"users": [{"id": 1}], "_metadata": {}}))

    assert cm.show_cache() == {"users": 1}


def test_show_cache_without_seed_returns_none(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    cm = CacheManager(generator=_GeneratorStub(), seed=None)

    assert cm.show_cache() is None


def test_show_cache_missing_file_returns_none(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    cm = CacheManager(generator=_GeneratorStub(), seed="no-such-seed")

    assert cm.show_cache() is None


def test_show_cache_empty_file_returns_empty_dict(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    cm = CacheManager(generator=_GeneratorStub(), seed="empty-seed")
    (tmp_path / "seed_empty-seed.json").write_text("{}")

    assert cm.show_cache() == {}


# ---------------------------------------------------------------------------
# xdist merge (_save_seed)
# ---------------------------------------------------------------------------

class TestSaveSeedXdistMerge:

    def _make_cm(self, tmp_path, monkeypatch, seed="shared"):
        from testdata_ai import cache_manager as cm_mod
        monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
        return CacheManager(generator=_GeneratorStub(), seed=seed)

    def test_contexts_from_disk_are_preserved_when_memory_is_empty(self, tmp_path, monkeypatch):
        cm = self._make_cm(tmp_path, monkeypatch)
        seed_file = tmp_path / "seed_shared.json"
        seed_file.write_text(json.dumps({"contextX": [{"id": 1}]}))

        cm._cache = {"contextY": [{"id": 2}]}
        cm._save_seed()

        result = json.loads(seed_file.read_text())
        assert result["contextX"] == [{"id": 1}]
        assert result["contextY"] == [{"id": 2}]

    def test_longer_in_memory_list_wins_over_disk(self, tmp_path, monkeypatch):
        cm = self._make_cm(tmp_path, monkeypatch)
        seed_file = tmp_path / "seed_shared.json"
        seed_file.write_text(json.dumps({"ctx": [{"id": 1}]}))

        cm._cache = {"ctx": [{"id": 1}, {"id": 2}, {"id": 3}]}
        cm._save_seed()

        result = json.loads(seed_file.read_text())
        assert len(result["ctx"]) == 3

    def test_longer_disk_list_wins_over_memory(self, tmp_path, monkeypatch):
        cm = self._make_cm(tmp_path, monkeypatch)
        seed_file = tmp_path / "seed_shared.json"
        seed_file.write_text(json.dumps({"ctx": [{"id": 1}, {"id": 2}, {"id": 3}]}))

        cm._cache = {"ctx": [{"id": 1}]}
        cm._save_seed()

        result = json.loads(seed_file.read_text())
        assert len(result["ctx"]) == 3

    def test_missing_seed_file_is_created_from_memory(self, tmp_path, monkeypatch):
        cm = self._make_cm(tmp_path, monkeypatch)
        seed_file = tmp_path / "seed_shared.json"
        assert not seed_file.exists()

        cm._cache = {"ctx": [{"id": 42}]}
        cm._save_seed()

        result = json.loads(seed_file.read_text())
        assert result == {"ctx": [{"id": 42}]}


# ---------------------------------------------------------------------------
# _load_seed edge cases
# ---------------------------------------------------------------------------

def test_load_seed_returns_early_when_no_seed(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    cm = CacheManager(generator=_GeneratorStub(), seed=None)
    cm._load_seed()
    assert cm._cache == {}


def test_load_seed_handles_file_disappearing_under_lock(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    seed_file = tmp_path / "seed_race.json"
    seed_file.write_text('{"ctx": []}')
    cm = CacheManager(generator=_GeneratorStub(), seed="race")
    with patch("builtins.open", side_effect=FileNotFoundError("gone")):
        cm._load_seed()
    assert cm._cache == {}


# ---------------------------------------------------------------------------
# _save_seed edge cases
# ---------------------------------------------------------------------------

def test_save_seed_is_noop_when_no_seed(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    cm = CacheManager(generator=_GeneratorStub(), seed=None)
    cm._save_seed()
    assert list(tmp_path.glob("*.json")) == []


def test_save_seed_raises_on_corrupted_disk_file(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    cm = CacheManager(generator=_GeneratorStub(), seed="corrupt")
    (tmp_path / "seed_corrupt.json").write_text("NOT JSON")
    cm._cache = {"ctx": [{"id": 1}]}
    with pytest.raises(RuntimeError, match="corrupted"):
        cm._save_seed()


def test_save_seed_cleans_tmp_file_on_error(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    cm = CacheManager(generator=_GeneratorStub(), seed="err-seed")
    cm._cache = {"ctx": [{"id": 1}]}
    with patch("testdata_ai.cache_manager.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            cm._save_seed()
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------

def test_finalize_persists_seed_and_last_seeds(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    cm = CacheManager(generator=_GeneratorStub(), seed="final-seed")
    cm._cache = {"ctx": [{"id": 1}]}
    cm._last_seeds = ["final-seed"]
    cm.finalize()
    assert (tmp_path / "seed_final-seed.json").exists()
    assert (tmp_path / "last_seeds.json").exists()


# ---------------------------------------------------------------------------
# get_data
# ---------------------------------------------------------------------------

def test_get_data_generates_missing_records(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    stub = _GeneratorStubWithData([{"id": i} for i in range(5)])
    cm = CacheManager(generator=stub, seed="gd-seed")
    result = cm.get_data("my-ctx", 3)
    assert len(result) == 3
    assert stub.call_count == 1


def test_get_data_uses_cache_on_repeated_calls(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    stub = _GeneratorStubWithData([{"id": i} for i in range(5)])
    cm = CacheManager(generator=stub, seed="gd-cache")
    cm.get_data("my-ctx", 3)
    cm.get_data("my-ctx", 3)
    assert stub.call_count == 1


def test_get_data_returns_deep_copy(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    stub = _GeneratorStubWithData([{"id": 1, "val": "orig"}])
    cm = CacheManager(generator=stub, seed="deep-copy")
    result = cm.get_data("ctx", 1)
    result[0]["val"] = "modified"
    cached = cm.get_data("ctx", 1)
    assert cached[0]["val"] == "orig"


def test_get_data_requests_only_missing_records(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    stub = _GeneratorStubWithData([{"id": i} for i in range(10)])
    cm = CacheManager(generator=stub, seed="partial")
    cm.get_data("ctx", 2)
    stub.call_count = 0
    cm.get_data("ctx", 5)
    assert stub.call_count == 1
    assert len(cm._cache["ctx"]) == 5
