import json
import logging
from unittest.mock import patch

import pytest

from testdata_ai.cache_manager import CacheManager, _validate_seed_name


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


class TestLoadLastSeed:

    def test_returns_none_when_no_seeds_exist(self, tmp_path, monkeypatch):
        from testdata_ai import cache_manager as cm_mod

        monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
        cm = CacheManager(generator=_GeneratorStub(), seed="TEMP-aaa")

        result = cm.load_last_seed()

        assert result is None

    def test_switches_seed_and_returns_name(self, tmp_path, monkeypatch):
        from testdata_ai import cache_manager as cm_mod

        monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
        cm = CacheManager(generator=_GeneratorStub(), seed="TEMP-bbb")
        # Populate the last-seeds queue manually via the public method.
        cm.add_to_last_seeds("named-seed")
        cm.add_to_last_seeds("recent-seed")

        result = cm.load_last_seed()

        assert result == "recent-seed"
        assert cm.seed == "recent-seed"

    def test_loads_cached_data_for_switched_seed(self, tmp_path, monkeypatch):
        from testdata_ai import cache_manager as cm_mod

        monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)

        # Write a seed file that load_last_seed() should pick up.
        seed_file = tmp_path / "seed_my-seed.json"
        seed_file.write_text('{"users": [{"id": 99}]}')

        cm = CacheManager(generator=_GeneratorStub(), seed="TEMP-ccc")
        cm.add_to_last_seeds("my-seed")

        cm.load_last_seed()

        assert cm._cache.get("users") == [{"id": 99}]


class TestSaveSeedXdistMerge:
    """_save_seed must merge with disk so concurrent xdist workers don't
    overwrite each other's generated data."""

    def _make_cm(self, tmp_path, monkeypatch, seed="shared"):
        from testdata_ai import cache_manager as cm_mod
        monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
        return CacheManager(generator=_GeneratorStub(), seed=seed)

    def test_contexts_from_disk_are_preserved_when_memory_is_empty(self, tmp_path, monkeypatch):
        """Worker B should keep data written by worker A even if B's in-memory
        cache doesn't know about that context."""
        cm = self._make_cm(tmp_path, monkeypatch)

        # Simulate worker A having already written contextX to disk.
        seed_file = tmp_path / "seed_shared.json"
        seed_file.write_text(json.dumps({"contextX": [{"id": 1}]}))

        # Worker B's in-memory cache only has contextY.
        cm._cache = {"contextY": [{"id": 2}]}
        cm._save_seed()

        result = json.loads(seed_file.read_text())
        assert result["contextX"] == [{"id": 1}], "contextX from disk was lost"
        assert result["contextY"] == [{"id": 2}], "contextY from memory was not written"

    def test_longer_in_memory_list_wins_over_disk(self, tmp_path, monkeypatch):
        """When the in-memory cache has more items than disk, our data wins."""
        cm = self._make_cm(tmp_path, monkeypatch)

        seed_file = tmp_path / "seed_shared.json"
        seed_file.write_text(json.dumps({"ctx": [{"id": 1}]}))

        cm._cache = {"ctx": [{"id": 1}, {"id": 2}, {"id": 3}]}
        cm._save_seed()

        result = json.loads(seed_file.read_text())
        assert len(result["ctx"]) == 3

    def test_longer_disk_list_wins_over_memory(self, tmp_path, monkeypatch):
        """When disk has more items (written by another worker after we loaded),
        the disk data is kept — we never shrink a context."""
        cm = self._make_cm(tmp_path, monkeypatch)

        seed_file = tmp_path / "seed_shared.json"
        seed_file.write_text(json.dumps({"ctx": [{"id": 1}, {"id": 2}, {"id": 3}]}))

        cm._cache = {"ctx": [{"id": 1}]}
        cm._save_seed()

        result = json.loads(seed_file.read_text())
        assert len(result["ctx"]) == 3

    def test_missing_seed_file_is_created_from_memory(self, tmp_path, monkeypatch):
        """Normal first-write: no disk file yet, memory is written as-is."""
        cm = self._make_cm(tmp_path, monkeypatch)
        seed_file = tmp_path / "seed_shared.json"
        assert not seed_file.exists()

        cm._cache = {"ctx": [{"id": 42}]}
        cm._save_seed()

        result = json.loads(seed_file.read_text())
        assert result == {"ctx": [{"id": 42}]}


# ---------------------------------------------------------------------------
# _validate_seed_name
# ---------------------------------------------------------------------------

class TestValidateSeedName:

    def test_valid_name_does_not_raise(self):
        _validate_seed_name("valid-seed_123")

    def test_space_raises(self):
        with pytest.raises(ValueError, match="Invalid seed name"):
            _validate_seed_name("with space")

    def test_slash_raises(self):
        with pytest.raises(ValueError, match="Invalid seed name"):
            _validate_seed_name("../traversal")

    def test_dot_raises(self):
        with pytest.raises(ValueError, match="Invalid seed name"):
            _validate_seed_name("bad.name")


# ---------------------------------------------------------------------------
# _load_last_seeds fallback
# ---------------------------------------------------------------------------

def test_load_last_seeds_returns_empty_on_corrupt_json(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    (tmp_path / "last_seeds.json").write_text("not json at all")
    cm = CacheManager(generator=_GeneratorStub())
    assert cm._last_seeds == []


# ---------------------------------------------------------------------------
# _save_last_seeds (merge with on-disk)
# ---------------------------------------------------------------------------

def test_save_last_seeds_merges_disk_and_memory(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    cm = CacheManager(generator=_GeneratorStub())
    (tmp_path / "last_seeds.json").write_text(json.dumps(["disk-seed"]))
    cm._last_seeds = ["mem-seed"]
    cm._save_last_seeds()
    result = json.loads((tmp_path / "last_seeds.json").read_text())
    assert "disk-seed" in result
    assert "mem-seed" in result


# ---------------------------------------------------------------------------
# add_to_last_seeds deduplication (line 72)
# ---------------------------------------------------------------------------

def test_add_to_last_seeds_moves_duplicate_to_front(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    cm = CacheManager(generator=_GeneratorStub())
    cm.add_to_last_seeds("seed-a")
    cm.add_to_last_seeds("seed-b")
    cm.add_to_last_seeds("seed-a")  # should move to front, not duplicate
    seeds = json.loads((tmp_path / "last_seeds.json").read_text())
    assert seeds[0] == "seed-a"
    assert seeds.count("seed-a") == 1


# ---------------------------------------------------------------------------
# list_seeds
# ---------------------------------------------------------------------------

def test_list_seeds_returns_empty_when_cache_dir_missing(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path / "nonexistent")
    cm = CacheManager(generator=_GeneratorStub())
    assert cm.list_seeds() == []


def test_list_seeds_returns_seed_names(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    (tmp_path / "seed_alpha.json").write_text("{}")
    (tmp_path / "seed_beta.json").write_text("{}")
    cm = CacheManager(generator=_GeneratorStub())
    seeds = cm.list_seeds()
    assert set(seeds) == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# delete_seed
# ---------------------------------------------------------------------------

def test_delete_seed_removes_file_and_entry(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    seed_file = tmp_path / "seed_my-seed.json"
    seed_file.write_text("{}")
    (tmp_path / "last_seeds.json").write_text(json.dumps(["my-seed", "other"]))
    cm = CacheManager(generator=_GeneratorStub())
    cm.delete_seed("my-seed")
    assert not seed_file.exists()
    seeds = json.loads((tmp_path / "last_seeds.json").read_text())
    assert "my-seed" not in seeds
    assert "other" in seeds


def test_delete_seed_logs_warning_when_not_found(tmp_path, monkeypatch, caplog):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    cm = CacheManager(generator=_GeneratorStub())
    with caplog.at_level(logging.WARNING, logger="testdata_ai"):
        cm.delete_seed("ghost-seed")
    assert "not found" in caplog.text


# ---------------------------------------------------------------------------
# delete_last_seed
# ---------------------------------------------------------------------------

def test_delete_last_seed_removes_first_seed(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    (tmp_path / "seed_first.json").write_text("{}")
    (tmp_path / "last_seeds.json").write_text(json.dumps(["first", "second"]))
    cm = CacheManager(generator=_GeneratorStub())
    cm.delete_last_seed()
    assert not (tmp_path / "seed_first.json").exists()


def test_delete_last_seed_noop_when_no_seeds(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    cm = CacheManager(generator=_GeneratorStub())
    cm.delete_last_seed()  # should not raise


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------

def test_clear_cache_removes_all_seed_files(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    (tmp_path / "seed_a.json").write_text("{}")
    (tmp_path / "seed_b.json").write_text("{}")
    (tmp_path / "last_seeds.json").write_text(json.dumps(["a", "b"]))
    cm = CacheManager(generator=_GeneratorStub())
    cm.clear_cache()
    assert list(tmp_path.glob("seed_*.json")) == []
    assert json.loads((tmp_path / "last_seeds.json").read_text()) == []


# ---------------------------------------------------------------------------
# _load_seed edge cases
# ---------------------------------------------------------------------------

def test_load_seed_returns_early_when_no_seed(tmp_path, monkeypatch):
    """Calling _load_seed() when self.seed is None is a no-op."""
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    cm = CacheManager(generator=_GeneratorStub(), seed=None)
    cm._load_seed()  # direct call with no seed set
    assert cm._cache == {}


def test_load_seed_handles_file_disappearing_under_lock(tmp_path, monkeypatch):
    """Race condition: file exists before lock but disappears inside open()."""
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    seed_file = tmp_path / "seed_race.json"
    seed_file.write_text('{"ctx": []}')
    cm = CacheManager(generator=_GeneratorStub(), seed="race")
    # Simulate file being deleted between existence check and open()
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
    # Write corrupted data AFTER init (so _load_seed doesn't fail)
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
    assert stub.call_count == 1  # second call served from cache


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
    cm.get_data("ctx", 2)   # generates 2
    stub.call_count = 0     # reset counter
    cm.get_data("ctx", 5)   # should generate only 3 more
    assert stub.call_count == 1
    # verify the generator was asked for the delta
    assert len(cm._cache["ctx"]) == 5
