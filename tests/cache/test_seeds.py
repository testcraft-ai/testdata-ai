"""Tests for testdata_ai.cache_manager — seed management (load, add, delete, list, clear)."""

import json
import logging
import pytest

from testdata_ai.cache_manager import CacheManager, _validate_seed_name


class _GeneratorStub:
    def generate(self, context, count, locale=None):
        return []


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
# load_last_seed
# ---------------------------------------------------------------------------

class TestLoadLastSeed:

    def test_returns_none_when_no_seeds_exist(self, tmp_path, monkeypatch):
        from testdata_ai import cache_manager as cm_mod
        monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
        cm = CacheManager(generator=_GeneratorStub(), seed="TEMP-aaa")

        assert cm.load_last_seed() is None

    def test_switches_seed_and_returns_name(self, tmp_path, monkeypatch):
        from testdata_ai import cache_manager as cm_mod
        monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
        cm = CacheManager(generator=_GeneratorStub(), seed="TEMP-bbb")
        cm.add_to_last_seeds("named-seed")
        cm.add_to_last_seeds("recent-seed")

        result = cm.load_last_seed()

        assert result == "recent-seed"
        assert cm.seed == "recent-seed"

    def test_loads_cached_data_for_switched_seed(self, tmp_path, monkeypatch):
        from testdata_ai import cache_manager as cm_mod
        monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)

        seed_file = tmp_path / "seed_my-seed.json"
        seed_file.write_text('{"users": [{"id": 99}]}')

        cm = CacheManager(generator=_GeneratorStub(), seed="TEMP-ccc")
        cm.add_to_last_seeds("my-seed")
        cm.load_last_seed()

        assert cm._cache.get("users") == [{"id": 99}]


# ---------------------------------------------------------------------------
# last_seeds persistence
# ---------------------------------------------------------------------------

def test_load_last_seeds_returns_empty_on_corrupt_json(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    (tmp_path / "last_seeds.json").write_text("not json at all")
    cm = CacheManager(generator=_GeneratorStub())
    assert cm._last_seeds == []


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


def test_add_to_last_seeds_moves_duplicate_to_front(tmp_path, monkeypatch):
    from testdata_ai import cache_manager as cm_mod
    monkeypatch.setattr(cm_mod, "CACHE_DIR", tmp_path)
    cm = CacheManager(generator=_GeneratorStub())
    cm.add_to_last_seeds("seed-a")
    cm.add_to_last_seeds("seed-b")
    cm.add_to_last_seeds("seed-a")
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
    cm.delete_last_seed()


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
