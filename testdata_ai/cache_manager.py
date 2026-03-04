import copy
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
import threading
import logging
from typing import Optional

from filelock import FileLock

logger = logging.getLogger("testdata_ai")
CACHE_DIR = Path(".testdata_ai_cache")

# Only allow safe characters in seed names to prevent path traversal.
_VALID_SEED_RE = re.compile(r"^[\w-]+$")


def _validate_seed_name(name: str) -> None:
    if not _VALID_SEED_RE.match(name):
        raise ValueError(
            f"Invalid seed name '{name}': only letters, digits, hyphens, "
            "and underscores are allowed."
        )


class CacheManager:
    def __init__(self, generator, seed=None):
        self.generator = generator
        self._cache_dir = CACHE_DIR
        self._cache = {}
        self._lock = threading.Lock()
        self._last_seeds_file = self._cache_dir / "last_seeds.json"
        self._last_seeds = self._load_last_seeds()
        self.seed = seed
        if seed:
            _validate_seed_name(seed)
            self._load_seed()

    def _load_last_seeds(self):
        if self._last_seeds_file.exists():
            try:
                return json.loads(self._last_seeds_file.read_text())
            except Exception:
                return []
        return []

    @contextmanager
    def _last_seeds_lock(self):
        """Acquire the FileLock for last_seeds.json."""
        self._cache_dir.mkdir(exist_ok=True)
        with FileLock(str(self._cache_dir / "last_seeds.lock")):
            yield

    def _save_last_seeds(self):
        self._cache_dir.mkdir(exist_ok=True)
        with self._last_seeds_lock():
            on_disk = self._load_last_seeds()
            merged = list(on_disk)
            for seed in self._last_seeds:
                if seed not in merged:
                    merged.append(seed)
            self._last_seeds = merged
            self._last_seeds_file.write_text(json.dumps(merged))

    def add_to_last_seeds(self, seed):
        with self._last_seeds_lock():
            # Re-read under lock so concurrent xdist workers don't overwrite each other.
            current = self._load_last_seeds()
            if seed in current:
                current.remove(seed)
            current.insert(0, seed)
            self._last_seeds = current
            self._last_seeds_file.write_text(json.dumps(current))

    def load_last_seed(self):
        """Switch to the most recent named seed.

        Sets ``self.seed`` to the first entry in the last-seeds queue and
        reloads the cache from disk.  Returns the seed name that was loaded,
        or ``None`` when the queue is empty.
        """
        if not self._last_seeds:
            return None
        last = self._last_seeds[0]
        _validate_seed_name(last)
        self.seed = last
        self._load_seed()
        return last

    def list_seeds(self):
        if not self._cache_dir.exists():
            return []
        return [f.stem.replace("seed_", "") for f in self._cache_dir.glob("seed_*.json")]

    def delete_seed(self, seed):
        _validate_seed_name(seed)
        path = self._cache_dir / f"seed_{seed}.json"
        if path.exists():
            path.unlink()
            logger.info(f"Deleted seed '{seed}'")
        else:
            logger.warning(f"Seed '{seed}' not found, nothing to delete")
        with self._last_seeds_lock():
            current = self._load_last_seeds()
            if seed in current:
                current.remove(seed)
            self._last_seeds = current
            self._last_seeds_file.write_text(json.dumps(current))

    def delete_last_seed(self):
        if self._last_seeds:
            self.delete_seed(self._last_seeds[0])

    def clear_cache(self):
        if self._cache_dir.exists():
            for f in self._cache_dir.glob("seed_*.json"):
                f.unlink()
        self._last_seeds = []
        with self._last_seeds_lock():
            self._last_seeds_file.write_text(json.dumps([]))
        logger.info("Cleared all seeds")

    def seed_path(self, seed=None) -> Path:
        """Return the cache file path for the given seed (or current seed)."""
        return self._cache_dir / f"seed_{seed or self.seed}.json"

    def show_cache(self, seed=None):
        """Return a dict of {context: item_count} for the given seed.

        Returns None when the seed file does not exist, an empty dict when
        the file exists but contains no context data, or a non-empty dict
        with context counts.
        """
        seed = seed or self.seed
        if not seed:
            return None
        path = self.seed_path(seed)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return {ctx: len(items) for ctx, items in data.items() if ctx != "_metadata"}

    @contextmanager
    def _seed_lock(self):
        """Acquire the FileLock for the current seed and yield the seed file path."""
        self._cache_dir.mkdir(exist_ok=True)
        seed_file = self._cache_dir / f"seed_{self.seed}.json"
        with FileLock(str(self._cache_dir / f"seed_{self.seed}.lock")):
            yield seed_file

    def _load_seed(self):
        if not self.seed:
            return

        # Fast path: avoid creating the cache dir for TEMP seeds that have no
        # persisted data yet. The existence check is intentionally outside the
        # lock; if the file disappears between here and the open() below we
        # catch FileNotFoundError inside the lock instead of crashing.
        seed_file = self._cache_dir / f"seed_{self.seed}.json"
        if not self._cache_dir.exists() or not seed_file.exists():
            self._cache = {}
            if self.seed.startswith("TEMP-"):
                logger.debug(f"Creating new seed '{self.seed}'")
            else:
                logger.info(f"Creating new seed '{self.seed}'")
            return

        with self._seed_lock() as file:
            try:
                with open(file, "r") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded seed '{self.seed}' from {file}")
            except FileNotFoundError:
                self._cache = {}
                logger.info(f"Seed '{self.seed}' removed before lock acquired, starting fresh")

    def _save_seed(self):
        if not self.seed:
            return

        with self._seed_lock() as file:
            # Read disk state inside the lock so concurrent xdist workers
            # don't overwrite each other's data.  Merge strategy: start from
            # whatever is on disk (written by peer workers), then overlay only
            # the contexts where we have *more* items than disk — we never
            # shrink an existing context.
            disk_cache: dict = {}
            if file.exists():
                try:
                    with open(file, "r") as f:
                        disk_cache = json.load(f)
                except Exception as exc:
                    raise RuntimeError(
                        f"Seed '{self.seed}' cache file is corrupted: {exc}"
                    ) from exc

            merged = dict(disk_cache)
            for context, items in self._cache.items():
                if len(items) > len(merged.get(context, [])):
                    merged[context] = items

            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(
                    dir=self._cache_dir, prefix=f"seed_{self.seed}_", suffix=".tmp"
                )
                with os.fdopen(fd, "w") as f:
                    json.dump(merged, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, file)
                logger.info(f"Seed '{self.seed}' saved successfully")
            except Exception as e:
                logger.warning(f"Failed to save seed {self.seed}: {e}")
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

    def finalize(self):
        """Persist seed data and last-seeds queue to disk. Call once at session end."""
        self._save_seed()
        self._save_last_seeds()

    def get_data(self, context: str, count: int, locale: Optional[str] = None):
        cache_key = f"{context}:{locale}" if locale else context
        with self._lock:
            existing = len(self._cache.get(cache_key, []))
            if existing < count:
                missing = count - existing
                logger.info(
                    f"Added {missing} missing records for {cache_key} to cache"
                )
                new_data = self.generator.generate(
                    context=context,
                    count=missing,
                    locale=locale,
                )

                self._cache.setdefault(cache_key, []).extend(new_data)
                self._save_seed()
            data = self._cache[cache_key][:count]
        return copy.deepcopy(data)
