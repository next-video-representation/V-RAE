"""Node-local, size-bounded cache of whole remote video files, shared by every rank and DataLoader worker.

Layout of storage.cache.dir (the cache only ever deletes inside it, and only files it published):
  NANOVVM_VIDEO_CACHE              marker: this directory belongs to the cache
  state.sqlite                     published entries (LRU clock), in-flight reservations, counters
  objects/<key[:2]>/<key><suffix>  published videos, complete and length-checked, renamed into place
  tmp/<key>.<pid>.<nonce>.part     downloads in flight
  locks/<stripe>.lock              producer locks, striped by key

Concurrency, for processes on one host (one cache directory per host; it is not for shared file systems):
  * One producer per video: a miss takes the key's stripe lock (flock LOCK_EX) and downloads; other
    processes asking for the same video wait on that lock, then find the published file.
  * Leases: a reader holds flock(LOCK_SH) on the published file from lookup until its decoder is done.
    Eviction takes flock(LOCK_EX | LOCK_NB) before unlinking, so a leased file is never removed; a reader
    that opened a file just as it was evicted sees st_nlink == 0 and treats it as a miss. The producer
    takes its lease on the temporary file before renaming it, so no eviction can slip in between publish and
    use. flock locks vanish with their process: a killed reader or producer leaves no stale pin or lock.
  * One budget for the node: a single SQLite write transaction (BEGIN IMMEDIATE) reserves a download's bytes,
    evicting least-recently-used unleased files as needed, so published bytes + in-flight reservations +
    the state database stay within capacity_bytes; it also bounds concurrent downloads and keeps
    reserve_bytes free on the file system. A reservation records its process (pid and start time); those of
    dead processes are dropped together with their partial files.
  * A video larger than the whole budget fails at once (CacheTooSmallError). When nothing evictable is left
    (everything leased or in flight) a demand read waits, polling, up to wait_timeout seconds and then fails
    with the cache's state (CacheFullError); it never exceeds the budget. Prefetch never waits.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

from vrae.storage import sqlite_fork

MARKER = "NANOVVM_VIDEO_CACHE"
BLOCK = 4096
STRIPES = 4096
TOUCH_INTERVAL = 30.0
FLUSH_INTERVAL = 2.0
WAL_LIMIT = 1 << 20  # the write-ahead log is truncated to this after checkpoints; counted as a fixed allowance


class CacheError(RuntimeError):
    pass


class CacheTooSmallError(CacheError):
    pass


class CacheFullError(CacheError):
    pass


class CacheBusy(Exception):
    """Prefetch only: the cache cannot take this download right now."""


def footprint(size: int) -> int:
    """Disk blocks a file of size bytes occupies, plus one block for its inode and directory entry."""
    return (size + BLOCK - 1) // BLOCK * BLOCK + BLOCK


def cache_key(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class CachedObject:
    """What to cache: a stable key (from the remote identity), its exact length and a file suffix."""
    key: str
    size: int
    suffix: str = ""
    description: str = ""
    meta: dict = field(default_factory=dict, compare=False, hash=False)


def _process_start(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/stat", "rb") as stream:
            data = stream.read()
    except OSError:
        return None
    return data.rsplit(b")", 1)[1].split()[19].decode()  # field 22: start time in clock ticks


def _alive(pid: int, start: str | None) -> bool:
    current = _process_start(pid)
    if current is None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    return start is None or current == start


SCHEMA = """
CREATE TABLE IF NOT EXISTS entries(key TEXT PRIMARY KEY, path TEXT NOT NULL, size INTEGER NOT NULL,
    footprint INTEGER NOT NULL, last_access REAL NOT NULL, prefetched INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS entries_lru ON entries(last_access);
CREATE TABLE IF NOT EXISTS reservations(key TEXT NOT NULL, footprint INTEGER NOT NULL, kind TEXT NOT NULL,
    pid INTEGER NOT NULL, start TEXT, tmp TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS counters(name TEXT PRIMARY KEY, value REAL NOT NULL);
"""


class Lease:
    """A published cache file pinned for reading; close() (or leaving the with-block) releases it."""

    def __init__(self, path: str, fd: int, status: str):
        self.path, self._fd, self.status = path, fd, status

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self.path

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


class VideoCache:
    def __init__(self, directory, capacity_bytes: int, *, reserve_bytes: int = 0, max_downloads: int = 16,
                 prefetch_max_downloads: int | None = None, prefetch_max_bytes: int | None = None,
                 wait_timeout: float = 600.0):
        self.directory = os.path.abspath(os.path.expanduser(str(directory)))
        if capacity_bytes <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity = int(capacity_bytes)
        self.reserve = int(reserve_bytes)
        self.max_downloads = max(1, int(max_downloads))
        self.prefetch_max_downloads = max(0, int(prefetch_max_downloads if prefetch_max_downloads is not None
                                                 else max(0, self.max_downloads - max(1, self.max_downloads // 4))))
        self.prefetch_max_bytes = int(prefetch_max_bytes if prefetch_max_bytes is not None else self.capacity // 8)
        self.wait_timeout = float(wait_timeout)
        self._local = threading.local()
        self._counter_lock = threading.Lock()
        self._counters, self._last_flush = {}, time.monotonic()
        self._initialize_directory()
        self._register_exit_flush()

    # ---- setup and state ----

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_local=None, _counter_lock=None, _counters={})
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local, self._counter_lock = threading.local(), threading.Lock()
        self._register_exit_flush()

    def _register_exit_flush(self):
        """Write pending counters when the process exits, including DataLoader workers (multiprocessing runs
        its finalizers on a worker's normal exit, where atexit does not run)."""
        import atexit
        from multiprocessing import util

        def flush():
            try:
                self.flush_counters()
            except Exception:  # noqa: BLE001 - statistics only
                pass

        atexit.register(flush)
        util.Finalize(None, flush, exitpriority=10)

    def _initialize_directory(self):
        marker = os.path.join(self.directory, MARKER)
        if os.path.isdir(self.directory) and not os.path.exists(marker) and os.listdir(self.directory):
            raise CacheError(f"{self.directory} is not empty and is not a video cache (no {MARKER} marker); "
                             "point storage.cache.dir at an empty or cache-owned directory")
        for sub in ("objects", "tmp", "locks"):
            os.makedirs(os.path.join(self.directory, sub), exist_ok=True)
        if not os.path.exists(marker):
            with open(marker, "a") as stream:
                stream.write("nanovvm video cache: every file below is disposable and may be evicted.\n")
        self._db()

    def _open_db(self) -> sqlite3.Connection:
        deadline = time.monotonic() + 120
        while True:
            db = sqlite_fork.connect(os.path.join(self.directory, "state.sqlite"), timeout=120, isolation_level=None)
            try:
                db.execute("PRAGMA busy_timeout = 120000")
                db.execute("PRAGMA synchronous = NORMAL")
                db.execute("PRAGMA wal_autocheckpoint = 128")
                db.execute(f"PRAGMA journal_size_limit = {WAL_LIMIT}")
                # Switching the journal mode can return BUSY without waiting; it is persistent, so only the
                # first connection of a new cache switches it (under the directory's init lock).
                if db.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                    with self._init_lock():
                        db.execute("PRAGMA journal_mode = WAL")
                db.executescript(SCHEMA)
                return db
            except sqlite3.OperationalError as error:
                db.close()
                if "locked" not in str(error) or time.monotonic() > deadline:
                    raise
                time.sleep(0.05)

    @contextmanager
    def _init_lock(self):
        fd = os.open(os.path.join(self.directory, "locks", "init.lock"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    @contextmanager
    def _connection(self):
        """This thread's connection, used under the fork guard (data/sqlite_fork.py)."""
        with sqlite_fork.GUARD:
            yield sqlite_fork.thread_connection(self._local, self._open_db)

    def _db(self):
        with self._connection():
            pass

    @contextmanager
    def _transaction(self):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")

    def object_path(self, obj: CachedObject) -> str:
        return os.path.join(self.directory, "objects", obj.key[:2], obj.key + obj.suffix)

    def _relative(self, path):
        return os.path.relpath(path, self.directory)

    # ---- counters ----

    def count(self, **values):
        """Accumulate counters in memory (safe inside transactions); maybe_flush() writes them out."""
        with self._counter_lock:
            for name, value in values.items():
                self._counters[name] = self._counters.get(name, 0) + value

    def maybe_flush(self):
        if time.monotonic() - self._last_flush > FLUSH_INTERVAL:
            self.flush_counters()

    def _drain_counters(self, db):
        with self._counter_lock:
            values, self._counters = self._counters, {}
            self._last_flush = time.monotonic()
        for name, value in values.items():
            db.execute("INSERT INTO counters(name, value) VALUES (?, ?) "
                       "ON CONFLICT(name) DO UPDATE SET value = value + excluded.value", (name, value))

    def flush_counters(self):
        with self._counter_lock:
            pending = bool(self._counters)
        if pending:
            with self._transaction() as db:
                self._drain_counters(db)

    def stats(self) -> dict:
        self.flush_counters()
        with self._connection() as db:
            counters = dict(db.execute("SELECT name, value FROM counters").fetchall())
            entries, used = db.execute("SELECT COUNT(*), COALESCE(SUM(footprint), 0) FROM entries").fetchone()
            inflight, reserved = db.execute("SELECT COUNT(*), COALESCE(SUM(footprint), 0) "
                                            "FROM reservations").fetchone()
        return {**counters, "entries": entries, "used_bytes": used, "inflight_downloads": inflight,
                "reserved_bytes": reserved, "state_bytes": self._state_bytes(), "capacity_bytes": self.capacity}

    def _state_bytes(self):
        """The state database counts against the budget: its main file plus a fixed WAL allowance."""
        try:
            main = os.path.getsize(os.path.join(self.directory, "state.sqlite"))
        except OSError:
            main = 0
        return footprint(main) + WAL_LIMIT

    # ---- leases ----

    def contains(self, obj: CachedObject) -> bool:
        try:
            return os.stat(self.object_path(obj)).st_size == obj.size
        except OSError:
            return False

    def _try_pin(self, obj: CachedObject) -> Lease | None:
        path = self.object_path(obj)
        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            stat = os.fstat(fd)
            if stat.st_nlink == 0 or stat.st_size != obj.size:
                os.close(fd)
                return None
        except BaseException:
            os.close(fd)
            raise
        lease = Lease(path, fd, "hit")
        try:
            self._touch(obj)
        except BaseException:
            lease.close()
            raise
        return lease

    def _touch(self, obj: CachedObject):
        now = time.time()
        with self._connection() as db:
            row = db.execute("SELECT last_access, prefetched FROM entries WHERE key = ?", (obj.key,)).fetchone()
        if row is None:
            return
        if row[1]:
            self.count(prefetch_used=1)
        if row[1] or now - row[0] > TOUCH_INTERVAL:
            with self._transaction() as db:
                db.execute("UPDATE entries SET last_access = ?, prefetched = 0 WHERE key = ?", (now, obj.key))

    @contextmanager
    def _producer_lock(self, obj: CachedObject, *, wait: bool):
        stripe = int(obj.key[:8], 16) % STRIPES
        fd = os.open(os.path.join(self.directory, "locks", f"{stripe:04d}.lock"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            started, delay = time.monotonic(), 0.002
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if not wait:
                        raise CacheBusy("another process is downloading this stripe") from None
                    waited = time.monotonic() - started
                    if waited > self.wait_timeout:
                        raise CacheFullError(f"waited {waited:.0f}s for another process to download "
                                             f"{obj.description or obj.key}; storage.cache.wait_timeout_seconds="
                                             f"{self.wait_timeout:g}") from None
                    time.sleep(delay)
                    delay = min(0.1, delay * 2)
            yield time.monotonic() - started
        finally:
            os.close(fd)

    def lease(self, obj: CachedObject, fetch) -> Lease:
        """Pin obj's cached file for reading, downloading it first (fetch(sink_file)) if needed."""
        self.count(demand_requests=1)
        lease = self._try_pin(obj)
        if lease is not None:
            self.count(demand_hits=1)
            self.maybe_flush()
            return lease
        started = time.monotonic()
        try:
            with self._producer_lock(obj, wait=True):
                lease = self._try_pin(obj)
                if lease is not None:
                    self.count(demand_hits=1, demand_waited_for_producer=1)
                    lease.status = "waited"
                    return lease
                self._discard_invalid(obj)
                lease = self._download(obj, fetch, "demand")
                self.count(demand_misses=1)
                return lease
        finally:
            self.count(demand_wait_seconds=time.monotonic() - started)
            self.maybe_flush()

    def prefetch(self, obj: CachedObject, fetch) -> str:
        """Download obj unless cached, in flight elsewhere, or over the prefetch limits. Never waits."""
        self.count(prefetch_requests=1)
        try:
            return self._prefetch(obj, fetch)
        finally:
            self.maybe_flush()

    def _prefetch(self, obj, fetch) -> str:
        if self.contains(obj):
            self.count(prefetch_already_cached=1)
            return "cached"
        try:
            with self._producer_lock(obj, wait=False):
                if self.contains(obj):
                    self.count(prefetch_already_cached=1)
                    return "cached"
                self._discard_invalid(obj)
                self._download(obj, fetch, "prefetch").close()
                return "downloaded"
        except CacheBusy:
            self.count(prefetch_skipped_busy=1)
            return "busy"

    def _discard_invalid(self, obj):
        """Under the producer lock: remove a published file whose length is wrong (never a leased one)."""
        path = self.object_path(obj)
        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if os.fstat(fd).st_nlink and os.fstat(fd).st_size != obj.size:
                with self._transaction() as db:
                    os.unlink(path)
                    db.execute("DELETE FROM entries WHERE key = ?", (obj.key,))
        except BlockingIOError:
            pass
        finally:
            os.close(fd)

    # ---- downloads ----

    def _download(self, obj: CachedObject, fetch, kind: str) -> Lease:
        if footprint(obj.size) + self._state_bytes() > self.capacity:
            raise CacheTooSmallError(
                f"{obj.description or obj.key} is {obj.size / 2**30:.2f} GiB, more than the whole video cache "
                f"({self.capacity / 2**30:.2f} GiB, storage.cache.capacity_gib); raise the capacity")
        tmp = os.path.join(self.directory, "tmp", f"{obj.key}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part")
        self._reserve(obj, kind, tmp)
        pin = None
        try:
            fd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "wb+") as stream:
                fetched = fetch(stream)
                fetched = fetched if isinstance(fetched, dict) else {}
                stream.flush()
                size = os.fstat(stream.fileno()).st_size
            if size != obj.size:
                raise CacheError(f"downloaded {size} bytes of {obj.description or obj.key}, expected {obj.size}")
            pin = os.open(tmp, os.O_RDONLY)
            fcntl.flock(pin, fcntl.LOCK_SH)  # the lease starts before the file becomes visible
            final = self.object_path(obj)
            os.makedirs(os.path.dirname(final), exist_ok=True)
            with self._transaction() as db:
                os.rename(tmp, final)
                db.execute("INSERT OR REPLACE INTO entries(key, path, size, footprint, last_access, prefetched) "
                           "VALUES (?, ?, ?, ?, ?, ?)", (obj.key, self._relative(final), obj.size,
                                                         footprint(obj.size), time.time(), int(kind == "prefetch")))
                db.execute("DELETE FROM reservations WHERE tmp = ?", (tmp,))
                self.count(**{f"{kind}_downloads": 1, f"{kind}_bytes": obj.size},
                           **{f"transport_{name}": value for name, value in fetched.items()})
                self._drain_counters(db)
            return Lease(final, pin, "downloaded")
        except BaseException:
            if pin is not None:
                os.close(pin)
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            with self._transaction() as db:
                db.execute("DELETE FROM reservations WHERE tmp = ?", (tmp,))
            self.count(download_errors=1)
            raise

    def _reserve(self, obj: CachedObject, kind: str, tmp: str):
        need = footprint(obj.size)
        started, delay = time.monotonic(), 0.01
        while True:
            with self._transaction() as db:
                problem = self._try_reserve(db, obj, kind, tmp, need)
            if problem is None:
                return
            if kind == "prefetch":
                raise CacheBusy(problem)
            waited = time.monotonic() - started
            if waited > self.wait_timeout:
                raise CacheFullError(f"no room to download {obj.description or obj.key} after {waited:.0f}s: "
                                     f"{problem}. Everything else is leased or in flight; raise "
                                     "storage.cache.capacity_gib or lower the number of concurrent readers.")
            time.sleep(delay)
            delay = min(1.0, delay * 2)

    def _purge_dead(self, db) -> int:
        dropped = 0
        for rowid, pid, start, tmp in db.execute("SELECT rowid, pid, start, tmp FROM reservations").fetchall():
            if not _alive(pid, start):
                db.execute("DELETE FROM reservations WHERE rowid = ?", (rowid,))
                try:
                    os.unlink(tmp)
                except FileNotFoundError:
                    pass
                dropped += 1
                self.count(stale_reservations_dropped=1)
        return dropped

    def _try_reserve(self, db, obj, kind, tmp, need) -> str | None:
        """Inside a write transaction: reserve need bytes, evicting LRU unleased files. None on success."""
        self._purge_dead(db)
        used = db.execute("SELECT COALESCE(SUM(footprint), 0) FROM entries").fetchone()[0]
        inflight, reserved = db.execute("SELECT COUNT(*), COALESCE(SUM(footprint), 0) FROM reservations").fetchone()
        if inflight >= self.max_downloads:
            return f"{inflight} downloads in flight (storage.cache.max_downloads={self.max_downloads})"
        if kind == "prefetch":
            count, pbytes = db.execute("SELECT COUNT(*), COALESCE(SUM(footprint), 0) FROM reservations "
                                       "WHERE kind = 'prefetch'").fetchone()
            if count >= self.prefetch_max_downloads or pbytes + need > self.prefetch_max_bytes:
                return "prefetch limits reached"
        total = used + reserved + self._state_bytes()
        disk = os.statvfs(self.directory)
        free = disk.f_bavail * disk.f_frsize - reserved  # in-flight files will still grow
        shortfall = max(total + need - self.capacity, self.reserve + need - free)
        if shortfall > 0:
            freed = self._evict(db, shortfall)
            if freed < shortfall:
                return (f"capacity {self.capacity / 2**30:.2f} GiB, published {used / 2**30:.2f} GiB, in flight "
                        f"{reserved / 2**30:.2f} GiB ({inflight} downloads), file system free "
                        f"{(free + reserved) / 2**30:.2f} GiB with reserve {self.reserve / 2**30:.2f} GiB; "
                        f"could evict only {freed / 2**30:.2f} of the {shortfall / 2**30:.2f} GiB needed")
        db.execute("INSERT INTO reservations(key, footprint, kind, pid, start, tmp, created) VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (obj.key, need, kind, os.getpid(), _process_start(os.getpid()), tmp, time.time()))
        return None

    def _evict(self, db, need: int) -> int:
        """Inside a write transaction: unlink least-recently-used unleased files until need bytes are free."""
        freed, skipped = 0, 0
        while freed < need:
            rows = db.execute("SELECT key, path, size, footprint FROM entries ORDER BY last_access LIMIT 64 OFFSET ?",
                              (skipped,)).fetchall()
            if not rows:
                break
            for key, relative, size, blocks in rows:
                path = os.path.join(self.directory, relative)
                try:
                    fd = os.open(path, os.O_RDONLY)
                except FileNotFoundError:
                    db.execute("DELETE FROM entries WHERE key = ?", (key,))
                    freed += blocks
                    continue
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    skipped += 1  # leased by a reader
                    os.close(fd)
                    continue
                try:
                    os.unlink(path)
                finally:
                    os.close(fd)
                db.execute("DELETE FROM entries WHERE key = ?", (key,))
                freed += blocks
                self.count(evictions=1, evicted_bytes=size)
                if freed >= need:
                    break
        return freed

    # ---- recovery and maintenance ----

    def recover(self, *, full: bool = False) -> dict:
        """Drop reservations and partial files of dead processes; with full=True also reconcile published
        files with the database (rows without files, files without rows)."""
        report = {"partials_removed": 0, "rows_without_file": 0, "orphans_removed": 0}
        with self._transaction() as db:
            report["partials_removed"] = self._purge_dead(db)
            # A download registers its reservation before creating its file, so any other file is a leftover.
            live = {row[0] for row in db.execute("SELECT tmp FROM reservations")}
            tmp_dir = os.path.join(self.directory, "tmp")
            for name in os.listdir(tmp_dir):
                path = os.path.join(tmp_dir, name)
                if path not in live:
                    try:
                        os.unlink(path)
                        report["partials_removed"] += 1
                    except FileNotFoundError:
                        pass
            if full:
                known = {}
                for key, relative in db.execute("SELECT key, path FROM entries").fetchall():
                    if os.path.exists(os.path.join(self.directory, relative)):
                        known[relative] = key
                    else:
                        db.execute("DELETE FROM entries WHERE key = ?", (key,))
                        report["rows_without_file"] += 1
                objects = os.path.join(self.directory, "objects")
                for directory, _, files in os.walk(objects):
                    for name in files:
                        path = os.path.join(directory, name)
                        if self._relative(path) not in known:
                            os.unlink(path)
                            report["orphans_removed"] += 1
        return report

    def clear(self) -> int:
        """Evict every unleased file (for benchmarks and tests)."""
        with self._transaction() as db:
            return self._evict(db, 1 << 62)

    def validate_budget(self) -> dict:
        """Startup check: capacity + reserve must fit in what this cache already uses plus the free space."""
        with self._connection() as db:
            used = db.execute("SELECT COALESCE(SUM(footprint), 0) FROM entries").fetchone()[0]
        disk = os.statvfs(self.directory)
        free = disk.f_bavail * disk.f_frsize
        available = free + used
        if self.capacity + self.reserve > available:
            raise CacheError(
                f"storage.cache: capacity {self.capacity / 2**30:.1f} GiB + reserve {self.reserve / 2**30:.1f} GiB "
                f"exceeds the {available / 2**30:.1f} GiB this cache can use on {self.directory} "
                f"({free / 2**30:.1f} GiB free + {used / 2**30:.1f} GiB already cached). Lower capacity_gib "
                "or free space.")
        return {"free_bytes": free, "cached_bytes": used, "capacity_bytes": self.capacity,
                "reserve_bytes": self.reserve}

