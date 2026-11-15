"""Where raw videos come from: local files (the default) or remote roots read through the node video cache.

Vendored from NanoVVM (data/storage.py, commit ca80730) with imports adapted; keep the two copies in sync, so
that NanoVVM and V-RAE runs on one node can share one cache directory and its budget.

Without a storage section (or with backend: local) every path is a local file, exactly as before, and no
AWS code is imported. With backend: remote, a video path below one of storage.roots[*].local is served from
the remote source mapped to that prefix; every other path stays local. Configs keep naming videos by their
usual local paths (<data root>/<dataset>_video/...), so recipes do not change; manifests, weights and
checkpoints stay local. vrae.storage.config loads the section from a file (VRAE_STORAGE_CONFIG) and turns
relative `local` and `index` paths into absolute ones below the NanoVVM data root.

storage:
  backend: remote                             # local (default) | remote
  s3:                                         # used by s3:// URLs
    region: us-west-2
    credentials: instance_role                # instance_role | default_chain (vrae/storage/remote_io.py)
    endpoint_url: null
    connect_timeout_seconds: 10
    read_timeout_seconds: 60
    max_attempts: 6                           # per request, with exponential backoff
    backoff_seconds: 0.5
    max_backoff_seconds: 20
    use_version_id: false                     # also pin reads to the indexed VersionId (needs s3:GetObjectVersion)
  roots:
    - local: /abs/data/bridgedatav2_video      # path prefix the config uses for these videos
      archive: s3://bucket/assets/bridgedatav2_video.tar   # an uncompressed tar ...
      member_prefix: data/bridgedatav2_video   # ... whose member <member_prefix>/<x> is <local>/<x>
      index: /abs/data/remote_index/bridgedatav2_video.tar.index.sqlite
    - local: /abs/data/other_video
      objects: s3://bucket/prefix/other_video  # or one object per video: <objects>/<x> is <local>/<x>
  cache:
    dir: /mnt/localssd/<user>/nanovvm/video_cache
    capacity_gib: 300                         # one budget for the whole node (all ranks and workers)
    reserve_gib: 100                          # free space always left on the cache's file system
    max_downloads: 16                         # concurrent downloads on the node
    wait_timeout_seconds: 600                 # longest a reader waits for room or for another producer
  prefetch:
    enabled: true
    depth: 64                                 # upcoming samples per rank to look ahead
    workers: 4                                # download threads per rank (in the rank's main process)
    max_downloads: 8                          # node-wide prefetch downloads (< cache.max_downloads)
    max_reserved_gib: 16                      # node-wide bytes reserved by in-flight prefetches

See README_nanovvm_video_pred.md, "Remote Video Storage".
"""
from __future__ import annotations

import math
import os
import posixpath
import threading
from contextlib import contextmanager
from pathlib import Path

from vrae.storage.remote_io import (FileSink, FileTransport, RemoteError, RetryPolicy, S3Transport, StaleRemoteError,
                            join_url, parse_url)
from vrae.storage.tar_index import BLOCK, TarIndex, TarIndexError, check_member_header
from vrae.storage.video_cache import CachedObject, CacheError, VideoCache, cache_key

VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")
COMPRESSED_SUFFIXES = (".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar.zst", ".tar.lz4", ".zip")
GIB = 2 ** 30
STORAGE_KEYS = {"backend", "s3", "roots", "cache", "prefetch"}
S3_DEFAULTS = {"region": None, "credentials": "instance_role", "endpoint_url": None, "connect_timeout_seconds": 10,
               "read_timeout_seconds": 60, "max_attempts": 6, "backoff_seconds": 0.5, "max_backoff_seconds": 20,
               "use_version_id": False}
CACHE_DEFAULTS = {"dir": None, "capacity_gib": 300, "reserve_gib": 100, "max_downloads": 16,
                  "wait_timeout_seconds": 600}
PREFETCH_DEFAULTS = {"enabled": True, "depth": 64, "workers": 4, "max_downloads": 8, "max_reserved_gib": 16}


def is_storage_error(error: BaseException) -> bool:
    """Failures of the storage layer (never a reason to drop a clip from a population)."""
    return isinstance(error, (RemoteError, CacheError, TarIndexError))


def _section(value, defaults, where):
    value = {} if value is None else value
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown = set(value) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {where} keys {sorted(unknown)}; expected {sorted(defaults)}")
    return {**defaults, **value}


def _positive(value, where, *, integer=False, zero=False):
    ok = not isinstance(value, bool) and isinstance(value, int if integer else (int, float))
    ok = ok and math.isfinite(value) and (value >= 0 if zero else value > 0)
    if not ok:
        kind = "integer" if integer else "number"
        raise ValueError(f"{where} must be a {'non-negative' if zero else 'positive'} {kind}, got {value!r}")
    return value


def _local_prefix(value, where):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty path")
    path = os.path.expanduser(value)
    if not os.path.isabs(path):
        raise ValueError(f"{where} must be an absolute path, got {value!r}")
    return os.path.normpath(path)


def validate_storage(storage) -> dict | None:
    """Normalized storage section (defaults filled in), or None for the local backend."""
    if storage is None:
        return None
    if not isinstance(storage, dict):
        raise ValueError("storage must be a mapping")
    unknown = set(storage) - STORAGE_KEYS
    if unknown:
        raise ValueError(f"Unknown storage keys {sorted(unknown)}; expected {sorted(STORAGE_KEYS)}")
    backend = storage.get("backend", "local")
    if backend not in ("local", "remote"):
        raise ValueError("storage.backend must be local or remote")
    if backend == "local":
        return None  # any remote settings are ignored: this is the switch back to local files
    s3 = _section(storage.get("s3"), S3_DEFAULTS, "storage.s3")
    cache = _section(storage.get("cache"), CACHE_DEFAULTS, "storage.cache")
    prefetch = _section(storage.get("prefetch"), PREFETCH_DEFAULTS, "storage.prefetch")
    roots = storage.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("storage.roots must be a non-empty list when storage.backend is remote")
    normalized, seen, schemes = [], set(), set()
    for index, root in enumerate(roots):
        where = f"storage.roots[{index}]"
        if not isinstance(root, dict):
            raise ValueError(f"{where} must be a mapping")
        kinds = [key for key in ("archive", "objects") if root.get(key) is not None]
        if len(kinds) != 1:
            raise ValueError(f"{where} needs exactly one of archive (indexed uncompressed tar) or objects (one object "
                             "per video)")
        kind = kinds[0]
        allowed = {"local", "archive", "member_prefix", "index"} if kind == "archive" else {"local", "objects"}
        if set(root) - allowed:
            raise ValueError(f"{where}: unknown keys {sorted(set(root) - allowed)} for an {kind} root")
        local = _local_prefix(root.get("local"), f"{where}.local")
        if local in seen:
            raise ValueError(f"{where}: local prefix {local} is mapped twice")
        seen.add(local)
        url = root[kind]
        if not isinstance(url, str):
            raise ValueError(f"{where}.{kind} must be a URL string")
        schemes.add(parse_url(url)[0])
        entry = {"local": local, "kind": kind, "url": url}
        if kind == "archive":
            if url.lower().endswith(COMPRESSED_SUFFIXES):
                raise ValueError(f"{where}.archive {url} is compressed; ranged reads need an uncompressed tar "
                                 "(README_nanovvm_video_pred.md, 'Remote Video Storage')")
            prefix = root.get("member_prefix", "")
            if not isinstance(prefix, str) or prefix.startswith("/") or ".." in prefix.split("/"):
                raise ValueError(f"{where}.member_prefix must be a relative path inside the archive")
            entry["member_prefix"] = "/".join(p for p in prefix.split("/") if p not in ("", "."))
            entry["index"] = _local_prefix(root.get("index"), f"{where}.index")
        normalized.append(entry)
    if "s3" in schemes:
        if not isinstance(s3["region"], str) or not s3["region"]:
            raise ValueError("storage.s3.region is required for s3:// roots")
        if s3["credentials"] not in ("instance_role", "default_chain"):
            raise ValueError("storage.s3.credentials must be instance_role or default_chain")
        for key in ("connect_timeout_seconds", "read_timeout_seconds", "backoff_seconds", "max_backoff_seconds"):
            _positive(s3[key], f"storage.s3.{key}")
        _positive(s3["max_attempts"], "storage.s3.max_attempts", integer=True)
        if not isinstance(s3["use_version_id"], bool):
            raise ValueError("storage.s3.use_version_id must be true or false")
    cache["dir"] = _local_prefix(cache["dir"], "storage.cache.dir")
    for root in normalized:
        inside = (cache["dir"] + os.sep).startswith(root["local"] + os.sep)
        contains = (root["local"] + os.sep).startswith(cache["dir"] + os.sep)
        if inside or contains:
            raise ValueError(f"storage.cache.dir {cache['dir']} must not overlap the video root {root['local']}")
    _positive(cache["capacity_gib"], "storage.cache.capacity_gib")
    _positive(cache["reserve_gib"], "storage.cache.reserve_gib", zero=True)
    _positive(cache["max_downloads"], "storage.cache.max_downloads", integer=True)
    _positive(cache["wait_timeout_seconds"], "storage.cache.wait_timeout_seconds")
    if not isinstance(prefetch["enabled"], bool):
        raise ValueError("storage.prefetch.enabled must be true or false")
    for key in ("depth", "workers"):
        _positive(prefetch[key], f"storage.prefetch.{key}", integer=True)
    _positive(prefetch["max_downloads"], "storage.prefetch.max_downloads", integer=True, zero=True)
    _positive(prefetch["max_reserved_gib"], "storage.prefetch.max_reserved_gib")
    if prefetch["max_downloads"] >= cache["max_downloads"]:
        raise ValueError("storage.prefetch.max_downloads must be below storage.cache.max_downloads, so foreground "
                         "reads always have a download slot")
    if prefetch["max_reserved_gib"] >= cache["capacity_gib"] / 2:
        raise ValueError("storage.prefetch.max_reserved_gib must be below half of storage.cache.capacity_gib, so "
                         "prefetched videos are not evicted before they are read")
    return {"backend": "remote", "s3": s3, "roots": normalized, "cache": cache, "prefetch": prefetch}


class TarMemberSink:
    """Receives [header][payload] of one tar member: checks the header's CRC, stores only the payload."""

    def __init__(self, stream, member, where):
        self.stream, self.member, self.where = stream, member, where
        self.header = b""

    def reset(self):
        self.stream.seek(0)
        self.stream.truncate()
        self.header = b""

    def write(self, data):
        if len(self.header) < BLOCK:
            need = BLOCK - len(self.header)
            self.header += data[:need]
            data = data[need:]
            if len(self.header) == BLOCK:
                check_member_header(self.header, self.member, self.where)
        if data:
            self.stream.write(data)


class StorageBackend:
    """Resolves video paths and reads remote ones through the node cache. Picklable (DataLoader workers);
    transports, indexes and SQLite handles are created lazily in each process."""

    def __init__(self, config=None, *, normalized=False):
        """config: a storage section (validated here), or validate_storage's output with normalized=True."""
        self.config = config if normalized else validate_storage(config)
        self.remote = self.config is not None
        roots = self.config["roots"] if self.remote else []
        self.roots = sorted(({**root, "aliases": sorted({root["local"], os.path.realpath(root["local"])},
                                                        key=len, reverse=True)} for root in roots),
                            key=lambda root: len(root["local"]), reverse=True)
        self._resources_by_pid = {}

    def __getstate__(self):
        return {"config": self.config}

    def __setstate__(self, state):
        self.__init__(state["config"], normalized=True)

    def _resources(self) -> dict:
        resources = self._resources_by_pid.get(os.getpid())
        if resources is None:
            # A fresh dict (and lock) per process: never reuse a lock a parent thread might have held at fork.
            resources = self._resources_by_pid.setdefault(os.getpid(), {
                "lock": threading.Lock(), "transports": {}, "indexes": {}, "objects": {}, "cache": None})
        return resources

    # ---- resolution ----

    def root_for(self, path):
        """(root, relative posix path) when path lies below a remote root, else None."""
        if not self.remote:
            return None
        text = os.path.normpath(os.path.abspath(os.path.expanduser(str(path))))
        for root in self.roots:
            for prefix in root["aliases"]:
                if text == prefix:
                    return root, ""
                if text.startswith(prefix + os.sep):
                    return root, text[len(prefix) + 1:].replace(os.sep, "/")
        return None

    def transport(self, url):
        scheme = parse_url(url)[0]
        resources = self._resources()
        with resources["lock"]:
            transport = resources["transports"].get(scheme)
            if transport is None:
                s3 = self.config["s3"]
                retry = RetryPolicy(s3["max_attempts"], s3["backoff_seconds"], s3["max_backoff_seconds"])
                if scheme == "s3":
                    transport = S3Transport(region=s3["region"], credentials=s3["credentials"],
                                            endpoint_url=s3["endpoint_url"],
                                            connect_timeout=s3["connect_timeout_seconds"],
                                            read_timeout=s3["read_timeout_seconds"],
                                            max_pool_connections=max(10, self.config["cache"]["max_downloads"] * 2),
                                            use_version_id=s3["use_version_id"], retry=retry)
                else:
                    transport = FileTransport(retry)
                resources["transports"][scheme] = transport
        return transport

    def index(self, root) -> TarIndex:
        resources = self._resources()
        with resources["lock"]:
            index = resources["indexes"].get(root["index"])
            if index is None:
                index = resources["indexes"][root["index"]] = TarIndex(root["index"])
                identity = index.remote_identity()
                if identity["url"] != root["url"]:
                    raise TarIndexError(f"{root['index']} indexes {identity['url']}, but storage maps "
                                        f"{root['local']} to {root['url']}")
        return index

    def _object_info(self, url):
        resources = self._resources()
        info = resources["objects"].get(url)
        if info is None:
            info = resources["objects"][url] = self.transport(url).head(url)
        return info

    def locate(self, path) -> CachedObject | None:
        """The cache object for a remote-backed path; None for a local path."""
        found = self.root_for(path)
        if found is None:
            return None
        root, relative = found
        if not relative:
            raise TarIndexError(f"{path} is the root of a remote source, not a video")
        suffix = posixpath.splitext(relative)[1].lower()
        if root["kind"] == "archive":
            index = self.index(root)
            member_name = f"{root['member_prefix']}/{relative}" if root["member_prefix"] else relative
            member = index.lookup(member_name)
            identity = index.remote_identity()
            key = cache_key({"v": 1, "kind": "tar", **identity, "offset": member.offset, "size": member.size,
                             "header_crc": member.header_crc})
            meta = {"kind": "tar", "url": root["url"], "start": member.offset - BLOCK, "length": member.size + BLOCK,
                    "version_id": identity["version_id"], "etag": identity["etag"], "member": member,
                    "archive_size": identity["size"]}
            description = f"{member_name} in {root['url']}"
        else:
            url = join_url(root["url"], relative)
            info = self._object_info(url)
            key = cache_key({"v": 1, "kind": "object", **info.identity()})
            meta = {"kind": "object", "url": url, "start": 0, "length": info.size, "version_id": info.version_id,
                    "etag": info.etag}
            description = url
            member = None
        size = member.size if member is not None else meta["length"]
        if size <= 0:
            raise RemoteError(f"{description} is empty")
        return CachedObject(key=key, size=size, suffix=suffix, description=description, meta=meta)

    # ---- local-equivalent queries ----

    def is_dir(self, path) -> bool:
        found = self.root_for(path)
        if found is None:
            return Path(path).expanduser().is_dir()
        root, relative = found
        if root["kind"] == "objects":
            return True
        member = "/".join(p for p in (root["member_prefix"], relative) if p)
        return self.index(root).is_dir(member) if member else True

    def is_file(self, path) -> bool:
        if self.root_for(path) is None:
            return Path(path).expanduser().is_file()
        try:
            self.locate(path)
            return True
        except (TarIndexError, RemoteError):
            return False

    def list_videos(self, root_path) -> list:
        """Sorted video files below root_path (local or remote), as local-equivalent Paths."""
        found = self.root_for(root_path)
        if found is None:
            root = Path(root_path).expanduser().resolve()
            return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
        root, relative = found
        base = Path(os.path.normpath(os.path.abspath(os.path.expanduser(str(root_path)))))
        if root["kind"] == "archive":
            prefix = "/".join(p for p in (root["member_prefix"], relative) if p)
            names = self.index(root).names(prefix)
            cut = len(prefix) + 1 if prefix else 0
            paths = [base / name[cut:] for name in names]
        else:
            listing = join_url(root["url"], relative) if relative else root["url"]
            cut = len(listing.rstrip("/")) + 1
            paths = [base / url[cut:] for url, _, _ in self.transport(listing).list(listing)]
        return sorted(p for p in paths if p.suffix.lower() in VIDEO_EXTENSIONS)

    def identity(self, path) -> dict | None:
        """Remote identity of a remote-backed path (FVD cache keys); None for local files."""
        video = self.locate(path)
        if video is None:
            return None
        meta = video.meta
        return {"path": os.path.normpath(os.path.abspath(os.path.expanduser(str(path)))), "size": video.size,
                "remote": {"url": meta["url"], "etag": meta["etag"], "version_id": meta["version_id"],
                           "offset": meta["start"] + (BLOCK if meta["kind"] == "tar" else 0)}}

    def check_videos(self, root_path, relative_paths, where="") -> int:
        """Preflight: every listed video below root_path resolves (index lookups only; objects: the first
        one, by HEAD). Returns how many were checked."""
        found = self.root_for(Path(root_path) / "x")
        if found is None:
            return 0
        root = found[0]
        paths = list(relative_paths) if root["kind"] == "archive" else list(relative_paths)[:1]
        failures = []
        for relative in paths:
            try:
                self.locate(Path(root_path) / relative)
            except (TarIndexError, RemoteError) as error:
                failures.append(str(error))
        if failures:
            raise TarIndexError(f"{where}{len(failures)} of {len(paths)} manifest videos cannot be served from "
                                f"{root['url']}; first: {failures[0]}")
        return len(paths)

    # ---- reading ----

    @property
    def cache(self) -> VideoCache:
        resources = self._resources()
        with resources["lock"]:
            if resources["cache"] is None:
                cache, prefetch = self.config["cache"], self.config["prefetch"]
                resources["cache"] = VideoCache(
                    cache["dir"], int(cache["capacity_gib"] * GIB), reserve_bytes=int(cache["reserve_gib"] * GIB),
                    max_downloads=cache["max_downloads"], prefetch_max_downloads=prefetch["max_downloads"],
                    prefetch_max_bytes=int(prefetch["max_reserved_gib"] * GIB),
                    wait_timeout=cache["wait_timeout_seconds"])
        return resources["cache"]

    def _fetcher(self, video: CachedObject):
        meta = video.meta
        transport = self.transport(meta["url"])

        def fetch(stream):
            if meta["kind"] == "tar":
                sink = TarMemberSink(stream, meta["member"], meta["url"])
            else:
                sink = FileSink(stream)
            transport.read_range(meta["url"], meta["start"], meta["length"], sink, version_id=meta["version_id"],
                                 if_match=meta["etag"])
            counters = transport.take_counters()
            return {"requests": counters["requests"], "retries": counters["retries"], "bytes": counters["bytes"]}

        return fetch

    def lease(self, video: CachedObject):
        return self.cache.lease(video, self._fetcher(video))

    def prefetch_video(self, video: CachedObject) -> str:
        return self.cache.prefetch(video, self._fetcher(video))

    @contextmanager
    def open(self, path):
        """Yield a local path to decode: the file itself, or a leased cache copy of a remote video."""
        video = self.locate(path)
        if video is None:
            yield str(path)
            return
        lease = self.lease(video)
        try:
            yield lease.path
        finally:
            lease.close()

    def prefetcher(self, video_for_item, key_for_item):
        """A Prefetcher for this process, or None (local backend or prefetch disabled)."""
        if not self.remote or not self.config["prefetch"]["enabled"]:
            return None
        from vrae.storage.prefetch import Prefetcher
        prefetch = self.config["prefetch"]
        return Prefetcher(video_for_item, key_for_item, self.prefetch_video, depth=prefetch["depth"],
                          workers=prefetch["workers"])

    @contextmanager
    def prefetch_along(self, paths):
        """For sequential readers (FVD): yields advance(position) that prefetches paths[position + 1 ...]."""
        paths = [str(p) for p in paths]
        prefetcher = self.prefetcher(lambda i: self.locate(paths[i]), lambda i: paths[i])
        if prefetcher is None:
            yield lambda position: None
            return
        prefetcher.start(list(range(len(paths))))
        try:
            prefetcher.advance(-1)
            yield prefetcher.advance
        finally:
            prefetcher.close()

    # ---- startup and monitoring ----

    def prepare(self, *, owner: bool) -> dict:
        """Before training: validate the disk budget; the node's owner process (local rank 0) also reconciles
        the cache with its database, dropping leftovers of interrupted jobs."""
        if not self.remote:
            return {}
        cache = self.cache
        report = cache.validate_budget()
        report["recovered"] = cache.recover(full=owner)
        return report

    def check_remote(self, paths=None) -> list:
        """Credentials and archive identity of the roots serving `paths` (default: every root): one HEAD per
        root; an index whose archive changed is stale. Roots no path uses are not touched."""
        if not self.remote:
            return []
        roots = self.roots
        if paths is not None:
            used = {id(found[0]) for found in (self.root_for(Path(p) / "x") for p in paths) if found}
            roots = [root for root in self.roots if id(root) in used]
        lines = []
        for root in roots:
            transport = self.transport(root["url"])
            if root["kind"] == "archive":
                index = self.index(root)
                expected = index.remote_identity()
                info = transport.head(root["url"])
                changed = [key for key in ("size", "etag", "version_id")
                           if expected.get(key) is not None and info.identity().get(key) != expected[key]]
                if changed:
                    raise StaleRemoteError(
                        f"{root['url']} changed since {root['index']} was built ({', '.join(changed)}: index "
                        f"{[expected[k] for k in changed]}, now {[info.identity()[k] for k in changed]}); "
                        "rebuild the index with python -m vrae.storage.remote_video index")
                lines.append(f"{root['local']} -> {root['url']} ({info.size} bytes, ETag {info.etag}, version "
                             f"{info.version_id}), index {root['index']} ({index.summary.get('indexed')} members)")
            else:
                lines.append(f"{root['local']} -> {root['url']} (objects)")
        if any(parse_url(root["url"])[0] == "s3" for root in roots):
            method = self.transport("s3://x/y").check_credentials()
            lines.append(f"S3 credentials: {self.config['s3']['credentials']} (provider: {method})")
        return lines

    def stats(self) -> dict:
        return self.cache.stats() if self.remote else {}


_ACTIVE = StorageBackend(None)


def configure_storage(config) -> StorageBackend:
    """Set (and return) the process-wide backend used by open_video, FVD and sampling."""
    global _ACTIVE
    _ACTIVE = config if isinstance(config, StorageBackend) else StorageBackend(config)
    return _ACTIVE


def active_storage() -> StorageBackend:
    return _ACTIVE


def open_video(path):
    return _ACTIVE.open(path)
