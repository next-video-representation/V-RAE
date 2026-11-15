"""Index of an uncompressed tar archive: member name -> payload offset and length, built from headers only.

The index is a small SQLite file. It is built from one of
  * a local seekable copy of the archive (fast; reads only the headers),
  * a tar stream (stdin), e.g. while the archive is created: `tar -cf - x | tee x.tar | ... index --from-stdin`,
  * the remote archive itself, by ranged GETs that skip every payload larger than the read window.
A remote build costs about one request per member larger than the window, plus the bytes of each window
(README_nanovvm_video_pred.md, "Remote Video Storage"); it never downloads payloads it can skip. Builds commit progress every few
thousand members, so an interrupted build resumes where it stopped (for the same archive identity).

The index pins the archive identity (URL, size, ETag, VersionId) and stores a CRC-32 of every member's
512-byte header. A fetch of a member reads its header together with its payload and checks that CRC, so a
stale or misapplied index can never deliver another member's bytes. ETags are compared as opaque strings
(multipart ETags are not MD5 checksums).

Supported: ustar / POSIX (prefix field), GNU (long names 'L', long link names 'K', base-256 sizes),
PAX extended ('x') and global ('g') headers (path, linkpath, size). Symlinks and hard links are resolved
inside the archive; absolute targets, targets escaping the archive root and loops are rejected. Member names
that are absolute or contain '..' are recorded as rejected and can never be looked up. A name that appears
twice for a non-directory is ambiguous and is rejected on lookup. Sparse files, devices and FIFOs are
unsupported. Compressed archives (gzip, bzip2, xz, zstd, lz4, zip) are detected and refused: plain tar
offsets do not exist in them.
"""
from __future__ import annotations

import fnmatch
import json
import os
import sqlite3
import threading
import time
import zlib
from dataclasses import dataclass
from functools import lru_cache

from vrae.storage import sqlite_fork

INDEX_FORMAT = 1
BLOCK = 512
REGULAR, HARDLINK, SYMLINK, DIRECTORY = "0", "1", "2", "5"
DATA_TYPES = {"0", "\0", "7"}  # regular, old-style regular, contiguous
NO_PAYLOAD_TYPES = {"1", "2", "3", "4", "5", "6"}
COMPRESSION_MAGIC = [
    (b"\x1f\x8b", "gzip"), (b"BZh", "bzip2"), (b"\xfd7zXZ\x00", "xz"), (b"\x28\xb5\x2f\xfd", "zstd"),
    (b"\x04\x22\x4d\x18", "lz4"), (b"PK\x03\x04", "zip"),
]
MAX_LINK_HOPS = 40


class TarIndexError(ValueError):
    pass


@dataclass(frozen=True)
class Member:
    """A regular file's payload inside the archive (links already resolved)."""
    name: str
    offset: int
    size: int
    header_crc: int


def detect_compression(head: bytes) -> str | None:
    for magic, name in COMPRESSION_MAGIC:
        if head.startswith(magic):
            return name
    return None


def compression_error(source: str, kind: str) -> TarIndexError:
    return TarIndexError(
        f"{source} is a {kind}-compressed archive: members of a compressed stream have no byte offsets, so it "
        "cannot be read by ranges. Prepare an uncompressed copy once (e.g. `zcat a.tar.gz > a.tar`, or "
        "re-create it with `tar -cf`) where the license allows a copy, upload it under a new key, and index "
        "that. Existing assets are never repacked automatically.")


def _number(field: bytes) -> int:
    if field and field[0] & 0x80:  # GNU base-256
        value = int.from_bytes(field, "big", signed=False) & ~(0x80 << (8 * (len(field) - 1)))
        return value
    text = field.split(b"\0", 1)[0].strip(b" \0")
    return int(text, 8) if text else 0


def _text(field: bytes) -> str:
    return field.split(b"\0", 1)[0].decode("utf-8", "surrogateescape")


def parse_header(block: bytes, where: str = "") -> dict | None:
    """Fields of one 512-byte header, or None for an all-zero block."""
    if len(block) != BLOCK:
        raise TarIndexError(f"truncated tar header{where}")
    if block == b"\0" * BLOCK:
        return None
    stored = _number(block[148:156])
    unsigned = sum(block[:148]) + 256 + sum(block[156:])
    signed = sum((b if b < 128 else b - 256) for b in block[:148]) + 256 + sum(
        (b if b < 128 else b - 256) for b in block[156:])
    if stored not in (unsigned, signed):
        raise TarIndexError(f"bad tar header checksum{where}: not an (uncompressed) tar archive, or the offset "
                            "is wrong")
    magic = block[257:265]
    name = _text(block[0:100])
    if magic[:6] == b"ustar\0":  # POSIX ustar: prefix field holds leading path components
        prefix = _text(block[345:500])
        if prefix:
            name = prefix + "/" + name
    return {"name": name, "size": _number(block[124:136]), "type": block[156:157].decode("latin-1") or "\0",
            "linkname": _text(block[157:257]), "crc": zlib.crc32(block)}


def _parse_pax(data: bytes) -> dict:
    values, position = {}, 0
    while position < len(data):
        space = data.find(b" ", position)
        if space < 0:
            break
        length = int(data[position:space])
        if length <= 0:
            raise TarIndexError("malformed PAX record")
        record = data[space + 1:position + length - 1]
        key, _, value = record.partition(b"=")
        values[key.decode("utf-8", "surrogateescape")] = value.decode("utf-8", "surrogateescape")
        position += length
    return values


def normalize_member(name: str) -> tuple[str | None, str | None]:
    """(normalized name, None) or (None, rejection reason)."""
    if name.startswith("/"):
        return None, "absolute path"
    parts = [part for part in name.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return None, "path traversal ('..')"
    if not parts:
        return None, "empty name"
    return "/".join(parts), None


def _pad(size: int) -> int:
    return (size + BLOCK - 1) // BLOCK * BLOCK


# ---- readers ----

class FileReader:
    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        self.size = os.fstat(self.fd).st_size
        self.requests = self.bytes = 0

    def read(self, offset, length):
        self.requests += 1
        data = os.pread(self.fd, length, offset)
        self.bytes += len(data)
        return data

    def close(self):
        os.close(self.fd)


class StreamReader:
    """Sequential stream: seeking forward reads and discards (only for streams already flowing locally)."""

    def __init__(self, stream):
        self.stream, self.position, self.size = stream, 0, None
        self.last_offset, self.last = 0, b""
        self.requests = self.bytes = 0

    def read(self, offset, length):
        if self.last_offset <= offset and offset + length <= self.last_offset + len(self.last):
            return self.last[offset - self.last_offset:offset - self.last_offset + length]
        if offset < self.position:
            raise TarIndexError("a tar stream cannot seek backwards")
        while self.position < offset:
            skipped = self.stream.read(min(1 << 20, offset - self.position))
            if not skipped:
                return b""
            self.position += len(skipped)
            self.bytes += len(skipped)
        data = b""
        while len(data) < length:
            chunk = self.stream.read(length - len(data))
            if not chunk:
                break
            data += chunk
        self.position += len(data)
        self.bytes += len(data)
        self.last_offset, self.last = offset, data
        return data

    def close(self):
        pass


class RangeReader:
    """Remote archive read by ranged GETs with an adaptive window: it widens while consecutive headers are
    small (directories, links, short files) and shrinks back after skipping a large payload."""

    def __init__(self, transport, info, min_window=16 << 10, max_window=1 << 20):
        self.transport, self.info, self.size = transport, info, info.size
        self.min_window, self.max_window, self.window = min_window, max_window, min_window
        self.buffer_start, self.buffer = 0, b""
        self.requests = self.bytes = 0

    def read(self, offset, length):
        end = offset + length
        if self.buffer_start <= offset and end <= self.buffer_start + len(self.buffer):
            return self.buffer[offset - self.buffer_start:end - self.buffer_start]
        contiguous = offset == self.buffer_start + len(self.buffer) and self.buffer
        self.window = min(self.max_window, self.window * 2) if contiguous else self.min_window
        fetch = min(max(length, self.window), self.size - offset)
        if fetch <= 0:
            return b""
        data = self.transport.read_bytes(self.info.url, offset, fetch, version_id=self.info.version_id,
                                         if_match=self.info.etag)
        self.requests += 1
        self.bytes += len(data)
        self.buffer_start, self.buffer = offset, data
        return data[:length]

    def close(self):
        pass


# ---- building ----

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS members(name TEXT PRIMARY KEY, type TEXT NOT NULL, offset INTEGER NOT NULL,
    size INTEGER NOT NULL, linkname TEXT, header_crc INTEGER NOT NULL, ambiguous INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS rejected(name TEXT NOT NULL, reason TEXT NOT NULL, header_offset INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS progress(id INTEGER PRIMARY KEY CHECK (id = 0), next_offset INTEGER NOT NULL,
    state TEXT NOT NULL);
"""


def _set_meta(db, key, value):
    db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, json.dumps(value)))


def _get_meta(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def build_index(reader, output, source: dict, *, built_from: str, restart=False, commit_every=5000,
                log=print) -> dict:
    """Walk the archive's headers into an index at output (via output + '.partial', resumable)."""
    partial = str(output) + ".partial"
    if restart and os.path.exists(partial):
        os.unlink(partial)
    db = sqlite3.connect(partial, isolation_level=None)
    try:
        db.executescript(SCHEMA)
        previous = _get_meta(db, "source")
        if previous is not None and previous != source:
            raise TarIndexError(f"{partial} was started for {previous}, not {source}; pass --restart")
        db.execute("BEGIN")
        _set_meta(db, "source", source)
        _set_meta(db, "format", INDEX_FORMAT)
        _set_meta(db, "built_from", built_from)
        row = db.execute("SELECT next_offset, state FROM progress WHERE id = 0").fetchone()
        offset, state = (row[0], json.loads(row[1])) if row else (0, {"global_pax": {}, "members": 0,
                                                                       "requests": 0, "bytes": 0})
        db.execute("COMMIT")
        if row:
            log(f"resuming at byte {offset} after {state['members']} members")
        else:
            head = reader.read(0, BLOCK)  # a full block, so a stream can serve the first header from it
            kind = detect_compression(head)
            if kind:
                raise compression_error(source.get("url") or source.get("path"), kind)
        base_requests, base_bytes = state["requests"], state["bytes"]
        pending = {}
        started = last_log = time.monotonic()
        db.execute("BEGIN")
        since_commit, end_offset = 0, None
        while True:
            block = reader.read(offset, BLOCK)
            if len(block) < BLOCK:
                if not block and reader.size is None:
                    raise TarIndexError(f"tar stream ended at byte {offset} without an end-of-archive marker")
                raise TarIndexError(f"archive truncated at byte {offset}")
            header = parse_header(block, f" at byte {offset}")
            if header is None:
                end_offset = offset
                break
            data_offset = offset + BLOCK
            size = header["size"]
            kind = header["type"]
            if kind in ("x", "g", "L", "K"):
                payload = reader.read(data_offset, size)
                if len(payload) < size:
                    raise TarIndexError(f"archive truncated inside the extended header at byte {offset}")
                if kind == "x":
                    pending["pax"] = _parse_pax(payload)
                elif kind == "g":
                    state["global_pax"].update(_parse_pax(payload))
                elif kind == "L":
                    pending["longname"] = payload.split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
                else:
                    pending["longlink"] = payload.split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
                offset = data_offset + _pad(size)
                continue
            pax = {**state["global_pax"], **pending.get("pax", {})}
            name = pax.get("path") or pending.get("longname") or header["name"]
            linkname = pax.get("linkpath") or pending.get("longlink") or header["linkname"]
            if "size" in pax:
                size = int(pax["size"])
            sparse = kind == "S" or any(key.startswith("GNU.sparse") for key in pax)
            pending = {}
            normalized, reason = normalize_member(name)
            if sparse:
                reason = reason or "sparse file (unsupported)"
            if reason:
                db.execute("INSERT INTO rejected(name, reason, header_offset) VALUES (?, ?, ?)", (name, reason, offset))
            else:
                stored_type = REGULAR if kind in DATA_TYPES else kind
                existing = db.execute("SELECT type FROM members WHERE name = ?", (normalized,)).fetchone()
                if existing is None:
                    db.execute("INSERT INTO members(name, type, offset, size, linkname, header_crc) "
                               "VALUES (?, ?, ?, ?, ?, ?)",
                               (normalized, stored_type, data_offset, size, linkname or None, header["crc"]))
                elif not (existing[0] == DIRECTORY and stored_type == DIRECTORY):
                    db.execute("UPDATE members SET ambiguous = 1 WHERE name = ?", (normalized,))
            state["members"] += 1
            since_commit += 1
            # Like tarfile: links, directories, devices and FIFOs carry no payload even if size is set.
            offset = data_offset + (_pad(size) if kind in DATA_TYPES or kind not in NO_PAYLOAD_TYPES else 0)
            if since_commit >= commit_every:
                state["requests"], state["bytes"] = base_requests + reader.requests, base_bytes + reader.bytes
                db.execute("INSERT OR REPLACE INTO progress(id, next_offset, state) VALUES (0, ?, ?)",
                           (offset, json.dumps(state)))
                db.execute("COMMIT")
                db.execute("BEGIN")
                since_commit = 0
                now = time.monotonic()
                if now - last_log >= 30:
                    total = f" of {reader.size}" if reader.size else ""
                    log(f"{state['members']} members, byte {offset}{total}, {state['requests']} requests, "
                        f"{state['bytes'] / 2**20:.1f} MiB read, {now - started:.0f}s")
                    last_log = now
        state["requests"], state["bytes"] = base_requests + reader.requests, base_bytes + reader.bytes
        summary = {
            "members": state["members"],
            "indexed": db.execute("SELECT COUNT(*) FROM members").fetchone()[0],
            "rejected": db.execute("SELECT COUNT(*) FROM rejected").fetchone()[0],
            "ambiguous": db.execute("SELECT COUNT(*) FROM members WHERE ambiguous = 1").fetchone()[0],
            "end_offset": end_offset, "header_requests": state["requests"], "header_bytes_read": state["bytes"],
            "seconds_this_run": round(time.monotonic() - started, 1),
        }
        _set_meta(db, "summary", summary)
        _set_meta(db, "complete", True)
        _set_meta(db, "created", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        db.execute("DELETE FROM progress")
        db.execute("COMMIT")
    finally:
        db.close()
    os.replace(partial, output)
    return summary


def build_from_local_tar(path, output, *, archive_url=None, restart=False, log=print, **kwargs) -> dict:
    stat = os.stat(path)
    source = {"kind": "local_tar", "path": os.path.abspath(path), "size": stat.st_size,
              "mtime_ns": stat.st_mtime_ns, "url": archive_url}
    reader = FileReader(path)
    try:
        return build_index(reader, output, source, built_from="local_tar", restart=restart, log=log, **kwargs)
    finally:
        reader.close()


def build_from_stream(stream, output, *, archive_url=None, log=print, **kwargs) -> dict:
    source = {"kind": "stream", "url": archive_url}
    return build_index(StreamReader(stream), output, source, built_from="stream", restart=True, log=log, **kwargs)


def build_from_remote(transport, url, output, *, restart=False, log=print, min_window=16 << 10,
                      max_window=1 << 20, **kwargs) -> dict:
    info = transport.head(url)
    source = {"kind": "remote", **info.identity()}
    reader = RangeReader(transport, info, min_window=min_window, max_window=max_window)
    return build_index(reader, output, source, built_from="remote_range", restart=restart, log=log, **kwargs)


def bind_remote(index_path, transport, url, *, samples=16, log=print) -> dict:
    """Pin an index built from a local copy or a stream to the remote archive: same size, and byte-identical
    headers at the first, last and `samples` evenly spaced members (a few small ranged GETs)."""
    info = transport.head(url)
    db = sqlite3.connect(index_path, isolation_level=None)
    try:
        source = _get_meta(db, "source")
        if not _get_meta(db, "complete"):
            raise TarIndexError(f"{index_path} is incomplete")
        if source.get("size") is not None and source["size"] != info.size:
            raise TarIndexError(f"{url} has {info.size} bytes, the indexed archive {source['size']}: not the same archive")
        end = (_get_meta(db, "summary") or {}).get("end_offset")
        if end is not None and end + 2 * BLOCK > info.size:
            raise TarIndexError(f"{url} has {info.size} bytes, but the indexed archive ends at byte {end}: "
                                "not the same archive (or the upload is incomplete)")
        rows = db.execute("SELECT offset, header_crc FROM members ORDER BY offset").fetchall()
        if not rows:
            raise TarIndexError(f"{index_path} has no members")
        step = max(1, len(rows) // max(1, samples))
        picks = sorted({0, len(rows) - 1, *range(0, len(rows), step)})
        for position in picks:
            offset, crc = rows[position]
            header = transport.read_bytes(url, offset - BLOCK, BLOCK, version_id=info.version_id, if_match=info.etag)
            if zlib.crc32(header) != crc:
                raise TarIndexError(f"{url}: the header at byte {offset - BLOCK} differs from the index; "
                                    "the local copy is not this archive")
        bound = {"kind": "remote", **info.identity(), "bound_from": source}
        db.execute("BEGIN")
        _set_meta(db, "source", bound)
        _set_meta(db, "bind_checked_headers", len(picks))
        db.execute("COMMIT")
        log(f"bound {index_path} to {url} (size {info.size}, ETag {info.etag}, version {info.version_id}); "
            f"{len(picks)} headers compared")
        return bound
    finally:
        db.close()


# ---- reading ----

class TarIndex:
    """Read-only view of a complete index; one SQLite connection per process (opened after fork)."""

    def __init__(self, path):
        self.path = os.path.abspath(os.path.expanduser(str(path)))
        if not os.path.isfile(self.path):
            raise TarIndexError(f"tar index not found: {self.path}. Build it with "
                                "python -m vrae.storage.remote_video index (README_nanovvm_video_pred.md, 'Remote Video Storage')")
        self._local = threading.local()
        self.format = self._query(lambda db: _get_meta(db, "format"))
        self.source = self._query(lambda db: _get_meta(db, "source"))
        self.summary = self._query(lambda db: _get_meta(db, "summary", {}))
        complete = self._query(lambda db: _get_meta(db, "complete"))
        if self.format != INDEX_FORMAT:
            raise TarIndexError(f"{self.path}: index format {self.format!r}, this code reads {INDEX_FORMAT}")
        if not complete:
            raise TarIndexError(f"{self.path}: index is incomplete; finish building it")
        self._lookup = lru_cache(maxsize=1 << 16)(self._resolve)

    def __getstate__(self):
        return {"path": self.path}

    def __setstate__(self, state):
        self.__init__(state["path"])

    def _open(self):
        return sqlite_fork.connect(f"file:{self.path}?mode=ro&immutable=1", uri=True)

    def _query(self, run):
        """run(connection) under the fork guard (data/sqlite_fork.py), on this thread's connection."""
        with sqlite_fork.GUARD:
            return run(sqlite_fork.thread_connection(self._local, self._open))

    def _conn(self):
        """For tools and tests only (no fork guard)."""
        with sqlite_fork.GUARD:
            return sqlite_fork.thread_connection(self._local, self._open)

    def remote_identity(self) -> dict:
        if not self.source or self.source.get("kind") != "remote":
            raise TarIndexError(f"{self.path} is not bound to a remote archive; run "
                                "python -m vrae.storage.remote_video index ... --bind <url>")
        return {key: self.source.get(key) for key in ("url", "size", "etag", "version_id")}

    def _row(self, name):
        return self._query(lambda db: db.execute(
            "SELECT type, offset, size, linkname, header_crc, ambiguous FROM members WHERE name = ?", (name,)
        ).fetchone())

    def lookup(self, name: str) -> Member:
        normalized, reason = normalize_member(name)
        if reason:
            raise TarIndexError(f"{name!r}: {reason}")
        return self._lookup(normalized)

    def _resolve(self, name: str) -> Member:
        parts, resolved, position, hops = name.split("/"), [], 0, 0
        while position < len(parts):
            candidate = "/".join(resolved + [parts[position]])
            last = position == len(parts) - 1
            row = self._row(candidate)
            if row is None:
                if not last and self._has_prefix(candidate + "/"):
                    resolved.append(parts[position])  # directory without its own header
                    position += 1
                    continue
                rejected = self._query(lambda db: db.execute(
                    "SELECT reason FROM rejected WHERE name = ? LIMIT 1", (candidate,)).fetchone())
                why = f" (rejected while indexing: {rejected[0]})" if rejected else ""
                raise TarIndexError(f"{name!r}: {candidate!r} is not in the archive{why}")
            kind, offset, size, linkname, crc, ambiguous = row
            if ambiguous:
                raise TarIndexError(f"{name!r}: the archive holds {candidate!r} more than once; it is ambiguous "
                                    "which copy a tar extraction keeps, so it is not served")
            if kind == SYMLINK or kind == HARDLINK:
                hops += 1
                if hops > MAX_LINK_HOPS:
                    raise TarIndexError(f"{name!r}: more than {MAX_LINK_HOPS} links (a loop?)")
                if not linkname or linkname.startswith("/"):
                    raise TarIndexError(f"{name!r}: link {candidate!r} -> {linkname!r} points outside the archive")
                if kind == HARDLINK:
                    if not last:
                        raise TarIndexError(f"{name!r}: hard link {candidate!r} used as a directory")
                    target = linkname.split("/")
                else:
                    target = resolved + linkname.split("/")
                stack = []
                for part in target:
                    if part in ("", "."):
                        continue
                    if part == "..":
                        if not stack:
                            raise TarIndexError(f"{name!r}: link {candidate!r} -> {linkname!r} escapes the archive root")
                        stack.pop()
                    else:
                        stack.append(part)
                parts, resolved, position = stack + parts[position + 1:], [], 0
                continue
            if kind == DIRECTORY:
                if last:
                    raise TarIndexError(f"{name!r} is a directory")
                resolved.append(parts[position])
                position += 1
                continue
            if kind != REGULAR:
                raise TarIndexError(f"{name!r}: member type {kind!r} is not supported (only regular files and links)")
            if not last:
                raise TarIndexError(f"{name!r}: {candidate!r} is a file, not a directory")
            return Member(candidate, offset, size, crc)
        raise TarIndexError(f"{name!r}: empty path")

    def _has_prefix(self, prefix):
        upper = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        return self._query(lambda db: db.execute("SELECT 1 FROM members WHERE name >= ? AND name < ? LIMIT 1",
                                                 (prefix, upper)).fetchone()) is not None

    def is_dir(self, name: str) -> bool:
        normalized, reason = normalize_member(name)
        if reason:
            return False
        row = self._row(normalized)
        if row is not None:
            if row[0] == DIRECTORY:
                return True
            if row[0] == SYMLINK:
                try:
                    self.lookup(normalized)
                    return False
                except TarIndexError:
                    return False
        return self._has_prefix(normalized + "/")

    def names(self, prefix: str = "", pattern: str | None = None):
        """Member names (files and links, not directories) below prefix, optionally fnmatch-filtered."""
        prefix = prefix.strip("/")
        low = prefix + "/" if prefix else ""
        high = low[:-1] + chr(ord(low[-1]) + 1) if low else "\U0010ffff"
        rows = self._query(lambda db: db.execute(
            "SELECT name, type FROM members WHERE name >= ? AND name < ? ORDER BY name", (low, high)).fetchall())
        for name, kind in rows:
            if kind == DIRECTORY:
                continue
            if pattern is None or fnmatch.fnmatchcase(name, pattern):
                yield name

    def stats(self) -> dict:
        rejected = self._query(lambda db: db.execute("SELECT name, reason FROM rejected LIMIT 5").fetchall())
        ambiguous = self._query(lambda db: db.execute("SELECT name FROM members WHERE ambiguous = 1 LIMIT 5").fetchall())
        return {"path": self.path, "source": self.source, "summary": self.summary,
                "rejected_examples": rejected, "ambiguous_examples": [r[0] for r in ambiguous]}


def check_member_header(header: bytes, member: Member, where: str):
    if zlib.crc32(header) != member.header_crc:
        from vrae.storage.remote_io import StaleRemoteError
        raise StaleRemoteError(f"{where}: the tar header before {member.name!r} does not match the index; the "
                               "archive changed or the index belongs to another archive. Rebuild the index.")

