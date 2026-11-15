"""Remote video storage (vrae.storage): tar indexes, ranged member reads, the shared node cache, prefetch,
the storage file, and their use by the nanovvm_video_pred data path, checker and training loop.

Everything runs on CPU against tiny synthetic archives through file:// (a local, range-capable
transport); no AWS access is needed. Ported from NanoVVM's tests/test_remote_video.py.

Run from the repository root:
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m unittest tests.test_remote_video_storage -v
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import multiprocessing
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from vrae.config import load_config  # noqa: E402
from vrae.paths import ProjectPaths  # noqa: E402
from vrae.storage.backend import (  # noqa: E402
    StorageBackend,
    TarMemberSink,
    configure_storage,
    validate_storage,
)
from vrae.storage.config import (  # noqa: E402
    STORAGE_ENV,
    expand_environment,
    load_storage_file,
    storage_section_from_environment,
)
from vrae.storage.prefetch import Prefetcher  # noqa: E402
from vrae.storage.remote_io import (  # noqa: E402
    FileTransport,
    InvalidRangeError,
    RemoteError,
    RemoteNotFound,
    RemotePermissionError,
    RetryPolicy,
    StaleRemoteError,
    _Transient,
)
from vrae.storage.tar_index import (  # noqa: E402
    FileReader,
    TarIndex,
    TarIndexError,
    bind_remote,
    build_from_local_tar,
    build_from_remote,
    build_from_stream,
    build_index,
)
from vrae.storage.video_cache import (  # noqa: E402
    CachedObject,
    CacheError,
    CacheFullError,
    CacheTooSmallError,
    VideoCache,
    footprint,
)
from vrae.training.common.sampler import StatefulDistributedBatchSampler  # noqa: E402

MIB = 2**20
LADDER_TINY = (
    ROOT
    / "configs/training/nanovvm_video_pred/scaling-ladder/tiny"
    / "vrae_eupe_mix5_tiny_xpred_interval3_cond1gen16_bs64_lr2e-4_wd0_ep1.yaml"
)


class StorageTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        configure_storage(None)
        self._tmp.cleanup()


# ---------------------------------------------------------------- synthetic archives


def add_bytes(tar, name, data, **fields):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 1_700_000_000
    for key, value in fields.items():
        setattr(info, key, value)
    tar.addfile(info, io.BytesIO(data))


def add_link(tar, name, target, kind=tarfile.SYMTYPE):
    info = tarfile.TarInfo(name)
    info.type, info.linkname, info.mtime = kind, target, 1_700_000_000
    tar.addfile(info)


def add_dir(tar, name):
    info = tarfile.TarInfo(name)
    info.type, info.mtime = tarfile.DIRTYPE, 1_700_000_000
    tar.addfile(info)


LONG = "x/" + "/".join(["d" * 60] * 4) + "/long_name_file.bin"  # > 255 characters: GNU or PAX names
USTAR = "p/" + "q" * 120 + "/ustar_prefix.bin"


def payload(name, size=3000):
    seed = sum(name.encode()) % 251
    return bytes((seed + i * 7) % 256 for i in range(size))


def make_archive(path, fmt=tarfile.GNU_FORMAT, extra=None):
    """Regular files, a long name, symlinks (file, directory, chained), a hard link, and `extra`."""

    with tarfile.open(path, "w", format=fmt) as tar:
        add_dir(tar, "data")
        add_dir(tar, "data/videos")
        add_bytes(tar, "data/videos/a.bin", payload("a"))
        add_bytes(tar, "data/videos/b.bin", payload("b", 70_000))
        add_bytes(tar, "data/videos/empty.bin", b"")
        add_bytes(tar, LONG, payload("long"))
        add_dir(tar, "data/by_class")
        add_link(tar, "data/by_class/a_link.bin", "../videos/a.bin")
        add_link(tar, "data/by_class/chain.bin", "a_link.bin")
        add_link(tar, "data/alias", "videos")  # directory symlink
        add_link(tar, "data/hard.bin", "data/videos/b.bin", tarfile.LNKTYPE)
        add_bytes(tar, "data/manifests/m.json", b'{"ok": true}\n')
        if extra:
            extra(tar)
    return path


def build_local(archive, directory, name="index.sqlite"):
    output = Path(directory) / name
    build_from_local_tar(archive, output, archive_url=f"file://{archive}", log=lambda *_: None)
    bind_remote(output, FileTransport(), f"file://{archive}", log=lambda *_: None)
    return output


EXPECTED = {
    "data/videos/a.bin": payload("a"),
    "data/videos/b.bin": payload("b", 70_000),
    LONG: payload("long"),
    "data/by_class/a_link.bin": payload("a"),
    "data/by_class/chain.bin": payload("a"),
    "data/alias/a.bin": payload("a"),
    "data/hard.bin": payload("b", 70_000),
    "data/manifests/m.json": b'{"ok": true}\n',
    USTAR: payload("prefix"),
}


def read_member(transport, url, member):
    sink = io.BytesIO()
    transport.read_range(url, member.offset - 512, member.size + 512, TarMemberSink(sink, member, url))
    return sink.getvalue()


def rows(path):
    return TarIndex(path)._conn().execute("SELECT * FROM members ORDER BY name").fetchall()


def storage_section(tmp, *, archive=None, index=None, local=None, capacity_gib=1.0, **cache):
    return {
        "backend": "remote",
        "roots": [
            {
                "local": str(local or Path(tmp) / "remote_root"),
                "archive": archive or "file:///x.tar",
                "member_prefix": "",
                "index": str(index or Path(tmp) / "x.index"),
            }
        ],
        "cache": {"dir": str(Path(tmp) / "cache"), "capacity_gib": capacity_gib, "reserve_gib": 0, **cache},
        "prefetch": {"enabled": True, "depth": 8, "workers": 2, "max_downloads": 2, "max_reserved_gib": 0.1},
    }


# ---------------------------------------------------------------- tar index


class TarIndexTest(StorageTestCase):
    def test_index_resolves_names_links_and_exact_member_bytes(self):
        for fmt in (tarfile.GNU_FORMAT, tarfile.PAX_FORMAT, tarfile.USTAR_FORMAT):
            with self.subTest(fmt=fmt):
                directory = self.tmp / str(fmt)
                directory.mkdir()
                archive = directory / "a.tar"
                if fmt == tarfile.USTAR_FORMAT:  # ustar: names up to 255 via the prefix field
                    with tarfile.open(archive, "w", format=fmt) as tar:
                        add_bytes(tar, USTAR, payload("prefix"))
                        add_bytes(tar, "data/videos/a.bin", payload("a"))
                        add_link(tar, "data/by_class/a_link.bin", "../videos/a.bin")
                    names = [USTAR, "data/videos/a.bin", "data/by_class/a_link.bin"]
                else:
                    make_archive(archive, fmt, lambda tar: add_bytes(tar, USTAR, payload("prefix")))
                    names = list(EXPECTED)
                index = TarIndex(build_local(archive, directory))
                transport = FileTransport()
                url = f"file://{archive}"
                for name in names:
                    self.assertEqual(read_member(transport, url, index.lookup(name)), EXPECTED[name], name)
                # one ranged GET per member: header + payload, never the archive
                before = dict(transport.counters)
                member = index.lookup("data/videos/a.bin")
                read_member(transport, url, member)
                self.assertEqual(transport.counters["requests"] - before["requests"], 1)
                fetched = transport.counters["bytes"] - before["bytes"]
                self.assertEqual(fetched, member.size + 512)
                self.assertLess(fetched, archive.stat().st_size)

    def test_remote_and_stream_builds_match_local_and_skip_payloads(self):
        archive = make_archive(self.tmp / "a.tar")
        local = build_local(archive, self.tmp, "local.sqlite")
        summary = build_from_remote(
            FileTransport(), f"file://{archive}", self.tmp / "remote.sqlite",
            log=lambda *_: None, min_window=1024, max_window=4096,
        )
        with open(archive, "rb") as stream:
            build_from_stream(stream, self.tmp / "stream.sqlite", log=lambda *_: None)
        self.assertEqual(rows(local), rows(self.tmp / "remote.sqlite"))
        self.assertEqual(rows(local), rows(self.tmp / "stream.sqlite"))
        # the 70 kB payload of b.bin was skipped, not downloaded
        self.assertLess(summary["header_bytes_read"], archive.stat().st_size - 60_000)
        self.assertEqual(TarIndex(self.tmp / "remote.sqlite").remote_identity()["url"], f"file://{archive}")

    def test_index_rejects_traversal_duplicates_and_bad_links(self):
        def extra(tar):
            add_bytes(tar, "../evil.bin", b"x")
            add_bytes(tar, "/abs.bin", b"x")
            add_bytes(tar, "data/dup.bin", b"first")
            add_bytes(tar, "data/dup.bin", b"second")
            add_link(tar, "data/escape.bin", "../../outside.bin")
            add_link(tar, "data/absolute.bin", "/etc/passwd")
            add_link(tar, "data/loop1", "loop2")
            add_link(tar, "data/loop2", "loop1")
            add_link(tar, "data/dangling.bin", "videos/missing.bin")

        index = TarIndex(build_local(make_archive(self.tmp / "a.tar", extra=extra), self.tmp))
        self.assertEqual((index.summary["rejected"], index.summary["ambiguous"]), (2, 1))
        cases = {
            "../evil.bin": "traversal",
            "data/dup.bin": "more than once",
            "data/escape.bin": "escapes",
            "data/absolute.bin": "outside the archive",
            "data/loop1": "loop",
            "data/dangling.bin": "not in the archive",
            "data/videos": "directory",
            "data/videos/a.bin/x": "not a directory",
        }
        for name, match in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(TarIndexError, match):
                index.lookup(name)

    def test_compressed_archive_is_refused_with_preparation_hint(self):
        with tarfile.open(self.tmp / "a.tar.gz", "w:gz") as tar:
            add_bytes(tar, "v.bin", b"x" * 100)
        with self.assertRaisesRegex(TarIndexError, "gzip-compressed.*uncompressed copy"):
            build_from_local_tar(self.tmp / "a.tar.gz", self.tmp / "i.sqlite", log=lambda *_: None)
        storage = storage_section(self.tmp, archive="s3://b/a.tar.gz")
        storage["s3"] = {"region": "us-west-2"}
        with self.assertRaisesRegex(ValueError, "compressed"):
            validate_storage(storage)

    def test_interrupted_index_build_resumes(self):
        archive = make_archive(self.tmp / "a.tar")

        class Interrupting(FileReader):
            reads = 0

            def read(self, offset, length):
                Interrupting.reads += 1
                if Interrupting.reads == 9:
                    raise KeyboardInterrupt
                return super().read(offset, length)

        source = {"kind": "local_tar", "path": str(archive)}
        with self.assertRaises(KeyboardInterrupt):
            build_index(Interrupting(str(archive)), self.tmp / "i.sqlite", source, built_from="t",
                        commit_every=2, log=lambda *_: None)
        self.assertTrue((self.tmp / "i.sqlite.partial").exists())
        self.assertFalse((self.tmp / "i.sqlite").exists())
        messages = []
        build_index(FileReader(str(archive)), self.tmp / "i.sqlite", source, built_from="t",
                    commit_every=2, log=messages.append)
        self.assertTrue(any("resuming" in m for m in messages))
        build_index(FileReader(str(archive)), self.tmp / "full.sqlite", source, built_from="t",
                    log=lambda *_: None)
        self.assertEqual(rows(self.tmp / "i.sqlite"), rows(self.tmp / "full.sqlite"))

    def test_stale_index_is_detected(self):
        archive = make_archive(self.tmp / "a.tar")
        index_path = build_local(archive, self.tmp)
        storage = StorageBackend(storage_section(
            self.tmp, archive=f"file://{archive}", index=index_path, local=self.tmp / "data_root"
        ))
        self.assertTrue(storage.check_remote())
        path = self.tmp / "data_root" / "data/videos/a.bin"
        os.utime(archive, ns=(1, 1))  # the archive changes (new ETag)
        with self.assertRaisesRegex(StaleRemoteError, "changed since"):
            storage.check_remote()
        with self.assertRaisesRegex(StaleRemoteError, "precondition"):  # every read carries If-Match
            storage.lease(storage.locate(path))
        # same size and ETag recorded, but a different archive: the per-member header CRC catches it
        other = self.tmp / "other.tar"
        with tarfile.open(other, "w", format=tarfile.GNU_FORMAT) as tar:
            add_bytes(tar, "data/videos/a.bin", payload("a"), mtime=1_600_000_000)
        member = TarIndex(index_path).lookup("data/videos/a.bin")
        with self.assertRaisesRegex(StaleRemoteError, "does not match the index"):
            read_member(FileTransport(), f"file://{other}", member)

    def test_only_roots_in_use_are_checked(self):
        archive = make_archive(self.tmp / "a.tar")
        index = build_local(archive, self.tmp)
        storage = storage_section(self.tmp, archive=f"file://{archive}", index=index, local=self.tmp / "used")
        storage["roots"].append({"local": str(self.tmp / "unbuilt"), "archive": f"file://{self.tmp}/missing.tar",
                                 "member_prefix": "", "index": str(self.tmp / "missing.index")})
        backend = StorageBackend(storage)
        self.assertEqual(len(backend.check_remote([str(self.tmp / "used" / "data" / "videos")])), 1)
        with self.assertRaisesRegex(TarIndexError, "tar index not found"):
            backend.check_remote()
        self.assertEqual(backend.list_videos(self.tmp / "used" / "data"), [])  # .bin files are not videos


# ---------------------------------------------------------------- storage configuration


class StorageConfigTest(StorageTestCase):
    def test_storage_validation(self):
        cases = [
            (lambda s: s.update(backend="s3"), "backend"),
            (lambda s: s.update(colour=1), "Unknown storage keys"),
            (lambda s: s["roots"][0].update(local="relative/path"), "absolute"),
            (lambda s: s["roots"][0].update(objects="s3://b/p"), "exactly one"),
            (lambda s: s["roots"][0].pop("index"), "index"),
            (lambda s: s["roots"].append(dict(s["roots"][0])), "mapped twice"),
            (lambda s: s["cache"].update(capacity_gib=0), "capacity_gib"),
            (lambda s: s["cache"].update(dir=s["roots"][0]["local"] + "/cache"), "overlap"),
            (lambda s: s["prefetch"].update(max_downloads=16), "below storage.cache.max_downloads"),
            (lambda s: s["prefetch"].update(max_reserved_gib=0.6), "half"),
            (lambda s: s["roots"][0].update(archive="s3://bucket/x.tar"), "region"),
            (lambda s: s["roots"][0].update(archive="http://x/y.tar"), "s3:// or file://"),
        ]
        for edit, match in cases:
            with self.subTest(match=match):
                storage = storage_section(self.tmp)
                edit(storage)
                with self.assertRaisesRegex(ValueError, match):
                    validate_storage(storage)

    def test_local_backend_is_default_and_never_imports_aws(self):
        self.assertIsNone(validate_storage(None))
        self.assertIsNone(validate_storage({"backend": "local", "roots": "ignored"}))
        code = (
            "import sys; "
            "import vrae.training.nanovvm_video_pred.train, vrae.training.nanovvm_video_pred.sample; "
            "import vrae.training.nanovvm_video_pred.check, vrae.training.nanovvm_video_pred.latent_stats; "
            "import vrae.storage.remote_video; "
            "from vrae.storage import active_storage; assert not active_storage().remote; "
            "assert active_storage().locate('/any/video.mp4') is None; "
            "assert 'botocore' not in sys.modules and 'boto3' not in sys.modules; print('ok')"
        )
        env = {**os.environ, STORAGE_ENV: "", "PYTHONPATH": str(ROOT / "src")}
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("ok", result.stdout)

    def test_storage_file_expands_environment_and_anchors_relative_paths(self):
        environ = {"A": "1", "EMPTY": ""}
        self.assertEqual(
            expand_environment("${A} ${B:-d} ${C:-/m/${EMPTY:-vrae}/c} ${A:-${UNSET}}", environ),
            "1 d /m/vrae/c 1",
        )
        with self.assertRaisesRegex(ValueError, "needs environment variable UNSET"):
            expand_environment("${UNSET}", environ)
        with self.assertRaisesRegex(ValueError, "unclosed"):
            expand_environment("${A", environ)
        path = self.tmp / "storage.yaml"
        path.write_text(
            "storage:\n"
            "  backend: remote\n"
            "  roots:\n"
            "    - {local: toy_video, archive: '${ASSETS:-file:///x}/toy.tar', member_prefix: data/toy_video,\n"
            "       index: remote_index/toy.sqlite}\n"
            "    - {local: /abs/other, archive: file:///o.tar, index: /abs/o.sqlite}\n"
            "  cache: {dir: '${CACHE}', capacity_gib: ${GIB:-2}, reserve_gib: 0}\n"
            "  prefetch: {max_downloads: 2, max_reserved_gib: 0.5}\n"
        )
        data_root = self.tmp / "data"
        section = load_storage_file(path, data_root=data_root, environ={"CACHE": str(self.tmp / "c")})
        first, second = section["roots"]
        self.assertEqual(first["local"], str(data_root / "toy_video"))
        self.assertEqual(first["index"], str(data_root / "remote_index/toy.sqlite"))
        self.assertEqual(first["archive"], "file:///x/toy.tar")
        self.assertEqual((second["local"], second["index"]), ("/abs/other", "/abs/o.sqlite"))
        self.assertEqual(section["cache"]["capacity_gib"], 2)  # expanded before parsing: a number
        validate_storage(section)
        for value in ("", "local"):
            self.assertIsNone(storage_section_from_environment(data_root=data_root, environ={STORAGE_ENV: value}))
        self.assertEqual(
            storage_section_from_environment(
                data_root=data_root, environ={STORAGE_ENV: str(path), "CACHE": str(self.tmp / "c")}
            ),
            section,
        )
        bad = self.tmp / "bad.yaml"
        bad.write_text("storage: {}\nextra: 1\n")
        with self.assertRaisesRegex(ValueError, "exactly one top-level key"):
            load_storage_file(bad, data_root=data_root)

    def test_repository_storage_file_maps_every_ladder_root(self):
        from vrae.training.nanovvm_video_pred import data as nv_data

        data_root = (self.tmp / "nanovvm").resolve()
        data_root.mkdir()
        environ = {"VRAE_VIDEO_CACHE": str(self.tmp / "video_cache")}
        section = load_storage_file("configs/storage/adobe_s3.yaml", data_root=data_root,
                                    project_root=ROOT, environ=environ)
        storage = StorageBackend(section)
        config = load_config(LADDER_TINY)
        for split in ("train", "val"):
            for entry in nv_data.split_source_entries(config, split):
                root = data_root / entry["root"]
                self.assertIsNotNone(storage.root_for(root), root)
        self.assertEqual(storage.config["cache"]["dir"], str(self.tmp / "video_cache"))
        self.assertEqual(storage.config["cache"]["capacity_gib"], 300)
        self.assertEqual(storage.config["s3"]["credentials"], "instance_role")
        default = load_storage_file("configs/storage/adobe_s3.yaml", data_root=data_root,
                                    project_root=ROOT, environ={})
        self.assertEqual(default["cache"]["dir"], "/mnt/localssd/vrae/vrae/video_cache")  # USER unset


# ---------------------------------------------------------------- transports


class TransportTest(StorageTestCase):
    def test_transport_failures_are_classified_and_bounded(self):
        transport = FileTransport(RetryPolicy(max_attempts=3, backoff_seconds=0.001))
        with self.assertRaisesRegex(RemoteNotFound, "no such object"):
            transport.read_bytes(f"file://{self.tmp}/missing.tar", 0, 10)
        self.assertEqual((transport.counters["requests"], transport.counters["retries"]), (1, 0))
        (self.tmp / "x.tar").write_bytes(b"0123456789")
        with self.assertRaises(InvalidRangeError):
            transport.read_bytes(f"file://{self.tmp}/x.tar", 5, 10)
        if os.geteuid() != 0:
            (self.tmp / "locked.tar").write_bytes(b"x")
            (self.tmp / "locked.tar").chmod(0)
            with self.assertRaises(RemotePermissionError):
                transport.read_bytes(f"file://{self.tmp}/locked.tar", 0, 1)

        class Flaky(FileTransport):
            failures = 2

            def _raw_read(self, *args):
                if Flaky.failures:
                    Flaky.failures -= 1
                    raise _Transient("connection reset")
                return super()._raw_read(*args)

        flaky = Flaky(RetryPolicy(max_attempts=3, backoff_seconds=0.001))
        self.assertEqual(flaky.read_bytes(f"file://{self.tmp}/x.tar", 2, 3), b"234")
        self.assertEqual(flaky.counters["retries"], 2)

        class Short(FileTransport):
            def _raw_read(self, url, start, length, sink, *args):
                sink.write(b"x")
                return 1

        short = Short(RetryPolicy(max_attempts=2, backoff_seconds=0.001))
        with self.assertRaisesRegex(RemoteError, "failed after 2 attempts: short body"):
            short.read_bytes(f"file://{self.tmp}/x.tar", 0, 5)

    def test_s3_transport_validates_ranges_and_isolates_credentials(self):
        try:
            import botocore.exceptions
        except ImportError:
            self.skipTest("botocore is not installed (pip extra s3)")
        from vrae.storage.remote_io import S3Transport

        overrides = {
            "AWS_PROFILE": "profile-that-does-not-exist",
            "AWS_ACCESS_KEY_ID": "AKIDEXAMPLE",
            "AWS_SECRET_ACCESS_KEY": "not-a-secret",
            "AWS_ROLE_ARN": "arn:aws:iam::000000000000:role/x",
            "AWS_WEB_IDENTITY_TOKEN_FILE": "/nonexistent",
        }
        with mock.patch.dict(os.environ, overrides):
            transport = S3Transport(region="us-west-2", credentials="instance_role")
            session = transport._session()
            providers = session.get_component("credential_provider").providers
            self.assertEqual([type(p).__name__ for p in providers], ["InstanceMetadataProvider"])
            self.assertIsNone(session.get_config_variable("profile"))
            self.assertEqual(os.environ["AWS_PROFILE"], "profile-that-does-not-exist")  # untouched

        class Body:
            def __init__(self, data):
                self.data = io.BytesIO(data)

            def read(self, n):
                return self.data.read(n)

            def close(self):
                pass

        class Client:
            def __init__(self, response=None, error=None):
                self.response, self.error = response, error

            def get_object(self, **kwargs):
                if self.error:
                    raise self.error
                return self.response

        def use(client):
            transport._client_obj, transport._client_pid = client, os.getpid()

        use(Client({"ContentRange": "bytes 0-99/1000", "ContentLength": 100, "Body": Body(b"x" * 100)}))
        with self.assertRaisesRegex(InvalidRangeError, "asked for bytes 10-19"):
            transport.read_bytes("s3://b/k", 10, 10)
        use(Client({"ContentLength": 1000, "Body": Body(b"x" * 1000)}))  # server ignored the Range header
        with self.assertRaisesRegex(InvalidRangeError, "whole object"):
            transport.read_bytes("s3://b/k", 0, 10)

        def client_error(code, status):
            return botocore.exceptions.ClientError(
                {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, "GetObject"
            )

        for code, status, kind in [("NoSuchKey", 404, RemoteNotFound), ("AccessDenied", 403, RemotePermissionError),
                                   ("PreconditionFailed", 412, StaleRemoteError)]:
            use(Client(error=client_error(code, status)))
            before = transport.counters["requests"]
            with self.assertRaises(kind):
                transport.read_bytes("s3://b/k", 0, 10)
            self.assertEqual(transport.counters["requests"] - before, 1)  # not retried
        transport.retry = RetryPolicy(max_attempts=2, backoff_seconds=0.001)
        use(Client(error=client_error("SlowDown", 503)))
        with self.assertRaisesRegex(RemoteError, "failed after 2 attempts"):
            transport.read_bytes("s3://b/k", 0, 10)


# ---------------------------------------------------------------- the node cache


def obj(name, size):
    return CachedObject(key=hashlib.sha256(name.encode()).hexdigest(), size=size, suffix=".bin", description=name)


def writer(data, delay=0.0, log=None):
    def fetch(stream):
        if log is not None:
            with open(log, "a") as out:
                out.write("fetch\n")
        time.sleep(delay)
        stream.write(data)

    return fetch


def must_not_download(stream):
    raise AssertionError("must not download")


STATE = (1 << 20) + (128 << 10)  # the cache's fixed WAL allowance plus a small state database


def objects_on_disk(cache):
    return sum(footprint(p.stat().st_size) for p in Path(cache.directory, "objects").rglob("*") if p.is_file())


def _lease_in_child(directory, name, size, log, queue):
    cache = VideoCache(directory, 10 * MIB)
    with cache.lease(obj(name, size), writer(b"z" * size, delay=0.5, log=log)) as path:
        queue.put(Path(path).read_bytes() == b"z" * size)


def _hold_lease(directory, name, size, capacity, ready, release):
    cache = VideoCache(directory, capacity)
    lease = cache.lease(obj(name, size), writer(b"p" * size))
    ready.set()
    release.wait(60)
    lease.close()


def _die_mid_download(directory, name, size):
    cache = VideoCache(directory, 10 * MIB)

    def fetch(stream):
        stream.write(b"partial")
        stream.flush()
        os._exit(17)

    cache.lease(obj(name, size), fetch)


class VideoCacheTest(StorageTestCase):
    def test_cache_hit_needs_no_second_download(self):
        cache = VideoCache(self.tmp / "c", 10 * MIB)
        calls = []
        item = obj("a", 1000)
        for _ in range(3):
            with cache.lease(item, lambda s: (calls.append(1), s.write(b"a" * 1000))) as path:
                self.assertEqual(Path(path).read_bytes(), b"a" * 1000)
        self.assertEqual(len(calls), 1)
        stats = cache.stats()
        self.assertEqual((stats["demand_hits"], stats["demand_misses"], stats["entries"]), (2, 1, 1))

    def test_concurrent_processes_download_once(self):
        log = self.tmp / "fetches.log"
        VideoCache(self.tmp / "c", 10 * MIB)
        context = multiprocessing.get_context("fork")
        queue = context.Queue()
        children = [context.Process(target=_lease_in_child, args=(self.tmp / "c", "same", 5000, log, queue))
                    for _ in range(4)]
        for child in children:
            child.start()
        results = [queue.get(timeout=60) for _ in children]
        for child in children:
            child.join(30)
        self.assertEqual(results, [True] * 4)
        self.assertEqual(log.read_text().count("fetch"), 1)

    def test_eviction_never_removes_a_leased_file(self):
        size = 300_000
        capacity = 3 * footprint(size) + STATE  # room for three objects and the state database
        VideoCache(self.tmp / "c", capacity)
        context = multiprocessing.get_context("fork")
        ready, release = context.Event(), context.Event()
        child = context.Process(target=_hold_lease, args=(self.tmp / "c", "pinned", size, capacity, ready, release))
        child.start()
        self.assertTrue(ready.wait(60))
        cache = VideoCache(self.tmp / "c", capacity)
        pinned_path = Path(cache.object_path(obj("pinned", size)))
        for k in range(8):  # far more than fits: forces eviction around the pinned file
            cache.lease(obj(f"filler{k}", size), writer(b"f" * size)).close()
            self.assertTrue(pinned_path.exists())
            self.assertLessEqual(objects_on_disk(cache) + cache._state_bytes(), capacity)
        release.set()
        child.join(30)
        for k in range(8, 12):
            cache.lease(obj(f"filler{k}", size), writer(b"f" * size)).close()
        self.assertFalse(pinned_path.exists())  # evictable once released
        self.assertGreaterEqual(cache.stats()["evictions"], 8)

    def test_reservations_count_against_the_budget_and_full_cache_fails_clearly(self):
        size = 400_000
        capacity = 2 * footprint(size) + STATE
        cache = VideoCache(self.tmp / "c", capacity, wait_timeout=1.0, max_downloads=4)
        held = cache.lease(obj("held", size), writer(b"h" * size))  # pinned
        started, finished = threading.Event(), threading.Event()

        def slow(stream):
            started.set()
            time.sleep(1.5)
            stream.write(b"s" * size)

        thread = threading.Thread(target=lambda: (cache.lease(obj("slow", size), slow).close(), finished.set()))
        thread.start()
        self.assertTrue(started.wait(10))
        began = time.monotonic()
        with self.assertRaisesRegex(CacheFullError, "leased or in flight"):  # held pinned, slow reserved
            cache.lease(obj("third", size), writer(b"t" * size))
        self.assertGreaterEqual(time.monotonic() - began, 1.0)
        thread.join(10)
        self.assertTrue(finished.is_set())
        self.assertLessEqual(objects_on_disk(cache), capacity)
        with self.assertRaisesRegex(CacheTooSmallError, "more than the whole video cache"):
            cache.lease(obj("huge", capacity), must_not_download)
        held.close()
        cache.lease(obj("third", size), writer(b"t" * size)).close()  # room again after release

    def test_killed_producer_leaves_nothing_stale(self):
        VideoCache(self.tmp / "c", 10 * MIB)
        context = multiprocessing.get_context("fork")
        child = context.Process(target=_die_mid_download, args=(self.tmp / "c", "victim", 5000))
        child.start()
        child.join(30)
        self.assertEqual(child.exitcode, 17)
        self.assertTrue(list(Path(self.tmp, "c", "tmp").iterdir()))  # leftover partial
        cache = VideoCache(self.tmp / "c", 10 * MIB, wait_timeout=5)
        self.assertEqual(cache.stats()["inflight_downloads"], 1)  # the dead process's reservation
        report = cache.recover(full=True)
        self.assertEqual(report["partials_removed"], 1)
        self.assertEqual(cache.stats()["inflight_downloads"], 0)
        with cache.lease(obj("victim", 5000), writer(b"v" * 5000)) as path:  # producer lock released too
            self.assertEqual(Path(path).read_bytes(), b"v" * 5000)

    def test_failed_download_is_not_published_and_wrong_length_rejected(self):
        cache = VideoCache(self.tmp / "c", 10 * MIB)

        def missing(stream):
            raise RemoteNotFound("s3://b/x: no such object")

        with self.assertRaises(RemoteNotFound):
            cache.lease(obj("x", 100), missing)
        with self.assertRaisesRegex(CacheError, "expected 100"):
            cache.lease(obj("x", 100), writer(b"short"))
        self.assertFalse(list(Path(self.tmp, "c", "objects").rglob("*.bin")))
        self.assertFalse(list(Path(self.tmp, "c", "tmp").iterdir()))
        self.assertEqual(cache.stats()["inflight_downloads"], 0)

    def test_cache_refuses_foreign_directories_and_prefetch_never_waits(self):
        (self.tmp / "data").mkdir()
        (self.tmp / "data" / "precious.mp4").write_bytes(b"x")
        with self.assertRaisesRegex(CacheError, "not a video cache"):
            VideoCache(self.tmp / "data", MIB)
        cache = VideoCache(self.tmp / "c", 10 * MIB, prefetch_max_downloads=0)
        self.assertEqual(cache.prefetch(obj("p", 10), writer(b"p" * 10)), "busy")
        cache = VideoCache(self.tmp / "c", 10 * MIB, prefetch_max_downloads=2)
        self.assertEqual(cache.prefetch(obj("p", 10), writer(b"p" * 10)), "downloaded")
        self.assertEqual(cache.prefetch(obj("p", 10), writer(b"p" * 10)), "cached")
        cache.lease(obj("p", 10), must_not_download).close()
        self.assertEqual(cache.stats()["prefetch_used"], 1)


# ---------------------------------------------------------------- prefetch and the sampler


class PrefetchTest(unittest.TestCase):
    def test_prefetcher_window_is_bounded_and_deduplicated(self):
        seen, gate = [], threading.Event()

        def prefetch_one(video):
            gate.wait(10)
            seen.append(video)

        prefetcher = Prefetcher(lambda i: f"video{i // 2}", lambda i: i // 2, prefetch_one, depth=4, workers=1)
        prefetcher.start(list(range(40)))
        for position in range(10):
            prefetcher.advance(position)
        self.assertLessEqual(prefetcher._queue.qsize(), 4)
        gate.set()
        prefetcher.advance(39)
        time.sleep(0.5)
        prefetcher.close()
        self.assertEqual(len(seen), len(set(seen)))  # each video once
        self.assertGreater(prefetcher.counters["deduplicated"], 0)

    def test_sampler_announces_exactly_the_batches_it_yields(self):
        class Recorder:
            def __init__(self):
                self.order, self.drawn = None, []

            def start(self, order):
                self.order = order

            def advance(self, position):
                self.drawn.append(position)

        for world_size, rank, accumulation, committed in [(1, 0, 1, 0), (2, 1, 2, 3), (4, 2, 1, 5)]:
            with self.subTest(world_size=world_size, rank=rank, committed=committed):
                plain = StatefulDistributedBatchSampler(
                    203, 3, rank=rank, world_size=world_size, seed=7, gradient_accumulation_steps=accumulation
                )
                observed = StatefulDistributedBatchSampler(
                    203, 3, rank=rank, world_size=world_size, seed=7, gradient_accumulation_steps=accumulation
                )
                for sampler in (plain, observed):  # resume inside the epoch
                    iterator = iter(sampler)
                    for _ in range(committed):
                        next(iterator)
                        sampler.commit_batch()
                    del iterator
                observed.prefetcher = Recorder()
                expected = list(plain)
                batches = list(observed)
                self.assertEqual(batches, expected)  # prefetch never changes what is yielded
                flat = [index for batch in batches for index in batch]
                order = observed.prefetcher.order
                self.assertEqual(len(order), len(flat))
                self.assertEqual([order[p] for p in range(len(order))], flat)
                self.assertEqual(order[1:4], flat[1:4])
                self.assertEqual(observed.prefetcher.drawn, [3 * (k + 1) - 1 for k in range(len(batches))])


# ---------------------------------------------------------------- remote-backed nanovvm_video_pred data


def write_video(path: Path, length: int = 100, shift: int = 0) -> None:
    """MJPG AVI whose frame i shows (i + shift) in binary as seven bands (robust to resizing)."""

    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (200, 126))
    for index in range(length):
        frame = np.zeros((126, 200, 3), dtype=np.uint8)
        for bit in range(7):
            if (index + shift) >> bit & 1:
                frame[bit * 18 : (bit + 1) * 18] = 255
        video.write(frame)
    video.release()


CAMERAS = ("cam_a", "cam_b")
VIDEOS = ("ep0.avi", "ep1.avi")


def remote_sources(split: str) -> list[dict]:
    return [
        {
            "name": f"toy/{camera}",
            "dataset": "toy",
            "root": f"toy_video/data/{camera}",
            "manifest": f"toy_video/manifests/{camera}/cond1gen16_interval3/split/{split}/manifest.json",
            "manifest_meta": f"toy_video/manifests/{camera}/cond1gen16_interval3/split/{split}/meta.yaml",
        }
        for camera in CAMERAS
    ]


class RemoteDataTest(StorageTestCase):
    """A NanoVVM-shaped data root: manifests local, videos only in a tar (member prefix data/toy_video)."""

    def setUp(self):
        super().setUp()
        from tests.test_nanovvm_video_pred import record, write_manifest

        staging = self.tmp / "staging"
        self.local_root = staging / "data"  # a complete local copy, for comparison
        for k, camera in enumerate(CAMERAS):
            for j, name in enumerate(VIDEOS):
                write_video(self.local_root / "toy_video" / "data" / camera / name, shift=10 * k + j)
        self.archive = self.tmp / "toy_video.tar"
        with tarfile.open(self.archive, "w", format=tarfile.GNU_FORMAT) as tar:
            tar.add(self.local_root / "toy_video", arcname="data/toy_video")
        self.remote_root = (self.tmp / "nanovvm").resolve()  # holds manifests and the index only
        for root in (self.local_root, self.remote_root):
            for camera in CAMERAS:
                manifests = root / "toy_video" / "manifests" / camera / "cond1gen16_interval3" / "split"
                write_manifest(manifests / "train",
                               [record("ep0.avi", 0), record("ep1.avi", 0), record("ep1.avi", 49)])
                write_manifest(manifests / "val", [record("ep0.avi", 49)], split="val")
        index = self.remote_root / "remote_index" / "toy_video.tar.index.sqlite"
        index.parent.mkdir(parents=True)
        build_local(self.archive, index.parent, index.name)
        self.storage_file = self.tmp / "storage.yaml"
        self.storage_file.write_text(
            "storage:\n"
            "  backend: remote\n"
            "  roots:\n"
            f"    - {{local: toy_video, archive: 'file://{self.archive}', member_prefix: data/toy_video,\n"
            "       index: remote_index/toy_video.tar.index.sqlite}\n"
            f"  cache: {{dir: '{self.tmp / 'cache'}', capacity_gib: 1, reserve_gib: 0}}\n"
            "  prefetch: {enabled: true, depth: 8, workers: 2, max_downloads: 2, max_reserved_gib: 0.1}\n"
        )

    def config(self):
        from tests.test_nanovvm_video_pred import task_config

        config = task_config()
        config["data"]["train"] = {"sources": remote_sources("train")}
        config["data"]["val"] = {"sources": remote_sources("val")}
        return config

    def remote(self, config=None):
        from vrae.training.nanovvm_video_pred import data as nv_data

        paths = ProjectPaths(project_root=self.tmp, datasets={"nanovvm": self.remote_root})
        storage = nv_data.configure_nanovvm_storage(config or self.config(), paths, override=str(self.storage_file))
        return paths, storage

    def local_paths(self):
        configure_storage(None)
        return ProjectPaths(project_root=self.tmp, datasets={"nanovvm": self.local_root})

    def test_remote_dataset_matches_local_files(self):
        from vrae.training.nanovvm_video_pred import data as nv_data

        config = self.config()
        paths, storage = self.remote(config)
        self.assertTrue(storage.remote)
        self.assertFalse((self.remote_root / "toy_video" / "data").exists())  # nothing staged
        remote = nv_data.build_nanovvm_dataset(config, paths, split="train")
        remote_items = [remote[index] for index in range(len(remote))]
        self.assertEqual(storage.stats()["entries"], len(CAMERAS) * len(VIDEOS))  # one file per video
        local = nv_data.build_nanovvm_dataset(config, self.local_paths(), split="train")
        self.assertFalse(local.storage.remote)
        self.assertEqual(len(local), len(remote))
        for index in range(len(local)):
            a, b = local[index], remote_items[index]
            self.assertEqual(a["sample_id"], b["sample_id"])
            for key in ("condition", "future", "frame_indices"):
                self.assertTrue(torch.equal(a[key], b[key]), (index, key))
        self.assertEqual(remote_items[2]["frame_indices"].tolist(), list(range(49, 98, 3)))

    def test_prefetch_preserves_batches(self):
        from vrae.training.nanovvm_video_pred import data as nv_data

        config = self.config()
        config["training"].update({"global_batch_size": 2, "gradient_accumulation_steps": 1})
        paths, storage = self.remote(config)

        def run(dataset, workers):
            config["training"]["num_workers"] = workers
            loader, sampler = nv_data.build_nanovvm_loader(dataset, config, rank=0, world_size=1)
            batches = []
            for batch in loader:
                batches.append((batch["index"].tolist(), batch["condition"].clone(), batch["future"].clone()))
                sampler.commit_batch()
            if sampler.prefetcher is not None:
                sampler.prefetcher.close()
            return sampler, batches

        for workers in (0, 2):
            with self.subTest(workers=workers):
                storage.cache.clear()
                remote = nv_data.build_nanovvm_dataset(config, paths, split="train")
                sampler, prefetched = run(remote, workers)
                self.assertIsNotNone(sampler.prefetcher)
                local = nv_data.build_nanovvm_dataset(config, self.local_paths(), split="train")
                plain_sampler, plain = run(local, workers)
                self.assertIsNone(plain_sampler.prefetcher)
                configure_storage(storage)
                self.assertEqual([b[0] for b in prefetched], [b[0] for b in plain])
                for (_, c1, f1), (_, c2, f2) in zip(prefetched, plain, strict=True):
                    self.assertTrue(torch.equal(c1, c2) and torch.equal(f1, f2))
                self.assertGreater(storage.stats().get("prefetch_requests", 0), 0)

    def test_checker_reports_storage_and_missing_indexes(self):
        from vrae.training.nanovvm_video_pred import data as nv_data
        from vrae.training.nanovvm_video_pred.check import check_config

        config = self.config()
        paths, storage = self.remote(config)
        report = check_config(config, paths, load_manifests=True, decode=2, storage=storage)
        self.assertTrue(any(f"file://{self.archive}" in line for line in report["storage"]))
        self.assertTrue(any(line.startswith("video cache:") for line in report["storage"]))
        self.assertEqual(report["storage_check"], "every manifest video resolves in its index")
        self.assertEqual(report["splits"]["train"]["validated_clips"], 6)
        self.assertEqual(len(report["splits"]["val"]["decoded"]), 2)
        # A manifest video missing from the archive fails before any decoding.
        from tests.test_nanovvm_video_pred import record, write_manifest

        manifests = self.remote_root / "toy_video/manifests/cam_a/cond1gen16_interval3/split/train"
        write_manifest(manifests, [record("ep0.avi", 0), record("missing.avi", 0)])
        with self.assertRaisesRegex(TarIndexError, "cannot be served"):
            nv_data.build_nanovvm_dataset(config, paths, split="train")
        # Without the index, the root is reported as a missing path.
        (self.remote_root / "remote_index" / "toy_video.tar.index.sqlite").unlink()
        paths, storage = self.remote(config)
        with self.assertRaisesRegex(FileNotFoundError, "tar index not found"):
            nv_data.resolve_split_sources(config, paths, "train")

    def test_training_reads_remote_sources(self):
        from tests.test_nanovvm_video_pred import fake_wandb_module, replaced_module, tiny_stage1
        from vrae.config import save_resolved_config
        from vrae.training.common.contracts import create_run_metadata
        from vrae.training.nanovvm_video_pred import data as nv_data
        from vrae.training.nanovvm_video_pred import latent_stats
        from vrae.training.nanovvm_video_pred import train as train_module

        config = self.config()
        config["training"].update({"global_batch_size": 2, "num_workers": 2, "gradient_accumulation_steps": 1})
        config["wandb"].update({"mode": "online", "sample_interval": 1000})
        paths, storage = self.remote(config)
        report = nv_data.storage_preflight(
            storage, [item["root"] for item in nv_data.resolve_split_sources(config, paths, "train")], owner=True
        )
        stage1 = tiny_stage1()
        wandb_module = fake_wandb_module()
        loader = mock.Mock(return_value=(stage1, {}))
        output = io.StringIO()
        with (
            replaced_module("wandb", wandb_module),
            mock.patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
            mock.patch.object(latent_stats, "load_frozen_stage1", loader),
            mock.patch.object(train_module, "load_frozen_stage1", loader),
            contextlib.redirect_stdout(output),
        ):
            os.environ.pop("WANDB_MODE", None)
            latent_stats.compute_nanovvm_latent_stats(config, paths)
            run = paths.training_run(config["task"], config["run_name"], create=True)
            for directory in ("checkpoints", "samples", "wandb"):
                (run / directory).mkdir()
            save_resolved_config(create_run_metadata(config), run / "run_metadata.yaml")
            train_module.train(config, paths, run, max_steps=2, storage_report=report)
        text = output.getvalue()
        self.assertIn("[storage] Remote video storage: video cache:", text)
        lines = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
        steps = [line for line in lines if "cache/hit_rate" in line]
        self.assertEqual([line["step"] for line in steps], [1, 2])
        for line in steps:
            self.assertGreaterEqual(line["data/wait_seconds_per_step"], 0.0)
        logged = [values for _, values in wandb_module.logs if "loss/train" in values]
        self.assertTrue(logged and all("cache/requests" in values for values in logged))
        self.assertTrue((run / "checkpoints" / "step-00000002.pt").is_file())
        self.assertFalse((self.remote_root / "toy_video" / "data").exists())
        stats = storage.stats()
        self.assertGreater(stats["demand_requests"], 0)
        self.assertGreaterEqual(stats["entries"], 1)


if __name__ == "__main__":
    unittest.main()
