"""Remote video sources: build and check tar indexes, fetch single members, maintain the node video cache.

Vendored from NanoVVM (data_processing/tools/remote_video.py, commit ca80730). The per-config check is
`python -m vrae.training.nanovvm_video_pred.check` here, so NanoVVM's check and bench commands are omitted.

    T="python -m vrae.storage.remote_video"
    # index a remote uncompressed tar from its headers only (ranged GETs; resumable)
    $T index --archive s3://bucket/data/x_video.tar --output $DATA/remote_index/x_video.tar.index.sqlite
    # ... or from a local copy / a stream, then pin it to the uploaded object
    $T index --archive s3://bucket/data/x_video.tar --from-local-tar /path/x_video.tar --output ...
    $T bind --index ... --archive s3://...     # pin a local/stream-built index to the uploaded object
    $T verify --index ...                      # identity vs the live object + header spot checks
    $T list --index ... --prefix data/x_video/manifests
    $T extract --index ... --pattern 'data/x_video/manifests/*' --strip-prefix data --dest $DATA
    $T upload-index --index ... --dest s3://bucket/assets/nanovvm/indexes/v1/x_video.tar.index.sqlite
    $T cache --storage configs/storage/adobe_s3.yaml --data-root $DATA --stats | --recover | --clear

S3 options (--region, --credentials) apply to index/verify/extract/upload-index; cache reads the storage file
(default: VRAE_STORAGE_CONFIG). Never deletes or overwrites remote objects.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from vrae.storage.remote_io import FileTransport, RemoteError, S3Transport, parse_url
from vrae.storage.tar_index import (BLOCK, TarIndex, TarIndexError, bind_remote, build_from_local_tar,
                                    build_from_remote, build_from_stream)
from vrae.storage.video_cache import CacheError


def transport_for(url, args):
    if parse_url(url)[0] == "s3":
        return S3Transport(region=args.region, credentials=args.credentials, use_version_id=args.use_version_id)
    return FileTransport()


def log(message):
    print(f"[remote_video {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---- index tools ----

def cmd_index(args):
    output = Path(args.output).expanduser()
    if output.exists() and not args.force:
        raise SystemExit(f"{output} exists; pass --force to rebuild it")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.from_local_tar:
        summary = build_from_local_tar(args.from_local_tar, output, archive_url=args.archive, restart=args.restart,
                                       log=log)
    elif args.from_stdin:
        summary = build_from_stream(sys.stdin.buffer, output, archive_url=args.archive, log=log)
    else:
        transport = transport_for(args.archive, args)
        summary = build_from_remote(transport, args.archive, output, restart=args.restart, log=log,
                                    min_window=args.min_window_kib << 10, max_window=args.max_window_kib << 10)
        log(f"remote header walk: {summary['header_requests']} ranged GETs, "
            f"{summary['header_bytes_read'] / 2**20:.1f} MiB read")
    log(f"indexed {summary['indexed']} members ({summary['rejected']} rejected, {summary['ambiguous']} ambiguous) "
        f"in {time.monotonic() - started:.0f}s -> {output}")
    if (args.from_local_tar or args.from_stdin) and not args.no_bind:
        bind_remote(output, transport_for(args.archive, args), args.archive, samples=args.bind_samples, log=log)
    print(json.dumps(summary, indent=2))


def cmd_bind(args):
    bind_remote(args.index, transport_for(args.archive, args), args.archive, samples=args.samples, log=log)


def cmd_verify(args):
    index = TarIndex(args.index)
    identity = index.remote_identity()
    transport = transport_for(identity["url"], args)
    info = transport.head(identity["url"])
    changed = [k for k in ("size", "etag", "version_id")
               if identity.get(k) is not None and info.identity().get(k) != identity[k]]
    if changed:
        raise SystemExit(f"STALE: {identity['url']} changed since indexing ({changed}); rebuild the index")
    db = index._conn()
    rows = db.execute("SELECT name, offset, header_crc FROM members WHERE type = '0' ORDER BY offset").fetchall()
    step = max(1, len(rows) // max(1, args.samples))
    import zlib
    checked = 0
    for name, offset, crc in rows[::step][:args.samples] + rows[-1:]:
        header = transport.read_bytes(identity["url"], offset - BLOCK, BLOCK, version_id=identity["version_id"],
                                      if_match=identity["etag"])
        if zlib.crc32(header) != crc:
            raise SystemExit(f"MISMATCH: header of {name} at byte {offset - BLOCK} differs from the index")
        checked += 1
    print(json.dumps({**index.stats(), "live": info.identity(), "headers_checked": checked,
                      "requests": transport.counters}, indent=2, default=str))


def cmd_list(args):
    index = TarIndex(args.index)
    for count, name in enumerate(index.names(args.prefix, args.pattern)):
        if args.limit and count >= args.limit:
            print(f"... (limit {args.limit})")
            break
        print(name)


def cmd_extract(args):
    """Fetch only the matching members (e.g. manifests) into dest, verifying each header; links resolved."""
    index = TarIndex(args.index)
    identity = index.remote_identity()
    transport = transport_for(identity["url"], args)
    strip = args.strip_prefix.strip("/")
    dest = Path(args.dest).expanduser().resolve()
    names = sorted({name for pattern in args.pattern for name in index.names(args.prefix, pattern)})
    if not names:
        raise SystemExit("no member matches " + " ".join(args.pattern))
    total = 0
    from vrae.storage.backend import TarMemberSink
    for name in names:
        member = index.lookup(name)
        if strip and not (name == strip or name.startswith(strip + "/")):
            raise SystemExit(f"{name} is not below --strip-prefix {strip}")
        relative = name[len(strip) + 1:] if strip else name
        target = (dest / relative).resolve()
        if dest not in target.parents:
            raise SystemExit(f"{name} would leave {dest}")
        if target.exists() and target.stat().st_size == member.size and not args.force:
            log(f"exists, kept: {target}")
            continue
        if args.dry_run:
            log(f"would fetch {name} ({member.size} bytes) -> {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.part")
        with open(temporary, "wb+") as stream:
            transport.read_range(identity["url"], member.offset - BLOCK, member.size + BLOCK,
                                 TarMemberSink(stream, member, identity["url"]),
                                 version_id=identity["version_id"], if_match=identity["etag"])
        if temporary.stat().st_size != member.size:
            temporary.unlink()
            raise SystemExit(f"{name}: short download")
        os.replace(temporary, target)
        total += member.size
        log(f"{name} -> {target} ({member.size} bytes)")
    log(f"fetched {total / 2**20:.2f} MiB in {transport.counters['requests']} requests")


def cmd_upload_index(args):
    index = TarIndex(args.index)
    index.remote_identity()
    info = transport_for(args.dest, args).put_new(args.dest, index.path)
    log(f"uploaded {index.path} -> {args.dest} ({info.size} bytes, create-only)")


# ---- the node video cache ----

def cmd_cache(args):
    from vrae.storage.backend import StorageBackend
    from vrae.storage.config import STORAGE_ENV, storage_section_from_environment
    section = storage_section_from_environment(data_root=args.data_root, override=args.storage)
    if section is None:
        raise SystemExit(f"no storage file: pass --storage or set {STORAGE_ENV}")
    if args.cache_dir:
        section.setdefault("cache", {})["dir"] = args.cache_dir
    cache = StorageBackend(section).cache
    if args.recover:
        log(f"recovered: {cache.recover(full=True)}")
    if args.clear:
        log(f"evicted {cache.clear() / 2**20:.1f} MiB (leased files are kept)")
    print(json.dumps(cache.stats(), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", default=os.environ.get("VRAE_S3_REGION", "us-west-2"))
    parser.add_argument("--credentials", choices=["instance_role", "default_chain"], default="instance_role")
    parser.add_argument("--use-version-id", action="store_true",
                        help="also pin reads to the VersionId (needs s3:GetObjectVersion)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="build a tar index (headers only)")
    p.add_argument("--archive", required=True, help="s3:// or file:// URL of the uncompressed tar")
    p.add_argument("--output", required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--from-local-tar", help="read headers from this local copy, then bind to --archive")
    source.add_argument("--from-stdin", action="store_true", help="read a tar stream from stdin")
    p.add_argument("--no-bind", action="store_true", help="skip binding a local/stream index to --archive")
    p.add_argument("--bind-samples", type=int, default=16)
    p.add_argument("--min-window-kib", type=int, default=16)
    p.add_argument("--max-window-kib", type=int, default=1024)
    p.add_argument("--restart", action="store_true", help="discard a partial build instead of resuming it")
    p.add_argument("--force", action="store_true", help="replace an existing index file")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("bind", help="pin a local/stream index to the uploaded archive (header spot checks)")
    p.add_argument("--index", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--samples", type=int, default=16)
    p.set_defaults(func=cmd_bind)

    p = sub.add_parser("verify", help="check an index against the live archive")
    p.add_argument("--index", required=True)
    p.add_argument("--samples", type=int, default=16)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("list", help="list member names")
    p.add_argument("--index", required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--pattern")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("extract", help="fetch matching members only")
    p.add_argument("--index", required=True)
    p.add_argument("--pattern", action="append", required=True, help="fnmatch pattern ('*' crosses '/')")
    p.add_argument("--prefix", default="", help="limit the search to members below this path")
    p.add_argument("--strip-prefix", default="")
    p.add_argument("--dest", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("upload-index", help="copy an index to a new remote key (never overwrites)")
    p.add_argument("--index", required=True)
    p.add_argument("--dest", required=True)
    p.set_defaults(func=cmd_upload_index)

    p = sub.add_parser("cache", help="inspect or maintain the node video cache")
    p.add_argument("--storage", default=None, help="storage file (default: VRAE_STORAGE_CONFIG)")
    p.add_argument("--data-root", required=True, help="NanoVVM data root, for relative roots and indexes")
    p.add_argument("--cache-dir")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--recover", action="store_true")
    p.add_argument("--clear", action="store_true")
    p.set_defaults(func=cmd_cache)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (TarIndexError, RemoteError, CacheError) as error:
        raise SystemExit(f"error: {type(error).__name__}: {error}") from None


if __name__ == "__main__":
    main()
