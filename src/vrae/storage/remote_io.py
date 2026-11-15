"""Range-capable readers of remote objects: S3 (botocore, optional dependency) and file:// (tests, benchmarks).

Every request is a read (HEAD, ranged GET, LIST), except one create-only upload for small index files.
Failures are classified: missing objects, permission problems, stale identities (412) and invalid range
responses fail at once with the URL in the message; network errors, throttling, 5xx and short bodies are
retried with bounded exponential backoff.

Credentials (storage.s3.credentials):
  instance_role  the EC2 instance profile only, through the instance metadata service. AWS_PROFILE,
                 AWS_ACCESS_KEY_ID/..., AWS_ROLE_ARN/AWS_WEB_IDENTITY_TOKEN_FILE, ~/.aws files and configured
                 endpoint URLs are ignored, exactly like the README's `env -u ... s5cmd` wrapper, but without
                 touching os.environ (the session is private to this module).
  default_chain  botocore's standard chain (environment, profile, web identity, instance role) for other machines.
"""
from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse

CHUNK = 1 << 20


class RemoteError(RuntimeError):
    """A remote read failed for a reason retrying will not fix (or after the retries ran out)."""


class RemoteNotFound(RemoteError):
    pass


class RemotePermissionError(RemoteError):
    pass


class StaleRemoteError(RemoteError):
    """The object no longer matches the identity (ETag / VersionId / size) an index or cache entry recorded."""


class InvalidRangeError(RemoteError):
    pass


class _Transient(Exception):
    """Internal: worth another attempt."""


@dataclass(frozen=True)
class ObjectInfo:
    url: str
    size: int
    etag: str | None = None
    version_id: str | None = None
    last_modified: str | None = None

    def identity(self) -> dict:
        return {"url": self.url, "size": self.size, "etag": self.etag, "version_id": self.version_id}


def parse_url(url: str):
    """('s3', bucket, key) or ('file', '', absolute path)."""
    parsed = urlparse(url)
    if parsed.scheme == "s3":
        if not parsed.netloc:
            raise ValueError(f"{url!r}: an s3:// URL needs a bucket")
        return "s3", parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost") or not parsed.path.startswith("/"):
            raise ValueError(f"{url!r}: use file:///absolute/path")
        return "file", "", parsed.path
    raise ValueError(f"{url!r}: remote URLs must start with s3:// or file://")


def join_url(prefix: str, relative: str) -> str:
    return prefix.rstrip("/") + "/" + relative.lstrip("/")


class FileSink:
    """Destination of a ranged read; reset() discards a failed attempt's bytes."""

    def __init__(self, stream):
        self.stream = stream

    def reset(self):
        self.stream.seek(0)
        self.stream.truncate()

    def write(self, data: bytes):
        self.stream.write(data)


class MemorySink:
    def __init__(self):
        self.parts = []

    def reset(self):
        self.parts.clear()

    def write(self, data: bytes):
        self.parts.append(bytes(data))

    def getvalue(self) -> bytes:
        return b"".join(self.parts)


@dataclass
class RetryPolicy:
    max_attempts: int = 6
    backoff_seconds: float = 0.5
    max_backoff_seconds: float = 20.0

    def delay(self, attempt: int) -> float:
        base = min(self.max_backoff_seconds, self.backoff_seconds * (2 ** (attempt - 1)))
        return base * (0.5 + random.random() / 2)


class Transport:
    """Counts requests, bytes and retries (per process); subclasses implement the _raw_* calls."""

    def __init__(self, retry: RetryPolicy | None = None):
        self.retry = retry or RetryPolicy()
        self._lock = threading.Lock()
        self.counters = {"requests": 0, "bytes": 0, "retries": 0}

    def _count(self, **values):
        with self._lock:
            for key, value in values.items():
                self.counters[key] = self.counters.get(key, 0) + value

    def take_counters(self) -> dict:
        with self._lock:
            values, self.counters = self.counters, {"requests": 0, "bytes": 0, "retries": 0}
        return values

    def _attempts(self, describe: str, call):
        last = None
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                self._count(requests=1)
                return call()
            except _Transient as error:
                last = error
                if attempt == self.retry.max_attempts:
                    break
                self._count(retries=1)
                time.sleep(self.retry.delay(attempt))
        raise RemoteError(f"{describe} failed after {self.retry.max_attempts} attempts: {last}")

    # ---- public API ----

    def head(self, url: str) -> ObjectInfo:
        return self._attempts(f"HEAD {url}", lambda: self._raw_head(url))

    def read_range(self, url: str, start: int, length: int, sink, *, version_id=None, if_match=None) -> int:
        """Write bytes [start, start + length) of url to sink, validating the response's range and length."""
        if start < 0 or length <= 0:
            raise ValueError(f"invalid range start={start} length={length}")
        describe = f"GET {url} bytes={start}-{start + length - 1}"

        def call():
            sink.reset()
            received = self._raw_read(url, start, length, sink, version_id, if_match)
            if received != length:
                raise _Transient(f"short body: received {received} of {length} bytes")
            return received

        received = self._attempts(describe, call)
        self._count(bytes=received)
        return received

    def read_bytes(self, url: str, start: int, length: int, **kwargs) -> bytes:
        sink = MemorySink()
        self.read_range(url, start, length, sink, **kwargs)
        return sink.getvalue()

    def list(self, prefix_url: str):
        """(url, size, etag) of every object below prefix_url/."""
        return self._raw_list(prefix_url)

    def put_new(self, url: str, path: str) -> ObjectInfo:
        """Upload a small local file to url, never replacing an existing object."""
        return self._raw_put_new(url, path)


class FileTransport(Transport):
    """file:// objects: a local file read by ranges. Identity: size, inode and mtime (a fake ETag)."""

    @staticmethod
    def _path(url):
        scheme, _, path = parse_url(url)
        if scheme != "file":
            raise ValueError(f"FileTransport cannot read {url}")
        return path

    def _stat(self, url):
        path = self._path(url)
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            raise RemoteNotFound(f"{url}: no such object") from None
        except PermissionError as error:
            raise RemotePermissionError(f"{url}: permission denied ({error})") from None
        if not os.path.isfile(path):
            raise RemoteNotFound(f"{url}: not a regular file")
        return path, stat

    def _raw_head(self, url):
        _, stat = self._stat(url)
        return ObjectInfo(url, stat.st_size, f"{stat.st_ino:x}-{stat.st_mtime_ns:x}-{stat.st_size:x}", None,
                          str(stat.st_mtime_ns))

    def _raw_read(self, url, start, length, sink, version_id, if_match):
        path, stat = self._stat(url)
        if if_match is not None and if_match != self._raw_head(url).etag:
            raise StaleRemoteError(f"{url}: precondition failed (ETag changed from {if_match})")
        if start + length > stat.st_size:
            raise InvalidRangeError(f"{url}: range {start}-{start + length - 1} exceeds the object size {stat.st_size}")
        received = 0
        try:
            stream = open(path, "rb")
        except PermissionError as error:
            raise RemotePermissionError(f"{url}: permission denied ({error})") from None
        with stream:
            stream.seek(start)
            while received < length:
                data = stream.read(min(CHUNK, length - received))
                if not data:
                    break
                sink.write(data)
                received += len(data)
        return received

    def _raw_list(self, prefix_url):
        root = self._path(prefix_url).rstrip("/")
        for directory, _, files in os.walk(root):
            for name in sorted(files):
                path = os.path.join(directory, name)
                info = self._raw_head("file://" + path)
                yield info.url, info.size, info.etag

    def _raw_put_new(self, url, path):
        destination = self._path(url)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        try:
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            raise RemoteError(f"{url} already exists; refusing to overwrite it") from None
        with os.fdopen(fd, "wb") as out, open(path, "rb") as source:
            while data := source.read(CHUNK):
                out.write(data)
        return self._raw_head(url)


_TRANSIENT_CODES = {"SlowDown", "Throttling", "ThrottlingException", "RequestTimeout", "RequestTimeTooSkewed",
                    "InternalError", "ServiceUnavailable", "OperationAborted"}


class S3Transport(Transport):
    """S3 through a private botocore session (imported lazily; `uv sync --group s3`)."""

    def __init__(self, *, region: str, credentials: str = "instance_role", endpoint_url: str | None = None,
                 connect_timeout: float = 10, read_timeout: float = 60, max_pool_connections: int = 32,
                 use_version_id: bool = False, retry: RetryPolicy | None = None):
        super().__init__(retry)
        # GET by VersionId needs s3:GetObjectVersion; without it, identity is enforced by If-Match (ETag) alone.
        self.use_version_id = use_version_id
        if credentials not in ("instance_role", "default_chain"):
            raise ValueError("storage.s3.credentials must be instance_role or default_chain")
        self.region, self.credentials, self.endpoint_url = region, credentials, endpoint_url
        self.connect_timeout, self.read_timeout = connect_timeout, read_timeout
        self.max_pool_connections = max_pool_connections
        self._client_lock = threading.Lock()
        self._client_pid, self._client_obj, self._session_obj = None, None, None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_client_lock=None, _client_pid=None, _client_obj=None, _session_obj=None, _lock=None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._client_lock, self._lock = threading.Lock(), threading.Lock()

    def _session(self):
        try:
            import botocore.session
            from botocore.credentials import CredentialResolver, InstanceMetadataFetcher, InstanceMetadataProvider
        except ImportError:
            raise RemoteError("s3:// video sources need botocore: `uv pip install -r pyproject.toml --extra s3` "
                              "(README_nanovvm_video_pred.md, 'Remote Video Storage')") from None
        if self.credentials == "default_chain":
            return botocore.session.Session()
        # Instance role only: no profile, no shared config/credential files, no environment credentials.
        session = botocore.session.Session(session_vars={
            "profile": (None, None, None, None),
            "config_file": (None, None, os.devnull, None),
            "credentials_file": (None, None, os.devnull, None),
        })
        fetcher = InstanceMetadataFetcher(timeout=2, num_attempts=3, env={})
        session.register_component("credential_provider",
                                   CredentialResolver([InstanceMetadataProvider(iam_role_fetcher=fetcher)]))
        return session

    def _client(self):
        pid = os.getpid()
        if self._client_pid == pid and self._client_obj is not None:
            return self._client_obj
        with self._client_lock:
            if self._client_pid != pid or self._client_obj is None:
                from botocore.config import Config
                config = Config(region_name=self.region, connect_timeout=self.connect_timeout,
                                read_timeout=self.read_timeout, retries={"total_max_attempts": 1},
                                max_pool_connections=self.max_pool_connections,
                                ignore_configured_endpoint_urls=self.credentials == "instance_role")
                self._session_obj = self._session()
                self._client_obj = self._session_obj.create_client(
                    "s3", region_name=self.region, endpoint_url=self.endpoint_url, config=config)
                self._client_pid = pid
        return self._client_obj

    def check_credentials(self) -> str:
        """Which provider supplied credentials (never the credentials themselves)."""
        self._client()
        credentials = self._session_obj.get_credentials()
        if credentials is None:
            raise RemotePermissionError(self._credential_hint("no credentials found"))
        return getattr(credentials, "method", "unknown")

    def _credential_hint(self, message):
        if self.credentials == "instance_role":
            return (f"{message}; storage.s3.credentials=instance_role reads only the EC2 instance profile through "
                    "the instance metadata service (169.254.169.254). Check that the node has an instance role "
                    "with s3:GetObject/s3:ListBucket on this prefix, or use credentials: default_chain elsewhere.")
        return f"{message}; storage.s3.credentials=default_chain uses botocore's standard credential chain."

    def _classify(self, error, url):
        import botocore.exceptions as be
        if isinstance(error, (be.NoCredentialsError, be.PartialCredentialsError)):
            raise RemotePermissionError(self._credential_hint(f"{url}: {error}")) from None
        if isinstance(error, be.ClientError):
            code = error.response.get("Error", {}).get("Code", "")
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 404 or code in ("NoSuchKey", "NotFound", "NoSuchVersion", "NoSuchBucket", "404"):
                raise RemoteNotFound(f"{url}: no such object or version ({code or status})") from None
            if status == 403 or code in ("AccessDenied", "403", "InvalidAccessKeyId", "SignatureDoesNotMatch"):
                raise RemotePermissionError(self._credential_hint(f"{url}: access denied ({code or status})")) from None
            if status == 412 or code == "PreconditionFailed":
                raise StaleRemoteError(f"{url}: precondition failed; the object changed since its identity was "
                                       "recorded (rebuild or re-verify the index)") from None
            if status == 416 or code == "InvalidRange":
                raise InvalidRangeError(f"{url}: invalid range ({code or status})") from None
            if (status or 0) >= 500 or code in _TRANSIENT_CODES:
                raise _Transient(f"{code or status}: {error}") from None
            raise RemoteError(f"{url}: {code or status}: {error}") from None
        transient = (be.EndpointConnectionError, be.ConnectionClosedError, be.ReadTimeoutError,
                     be.ConnectTimeoutError, be.IncompleteReadError, be.ResponseStreamingError, ConnectionError,
                     TimeoutError, OSError)
        if isinstance(error, transient) or type(error).__name__ in ("ProtocolError", "SSLError"):
            raise _Transient(f"{type(error).__name__}: {error}") from None
        raise error

    def _raw_head(self, url):
        _, bucket, key = parse_url(url)
        try:
            response = self._client().head_object(Bucket=bucket, Key=key)
        except Exception as error:  # noqa: BLE001 - classified below
            self._classify(error, url)
        return ObjectInfo(url, int(response["ContentLength"]), response.get("ETag"), response.get("VersionId"),
                          str(response.get("LastModified")))

    def _raw_read(self, url, start, length, sink, version_id, if_match):
        _, bucket, key = parse_url(url)
        end = start + length - 1
        kwargs = {"Bucket": bucket, "Key": key, "Range": f"bytes={start}-{end}"}
        if version_id and self.use_version_id:
            kwargs["VersionId"] = version_id
        if if_match:
            kwargs["IfMatch"] = if_match
        try:
            response = self._client().get_object(**kwargs)
            content_range = response.get("ContentRange") or ""
            expected = f"bytes {start}-{end}/"
            if not content_range.startswith(expected):
                response["Body"].close()
                raise InvalidRangeError(f"{url}: asked for bytes {start}-{end}, the response covers "
                                        f"{content_range or 'the whole object (no Content-Range)'}")
            if int(response.get("ContentLength", -1)) != length:
                response["Body"].close()
                raise InvalidRangeError(f"{url}: Content-Length {response.get('ContentLength')} for a "
                                        f"{length}-byte range")
            body, received = response["Body"], 0
            try:
                while received < length:
                    data = body.read(min(CHUNK, length - received))
                    if not data:
                        break
                    sink.write(data)
                    received += len(data)
            finally:
                body.close()
            return received
        except (RemoteError, _Transient):
            raise
        except Exception as error:  # noqa: BLE001 - classified below
            self._classify(error, url)

    def _raw_list(self, prefix_url):
        _, bucket, key = parse_url(prefix_url)
        prefix = key.rstrip("/") + "/" if key else ""
        token = None
        while True:
            kwargs = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            response = self._attempts(f"LIST {prefix_url}", lambda: self._list_page(prefix_url, kwargs))
            for item in response.get("Contents", []):
                yield f"s3://{bucket}/{item['Key']}", int(item["Size"]), item.get("ETag")
            if not response.get("IsTruncated"):
                return
            token = response["NextContinuationToken"]

    def _list_page(self, url, kwargs):
        try:
            return self._client().list_objects_v2(**kwargs)
        except Exception as error:  # noqa: BLE001
            self._classify(error, url)

    def _raw_put_new(self, url, path):
        _, bucket, key = parse_url(url)
        try:
            self._raw_head(url)
        except RemoteNotFound:
            pass
        else:
            raise RemoteError(f"{url} already exists; refusing to overwrite it")
        try:
            with open(path, "rb") as body:
                self._client().put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch="*")
        except Exception as error:  # noqa: BLE001
            import botocore.exceptions as be
            if isinstance(error, be.ClientError) and error.response.get("Error", {}).get("Code") in (
                    "PreconditionFailed", "ConditionalRequestConflict"):
                raise RemoteError(f"{url} already exists; refusing to overwrite it") from None
            self._classify(error, url)
        return self._raw_head(url)
