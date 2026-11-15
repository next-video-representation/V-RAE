"""Raw-video storage: local files by default, or uncompressed tars on S3 read on demand through a bounded
node-local cache (README_nanovvm_video_pred.md, "Remote Video Storage").

The modules remote_io, tar_index, video_cache, prefetch, sqlite_fork and backend are vendored from NanoVVM
(data/, commit ca80730) with imports adapted. The cache directory format is shared, so NanoVVM and V-RAE on
one node can use one cache. Importing this package never imports botocore; only s3:// reads do.
"""

from vrae.storage.backend import (
    StorageBackend,
    active_storage,
    configure_storage,
    is_storage_error,
    open_video,
    validate_storage,
)
from vrae.storage.config import STORAGE_ENV, load_storage_file, storage_section_from_environment

__all__ = [
    "STORAGE_ENV",
    "StorageBackend",
    "active_storage",
    "configure_storage",
    "is_storage_error",
    "load_storage_file",
    "open_video",
    "storage_section_from_environment",
    "validate_storage",
]
