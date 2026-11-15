"""Load a machine-specific storage file and anchor it to the NanoVVM data root.

V-RAE configs are plain PyYAML and name NanoVVM videos relative to the data root, so the storage section
lives in its own file, selected per machine by VRAE_STORAGE_CONFIG (unset or ``local``: local files):

    VRAE_STORAGE_CONFIG=configs/storage/adobe_s3.yaml

The file holds exactly one top-level key, ``storage`` (vrae/storage/backend.py). Before parsing, the file's
text expands ``${NAME}`` and ``${NAME:-default}`` from the environment (defaults may nest one variable),
so one file serves every user and node. In storage.roots, a relative ``local`` or ``index`` path is taken
below the data root (data.data_root or datasets.nanovvm of --paths); absolute paths are kept. The training
config itself never changes, so a run and its resume behave the same with local or remote videos.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

STORAGE_ENV = "VRAE_STORAGE_CONFIG"
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _closing_brace(text: str, start: int) -> int:
    """Index of the ``}`` that closes the ``${`` opened just before ``start`` (nesting allowed)."""

    depth = 1
    index = start
    while index < len(text):
        if text.startswith("${", index):
            depth += 1
            index += 2
            continue
        if text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise ValueError(f"unclosed ${{ in storage file: {text[start - 2:start + 40]!r}")


def expand_environment(text: str, environ: Mapping[str, str] | None = None) -> str:
    """Replace ${NAME} and ${NAME:-default} (defaults may nest); an unset NAME without a
    default is an error. A default is only expanded when it is used."""

    environ = os.environ if environ is None else environ
    out: list[str] = []
    index = 0
    while True:
        found = text.find("${", index)
        if found < 0:
            out.append(text[index:])
            return "".join(out)
        out.append(text[index:found])
        end = _closing_brace(text, found + 2)
        body = text[found + 2 : end]
        name_match = _NAME.match(body)
        if name_match is None:
            raise ValueError(f"bad variable reference ${{{body}}} in storage file")
        name, rest = name_match.group(0), body[name_match.end() :]
        if rest and not rest.startswith(":-"):
            raise ValueError(f"bad variable reference ${{{body}}}: use ${{NAME}} or ${{NAME:-default}}")
        value = environ.get(name)
        if value not in (None, ""):
            out.append(value)
        elif rest:
            out.append(expand_environment(rest[2:], environ))
        else:
            raise ValueError(
                f"storage file needs environment variable {name} (or write ${{{name}:-default}})"
            )
        index = end + 1


def resolve_storage_file(value: str | os.PathLike, project_root: str | os.PathLike | None = None) -> Path:
    """The storage file: as given, else relative to the repository root."""

    path = Path(value).expanduser()
    if path.is_file() or path.is_absolute() or project_root is None:
        return path
    candidate = Path(project_root) / path
    return candidate if candidate.is_file() else path


def load_storage_file(
    value: str | os.PathLike,
    *,
    data_root: str | os.PathLike,
    project_root: str | os.PathLike | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The file's storage section with relative roots and indexes anchored at data_root (not validated)."""

    path = resolve_storage_file(value, project_root)
    if not path.is_file():
        raise FileNotFoundError(f"{STORAGE_ENV}={value}: no such storage file")
    loaded = yaml.safe_load(expand_environment(path.read_text(), environ))
    if not isinstance(loaded, dict) or set(loaded) != {"storage"}:
        raise ValueError(f"{path} must hold exactly one top-level key, storage")
    storage = loaded["storage"]
    if not isinstance(storage, dict):
        raise ValueError(f"{path}: storage must be a mapping")
    storage = dict(storage)
    base = Path(os.path.abspath(os.path.expanduser(str(data_root))))
    roots = storage.get("roots")
    if isinstance(roots, list):
        anchored = []
        for root in roots:
            if isinstance(root, dict):
                root = dict(root)
                for key in ("local", "index"):
                    if isinstance(root.get(key), str) and root[key]:
                        candidate = Path(root[key]).expanduser()
                        root[key] = str(candidate if candidate.is_absolute() else base / candidate)
            anchored.append(root)
        storage["roots"] = anchored
    return storage


def storage_section_from_environment(
    *,
    data_root: str | os.PathLike,
    project_root: str | os.PathLike | None = None,
    override: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """The storage section VRAE_STORAGE_CONFIG (or override) selects; None means local files."""

    environ = os.environ if environ is None else environ
    value = environ.get(STORAGE_ENV, "") if override is None else override
    if not value or value == "local":
        return None
    return load_storage_file(value, data_root=data_root, project_root=project_root, environ=environ)


__all__ = [
    "STORAGE_ENV",
    "expand_environment",
    "load_storage_file",
    "resolve_storage_file",
    "storage_section_from_environment",
]
