"""Check a NanoVVM prediction config against the files it names.

Without flags: resolve every train/val source against the data root and check
that each video root, manifest, and meta.yaml exists, reading only meta.yaml.
With ``stage1.type: wan22_vae`` it also checks that the Wan2.2 VAE weights exist.
--load-manifests also runs the full manifest validation (schema, sampling,
frame ranges, record count, SHA-256) and checks each source's first video.
--decode N decodes the first clip of each source for N sources and reports the
source frames of the condition and the future.

With VRAE_STORAGE_CONFIG (remote video storage) it also checks the S3 credentials,
the identity of every archive in use against its index, and the cache budget;
--load-manifests then resolves every manifest video in its index (no downloads),
and --decode fetches the decoded videos through the node cache. It ends with
"storage check passed".
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from vrae.config import load_config
from vrae.paths import find_project_root, load_project_paths
from vrae.storage.backend import StorageBackend, active_storage
from vrae.training.common.engine import stage1_type
from vrae.training.nanovvm_video_pred.data import (
    NanoVVMVideoPredictionDataset,
    configure_nanovvm_storage,
    load_split_sources,
    resolve_data_root,
    resolve_split_sources,
    storage_preflight,
)
from vrae.training.nanovvm_video_pred.manifest import read_meta
from vrae.training.nanovvm_video_pred.train import validate_build


def check_config(
    config: dict[str, Any],
    paths: Any,
    *,
    load_manifests: bool = False,
    decode: int = 0,
    storage: StorageBackend | None = None,
) -> dict[str, Any]:
    validate_build(config)
    storage = active_storage() if storage is None else storage
    report: dict[str, Any] = {"data_root": str(resolve_data_root(config, paths)), "splits": {}}
    if stage1_type(config) == "wan22_vae":
        # The Wan2.2 VAE weights (Wan2.2_VAE.pth of Wan-AI/Wan2.2-TI2V-5B, a symlink under ckpts/).
        weights = paths.checkpoint(config["stage1"]["checkpoint"])
        if not weights.resolve().is_file():
            raise FileNotFoundError(f"Wan2.2 VAE weights are not a file: {weights}")
        report["stage1"] = {
            "type": "wan22_vae",
            "checkpoint": str(weights),
            "resolved": str(weights.resolve()),
            "bytes": weights.resolve().stat().st_size,
        }
    splits = ["train"] + (["val"] if config["data"].get("val") is not None else [])
    if storage.remote:
        roots = [
            item["root"]
            for split in splits
            for item in resolve_split_sources(config, paths, split, storage)
        ]
        report["storage"] = storage_preflight(storage, roots, owner=False)
    for split in splits:
        entries = []
        for item in resolve_split_sources(config, paths, split, storage):
            meta = read_meta(item["manifest_meta"])
            entries.append(
                {
                    "name": item["name"],
                    "dataset": item["dataset"],
                    "root": str(item["root"]),
                    "manifest": str(item["manifest"]),
                    "manifest_meta": str(item["manifest_meta"]),
                    "meta_split": meta.get("split"),
                    "meta_total_clips": meta.get("statistics", {}).get("total_clips"),
                    "meta_sampling": {
                        key: meta.get("sampling", {}).get(key)
                        for key in ("rgb_seq_len", "frame_interval", "source_frame_span")
                    },
                }
            )
        summary: dict[str, Any] = {
            "sources": entries,
            "meta_total_clips": sum(int(entry["meta_total_clips"] or 0) for entry in entries),
        }
        if load_manifests or decode:
            sources = load_split_sources(config, paths, split, storage)
            dataset = NanoVVMVideoPredictionDataset(
                sources,
                split=split,
                image_size=int(config["data"].get("image_size", 256)),
                preprocessing=config["data"].get("preprocessing"),
                storage=storage,
            )
            summary["validated_clips"] = len(dataset)
            summary["shares"] = dataset.summary()
            decoded = []
            offset = 0
            for source in sources[: max(0, int(decode))]:
                item = dataset[offset]
                decoded.append(
                    {
                        "sample_id": item["sample_id"],
                        "condition_source_frame": int(item["frame_indices"][0]),
                        "future_source_frames": [int(v) for v in item["frame_indices"][1:]],
                        "condition": [list(item["condition"].shape), str(item["condition"].dtype)],
                        "future": [list(item["future"].shape), str(item["future"].dtype)],
                    }
                )
                offset += len(source)
            if decoded:
                summary["decoded"] = decoded
        report["splits"][split] = summary
    if storage.remote:
        report["storage_check"] = (
            "every manifest video resolves in its index"
            if load_manifests or decode
            else "roots and archives only; add --load-manifests to resolve every manifest video"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check NanoVVM prediction data paths and manifests"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default=None)
    parser.add_argument("--load-manifests", action="store_true")
    parser.add_argument("--decode", type=int, default=0, help="decode the first clip of N sources")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    paths = load_project_paths(
        config, override=arguments.paths, project_root=find_project_root(arguments.config)
    )
    storage = configure_nanovvm_storage(config, paths)
    report = check_config(
        config,
        paths,
        load_manifests=arguments.load_manifests,
        decode=arguments.decode,
        storage=storage,
    )
    if storage.remote:
        report["cache_stats"] = storage.stats()
    print(json.dumps(report, indent=2))
    if storage.remote:
        print(f"storage check passed ({report['storage_check']})", flush=True)


if __name__ == "__main__":
    main()
