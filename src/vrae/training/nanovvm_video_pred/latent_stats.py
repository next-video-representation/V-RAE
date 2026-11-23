"""Per-channel statistics of the train-split future latents (never the first frame).

With a Wan2.2 VAE stage 1 the future latents come from the joint 17-frame encoding, and the same
statistics normalize the first-frame latent z0 the DiT reads. The per-channel mean/std of z0 over
the same clips is recorded in the metadata (``condition_latent``) for comparison only.

The normalizer records what it was fit on: the stage-1 structure and checkpoint,
the identity of every train manifest (configured path, SHA-256 and clip count
from its ``meta.yaml``), the image size and preprocessing, the frame protocol,
and the clip subset. ``validate_nanovvm_normalizer`` rejects a file whose
identity differs from the current config, so a stale file is never reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from vrae.config import load_config
from vrae.paths import ProjectPaths, find_project_root, load_project_paths
from vrae.training.common.contracts import structural_stage1_metadata
from vrae.training.common.distributed import initialize_distributed
from vrae.training.common.engine import load_frozen_latent_backend, load_frozen_stage1, stage1_type
from vrae.training.common.latent_norm import (
    DistributedLatentStats,
    LatentNormalizer,
    validate_normalizer_compatibility,
)
from vrae.training.nanovvm_video_pred.data import (
    DATASET_KEY,
    PREPROCESSING,
    build_nanovvm_dataset,
    collate_nanovvm,
    configure_nanovvm_storage,
    frame_protocol_metadata,
    resolve_split_sources,
)
from vrae.training.nanovvm_video_pred.latents import (
    encode_condition_and_future,
    encode_future,
    is_wan_backend,
)
from vrae.training.nanovvm_video_pred.manifest import read_meta

FUTURE_RELATIVE_FRAMES = [1, 16]


def train_source_identity(config: Mapping[str, Any], paths: ProjectPaths) -> list[dict[str, Any]]:
    """Identity of the train manifests from their ``meta.yaml`` files (no manifest load).

    The recorded SHA-256 is the one training checks against ``manifest.json``.
    """

    identity = []
    for item in resolve_split_sources(config, paths, "train"):
        meta = read_meta(item["manifest_meta"])
        identity.append(
            {
                "name": item["name"],
                "dataset": item["dataset"],
                "manifest": item["configured"]["manifest"],
                "manifest_sha256": str(meta.get("manifest", {}).get("sha256")),
                "manifest_split": meta.get("split"),
                "total_clips": int(meta.get("statistics", {}).get("total_clips", -1)),
            }
        )
    return identity


def identity_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def normalizer_context(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    stage1_metadata: Mapping[str, Any],
    future_latent_shape: list[int] | None = None,
) -> dict[str, Any]:
    """Everything a reusable future normalizer must agree on."""

    sources = train_source_identity(config, paths)
    image_size = config["data"].get("image_size", 256)
    if isinstance(image_size, int):
        image_size = [image_size, image_size]
    context = {
        "dataset": DATASET_KEY,
        "split": "train",
        "scope": "future",
        "future_relative_frames": FUTURE_RELATIVE_FRAMES,
        "clean_latent": True,
        "normalized": False,
        "stage1": structural_stage1_metadata(stage1_metadata),
        "stage1_checkpoint": str(config["stage1"]["checkpoint"]),
        "stage1_weights": str(
            config["stage1"].get("weights", "ema" if stage1_type(config) == "vrae" else "official")
        ),
        "stage1_precision": str(config["stage1"].get("precision", "fp32")),
        "train_sources": sources,
        "train_sources_sha256": identity_digest(sources),
        "image_size": [int(value) for value in image_size],
        "preprocessing": dict(config["data"].get("preprocessing", PREPROCESSING)),
        "frame_protocol": frame_protocol_metadata(),
    }
    if future_latent_shape is not None:
        context["future_latent_shape"] = [int(value) for value in future_latent_shape]
    return context


IDENTITY_FIELDS = (
    "stage1_weights",
    "stage1_precision",
    "train_sources_sha256",
    "image_size",
    "preprocessing",
    "frame_protocol",
)


def validate_nanovvm_normalizer(
    normalizer: LatentNormalizer,
    config: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    stage1_metadata: Mapping[str, Any],
) -> None:
    validate_normalizer_compatibility(
        normalizer,
        stage1_metadata=stage1_metadata,
        stage1_checkpoint=str(config["stage1"]["checkpoint"]),
        dataset=DATASET_KEY,
        split="train",
        scope="future",
    )
    expected = normalizer_context(config, paths, stage1_metadata=stage1_metadata)
    mismatches = [
        f"{field}: file={normalizer.metadata.get(field)!r}, config={expected[field]!r}"
        for field in IDENTITY_FIELDS
        if normalizer.metadata.get(field) != expected[field]
    ]
    if mismatches:
        raise ValueError(
            "NanoVVM future normalizer was fit on different data or stage-1 settings; delete it "
            "or point latent_normalizer.path elsewhere:\n" + "\n".join(mismatches)
        )
    channels = latent_width(stage1_metadata)
    if normalizer.mean.numel() != channels:
        raise ValueError(f"normalizer has {normalizer.mean.numel()} channels, expected {channels}")


def latent_width(stage1_metadata: Mapping[str, Any]) -> int:
    """Channels of the stage-1 latents: V-RAE hidden_size or Wan2.2 latent_channels."""

    if stage1_metadata.get("type") == "wan22_vae":
        return int(stage1_metadata["latent_channels"])
    return int(stage1_metadata["hidden_size"])


def fit_subset_indices(population: int, clips: object, seed: int) -> list[int]:
    """All indices, or a seeded uniform subset of ``clips`` indices in sorted order."""

    if clips is None or clips == "all":
        return list(range(population))
    count = int(clips)
    if count <= 0:
        raise ValueError("latent_normalizer.fit_clips must be positive or 'all'")
    if count >= population:
        return list(range(population))
    generator = torch.Generator().manual_seed(int(seed))
    return sorted(torch.randperm(population, generator=generator)[:count].tolist())


def compute_nanovvm_latent_stats(config: Mapping[str, Any], paths: ProjectPaths) -> Path | None:
    distributed = initialize_distributed()
    if stage1_type(config) == "vrae":
        stage1, _ = load_frozen_stage1(config, paths, distributed.device)
    else:
        stage1, _ = load_frozen_latent_backend(config, paths, distributed.device)
    stage1_metadata = stage1.metadata()
    wan = is_wan_backend(stage1)
    dataset = build_nanovvm_dataset(config, paths, split="train")
    settings = config["latent_normalizer"]
    selected = fit_subset_indices(
        len(dataset),
        settings.get("fit_clips", "all"),
        int(settings.get("fit_seed", 0)),
    )
    local_dataset = Subset(dataset, selected[distributed.rank :: distributed.world_size])
    loader = DataLoader(
        local_dataset,
        batch_size=int(settings.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 4)),
        pin_memory=bool(config["training"].get("pin_memory", True)),
        collate_fn=collate_nanovvm,
    )
    channels = latent_width(stage1_metadata)
    accumulator = DistributedLatentStats(channels, device=distributed.device)
    # Wan2.2: z0's own statistics, recorded for comparison (z0 is normalized with the future's).
    condition_accumulator = DistributedLatentStats(channels, device=distributed.device) if wan else None
    shape: list[int] | None = None
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if wan:
                condition, future = encode_condition_and_future(
                    stage1,
                    batch["condition"].to(distributed.device, non_blocking=True),
                    batch["future"].to(distributed.device, non_blocking=True),
                )
                condition_accumulator.update(condition[:, None])
            else:
                future = encode_future(
                    stage1, batch["future"].to(distributed.device, non_blocking=True)
                )
            shape = list(future.shape[1:])
            accumulator.update(future)
            if distributed.is_main and index % 50 == 0:
                print(
                    f"[latent-stats] batch {index + 1}/{len(loader)} future {tuple(future.shape)}",
                    flush=True,
                )
    metadata = normalizer_context(
        config,
        paths,
        stage1_metadata=stage1_metadata,
        future_latent_shape=shape,
    )
    metadata["fit"] = {
        "population": len(dataset),
        "clips": len(selected),
        "seed": int(settings.get("fit_seed", 0)),
        "selection": "all" if len(selected) == len(dataset) else "seeded_uniform_subset",
    }
    if condition_accumulator is not None:
        condition_stats = condition_accumulator.finalize()
        metadata["condition_latent"] = {
            "scope": "first_frame_z0",
            "relative_frame": 0,
            "use": "recorded only; the DiT input z0 is normalized with the future statistics",
            "mean": [float(value) for value in condition_stats.mean.tolist()],
            "std": [float(value) for value in condition_stats.std.tolist()],
        }
    normalizer = accumulator.finalize(metadata=metadata)
    if not distributed.is_main:
        return None
    output = paths.checkpoint(settings["path"], require_exists=False)
    return normalizer.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit NanoVVM train-future latent statistics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default=None)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    paths = load_project_paths(
        config, override=arguments.paths, project_root=find_project_root(arguments.config)
    )
    configure_nanovvm_storage(config, paths)  # VRAE_STORAGE_CONFIG: videos from S3 via the cache
    output = compute_nanovvm_latent_stats(config, paths)
    if output is not None:
        print(f"saved NanoVVM train-future latent statistics to {output}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "compute_nanovvm_latent_stats",
    "fit_subset_indices",
    "latent_width",
    "normalizer_context",
    "train_source_identity",
    "validate_nanovvm_normalizer",
]
