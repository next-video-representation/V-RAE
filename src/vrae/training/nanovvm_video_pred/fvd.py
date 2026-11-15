"""In-training and offline FVD of first-frame conditioned NanoVVM prediction, with NanoVVM's protocol.

The ``fvd`` section of a config follows NanoVVM's (configs/scaling-ladder/*: interval 10000, the val
and train_eval holdout manifests, I3D / VideoMAE-v2 / V-JEPA 2.1, controls, 16 scored frames), with
V-RAE's path conventions: source paths are relative to the NanoVVM data root, like data.train/val,
and encoder checkpoints are resolved like stage1.checkpoint (``ckpts/...``). The engine is
vrae.evaluation.nanovvm_fvd.fvd. Offline, for an existing checkpoint:

    torchrun --standalone --nproc_per_node=<GPUs> -m vrae.training.nanovvm_video_pred.fvd \\
      --config <config> --paths <paths file> --checkpoint ckpts/nanovvm_video_pred/<run>/checkpoints/step-00010000.pt
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vrae.evaluation.nanovvm_fvd.fvd import (
    run_fvd_evaluation,
    split_configs,
    validate_fvd_config,
    validate_fvd_settings,
)
from vrae.evaluation.nanovvm_fvd.metrics import weight_identity
from vrae.paths import ProjectPaths
from vrae.training.nanovvm_video_pred.data import (
    FRAME_INTERVAL,
    IMAGE_SIZE,
    RGB_SEQ_LEN,
    _reject_interpolation,
    _resolve,
    resolve_data_root,
)
from vrae.training.nanovvm_video_pred.latents import (
    decode_future_tokens,
    encode_first_frame,
    encode_future,
)


def validate_fvd_section(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """File-free checks of config['fvd'] (None when the config has no FVD)."""

    section = config.get("fvd")
    if section is None:
        return None
    section = validate_fvd_settings(
        copy.deepcopy(dict(section)), frame_interval=FRAME_INTERVAL, num_frames=RGB_SEQ_LEN
    )
    if int(section.get("image_size", IMAGE_SIZE)) != IMAGE_SIZE:
        raise ValueError(f"fvd.image_size must be {IMAGE_SIZE}: frame 0 is the model's condition")
    return section


def resolve_fvd_config(config: Mapping[str, Any], paths: ProjectPaths) -> dict[str, Any] | None:
    """config['fvd'] with absolute paths: sources and raw roots below the data root, encoder
    weights like stage1.checkpoint. Checks every file the evaluation will read."""

    section = validate_fvd_section(config)
    if section is None:
        return None
    base = resolve_data_root(config, paths)
    for name, split in split_configs(section).items():
        if split.get("sources") is not None:
            resolved = []
            for index, source in enumerate(split["sources"]):
                item = dict(source)
                for key in ("root", "manifest", "manifest_meta"):
                    _reject_interpolation(str(item[key]), f"fvd.splits.{name}.sources[{index}].{key}")
                    item[key] = str(_resolve(str(item[key]), base))
                resolved.append(item)
            split["sources"] = resolved
        else:
            split["raw_video_root"] = str(_resolve(str(split["raw_video_root"]), base))
    for params in section["encoders"].values():
        params["checkpoint"] = str(paths.checkpoint(params["checkpoint"]))
        if params.get("source_path"):
            params["source_path"] = str(paths.checkpoint(params["source_path"]))
    if section.get("cache_dir"):
        section["cache_dir"] = str(paths.checkpoint(section["cache_dir"], require_exists=False))
    return validate_fvd_config(section, FRAME_INTERVAL, RGB_SEQ_LEN)


class VRAEFuturePredictor:
    """The frozen V-RAE and a first-frame DiT, in the engine's predictor interface."""

    def __init__(
        self,
        *,
        stage1: Any,
        dit: nn.Module,
        normalizer: Any,
        transport: Any,
        sampling: Mapping[str, Any],
        base_seed: int,
        autocast: Any,
        stage1_checkpoint: str | Path,
        device: torch.device,
    ) -> None:
        self.stage1, self.dit, self.normalizer, self.transport = stage1, dit, normalizer, transport
        self.sampling, self.base_seed, self.autocast = dict(sampling), int(base_seed), autocast
        self.stage1_checkpoint, self.device = Path(stage1_checkpoint), torch.device(device)

    def identity(self) -> dict[str, Any]:
        from vrae.training.common.contracts import structural_stage1_metadata

        return {
            "vrae_stage1": weight_identity(self.stage1_checkpoint),
            "structure": structural_stage1_metadata(self.stage1.metadata()),
        }

    @torch.no_grad()
    def encode_condition(self, rgb: torch.Tensor) -> torch.Tensor:
        return encode_first_frame(self.stage1, rgb.to(self.device))

    @torch.no_grad()
    def generate(self, frame_features: torch.Tensor, sample_indices: Sequence[int]) -> torch.Tensor:
        from vrae.training.nanovvm_video_pred.sample import predict_future_tokens

        with self.autocast():
            tokens = predict_future_tokens(
                self.dit,
                frame_features,
                self.normalizer,
                sample_indices=list(sample_indices),
                base_seed=self.base_seed,
                steps=int(self.sampling["steps"]),
                cfg_scale=float(self.sampling.get("cfg_scale", 1.0)),
                internal_guidance_scale=float(self.sampling.get("internal_guidance_scale", 1.0)),
                transport=self.transport,
            )
            return decode_future_tokens(self.stage1, tokens, grid_size=tuple(self.dit.grid_size))

    @torch.no_grad()
    def reconstruct_future(self, future_rgb: torch.Tensor) -> torch.Tensor:
        with self.autocast():
            tokens = encode_future(self.stage1, future_rgb.to(self.device))
            return decode_future_tokens(self.stage1, tokens, grid_size=tuple(self.dit.grid_size))


def evaluate(
    *,
    config: Mapping[str, Any],
    paths: ProjectPaths,
    fvd_config: Mapping[str, Any],
    stage1: Any,
    dit: nn.Module,
    normalizer: Any,
    transport: Any,
    autocast: Any,
    rank: int,
    world_size: int,
    device: torch.device,
    run_dir: Path,
    step: int,
    model_name: str,
) -> dict[str, float] | None:
    """One FVD evaluation with ``dit`` as it is (the caller swaps in EMA weights if wanted)."""

    predictor = VRAEFuturePredictor(
        stage1=stage1,
        dit=dit,
        normalizer=normalizer,
        transport=transport,
        sampling=config["sampling"],
        base_seed=int(fvd_config.get("seed", 42)),
        autocast=autocast,
        # stage 1 was loaded from this file, which therefore exists; hashed once for cache keys.
        stage1_checkpoint=paths.checkpoint(config["stage1"]["checkpoint"], require_exists=False),
        device=device,
    )
    was_training = dit.training
    dit.eval()
    try:
        return run_fvd_evaluation(
            predictor=predictor,
            rank=rank,
            world_size=world_size,
            device=device,
            run_dir=run_dir,
            global_step=step,
            fvd_config=fvd_config,
            num_frames=RGB_SEQ_LEN,
            frame_interval=FRAME_INTERVAL,
            model_name=model_name,
        )
    finally:
        dit.train(was_training)


def main() -> None:
    from vrae.config import load_config
    from vrae.paths import find_project_root, load_project_paths
    from vrae.training.common.distributed import initialize_distributed
    from vrae.training.common.precision import PrecisionPolicy
    from vrae.training.nanovvm_video_pred.data import configure_nanovvm_storage
    from vrae.training.nanovvm_video_pred.sample import load_sampling_stack
    from vrae.training.nanovvm_video_pred.train import build_flow_transport

    parser = argparse.ArgumentParser(description="NanoVVM-protocol FVD of a nanovvm_video_pred checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None,
                        help="directory for fvd_eval/ and fvd_cache/ (default: the checkpoint's run directory)")
    parser.add_argument("--ema", choices=("config", "on", "off"), default="config",
                        help="EMA weights: fvd.use_ema (config), or force on/off")
    parser.add_argument("--num-videos", type=int, default=None, help="score a seeded subset per split")
    parser.add_argument("--splits", nargs="+", default=None, help="only these fvd.splits")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    paths = load_project_paths(
        config, override=arguments.paths, project_root=find_project_root(arguments.config)
    )
    configure_nanovvm_storage(config, paths)
    fvd_config = resolve_fvd_config(config, paths)
    if fvd_config is None:
        raise SystemExit(f"{arguments.config} has no fvd section")
    if arguments.splits:
        fvd_config["splits"] = {name: fvd_config["splits"][name] for name in arguments.splits}
    if arguments.num_videos is not None:
        for split in fvd_config["splits"].values():
            split["num_videos"] = int(arguments.num_videos)
    use_ema = {"on": True, "off": False}.get(arguments.ema, bool(fvd_config.get("use_ema", False)))
    distributed = initialize_distributed()
    checkpoint = paths.checkpoint(arguments.checkpoint)
    stage1, dit, normalizer = load_sampling_stack(
        config, paths, checkpoint, distributed.device, use_ema=use_ema
    )
    step = int(torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)["step"])
    precision = PrecisionPolicy(str(config["training"].get("precision", "bf16")), distributed.device.type)
    output = Path(arguments.output) if arguments.output else checkpoint.parent.parent
    metrics = evaluate(
        config=config,
        paths=paths,
        fvd_config=fvd_config,
        stage1=stage1,
        dit=dit,
        normalizer=normalizer,
        transport=build_flow_transport(config),
        autocast=precision.autocast,
        rank=distributed.rank,
        world_size=distributed.world_size,
        device=distributed.device,
        run_dir=output,
        step=step,
        model_name="ema" if use_ema else "model",
    )
    if distributed.is_main:
        print(json.dumps({"step": step, **(metrics or {})}, indent=2), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["VRAEFuturePredictor", "evaluate", "resolve_fvd_config", "validate_fvd_section"]
