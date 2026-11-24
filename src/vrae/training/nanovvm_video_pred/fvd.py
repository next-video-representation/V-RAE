"""In-training and offline FVD of first-frame conditioned NanoVVM prediction, with NanoVVM's protocol.

The ``fvd`` section of a config follows NanoVVM's (configs/scaling-ladder/*: interval 10000, the val
and train_eval holdout manifests, I3D / VideoMAE-v2 / V-JEPA 2.1, controls, 16 scored frames), with
V-RAE's path conventions: source paths are relative to the NanoVVM data root, like data.train/val,
and encoder checkpoints are resolved like stage1.checkpoint (``ckpts/...``). The engine is
vrae.evaluation.nanovvm_fvd.fvd. Offline, for an existing checkpoint:

    torchrun --standalone --nproc_per_node=<GPUs> -m vrae.training.nanovvm_video_pred.fvd \\
      --config <config> --paths <paths file> --checkpoint ckpts/nanovvm_video_pred/<run>/checkpoints/step-00010000.pt

The stage-1 controls alone (ae_reconstruction and static vs real; no DiT checkpoint, no latent
statistics), e.g. the reconstruction ceiling of a stage 1 at the config's resolution:

    torchrun --standalone --nproc_per_node=<GPUs> -m vrae.training.nanovvm_video_pred.fvd \\
      --config <config> --paths <paths file> --controls-only --splits val --output <directory>

With a Wan2.2 VAE stage 1 the controls encode and decode frame 0 with the 16 frames as one 17-frame
clip (the VAE is causal); the static control is frame 0 repeated 17 times.
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
    RGB_SEQ_LEN,
    _reject_interpolation,
    _resolve,
    configured_image_size,
    resolve_data_root,
)
from vrae.training.nanovvm_video_pred.latents import (
    condition_features,
    decode_future_tokens,
    encode_condition_and_future,
    encode_first_frame,
    encode_future,
    is_wan_backend,
)


def validate_fvd_section(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """File-free checks of config['fvd'] (None when the config has no FVD)."""

    section = config.get("fvd")
    if section is None:
        return None
    section = validate_fvd_settings(
        copy.deepcopy(dict(section)), frame_interval=FRAME_INTERVAL, num_frames=RGB_SEQ_LEN
    )
    data_size = configured_image_size(config.get("data") or {})
    section.setdefault("image_size", data_size)  # an unset fvd.image_size follows the training data
    if int(section["image_size"]) != data_size:
        raise ValueError(
            f"fvd.image_size must equal data.image_size ({data_size}): frame 0 is the model's condition"
        )
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
    """The frozen stage 1 (V-RAE or Wan2.2 VAE) and a first-frame DiT, in the engine's predictor
    interface. ``dit`` may be None for the controls alone (then ``grid_size`` is required)."""

    def __init__(
        self,
        *,
        stage1: Any,
        dit: nn.Module | None,
        normalizer: Any,
        transport: Any,
        sampling: Mapping[str, Any],
        base_seed: int,
        autocast: Any,
        stage1_checkpoint: str | Path,
        device: torch.device,
        grid_size: Sequence[int] | None = None,
    ) -> None:
        self.stage1, self.dit, self.normalizer, self.transport = stage1, dit, normalizer, transport
        self.sampling, self.base_seed, self.autocast = dict(sampling), int(base_seed), autocast
        self.stage1_checkpoint, self.device = Path(stage1_checkpoint), torch.device(device)
        if grid_size is None:
            if dit is None:
                raise ValueError("grid_size is required without a DiT")
            grid_size = dit.grid_size
        self.grid_size = tuple(int(value) for value in grid_size)

    def identity(self) -> dict[str, Any]:
        from vrae.training.common.contracts import structural_stage1_metadata

        structure = structural_stage1_metadata(self.stage1.metadata())
        if is_wan_backend(self.stage1):
            # Distinct from the V-RAE identity, so the control caches never mix.
            return {
                "backend": "wan22_vae",
                "wan22_vae": weight_identity(self.stage1_checkpoint),
                "structure": structure,
            }
        return {"vrae_stage1": weight_identity(self.stage1_checkpoint), "structure": structure}

    @torch.no_grad()
    def encode_condition(self, rgb: torch.Tensor) -> torch.Tensor:
        """V-RAE frame features, or the raw Wan2.2 z0."""

        return encode_first_frame(self.stage1, rgb.to(self.device))

    @torch.no_grad()
    def generate(self, condition: torch.Tensor, sample_indices: Sequence[int]) -> torch.Tensor:
        from vrae.training.nanovvm_video_pred.sample import predict_future_tokens

        if self.dit is None:
            raise RuntimeError("this predictor has no DiT (controls only)")
        frame_features = condition_features(self.stage1, condition, self.normalizer)
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
                # MeanFlow only; None (absent) leaves a flow-matching call unchanged.
                guidance_interval=self.sampling.get("guidance_interval"),
                time_grid=self.sampling.get("time_grid"),
            )
            return decode_future_tokens(
                self.stage1, tokens, grid_size=self.grid_size, condition=condition
            )

    @torch.no_grad()
    def reconstruct_future(
        self, future_rgb: torch.Tensor, condition_rgb: torch.Tensor | None = None
    ) -> torch.Tensor:
        """16 frames through the frozen stage 1. The V-RAE ignores ``condition_rgb``; the Wan2.2
        VAE encodes and decodes it with the 16 frames as one causal 17-frame clip."""

        with self.autocast():
            if is_wan_backend(self.stage1):
                if condition_rgb is None:
                    raise ValueError("the Wan2.2 VAE reconstructs the future together with frame 0")
                condition, tokens = encode_condition_and_future(
                    self.stage1, condition_rgb.to(self.device), future_rgb.to(self.device)
                )
                return decode_future_tokens(
                    self.stage1, tokens, grid_size=self.grid_size, condition=condition
                )
            tokens = encode_future(self.stage1, future_rgb.to(self.device))
            return decode_future_tokens(self.stage1, tokens, grid_size=self.grid_size)


def evaluate(
    *,
    config: Mapping[str, Any],
    paths: ProjectPaths,
    fvd_config: Mapping[str, Any],
    stage1: Any,
    dit: nn.Module | None,
    normalizer: Any,
    transport: Any,
    autocast: Any,
    rank: int,
    world_size: int,
    device: torch.device,
    run_dir: Path,
    step: int,
    model_name: str,
    controls_only: bool = False,
) -> dict[str, float] | None:
    """One FVD evaluation with ``dit`` as it is (the caller swaps in EMA weights if wanted).

    ``controls_only``: no DiT (``dit`` may be None); only ae_reconstruction and static vs real.
    """

    grid_size = None
    if dit is None:
        from vrae.training.nanovvm_video_pred.train import task_geometry  # train imports this module

        grid_size = task_geometry(config)["latent_grid"]
    predictor = VRAEFuturePredictor(
        stage1=stage1,
        dit=dit,
        normalizer=normalizer,
        transport=transport,
        sampling=config["sampling"],
        base_seed=int(fvd_config.get("seed", 42)),
        autocast=autocast,
        # stage 1 was loaded from this file, which therefore exists; hashed once for cache keys.
        # The Wan2.2 weights are a symlink into ckpts/, which only an existing-path lookup follows.
        stage1_checkpoint=paths.checkpoint(
            config["stage1"]["checkpoint"], require_exists=is_wan_backend(stage1)
        ),
        device=device,
        grid_size=grid_size,
    )
    if controls_only:
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
            controls_only=True,
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
    from vrae.training.nanovvm_video_pred.sample import (
        add_meanflow_sampling_arguments,
        load_sampling_stack,
    )
    from vrae.training.nanovvm_video_pred.train import build_flow_transport, load_latent_backend

    parser = argparse.ArgumentParser(description="NanoVVM-protocol FVD of a nanovvm_video_pred checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default=None)
    parser.add_argument("--checkpoint", default=None, help="DiT checkpoint (not with --controls-only)")
    parser.add_argument("--controls-only", action="store_true",
                        help="score only the stage-1 controls (ae_reconstruction, static) against real; "
                             "needs no DiT checkpoint and no latent statistics; requires --output")
    parser.add_argument("--output", default=None,
                        help="directory for fvd_eval/ and fvd_cache/ (default: the checkpoint's run directory)")
    parser.add_argument("--ema", choices=("config", "on", "off"), default="config",
                        help="EMA weights: fvd.use_ema (config), or force on/off")
    parser.add_argument("--num-videos", type=int, default=None, help="score a seeded subset per split")
    parser.add_argument("--splits", nargs="+", default=None, help="only these fvd.splits")
    add_meanflow_sampling_arguments(parser)
    arguments = parser.parse_args()
    if arguments.controls_only == (arguments.checkpoint is not None):
        parser.error("pass exactly one of --checkpoint or --controls-only")
    if arguments.controls_only and arguments.output is None:
        parser.error("--controls-only needs --output (there is no checkpoint run directory)")
    config = load_config(arguments.config)
    if arguments.guidance_interval is not None:
        config["sampling"]["guidance_interval"] = list(arguments.guidance_interval)
    if arguments.time_grid is not None:
        config["sampling"]["time_grid"] = arguments.time_grid
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
    precision = PrecisionPolicy(str(config["training"].get("precision", "bf16")), distributed.device.type)
    if arguments.controls_only:
        stage1, _ = load_latent_backend(config, paths, distributed.device)
        metrics = evaluate(
            config=config,
            paths=paths,
            fvd_config=fvd_config,
            stage1=stage1,
            dit=None,
            normalizer=None,
            transport=None,
            autocast=precision.autocast,
            rank=distributed.rank,
            world_size=distributed.world_size,
            device=distributed.device,
            run_dir=Path(arguments.output),
            step=0,
            model_name="none",
            controls_only=True,
        )
        if distributed.is_main:
            print(json.dumps({"controls_only": True, **(metrics or {})}, indent=2), flush=True)
        return
    checkpoint = paths.checkpoint(arguments.checkpoint)
    stage1, dit, normalizer = load_sampling_stack(
        config, paths, checkpoint, distributed.device, use_ema=use_ema
    )
    step = int(torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)["step"])
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
