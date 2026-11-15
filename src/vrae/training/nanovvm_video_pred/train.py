"""First-frame conditioned 16-frame future prediction on NanoVVM clip manifests.

One observed RGB frame conditions the DiT through spatial AdaLN; the DiT
generates the four V-RAE latent chunks of the next sixteen frames. Noise,
supervision, and the full/base flow-matching losses act on the future chunks only.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from vrae.config import load_config
from vrae.paths import ProjectPaths, find_project_root, load_project_paths
from vrae.registry import DIT_MODELS, register_builtin_models
from vrae.storage.backend import active_storage
from vrae.storage.metrics import cache_metrics, timed_batches
from vrae.training.common.accumulation import GradientAccumulator, clip_and_step
from vrae.training.common.checkpoint import (
    build_training_checkpoint,
    load_model_init,
    resume_training,
    save_training_checkpoint,
)
from vrae.training.common.contracts import (
    load_run_metadata,
    resolve_batch_contract,
    update_decoder_execution_metadata,
    update_run_metadata,
)
from vrae.training.common.distributed import gather_rng_states, initialize_distributed, reduce_mean
from vrae.training.common.ema import ExponentialMovingAverage
from vrae.training.common.engine import (
    build_flow_transport,
    dit_architecture_metadata,
    flow_transport_metadata,
    load_frozen_stage1,
    parse_config_argument,
    prepare_run,
    seed_everything,
)
from vrae.training.common.latent_norm import LatentNormalizer
from vrae.training.common.optim import (
    build_optimizer,
    build_scheduler,
    resolve_scheduler_steps,
    uses_ratio_schedule,
    validate_ratio_schedule,
)
from vrae.training.common.precision import PrecisionPolicy
from vrae.training.common.visualization import comparison_video, save_sample_tensors
from vrae.training.common.wandb import WandbLogger
from vrae.training.nanovvm_video_pred.data import (
    CONDITION_FRAMES,
    DATASET_KEY,
    FUTURE_FRAMES,
    IMAGE_SIZE,
    PREPROCESSING,
    RGB_SEQ_LEN,
    VIDEO_DECODERS,
    build_nanovvm_dataset,
    build_nanovvm_loader,
    configure_nanovvm_storage,
    frame_protocol_metadata,
    resolve_split_sources,
    split_source_entries,
    storage_preflight,
    validate_frame_protocol,
    validate_preprocessing,
)
from vrae.training.nanovvm_video_pred.latents import encode_condition_and_future

TASK = "nanovvm_video_pred"
DIT_NAME = "vrae_first_frame_prediction_dit"
TEMPORAL_COMPRESSION = 4
SINGLE_IMAGE_ENCODERS = ("eupe", "dinov3", "siglip2")


def _image_size(data: Mapping[str, Any]) -> int:
    value = data.get("image_size", IMAGE_SIZE)
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or int(value[0]) != int(value[1]):
            raise ValueError("data.image_size must be square for the 256x256 center crop")
        value = value[0]
    return int(value)


def task_geometry(config: Mapping[str, Any]) -> dict[str, Any]:
    """Shapes derived from the encoder, image size, and future frame count."""

    encoder = config["model"]["encoder"]
    image_size = _image_size(config["data"])
    patch_size = int(encoder["patch_size"])
    if image_size % patch_size:
        raise ValueError("image_size must be divisible by the encoder patch size")
    tubelet = int(encoder.get("encoder_tubelet_size", 1))
    group = int(config["model"]["pooling"]["group_size"])
    if tubelet * group != TEMPORAL_COMPRESSION:
        raise ValueError("V-RAE must compress time by four")
    if FUTURE_FRAMES % TEMPORAL_COMPRESSION:
        raise ValueError("future frames must fill whole V-RAE chunks")
    grid = (image_size // patch_size, image_size // patch_size)
    channels = int(encoder["hidden_size"])
    chunks = FUTURE_FRAMES // TEMPORAL_COMPRESSION
    tokens = grid[0] * grid[1]
    dit_patch = config["dit"].get("patch_size", 1)
    dit_patch = (
        (int(dit_patch), int(dit_patch))
        if isinstance(dit_patch, int)
        else tuple(int(value) for value in dit_patch)
    )
    if grid[0] % dit_patch[0] or grid[1] % dit_patch[1]:
        raise ValueError("the latent grid must be divisible by dit.patch_size")
    patches = (grid[0] // dit_patch[0]) * (grid[1] // dit_patch[1])
    hidden = config["dit"].get("hidden_size", [1536, 2048])
    encoder_width = int(hidden if isinstance(hidden, int) else hidden[0])
    return {
        "image_size": image_size,
        "latent_grid": list(grid),
        "latent_channels": channels,
        "future_chunks": chunks,
        "tokens_per_chunk": tokens,
        "dit_patch_size": list(dit_patch),
        "dit_patches_per_chunk": patches,
        "dit_encoder_width": encoder_width,
        "time_dist_shift": math.sqrt(chunks * tokens * channels / 4096),
    }


def validate_training_recipe(training: Mapping[str, Any]) -> None:
    """Checks that need no data: the ratio schedule's bounds and the clipping norm."""

    scheduler = training.get("scheduler", {})
    if uses_ratio_schedule(scheduler):
        validate_ratio_schedule(
            scheduler, peak_lr=float(training.get("optimizer", {}).get("lr", 1.0e-4))
        )
        epochs = training.get("epochs", 1)
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("training.epochs must be a positive integer")
    clip_grad = training.get("clip_grad")
    if clip_grad is not None and not float(clip_grad) > 0:
        raise ValueError("training.clip_grad must be positive (or null for no clipping)")


def validate_build(config: Mapping[str, Any]) -> dict[str, Any]:
    """Static checks and the shape contract; reads no files (see check_data for paths)."""

    from vrae.training.common.wandb import validate_wandb_config

    if "wandb" in config:
        validate_wandb_config(config["wandb"])
    if str(config.get("task")) != TASK:
        raise ValueError(f"NanoVVM entry requires task={TASK}")
    data = config["data"]
    if str(data.get("dataset")) != DATASET_KEY:
        raise ValueError(f"data.dataset must be {DATASET_KEY}")
    validate_frame_protocol(data)
    preprocessing = data.get("preprocessing", PREPROCESSING)
    validate_preprocessing(preprocessing)
    if dict(preprocessing) != PREPROCESSING:
        raise ValueError(
            "data.preprocessing must be the NanoVVM protocol for aligned condition/future frames: "
            f"{PREPROCESSING}"
        )
    if _image_size(data) != IMAGE_SIZE:
        raise ValueError(f"data.image_size must be {IMAGE_SIZE}")
    if str(data.get("video_decoder", "decord")) not in VIDEO_DECODERS:
        raise ValueError(f"data.video_decoder must be one of {VIDEO_DECODERS}")
    train_sources = split_source_entries(config, "train")
    val_sources = split_source_entries(config, "val") if data.get("val") is not None else []
    encoder = config["model"]["encoder"]
    if str(encoder.get("name")) not in SINGLE_IMAGE_ENCODERS:
        raise ValueError(
            f"first-frame conditioning needs a per-frame image encoder {SINGLE_IMAGE_ENCODERS}, "
            f"got {encoder.get('name')!r}"
        )
    dit = config["dit"]
    if str(dit.get("name")) != DIT_NAME:
        raise ValueError(f"NanoVVM future prediction must use {DIT_NAME}")
    geometry = task_geometry(config)
    channels = geometry["latent_channels"]
    for key in ("input_dim", "in_channels", "frame_channels"):
        if key in dit and int(dit[key]) != channels:
            raise ValueError(f"dit.{key} must match the V-RAE hidden size {channels}")
    for key in ("future_chunks", "num_future_chunks", "num_chunks", "num_frames"):
        if key in dit and int(dit[key]) != geometry["future_chunks"]:
            raise ValueError(f"dit.{key} must be {geometry['future_chunks']} future chunks")
    for key in ("grid_size", "input_size", "height", "width"):
        if key in dit:
            raise ValueError(f"dit.{key} is derived from the encoder grid; remove it")
    validate_training_recipe(config["training"])
    transport = flow_transport_metadata(config)
    if transport["prediction"] != "x":
        raise ValueError("NanoVVM future prediction uses x-prediction")
    if abs(transport["time_dist_shift"] - geometry["time_dist_shift"]) > 1e-9:
        raise ValueError(
            "transport.time_dist_shift must be sqrt(future_chunks*h*w*C/4096) = "
            f"{geometry['time_dist_shift']!r} for the future latents"
        )
    batch = "B"
    grid = geometry["latent_grid"]
    patches = geometry["dit_patches_per_chunk"]
    width = geometry["dit_encoder_width"]
    return {
        "task": TASK,
        "dit": DIT_NAME,
        "frame_protocol": frame_protocol_metadata(),
        "image_size": [geometry["image_size"], geometry["image_size"]],
        "latent_grid": grid,
        "future_chunks": geometry["future_chunks"],
        "transport": transport,
        "expected_time_dist_shift": geometry["time_dist_shift"],
        "shapes": {
            "clip_rgb": [batch, RGB_SEQ_LEN, 3, geometry["image_size"], geometry["image_size"]],
            "condition_rgb": [batch, 3, geometry["image_size"], geometry["image_size"]],
            "future_rgb": [batch, FUTURE_FRAMES, 3, geometry["image_size"], geometry["image_size"]],
            "frame_features": [batch, geometry["tokens_per_chunk"], channels],
            "z_future": [batch, geometry["future_chunks"], geometry["tokens_per_chunk"], channels],
            "frame_condition": [batch, patches, width],
            "time_condition": [batch, 1, width],
            "adaln_condition": [batch, geometry["future_chunks"] * patches, width],
            "dit_sequence": [batch, geometry["future_chunks"] * patches, width],
        },
        "condition_frames": CONDITION_FRAMES,
        "train_sources": [entry["name"] for entry in train_sources],
        "val_sources": [entry["name"] for entry in val_sources],
    }


def ensure_latent_stats(config: Mapping[str, Any], paths: ProjectPaths) -> Path:
    """Fit the train-future normalizer when absent; an existing file is identity-checked later."""

    from vrae.training.common.distributed import barrier, broadcast_object
    from vrae.training.nanovvm_video_pred.latent_stats import compute_nanovvm_latent_stats

    distributed = initialize_distributed()
    output = paths.checkpoint(config["latent_normalizer"]["path"], require_exists=False)
    missing = None
    if distributed.is_main:
        missing = not output.is_file() or output.stat().st_size == 0
    missing = bool(broadcast_object(missing))
    if not missing:
        if distributed.is_main:
            print(f"[latent-stats] using existing file: {output}", flush=True)
        return output
    if distributed.is_main:
        print(
            f"[latent-stats] file missing; fitting NanoVVM train-future statistics: {output}",
            flush=True,
        )
    compute_nanovvm_latent_stats(config, paths)
    barrier()
    failure = None
    if distributed.is_main and (not output.is_file() or output.stat().st_size == 0):
        failure = f"Latent statistics were not created successfully: {output}"
    failure = broadcast_object(failure)
    if failure is not None:
        raise RuntimeError(str(failure))
    return output


def build_nanovvm_dit(config: Mapping[str, Any], stage1_metadata: Mapping[str, Any]) -> nn.Module:
    register_builtin_models()
    image_size = _image_size(config["data"])
    patch_size = int(stage1_metadata["patch_size"])
    if image_size % patch_size:
        raise ValueError("image size must be divisible by the V-RAE patch size")
    compression = int(stage1_metadata["temporal_compression_ratio"])
    if FUTURE_FRAMES % compression:
        raise ValueError("future frames must fill whole V-RAE chunks")
    grid_size = (image_size // patch_size, image_size // patch_size)
    channels = int(stage1_metadata["hidden_size"])
    return DIT_MODELS.build(
        config["dit"],
        grid_size=grid_size,
        in_channels=channels,
        frame_channels=channels,
        future_chunks=FUTURE_FRAMES // compression,
    )


def nanovvm_model_metadata(
    config: Mapping[str, Any],
    stage1_metadata: Mapping[str, Any],
    dit: nn.Module,
    normalizer: LatentNormalizer | None = None,
) -> dict[str, Any]:
    from vrae.training.common.contracts import structural_stage1_metadata
    from vrae.training.common.latent_norm import normalizer_identity

    return {
        "stage1": structural_stage1_metadata(stage1_metadata),
        "stage1_checkpoint": str(config["stage1"]["checkpoint"]),
        "latent_normalizer_source": str(config["latent_normalizer"]["path"]),
        "dit_name": DIT_NAME,
        "dit_input_dim": int(dit.in_channels),
        "dit_frame_channels": int(dit.frame_channels),
        "dit_grid_size": list(dit.grid_size),
        "future_chunks": int(dit.future_chunks),
        "frame_protocol": frame_protocol_metadata(),
        "image_size": [_image_size(config["data"])] * 2,
        "first_frame_condition": "pre_pool_patch_features_spatial_adaln",
        **dit_architecture_metadata(dit),
        "transport": flow_transport_metadata(config),
        "latent_normalizer_identity": (
            {
                **normalizer_identity(normalizer),
                "train_sources_sha256": normalizer.metadata.get("train_sources_sha256"),
            }
            if normalizer is not None
            else None
        ),
    }


def load_task_normalizer(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    stage1_metadata: Mapping[str, Any],
    device: torch.device,
) -> LatentNormalizer:
    from vrae.training.nanovvm_video_pred.latent_stats import validate_nanovvm_normalizer

    normalizer = LatentNormalizer.load(paths.checkpoint(config["latent_normalizer"]["path"]))
    validate_nanovvm_normalizer(normalizer, config, paths, stage1_metadata=stage1_metadata)
    return normalizer.to(device)


@torch.no_grad()
def prepare_nanovvm_latents(
    stage1: Any,
    batch: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RGB batch -> (first-frame features ``[B,h*w,C]``, raw future latents ``[B,4,h*w,C]``)."""

    condition = batch["condition"].to(device, non_blocking=True)
    future = batch["future"].to(device, non_blocking=True)
    return encode_condition_and_future(stage1, condition, future)


def rgb_unit(video: torch.Tensor) -> torch.Tensor:
    return video.float().div(255.0) if video.dtype == torch.uint8 else video.float()


# NanoVVM metric names (nanovvm docs/wandb_metrics.md) for this trainer's loss terms.
LOSS_TERMS = ("loss", "loss_full", "loss_base")
LOSS_METRICS = {
    "loss": "loss/train",
    "loss_full": "loss/train_full",
    "loss_base": "loss/train_base",
}


class TrainMetrics:
    """NanoVVM's running training metrics, kept on the device between log steps.

    A step's loss is the mean of its accumulation micro-batch losses, and ``loss/train``
    is the mean of the step losses since the previous log, averaged over ranks. The
    gradient statistics are window means of the pre-clip total norm, its RMS over the
    gradient elements, and ``clip_grad / max(norm, clip_grad)``. ``loss/train_epoch`` is
    the mean micro-batch loss of the epoch. ``log_values`` and ``end_epoch`` are
    collectives: every rank calls them at the same steps.
    """

    def __init__(self, device: torch.device, accumulation_steps: int) -> None:
        self.accumulation_steps = int(accumulation_steps)
        self._step = torch.zeros(len(LOSS_TERMS), dtype=torch.float64, device=device)
        self._window = torch.zeros_like(self._step)
        self._epoch = torch.zeros_like(self._step)
        self._grad = torch.zeros(3, dtype=torch.float64, device=device)
        self.window_steps = 0
        self.grad_steps = 0
        self.epoch_batches = 0

    def add_microbatch(self, losses: Mapping[str, torch.Tensor]) -> None:
        values = torch.stack([losses[name].detach().reshape(()) for name in LOSS_TERMS])
        self._step += values.double()
        self._epoch += values.double()
        self.epoch_batches += 1

    def end_step(
        self,
        *,
        stepped: bool,
        grad_norm: torch.Tensor | None,
        grad_numel: int,
        clip_grad: float | None,
    ) -> None:
        if stepped:
            self._window += self._step / self.accumulation_steps
            self.window_steps += 1
            if grad_norm is not None and clip_grad:
                norm = grad_norm.detach().double().reshape(())
                self._grad += torch.stack(
                    (
                        norm,
                        norm / math.sqrt(max(int(grad_numel), 1)),
                        float(clip_grad) / norm.clamp(min=float(clip_grad)),
                    )
                )
                self.grad_steps += 1
        self._step.zero_()

    def log_values(self, *, clip_grad: float | None) -> dict[str, float]:
        losses = (reduce_mean(self._window) / max(self.window_steps, 1)).tolist()
        values = {LOSS_METRICS[name]: value for name, value in zip(LOSS_TERMS, losses)}
        if clip_grad:
            # Gradients are identical on every rank after the DDP all-reduce.
            grad = (self._grad / max(self.grad_steps, 1)).tolist()
            values["optimizer/grad_norm"] = grad[0]
            values["optimizer/grad_rms"] = grad[1]
            values["optimizer/grad_clip_scale"] = grad[2]
        self._window.zero_()
        self._grad.zero_()
        self.window_steps = 0
        self.grad_steps = 0
        return values

    def end_epoch(self) -> dict[str, float] | None:
        batches = self.epoch_batches
        total = float(reduce_mean(self._epoch[0]).item())
        self._epoch.zero_()
        self.epoch_batches = 0
        if batches == 0:
            return None
        return {"loss/train_epoch": total / batches}


@torch.no_grad()
def parameter_metrics(module: nn.Module) -> dict[str, float]:
    """L2 norm and RMS over the trainable parameters (NanoVVM ``parameters/*``)."""

    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p.detach().float()) for p in parameters])
    ).item()
    numel = sum(parameter.numel() for parameter in parameters)
    return {"parameters/norm": norm, "parameters/rms": norm / math.sqrt(max(numel, 1))}


def _checkpoint_payload(
    *,
    sampler: Any,
    step: int,
    dit: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    accumulation: GradientAccumulator,
    config: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    rng_state: Mapping[str, Any],
) -> dict[str, Any]:
    return build_training_checkpoint(
        task=TASK,
        epoch=sampler.epoch,
        step=step,
        model=dit,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        data_state=sampler.state_dict(gradient_accumulation_microstep=accumulation.microstep),
        resolved_config=config,
        model_metadata=model_metadata,
        rng_state=rng_state,
    )


def train(
    config: dict[str, Any],
    paths: ProjectPaths,
    run: Path,
    *,
    max_steps: int | None = None,
    storage_report: list[str] | None = None,
) -> None:
    distributed = initialize_distributed()
    storage = active_storage()
    seed_everything(int(config["data"].get("seed", 3407)), rank=distributed.rank)
    stage1, _ = load_frozen_stage1(config, paths, distributed.device)
    stage1_metadata = stage1.metadata()
    if not stage1.supports_single_image:
        raise ValueError("the frozen V-RAE encoder has no single-image feature path")
    dit = build_nanovvm_dit(config, stage1_metadata).to(distributed.device)
    dit.requires_grad_(True)
    model: nn.Module = dit
    if distributed.world_size > 1:
        model = DistributedDataParallel(
            dit,
            device_ids=[distributed.local_rank] if distributed.device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    normalizer = load_task_normalizer(
        config, paths, stage1_metadata=stage1_metadata, device=distributed.device
    )
    model_metadata = nanovvm_model_metadata(config, stage1_metadata, dit, normalizer)
    optimizer = build_optimizer(dit.parameters(), config["training"]["optimizer"])
    ema = ExponentialMovingAverage(dit, decay=float(config["training"]["ema"]["decay"]))
    precision = PrecisionPolicy(
        str(config["training"].get("precision", "bf16")), distributed.device.type
    )
    scaler = precision.make_scaler()
    transport = build_flow_transport(config)
    dataset = build_nanovvm_dataset(config, paths, split="train")
    if distributed.is_main:
        print(f"[data] train: {len(dataset)} clips from {len(dataset.sources)} sources", flush=True)
        for line in dataset.summary():
            print(f"[data]   {line}", flush=True)
        for line in storage_report or []:
            print(f"[storage] Remote video storage: {line}", flush=True)
    loader, sampler = build_nanovvm_loader(
        dataset,
        config,
        rank=distributed.rank,
        world_size=distributed.world_size,
        shuffle=True,
        drop_last=True,
    )
    accumulation = GradientAccumulator(
        int(config["training"].get("gradient_accumulation_steps", 1))
    )
    # Built here, before any resume: the loader is fresh, so len(loader) is a full epoch.
    epochs = int(config["training"].get("epochs", 1))
    steps_per_epoch = len(loader) // accumulation.steps
    schedule_steps = resolve_scheduler_steps(
        config["training"]["scheduler"],
        peak_lr=float(optimizer.param_groups[0]["lr"]),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
    )
    scheduler = build_scheduler(
        optimizer,
        config["training"]["scheduler"],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
    )
    lr_schedule = {
        "name": schedule_steps["name"],
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "total_steps": epochs * steps_per_epoch,
        "warmup_steps": schedule_steps["warmup_steps"],
        "decay_end_steps": schedule_steps["decay_end_steps"],
        "peak_lr": schedule_steps["peak_lr"],
        "final_lr": schedule_steps["final_lr"],
        "warmup_from_zero": schedule_steps["warmup_from_zero"],
    }
    batch_contract = resolve_batch_contract(config["training"], world_size=distributed.world_size)
    if distributed.is_main:
        print(
            "[schedule] "
            + ", ".join(f"{name}={value}" for name, value in lr_schedule.items()),
            flush=True,
        )
        update_run_metadata(
            run,
            {
                "world_size": distributed.world_size,
                "local_micro_batch_size": batch_contract.local_micro_batch_size,
                "global_micro_batch_size": batch_contract.global_micro_batch_size,
                "steps_per_epoch": steps_per_epoch,
                "lr_schedule": lr_schedule,
                "train_sources": dataset.identity(),
            },
        )
        wandb_run_id = str(load_run_metadata(run)["wandb_run_id"])
    else:
        wandb_run_id = None
    logger = WandbLogger(
        config["wandb"],
        enabled=distributed.is_main,
        run_dir=str(run / "wandb"),
        run_id=wandb_run_id,
        exact_resume=bool(config["training"].get("resume")),
        run_config=config,
    )

    start_epoch = 0
    step = 0
    metadata_fields = tuple(model_metadata)
    init_path = config["training"].get("init_from")
    resume_path = config["training"].get("resume")
    if init_path:
        load_model_init(
            paths.checkpoint(init_path),
            dit,
            expected_metadata=model_metadata,
            metadata_fields=metadata_fields,
        )
    if resume_path:
        # Opt-in, as in the other V-RAE tasks: resume on a different GPU count. The global
        # batches stay the same (same global micro-batch); rank-local RNG streams are re-seeded.
        allow_world_size_change = os.environ.get(
            "VRAE_ALLOW_WORLD_SIZE_CHANGE_ON_RESUME", ""
        ).lower() in {"1", "true", "yes", "on"}
        if distributed.is_main and allow_world_size_change:
            print(
                "[resume] allowing a world-size change (same global batches); "
                "rank-local RNG streams will be re-seeded",
                flush=True,
            )
        payload = resume_training(
            paths.checkpoint(resume_path),
            model=dit,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            sampler=sampler,
            expected_metadata=model_metadata,
            metadata_fields=metadata_fields,
            current_config=config,
            run_directory=run,
            allow_world_size_change=allow_world_size_change,
        )
        accumulation.load_data_state(payload["data_state"])
        start_epoch = sampler.epoch
        step = int(payload["step"])
    if max_steps is not None and step >= int(max_steps):
        logger.finish()
        return

    checkpoint_interval = int(config["training"].get("checkpoint_interval", 1000))
    log_interval = int(config["wandb"]["log_interval"])
    sample_interval = int(config["wandb"]["sample_interval"])
    clip_grad = config["training"].get("clip_grad")
    metrics = TrainMetrics(distributed.device, accumulation.steps)
    data_wait = [0.0]  # seconds this rank waited for batches since the last log step
    cache_previous = storage.stats() if storage.remote and distributed.is_main else {}
    last_saved_step = step
    stop = False
    for epoch in range(start_epoch, epochs):
        for batch in timed_batches(loader, data_wait):
            if accumulation.is_first_microstep:
                optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                frame_features, raw_future = prepare_nanovvm_latents(
                    stage1, batch, device=distributed.device
                )
                future = normalizer.normalize(raw_future)
            with accumulation.sync_context(model):
                with precision.autocast():
                    losses = transport.training_losses(model, future, frame_features)
                    loss = losses["loss"]
                accumulation.backward(loss, scaler)
            metrics.add_microbatch(losses)
            sampler.commit_batch()
            if not accumulation.advance():
                continue
            stepped, grad_norm = clip_and_step(
                optimizer,
                scaler=scaler,
                max_grad_norm=clip_grad,
                parameters=dit.parameters(),
            )
            metrics.end_step(
                stepped=stepped,
                grad_norm=grad_norm,
                grad_numel=sum(p.numel() for p in dit.parameters() if p.grad is not None),
                clip_grad=clip_grad,
            )
            if not stepped:
                continue
            scheduler.step()
            ema.update(dit)
            step += 1
            if step % log_interval == 0:
                values = metrics.log_values(clip_grad=clip_grad)
                if distributed.is_main:
                    values.update(parameter_metrics(dit))
                    values.update(
                        {
                            "optimizer/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch,
                            "train/global_batch_size": batch_contract.global_batch_size,
                            "data/wait_seconds_per_step": data_wait[0] / log_interval,
                        }
                    )
                    if storage.remote:
                        cache_stats = storage.stats()
                        values.update(cache_metrics(cache_stats, cache_previous))
                        cache_previous = cache_stats
                    print(
                        json.dumps({"task": TASK, "epoch": epoch, "step": step, **values}),
                        flush=True,
                    )
                    logger.log(values, step=step)
                data_wait[0] = 0.0
            if distributed.is_main and step % sample_interval == 0:
                from vrae.training.nanovvm_video_pred.sample import (
                    decode_future_tokens,
                    predict_future_tokens,
                )

                count = min(int(config["wandb"].get("sample_count", 2)), frame_features.shape[0])
                indices = [int(value) for value in batch["index"][:count].tolist()]
                was_training = dit.training
                dit.eval()
                with ema.average_parameters(dit), precision.autocast():
                    predicted_tokens = predict_future_tokens(
                        dit,
                        frame_features[:count],
                        normalizer,
                        sample_indices=indices,
                        base_seed=int(config["sampling"]["base_seed"]),
                        steps=int(config["sampling"]["steps"]),
                        cfg_scale=float(config["sampling"]["cfg_scale"]),
                        internal_guidance_scale=float(
                            config["sampling"]["internal_guidance_scale"]
                        ),
                        transport=transport,
                    )
                    predicted = decode_future_tokens(
                        stage1, predicted_tokens, grid_size=tuple(dit.grid_size)
                    )
                dit.train(was_training)
                # Ground truth is the RGB clip[:, 1:17], not a V-RAE reconstruction.
                real = rgb_unit(batch["future"][:count]).to(predicted.device)
                condition = rgb_unit(batch["condition"][:count]).to(predicted.device)
                save_sample_tensors(
                    {
                        "condition_rgb": condition,
                        "real_future_rgb": real,
                        "predicted_future": predicted,
                    },
                    run / "samples" / f"step-{step:08d}.pt",
                )
                logger.log_video(
                    "samples/future16_real_vs_predicted",
                    comparison_video(real, predicted),
                    step=step,
                    fps=int(config["data"].get("fps", 10)),
                )
                logger.log_video(
                    "samples/display17_first_frame_plus_future",
                    comparison_video(
                        torch.cat((condition[:, None], real), dim=1),
                        torch.cat((condition[:, None], predicted.float().clamp(0, 1)), dim=1),
                    ),
                    step=step,
                    fps=int(config["data"].get("fps", 10)),
                )
                update_decoder_execution_metadata(
                    run, stage1.decoder_execution_metadata(), training_dtype=precision.name
                )
            if step % checkpoint_interval == 0:
                rng_state = gather_rng_states()
                if distributed.is_main:
                    save_training_checkpoint(
                        _checkpoint_payload(
                            sampler=sampler,
                            step=step,
                            dit=dit,
                            ema=ema,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            accumulation=accumulation,
                            config=config,
                            model_metadata=model_metadata,
                            rng_state=rng_state,
                        ),
                        run / "checkpoints",
                    )
                last_saved_step = step
            if max_steps is not None and step >= int(max_steps):
                stop = True
                break
        epoch_values = metrics.end_epoch()
        if distributed.is_main and epoch_values is not None:
            print(
                json.dumps({"task": TASK, "epoch": epoch, "step": step, **epoch_values}),
                flush=True,
            )
            logger.log(epoch_values, step=step)
        if stop:
            break
    if step > 0 and step != last_saved_step:
        accumulation.require_boundary()
        rng_state = gather_rng_states()
        if distributed.is_main:
            save_training_checkpoint(
                _checkpoint_payload(
                    sampler=sampler,
                    step=step,
                    dit=dit,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    accumulation=accumulation,
                    config=config,
                    model_metadata=model_metadata,
                    rng_state=rng_state,
                ),
                run / "checkpoints",
            )
    if sampler.prefetcher is not None:
        sampler.prefetcher.close()
    if storage.remote:
        storage.cache.flush_counters()
    logger.finish()


def main() -> None:
    arguments = parse_config_argument("Train first-frame conditioned NanoVVM future prediction")
    initial_config = load_config(arguments.config)
    if arguments.build_only:
        print(json.dumps(validate_build(initial_config), indent=2))
        return
    validate_build(initial_config)
    initial_paths = load_project_paths(
        initial_config, override=arguments.paths, project_root=find_project_root(arguments.config)
    )
    # Raw-video storage: local files unless VRAE_STORAGE_CONFIG maps the video roots to S3 tars.
    # Credentials, archive identities and the cache budget are checked before any weights load.
    storage = configure_nanovvm_storage(initial_config, initial_paths)
    storage_report = (
        storage_preflight(
            storage,
            [item["root"] for item in resolve_split_sources(initial_config, initial_paths, "train")],
            owner=int(os.environ.get("LOCAL_RANK", "0")) == 0,  # one reconciling process per node
        )
        if storage.remote
        else []
    )
    ensure_latent_stats(initial_config, initial_paths)
    config, paths, run = prepare_run(arguments.config, arguments.paths)
    validate_build(config)
    train(config, paths, run, max_steps=arguments.max_steps, storage_report=storage_report)


if __name__ == "__main__":
    main()


__all__ = [
    "TrainMetrics",
    "build_nanovvm_dit",
    "ensure_latent_stats",
    "load_task_normalizer",
    "nanovvm_model_metadata",
    "parameter_metrics",
    "prepare_nanovvm_latents",
    "task_geometry",
    "train",
    "validate_build",
    "validate_training_recipe",
]
