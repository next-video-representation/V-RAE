from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import torch
from torch import nn


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters were found")
    return parameters


def build_optimizer(
    parameters: Iterable[nn.Parameter], config: Mapping[str, object]
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if not parameters:
        raise ValueError("Optimizer would have no trainable parameters")
    name = str(config.get("name", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer: {name}")
    betas_value = config.get("betas", (0.9, 0.95))
    betas = tuple(float(value) for value in betas_value)
    if len(betas) != 2:
        raise ValueError("AdamW betas must contain two values")
    fused_value = config.get("fused")
    fused = (
        all(parameter.is_cuda for parameter in parameters)
        if fused_value is None
        else bool(fused_value)
    )
    return torch.optim.AdamW(
        parameters,
        lr=float(config.get("lr", 1.0e-4)),
        betas=betas,
        eps=float(config.get("eps", 1.0e-8)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        fused=fused,
    )


# NanoVVM's ratio schedule (nanovvm optim.py::build_scheduler): warmup and decay end are
# fractions of the whole optimizer-step budget, epochs * steps_per_epoch, rounded with round().
RATIO_SCHEDULE_KEYS = ("warmup_ratio", "decay_end_ratio")
STEP_SCHEDULE_KEYS = ("warmup_steps", "warmup_epochs", "decay_end_steps", "decay_end_epoch")
RATIO_SCHEDULER_FIELDS = frozenset(
    {"name", "warmup_ratio", "decay_end_ratio", "final_lr", "warmup_from_zero"}
)


def _setting(config: Mapping[str, object], key: str, default: object) -> object:
    value = config.get(key)
    return default if value is None else value


def uses_ratio_schedule(config: Mapping[str, object]) -> bool:
    return any(config.get(key) is not None for key in RATIO_SCHEDULE_KEYS)


def validate_ratio_schedule(config: Mapping[str, object], *, peak_lr: float) -> None:
    """NanoVVM's validate_training_schedule for the scheduler mapping.

    A step/epoch key set to null counts as absent, as it already does for
    ``warmup_steps``/``decay_end_steps``; config includes merge mappings, so a derived
    config can only switch an inherited step schedule off by writing ``null``.
    """

    mixed = [key for key in STEP_SCHEDULE_KEYS if config.get(key) is not None]
    if mixed:
        raise ValueError(
            f"scheduler mixes ratio keys {list(RATIO_SCHEDULE_KEYS)} with step/epoch keys {mixed}"
        )
    unknown = sorted(
        key
        for key, value in config.items()
        if key not in RATIO_SCHEDULER_FIELDS and not (key in STEP_SCHEDULE_KEYS and value is None)
    )
    if unknown:
        raise ValueError(f"Unknown keys for a ratio scheduler: {unknown}")
    name = str(config.get("name", "constant")).lower()
    if name not in {"linear", "cosine"}:
        raise ValueError(f"Ratio keys need a linear or cosine scheduler, got {name}")
    if not math.isfinite(peak_lr) or peak_lr <= 0:
        raise ValueError("optimizer lr must be finite and positive")
    warmup = float(_setting(config, "warmup_ratio", 0.0))
    end = float(_setting(config, "decay_end_ratio", 1.0))
    if not 0 <= warmup <= end <= 1:
        raise ValueError("Require 0 <= warmup_ratio <= decay_end_ratio <= 1")
    final_lr = float(_setting(config, "final_lr", peak_lr))
    if not math.isfinite(final_lr) or not 0 <= final_lr <= peak_lr:
        raise ValueError("scheduler.final_lr must be finite and between zero and optimizer lr")
    if not isinstance(config.get("warmup_from_zero", False), bool):
        raise ValueError("scheduler.warmup_from_zero must be boolean")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def resolve_scheduler_steps(
    config: Mapping[str, object],
    *,
    peak_lr: float,
    steps_per_epoch: int | None = None,
    epochs: int | None = None,
) -> dict[str, object]:
    """Resolved warmup/decay boundaries in optimizer steps, as build_scheduler uses them."""

    name = str(config.get("name", "constant")).lower()
    if uses_ratio_schedule(config):
        validate_ratio_schedule(config, peak_lr=float(peak_lr))
        steps_per_epoch = _positive_int(steps_per_epoch, "steps_per_epoch")
        epochs = _positive_int(epochs, "epochs")
        total_steps = epochs * steps_per_epoch
        warmup_steps = round(float(_setting(config, "warmup_ratio", 0.0)) * total_steps)
        decay_end_steps = round(float(_setting(config, "decay_end_ratio", 1.0)) * total_steps)
        if not 0 <= warmup_steps <= decay_end_steps <= total_steps:
            raise ValueError("Invalid resolved scheduler step counts")
        final_lr = float(_setting(config, "final_lr", peak_lr))
        return {
            "name": name,
            "mode": "ratio",
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "decay_end_steps": decay_end_steps,
            "peak_lr": float(peak_lr),
            "final_lr": final_lr,
            "final_ratio": final_lr / float(peak_lr),
            "warmup_from_zero": bool(config.get("warmup_from_zero", False)),
        }
    total_steps = (
        int(epochs) * int(steps_per_epoch)
        if epochs is not None and steps_per_epoch is not None
        else None
    )
    if name == "constant":
        warmup_steps = int(config.get("warmup_steps", 0))
        if warmup_steps != 0:
            raise ValueError("The formal constant recipe has no implicit warmup")
        return {
            "name": name,
            "mode": "constant",
            "total_steps": total_steps,
            "warmup_steps": 0,
            "decay_end_steps": 0,
            "peak_lr": float(peak_lr),
            "final_lr": float(peak_lr),
            "final_ratio": 1.0,
            "warmup_from_zero": False,
        }
    if name not in {"linear", "cosine"}:
        raise ValueError(f"Unsupported scheduler: {name}")
    if steps_per_epoch is None or int(steps_per_epoch) <= 0:
        raise ValueError(f"{name} scheduler requires positive steps_per_epoch")

    base_lr = float(config.get("base_lr", peak_lr))
    final_lr = float(config.get("final_lr", base_lr))
    final_ratio = final_lr / base_lr if base_lr > 0 else 1.0
    warmup_steps_value = config.get("warmup_steps")
    warmup_steps = (
        int(warmup_steps_value)
        if warmup_steps_value is not None
        else int(float(config.get("warmup_epochs", 0)) * int(steps_per_epoch))
    )
    decay_end_steps_value = config.get("decay_end_steps")
    decay_end_steps = (
        int(decay_end_steps_value)
        if decay_end_steps_value is not None
        else int(float(config["decay_end_epoch"]) * int(steps_per_epoch))
    )
    warmup_steps = max(warmup_steps, 0)
    decay_end_steps = max(decay_end_steps, warmup_steps)
    return {
        "name": name,
        "mode": "steps",
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "decay_end_steps": decay_end_steps,
        "peak_lr": base_lr,
        "final_lr": final_lr,
        "final_ratio": final_ratio,
        "warmup_from_zero": bool(config.get("warmup_from_zero", False)),
    }


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, object],
    *,
    steps_per_epoch: int | None = None,
    epochs: int | None = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Constant, or warmup + linear/cosine decay, stepped once per optimizer step.

    Boundaries come either from ``warmup_steps``/``warmup_epochs`` and
    ``decay_end_steps``/``decay_end_epoch``, or (NanoVVM) from ``warmup_ratio`` and
    ``decay_end_ratio`` of ``epochs * steps_per_epoch``; the two forms cannot be mixed.
    In the ratio form the peak LR is the optimizer LR.
    """

    resolved = resolve_scheduler_steps(
        config,
        peak_lr=float(optimizer.param_groups[0]["lr"]),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
    )
    name = str(resolved["name"])
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    base_lr = float(resolved["peak_lr"])
    final_ratio = float(resolved["final_ratio"])
    warmup_steps = int(resolved["warmup_steps"])
    decay_end_steps = int(resolved["decay_end_steps"])
    total_decay_steps = max(decay_end_steps - warmup_steps, 1)
    warmup_from_zero = bool(resolved["warmup_from_zero"])
    for group in optimizer.param_groups:
        group["lr"] = base_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps if warmup_from_zero else 1.0
        if step >= decay_end_steps:
            return final_ratio
        progress = (step - warmup_steps) / total_decay_steps
        if name == "linear":
            return 1.0 - (1.0 - final_ratio) * progress
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_ratio + (1.0 - final_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
