"""Frozen stage-1 encodings for first-frame conditioned future prediction.

V-RAE (``stage1.type: vrae``, the default): the observed frame and the sixteen future frames are
encoded by separate calls:

* condition: ``encode_image_grid`` -> pre-pool fused patch features of frame 0,
  ``[B,h*w,C]`` (one image per sample, no temporal pooling);
* target: ``encode_grid`` on the future frames only -> pooled latents,
  ``[B,4,h*w,C]`` (four 4-frame chunks).

Neither path sees the other's pixels, so replacing the future cannot change the
condition and replacing the first frame cannot change the clean target.

Wan2.2 VAE (``stage1.type: wan22_vae``, vrae.models.video_vae.backend): frame 0 and the sixteen
future frames are encoded jointly as one 17-frame clip; latent 0 is the condition ``z0`` and
latents 1..4 are the target ``[B,4,h*w,48]``. The VAE is causal in time, so ``z0`` equals the
encoding of frame 0 alone and replacing the future cannot change it; the target, however, depends
on the pixels of frame 0 (as in Wan2.2-TI2V), so the second statement above does not hold. Its
decoder needs ``z0`` as well: ``decode_future_tokens`` takes the raw ``z0`` as ``condition``.

The DiT receives ``condition_features``: the V-RAE features unchanged, or the Wan ``z0`` normalized
with the task's future-latent statistics.
"""

from __future__ import annotations

from typing import Any

import torch

from vrae.training.nanovvm_video_pred.data import FUTURE_FRAMES


def _check_rgb(value: torch.Tensor, *, name: str, ndim: int) -> None:
    if value.ndim != ndim or value.shape[-3] != 3:
        layout = "[B,3,H,W]" if ndim == 4 else f"[B,{FUTURE_FRAMES},3,H,W]"
        raise ValueError(f"{name} must be RGB {layout}, got {tuple(value.shape)}")


def is_wan_backend(stage1: Any) -> bool:
    """True for the Wan2.2 VAE backend (``stage1.type: wan22_vae``)."""

    return getattr(stage1, "backend_type", "vrae") == "wan22_vae"


@torch.no_grad()
def encode_first_frame(stage1: Any, condition_rgb: torch.Tensor) -> torch.Tensor:
    """``[B,3,H,W]`` -> pre-pool patch features ``[B,h*w,C]`` (row-major grid order).

    Wan2.2 VAE: the raw first-frame latent ``z0`` ``[B,h*w,48]``.
    """

    _check_rgb(condition_rgb, name="condition_rgb", ndim=4)
    if is_wan_backend(stage1):
        return stage1.encode_condition(condition_rgb)
    grid = stage1.encode_image_grid(condition_rgb)
    return stage1.image_grid_to_tokens(grid)


@torch.no_grad()
def encode_future(stage1: Any, future_rgb: torch.Tensor) -> torch.Tensor:
    """``[B,16,3,H,W]`` -> pooled future latents ``[B,4,h*w,C]``."""

    _check_rgb(future_rgb, name="future_rgb", ndim=5)
    if is_wan_backend(stage1):
        raise ValueError(
            "the Wan2.2 VAE encodes the future together with frame 0: use encode_condition_and_future"
        )
    if future_rgb.shape[1] != FUTURE_FRAMES:
        raise ValueError(
            f"future_rgb must contain {FUTURE_FRAMES} frames, got {future_rgb.shape[1]}"
        )
    return stage1.grid_to_tokens(stage1.encode_grid(future_rgb))


@torch.no_grad()
def encode_condition_and_future(
    stage1: Any,
    condition_rgb: torch.Tensor,
    future_rgb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (V-RAE frame features or raw Wan z0 ``[B,h*w,C]``, raw future latents ``[B,4,h*w,C]``)."""

    if condition_rgb.shape[0] != future_rgb.shape[0]:
        raise ValueError("condition and future batch sizes must match")
    if is_wan_backend(stage1):
        _check_rgb(condition_rgb, name="condition_rgb", ndim=4)
        _check_rgb(future_rgb, name="future_rgb", ndim=5)
        return stage1.encode_clip(condition_rgb, future_rgb)
    frame_features = encode_first_frame(stage1, condition_rgb)
    future_tokens = encode_future(stage1, future_rgb)
    if future_tokens.shape[2:] != frame_features.shape[1:]:
        raise ValueError(
            "first-frame features and future latents must share the patch grid and width: "
            f"{tuple(frame_features.shape)} vs {tuple(future_tokens.shape)}"
        )
    return frame_features, future_tokens


def condition_features(stage1: Any, condition: torch.Tensor, normalizer: Any) -> torch.Tensor:
    """The DiT's first-frame input from ``encode_first_frame`` / ``encode_condition_and_future``.

    V-RAE: the pre-pool features themselves. Wan2.2 VAE: ``z0`` normalized with the task's
    future-latent statistics (the same per-channel normalizer as the target).
    """

    if not is_wan_backend(stage1):
        return condition
    if condition.ndim != 3:
        raise ValueError(f"z0 must be [B,h*w,C], got {tuple(condition.shape)}")
    return normalizer.normalize(condition[:, None])[:, 0]


@torch.no_grad()
def decode_future_tokens(
    stage1: Any,
    future_tokens: torch.Tensor,
    *,
    grid_size: tuple[int, int],
    condition: torch.Tensor | None = None,
) -> torch.Tensor:
    """``[B,4,h*w,C]`` -> RGB ``[B,16,3,H,W]`` through the frozen V-RAE decoder.

    Wan2.2 VAE: decoded causally together with ``condition``, the raw ``z0`` ``[B,h*w,48]``
    (Wan-normalized, not task-normalized); frame 0 of the 17 decoded frames is dropped. V-RAE
    ignores ``condition``.
    """

    if is_wan_backend(stage1):
        if condition is None:
            raise ValueError("the Wan2.2 VAE decodes the future together with z0: pass condition=")
        if tuple(grid_size) != tuple(stage1.grid_size):
            raise ValueError(f"grid_size {tuple(grid_size)} != the Wan latent grid {stage1.grid_size}")
        return stage1.decode_future(condition, future_tokens)
    grid = stage1.tokens_to_grid(future_tokens, height=int(grid_size[0]), width=int(grid_size[1]))
    return stage1.decode_grid(grid)


__all__ = [
    "condition_features",
    "decode_future_tokens",
    "encode_condition_and_future",
    "encode_first_frame",
    "encode_future",
    "is_wan_backend",
]
