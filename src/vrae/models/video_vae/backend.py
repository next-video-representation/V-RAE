"""Frozen Wan2.2 VAE as the latent space of first-frame conditioned prediction (``stage1.type: wan22_vae``).

The Wan2.2 VAE (z_dim 48, 2x2 pixel patchify + three stride-2 stages: 16x spatial, 4x temporal) is
causal in time: latent 0 depends only on RGB frame 0 and equals the encoding of frame 0 alone, and
latent k >= 1 depends on frames <= 4k. Its encoder silently drops frames beyond 4n+1. Hence:

* a 17-frame clip (frame 0 + 16 future frames) is encoded jointly: latent 0 is the condition ``z0``,
  latents 1..4 are the four future targets. A clip of any other length is rejected; the 16 future
  frames are never encoded on their own;
* the future latents depend on the pixels of frame 0 (as in Wan2.2-TI2V); ``z0`` does not depend on
  the future;
* decoding is causal too: ``z0`` is decoded together with the future latents (Wan's ``first_chunk``)
  and frame 0 of the 17 decoded frames is dropped. ``z0`` here is the raw latent (Wan's official
  per-channel normalization applied), never the task-normalized one.

Latents are posterior means with Wan's official normalization ``(mu - mean) / std``. The weights
stay fp32; ``precision`` (``stage1.precision``) sets the compute: ``bf16`` runs the VAE under bf16
autocast on CUDA (as V-RAE's ``stage1.precision: bf16``; fp32 on the CPU, like
``VRAELatentAdapter``), ``fp32`` runs it in fp32 with TF32 off. Either way the caller's autocast is
replaced and the latents/RGB are returned as fp32. Token layout matches ``VRAELatentAdapter``:
``[B, T, h*w, C]``, row-major over the latent grid.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

WAN22_REVISION = "1ea34ff48f87168174e12956e200b1d908b1c5ff"
BACKEND_TYPE = "wan22_vae"
CLIP_FRAMES = 17  # frame 0 + 16 future frames = 4n + 1 with n = 4
LATENT_CHANNELS = 48
SPATIAL_COMPRESSION = 16
TEMPORAL_COMPRESSION = 4


PRECISIONS = {"fp32": "float32", "bf16": "bfloat16"}  # stage1.precision -> metadata dtype


@contextmanager
def vae_precision(device_type: str, precision: str = "fp32"):
    """VAE compute independent of the caller's TF32 and autocast state (nanovvm wrapper): fp32 with
    TF32 off, or bf16 autocast on CUDA (fp32 elsewhere)."""

    if precision not in PRECISIONS:
        raise ValueError(f"Wan2.2 VAE precision must be one of {sorted(PRECISIONS)}, got {precision!r}")
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type=device_type, enabled=False):
            if precision == "bf16" and device_type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    yield
            else:
                yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags


_SHA256: dict[tuple[str, int, int], str] = {}


def file_sha256(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    if key not in _SHA256:
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        _SHA256[key] = digest.hexdigest()
    return _SHA256[key]


def _unit_rgb(value: torch.Tensor) -> torch.Tensor:
    if value.dtype == torch.uint8:
        return value.float().div(255.0)
    if not value.is_floating_point():
        raise TypeError("RGB must be uint8 or floating point in [0,1]")
    return value.float()


class Wan22LatentBackend(nn.Module):
    """Frozen Wan2.2 VAE with the first-frame/future interface of nanovvm_video_pred."""

    backend_type = BACKEND_TYPE
    supports_single_image = True
    latent_channels = LATENT_CHANNELS
    spatial_compression = SPATIAL_COMPRESSION
    temporal_compression = TEMPORAL_COMPRESSION

    def __init__(
        self,
        model: nn.Module,
        scale: Sequence[torch.Tensor],
        *,
        image_size: int,
        weights_sha256: str,
        revision: str = WAN22_REVISION,
        precision: str = "fp32",
    ) -> None:
        super().__init__()
        if str(precision) not in PRECISIONS:
            raise ValueError(f"Wan2.2 VAE precision must be one of {sorted(PRECISIONS)}, got {precision!r}")
        self.precision = str(precision)
        image_size = int(image_size)
        if image_size <= 0 or image_size % SPATIAL_COMPRESSION:
            raise ValueError(f"image_size must be a positive multiple of {SPATIAL_COMPRESSION}")
        mean, inv_std = (torch.as_tensor(value, dtype=torch.float32).flatten() for value in scale)
        if mean.numel() != LATENT_CHANNELS or inv_std.numel() != LATENT_CHANNELS:
            raise ValueError(f"Wan2.2 normalization must have {LATENT_CHANNELS} channels")
        self.model = model.float().requires_grad_(False).eval()
        self.register_buffer("latent_mean", mean)
        self.register_buffer("latent_inv_std", inv_std)
        self.image_size = image_size
        self.grid_size = (image_size // SPATIAL_COMPRESSION, image_size // SPATIAL_COMPRESSION)
        self.weights_sha256 = str(weights_sha256)
        self.revision = str(revision)

    def train(self, mode: bool = True) -> Wan22LatentBackend:
        super().train(mode)
        self.model.eval()
        return self

    @property
    def _scale(self) -> list[torch.Tensor]:
        return [self.latent_mean, self.latent_inv_std]

    def _check_frames(self, value: torch.Tensor, *, name: str, ndim: int) -> None:
        if value.ndim != ndim or value.shape[-3] != 3:
            layout = "[B,3,H,W]" if ndim == 4 else "[B,T,3,H,W]"
            raise ValueError(f"{name} must be RGB {layout}, got {tuple(value.shape)}")
        if tuple(value.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(
                f"{name} must be {self.image_size}x{self.image_size} (the latent grid "
                f"{list(self.grid_size)}), got {tuple(value.shape[-2:])}"
            )

    @torch.no_grad()
    def _encode(self, video: torch.Tensor) -> torch.Tensor:
        """RGB ``[B,T,3,H,W]`` (T = 1 or 17) -> raw latents ``[B,T_lat,h*w,48]``."""

        if video.shape[1] not in (1, CLIP_FRAMES):
            raise ValueError(
                f"the Wan2.2 VAE encodes either frame 0 alone or the whole {CLIP_FRAMES}-frame clip; "
                f"got {video.shape[1]} frames (its encoder would silently drop frames beyond 4n+1)"
            )
        x = _unit_rgb(video).mul(2).sub(1).permute(0, 2, 1, 3, 4).contiguous()  # [B,3,T,H,W]
        with vae_precision(x.device.type, self.precision):
            z = self.model.encode(x, self._scale)
        return z.float().permute(0, 2, 3, 4, 1).flatten(2, 3).contiguous()

    @torch.no_grad()
    def encode_clip(
        self, condition_rgb: torch.Tensor, future_rgb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Frame 0 ``[B,3,H,W]`` + future ``[B,16,3,H,W]`` -> (z0 ``[B,h*w,48]``, future ``[B,4,h*w,48]``)."""

        self._check_frames(condition_rgb, name="condition_rgb", ndim=4)
        self._check_frames(future_rgb, name="future_rgb", ndim=5)
        if future_rgb.shape[1] != CLIP_FRAMES - 1:
            raise ValueError(
                f"future_rgb must hold {CLIP_FRAMES - 1} frames (encoded with frame 0 as one "
                f"{CLIP_FRAMES}-frame clip), got {future_rgb.shape[1]}"
            )
        if condition_rgb.shape[0] != future_rgb.shape[0]:
            raise ValueError("condition and future batch sizes must match")
        clip = torch.cat((_unit_rgb(condition_rgb)[:, None], _unit_rgb(future_rgb)), dim=1)
        latents = self._encode(clip)
        return latents[:, 0], latents[:, 1:]

    @torch.no_grad()
    def encode_condition(self, condition_rgb: torch.Tensor) -> torch.Tensor:
        """Frame 0 ``[B,3,H,W]`` -> z0 ``[B,h*w,48]`` (equal to latent 0 of ``encode_clip``)."""

        self._check_frames(condition_rgb, name="condition_rgb", ndim=4)
        return self._encode(condition_rgb[:, None])[:, 0]

    @torch.no_grad()
    def decode_future(self, condition_latent: torch.Tensor, future_latent: torch.Tensor) -> torch.Tensor:
        """Raw z0 ``[B,h*w,48]`` + raw future ``[B,4,h*w,48]`` -> future RGB ``[B,16,3,H,W]`` in [0,1]."""

        tokens = self.grid_size[0] * self.grid_size[1]
        if condition_latent.ndim != 3 or tuple(condition_latent.shape[1:]) != (tokens, LATENT_CHANNELS):
            raise ValueError(
                f"z0 must be [B,{tokens},{LATENT_CHANNELS}], got {tuple(condition_latent.shape)}"
            )
        expected = ((CLIP_FRAMES - 1) // TEMPORAL_COMPRESSION, tokens, LATENT_CHANNELS)
        if future_latent.ndim != 4 or tuple(future_latent.shape[1:]) != expected:
            raise ValueError(f"future latents must be [B,{expected}], got {tuple(future_latent.shape)}")
        if condition_latent.shape[0] != future_latent.shape[0]:
            raise ValueError("z0 and future latent batch sizes must match")
        latents = torch.cat((condition_latent[:, None], future_latent), dim=1).float()
        batch, frames = latents.shape[:2]
        z = latents.reshape(batch, frames, *self.grid_size, LATENT_CHANNELS).permute(0, 4, 1, 2, 3)
        with vae_precision(z.device.type, self.precision):
            video = self.model.decode(z.contiguous(), self._scale)
        video = video.float().clamp(-1, 1).add(1).mul(0.5).permute(0, 2, 1, 3, 4)
        return video[:, 1:].contiguous()

    def decoder_execution_metadata(self) -> dict[str, Any]:
        return {
            "backend": BACKEND_TYPE,
            "dtype": PRECISIONS[self.precision],
            "tf32": False,
            "autocast": "bf16_cuda" if self.precision == "bf16" else False,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "type": BACKEND_TYPE,
            "revision": self.revision,
            "weights_sha256": self.weights_sha256,
            "latent_channels": LATENT_CHANNELS,
            "spatial_compression": SPATIAL_COMPRESSION,
            "temporal_compression": TEMPORAL_COMPRESSION,
            "latent_grid": list(self.grid_size),
            "normalization": "official",
            "posterior": "mean",
            "dtype": PRECISIONS[self.precision],
        }


def load_wan22_backend(
    checkpoint: str | Path,
    *,
    image_size: int,
    precision: str = "fp32",
    revision: str | None = None,
) -> Wan22LatentBackend:
    """Wan2.2_VAE.pth (Wan-AI/Wan2.2-TI2V-5B) -> frozen backend on the CPU."""

    from vrae.models.video_vae.wan22 import Wan2_2_VAE

    if str(precision) not in PRECISIONS:
        raise ValueError(f"stage1.precision must be fp32 or bf16 for the Wan2.2 VAE, got {precision!r}")
    if revision is not None and str(revision) != WAN22_REVISION:
        raise ValueError(
            f"stage1.revision {revision!r} does not match the vendored Wan2.2 VAE ({WAN22_REVISION})"
        )
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Wan2.2 VAE weights not found: {path}")
    reference = Wan2_2_VAE(vae_pth=str(path), device="cpu", dtype=torch.float32)
    return Wan22LatentBackend(
        reference.model,
        reference.scale,
        image_size=image_size,
        weights_sha256=file_sha256(path),
        precision=str(precision),
    )


__all__ = [
    "BACKEND_TYPE",
    "CLIP_FRAMES",
    "WAN22_REVISION",
    "Wan22LatentBackend",
    "file_sha256",
    "load_wan22_backend",
    "vae_precision",
]
