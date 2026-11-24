"""Wan2.2-TI2V DiT for first-frame conditioned future prediction in the Wan2.2 VAE latent space.

Ported from Wan-Video/Wan2.2 commit 1ea34ff48f87168174e12956e200b1d908b1c5ff, ``wan/modules/model.py``
(``WanModel`` with ``WanAttentionBlock`` and ``Head``; Apache-2.0, third_party/licenses/Wan2.2-LICENSE),
and the first-frame injection of ``wan/textimage2video.py`` (``WanTI2V.i2v``):

* the clean first-frame latent ``z0`` is latent frame 0 of the DiT sequence, the noisy future
  latents are frames 1..4: ``[z0, x_t^1, .., x_t^4]``, 3D RoPE time positions 0..4, full attention;
* per-token timesteps: the ``z0`` tokens use t = 0, the future tokens the transport's t
  (TI2V: ``temp_ts = mask2 * timestep``); no image encoder and no channel concatenation;
* the transport only noises and supervises the future latents; ``z0`` arrives clean through
  ``context`` / ``frame_features`` at every training and sampling step (TI2V replaces latent 0 with
  the clean image latent after every step) and only the future part of the output is returned.

Same as Wan2.2 (kept on purpose): patch embedding ``Conv3d(kernel = stride = patch)``; sinusoidal
time embedding of ``1000 * t`` (cos then sin, float64) -> ``Linear -> SiLU -> Linear`` and a shared
``SiLU -> Linear(d, 6d)`` projection; per block a learned ``[1,6,d]`` modulation bias added in fp32;
``LayerNorm(elementwise_affine=False, eps)`` pre-norms; q/k/v/o ``Linear(d, d)`` with bias, q and k
``RMSNorm(d)`` across all heads before the head split; 3D RoPE on (latent frame, row, column) with
the head_dim split ``d - 4*(d//6)`` / ``2*(d//6)`` / ``2*(d//6)`` (44/42/42 for head_dim 128) and
adjacent-pair rotation; non-causal attention; ``Linear(d, f) -> GELU(tanh) -> Linear(f, d)``; the
output head modulated by the time embedding ``e`` (not ``e0``) and computed in fp32; Wan's
``init_weights`` (xavier Linear, zero bias, xavier patch embedding on its flattened weight with the
default Conv3d bias init, normal(0.02) time embedding, zero output head, ``randn/sqrt(d)``
modulations).

Deliberate deviations from Wan2.2-TI2V-5B (each also noted where it is implemented):

1. No text conditioning: the cross-attention, its ``norm3``, and ``text_embedding`` are removed.
   An action-conditioning cross-attention can be added back at the same place in the block.
2. ``patch_size`` is configurable. At 128 px the 8x8 latent grid uses (1,1,1) (64 tokens per
   frame); Wan's (1,2,2) gives the same 64 tokens per frame at 256 px.
3. Internal guidance: an independent head of the same form as the output head reads the hidden
   state after block ``base_depth = max(1, round(2L/7))`` (V-RAE's encoder_depth/depth = 8/28) and
   predicts the base output; the transport trains it with ``base_model_coeff``.
4. Classifier-free guidance: during training ``z0`` is replaced with probability ``frame_dropout`` by
   a learned null latent ``[1,1,h*w,C]`` (zero-initialized, in the task-normalized latent space);
   its tokens keep t = 0. ``condition_drop_mask=True`` selects the unconditional branch.
5. Only the two distinct timesteps {0, t} are embedded and applied per latent frame (frame 0 at
   t = 0, frames 1..4 at t), which equals Wan's per-token embedding without materializing
   ``[B, L, 6, d]``. RoPE angles are computed in float64 and applied in float32 (Wan rotates in
   float64).

MeanFlow / improved MeanFlow additions (vrae.models.dit.meanflow), all off by default; with the
defaults the module, its state_dict and ``architecture_metadata()`` are unchanged:

6. ``time_delta_conditioning``: the interval ``h = t - r`` enters through ``time_delta_embedding``
   (same form as ``time_embedding``, frame 0 at h = 0, the future frames at h), added to the time
   embedding before ``time_projection``. Wan keeps embedding t per frame (it tells the clean z0 frame
   from the noisy ones), so the network reads both t and h.
7. ``guidance_conditioning``: the iMF guidance scale (embedded as ``1 - 1/omega``) and the guidance
   interval ``(t_min, t_max)``, each through its own MLP of the same form, added to the time
   embedding of every frame (z0 included) instead of iMF's in-context tokens.
   The last Linear of every embedding in 6-7 is zero-initialized, so a new model computes exactly
   what the same weights compute without them.
8. ``aux_v_head_depth = K``: an auxiliary instantaneous-velocity branch for iMF, K independent blocks
   and a head reading the hidden state after block L - K (the last K blocks + ``head`` are the u
   branch). Only ``return_aux_v=True`` evaluates it.

Float64 inputs keep float64 where Wan casts to float32 (``_upcast``), for finite-difference checks;
bf16 / fp32 runs are bit-identical to Wan's casts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from vrae.models.dit.conditioning import _resolve_drop_mask
from vrae.models.dit.video_dit import _factory_parameters
from vrae.registry import DIT_MODELS

WAN22_REVISION = "1ea34ff48f87168174e12956e200b1d908b1c5ff"
QK_NORM = "rms_norm_across_heads"
TIME_SCALE = 1000.0  # Wan embeds timesteps on its 0..1000 training scale


def wan_base_depth(num_layers: int) -> int:
    """Block after which the internal-guidance head reads: max(1, round(2L/7))."""

    return max(1, round(2 * int(num_layers) / 7))


def rope_split(head_dim: int) -> tuple[int, int, int]:
    """Wan's real-valued RoPE split of one head over (time, height, width)."""

    third = 2 * (head_dim // 6)
    return head_dim - 2 * third, third, third


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Wan's ``sinusoidal_embedding_1d``: [N] -> [N, dim] (cos half, then sin half), float64."""

    if dim % 2:
        raise ValueError("freq_dim must be even")
    half = dim // 2
    position = position.to(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half, device=position.device).to(position).div(half))
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def _rope_angles(positions: int, dim: int, theta: float, device: torch.device) -> torch.Tensor:
    """Wan's ``rope_params`` as angles: [positions, dim/2], float64."""

    exponents = torch.arange(0, dim, 2, device=device, dtype=torch.float64).div(dim)
    return torch.outer(
        torch.arange(positions, device=device, dtype=torch.float64), 1.0 / torch.pow(theta, exponents)
    )


def rope_cos_sin(
    grid: tuple[int, int, int], head_dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin ``[L, head_dim/2]`` for tokens in (frame, row, column) raster order."""

    frames, height, width = grid
    t_dim, h_dim, w_dim = rope_split(head_dim)
    angles_t = _rope_angles(frames, t_dim, theta, device)
    angles_h = _rope_angles(height, h_dim, theta, device)
    angles_w = _rope_angles(width, w_dim, theta, device)
    angles = torch.cat(
        [
            angles_t[:, None, None].expand(frames, height, width, -1),
            angles_h[None, :, None].expand(frames, height, width, -1),
            angles_w[None, None, :].expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(frames * height * width, head_dim // 2)
    return angles.cos().float(), angles.sin().float()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent pairs ``(x[2i], x[2i+1])`` of ``[B,L,H,D]`` (Wan's view_as_complex layout)."""

    pairs = _upcast(x).reshape(*x.shape[:-1], -1, 2)
    even, odd = pairs.unbind(-1)
    cos = cos[None, :, None]
    sin = sin[None, :, None]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def _upcast(x: torch.Tensor) -> torch.Tensor:
    """Wan's ``.float()``; float64 stays float64 (finite-difference checks), the rest is identical."""

    return x if x.dtype == torch.float64 else x.float()


def _fp32_region(x: torch.Tensor):
    # Wan wraps its modulation, time embedding and head in autocast(dtype=float32).
    return torch.autocast(device_type=x.device.type, enabled=False) if torch.is_autocast_enabled(
        x.device.type
    ) else nullcontext()


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = _upcast(x)
        return (value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + self.eps)).type_as(
            x
        ) * self.weight


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__(dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(_upcast(x)).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        # qk_norm: RMSNorm over all heads (width d), before the head split.
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, length = x.shape[:2]
        shape = (batch, length, self.num_heads, self.head_dim)
        q = self.norm_q(self.q(x)).view(shape)
        k = self.norm_k(self.k(x)).view(shape)
        v = self.v(x).view(shape)
        q = apply_rope(q, cos, sin).to(v.dtype)
        k = apply_rope(k, cos, sin).to(v.dtype)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        )
        return self.o(out.transpose(1, 2).reshape(batch, length, self.dim))


class WanTI2VBlock(nn.Module):
    """``WanAttentionBlock`` without cross-attention (deviation 1; text-free)."""

    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, eps)
        # Wan2.2 has `norm3` + `cross_attn` (text) here; an action cross-attention would go here.
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e0: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """x ``[B, T*P, d]``; e0 ``[B, T, 6, d]`` fp32 (one modulation per latent frame)."""

        batch, frames = e0.shape[:2]
        width = x.shape[-1]
        with _fp32_region(x):
            # [B,T,1,d] chunks broadcast over the P tokens of each latent frame.
            e = (_upcast(self.modulation)[:, None] + _upcast(e0)).unsqueeze(3).unbind(2)

        def framed(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, frames, -1, width)

        y = self.self_attn(
            (framed(_upcast(self.norm1(x))) * (1 + e[1]) + e[0]).reshape(batch, -1, width), cos, sin
        )
        with _fp32_region(x):
            x = (framed(x) + framed(y) * e[2]).reshape(batch, -1, width)
        y = self.ffn((framed(_upcast(self.norm2(x))) * (1 + e[4]) + e[3]).reshape(batch, -1, width))
        with _fp32_region(x):
            x = (framed(x) + framed(y) * e[5]).reshape(batch, -1, width)
        return x


class WanTI2VHead(nn.Module):
    """Wan2.2 ``Head``: fp32, modulated by the time embedding ``e``."""

    def __init__(self, dim: int, out_dim: int, patch_size: Sequence[int], eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """x ``[B, T*P, d]``; e ``[B, T, d]`` fp32 -> ``[B, T*P, prod(patch)*C]`` fp32."""

        batch, frames, width = e.shape
        with _fp32_region(x):
            shift, scale = (_upcast(self.modulation)[:, None] + _upcast(e).unsqueeze(2)).unsqueeze(
                3
            ).unbind(2)
            normed = self.norm(_upcast(x)).reshape(batch, frames, -1, width)
            return self.head((normed * (1 + scale) + shift).reshape(batch, -1, width))


class WanTI2VDiT(nn.Module):
    """Wan2.2-TI2V DiT over ``[z0, noisy future]`` returning the future part only.

    ``forward(noisy_future [B,F,h*w,C], time [B], context=z0 [B,h*w,C])`` -> ``[B,F,h*w,C]``
    (``(full, base)`` with ``return_base=True``). ``context`` and ``frame_features`` are aliases.
    MeanFlow inputs (deviations 6-8): ``time_delta`` [B], ``guidance_scale`` and
    ``guidance_interval=(t_min, t_max)`` (scalars or [B]); ``return_dict=True`` returns
    ``{"full", "base" (return_base), "aux_v" (return_aux_v)}``.
    """

    def __init__(
        self,
        *,
        future_chunks: int = 4,
        grid_size: int | Sequence[int] = (8, 8),
        in_channels: int = 48,
        frame_channels: int | None = None,
        dim: int = 3072,
        ffn_dim: int = 14336,
        num_heads: int = 24,
        num_layers: int = 30,
        base_depth: int | None = None,
        patch_size: int | Sequence[int] = (1, 1, 1),
        freq_dim: int = 256,
        eps: float = 1e-6,
        qk_norm: str = QK_NORM,
        rope_theta: float = 10_000.0,
        frame_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        use_gradient_checkpointing: bool | None = None,
        time_delta_conditioning: bool = False,
        guidance_conditioning: bool = False,
        aux_v_head_depth: int = 0,
    ) -> None:
        super().__init__()
        grid = (int(grid_size), int(grid_size)) if isinstance(grid_size, int) else tuple(
            int(value) for value in grid_size
        )
        if len(grid) != 2 or min(grid) <= 0:
            raise ValueError("grid_size must be (h, w)")
        patch = (
            (1, int(patch_size), int(patch_size))
            if isinstance(patch_size, int)
            else tuple(int(value) for value in patch_size)
        )
        if len(patch) != 3 or min(patch) <= 0:
            raise ValueError("patch_size must be (t, h, w)")
        if patch[0] != 1:
            raise ValueError("patch_size[0] must be 1: z0 is a latent frame of its own")
        if grid[0] % patch[1] or grid[1] % patch[2]:
            raise ValueError("the latent grid must be divisible by the spatial patch size")
        frame_channels = int(in_channels if frame_channels is None else frame_channels)
        if frame_channels != int(in_channels):
            raise ValueError("z0 is a latent frame: frame_channels must equal in_channels")
        if str(qk_norm) != QK_NORM:
            raise ValueError(f"qk_norm must be {QK_NORM!r} (Wan2.2)")
        if dim % num_heads or (dim // num_heads) % 2:
            raise ValueError("dim must split into num_heads heads of even width")
        num_layers = int(num_layers)
        base_depth = wan_base_depth(num_layers) if base_depth is None else int(base_depth)
        if not 1 <= base_depth <= num_layers:
            raise ValueError("base_depth must be in [1, num_layers]")
        if not 0 <= float(frame_dropout) <= 1:
            raise ValueError("frame_dropout must be in [0, 1]")
        aux_v_head_depth = int(aux_v_head_depth)
        if aux_v_head_depth < 0:
            raise ValueError("aux_v_head_depth must be non-negative")
        if aux_v_head_depth and base_depth > num_layers - aux_v_head_depth:
            raise ValueError(
                f"base_depth {base_depth} must be <= num_layers - aux_v_head_depth = "
                f"{num_layers - aux_v_head_depth}: the base head reads the shared trunk"
            )
        self.num_chunks = int(future_chunks)
        self.grid_size = grid
        self.in_channels = int(in_channels)
        self.frame_channels = frame_channels
        self.patch_size = patch
        self.dim = int(dim)
        self.ffn_dim = int(ffn_dim)
        self.num_heads = int(num_heads)
        self.num_layers = num_layers
        self.base_depth = base_depth
        self.freq_dim = int(freq_dim)
        self.eps = float(eps)
        self.qk_norm = str(qk_norm)
        self.rope_theta = float(rope_theta)
        self.frame_dropout = float(frame_dropout)
        self.gradient_checkpointing = bool(
            gradient_checkpointing if use_gradient_checkpointing is None else use_gradient_checkpointing
        )
        self.time_delta_conditioning = bool(time_delta_conditioning)
        self.guidance_conditioning = bool(guidance_conditioning)
        self.aux_v_head_depth = aux_v_head_depth

        self.patch_embedding = nn.Conv3d(self.in_channels, self.dim, kernel_size=patch, stride=patch)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.dim, self.dim * 6))
        self.blocks = nn.ModuleList(
            WanTI2VBlock(self.dim, self.ffn_dim, self.num_heads, self.eps) for _ in range(num_layers)
        )
        self.head = WanTI2VHead(self.dim, self.in_channels, patch, self.eps)
        # Deviation 3: the internal-guidance base head, same form, independent parameters.
        self.base_head = WanTI2VHead(self.dim, self.in_channels, patch, self.eps)
        # Deviation 4: learned null z0 for classifier-free guidance (task-normalized space).
        self.null_condition = nn.Parameter(torch.zeros(1, 1, self.frame_tokens, self.in_channels))
        # Deviations 6-8 (MeanFlow); registered last, absent with the defaults.
        if self.time_delta_conditioning:
            self.time_delta_embedding = self._embedding_mlp()
        if self.guidance_conditioning:
            self.guidance_scale_embedding = self._embedding_mlp()
            self.guidance_start_embedding = self._embedding_mlp()
            self.guidance_end_embedding = self._embedding_mlp()
        if self.aux_v_head_depth:
            self.aux_v_blocks = nn.ModuleList(
                WanTI2VBlock(self.dim, self.ffn_dim, self.num_heads, self.eps)
                for _ in range(self.aux_v_head_depth)
            )
            self.aux_v_head = WanTI2VHead(self.dim, self.in_channels, patch, self.eps)
        self.init_weights()

    def _embedding_mlp(self) -> nn.Sequential:
        """Same form as ``time_embedding``: ``Linear(freq_dim, d) -> SiLU -> Linear(d, d)``."""

        return nn.Sequential(nn.Linear(self.freq_dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim))

    def condition_embeddings(self) -> dict[str, nn.Sequential]:
        """The zero-initialized MeanFlow condition embeddings present in this model (deviations 6-7)."""

        names = ["time_delta_embedding"] if self.time_delta_conditioning else []
        if self.guidance_conditioning:
            names += ["guidance_scale_embedding", "guidance_start_embedding", "guidance_end_embedding"]
        return {name: getattr(self, name) for name in names}

    # ----------------------------------------------------------------- interface

    @property
    def future_chunks(self) -> int:
        return self.num_chunks

    @property
    def frame_tokens(self) -> int:
        return self.grid_size[0] * self.grid_size[1]

    @property
    def tokens_per_chunk(self) -> int:
        return self.frame_tokens

    @property
    def future_latent_shape(self) -> tuple[int, int, int]:
        return (self.num_chunks, self.frame_tokens, self.in_channels)

    @property
    def patches_per_frame(self) -> int:
        return (self.grid_size[0] // self.patch_size[1]) * (self.grid_size[1] // self.patch_size[2])

    @property
    def sequence_length(self) -> int:
        return (1 + self.num_chunks) * self.patches_per_frame

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "dit_family": "wan2.2_ti2v",
            "dit_wan_revision": WAN22_REVISION,
            "dit_dim": self.dim,
            "dit_ffn_dim": self.ffn_dim,
            "dit_num_heads": self.num_heads,
            "dit_num_layers": self.num_layers,
            "dit_base_depth": self.base_depth,
            "dit_patch_size": list(self.patch_size),
            "dit_freq_dim": self.freq_dim,
            "dit_eps": self.eps,
            "dit_qk_norm": self.qk_norm,
            "dit_rope_theta": self.rope_theta,
            "dit_rope_split": list(rope_split(self.dim // self.num_heads)),
            "dit_time_scale": TIME_SCALE,
            "dit_sequence": "z0_t0_then_future",
            # MeanFlow keys only when enabled, so the flow-matching metadata is unchanged.
            **({"dit_time_delta_conditioning": True} if self.time_delta_conditioning else {}),
            **({"dit_guidance_conditioning": True} if self.guidance_conditioning else {}),
            **({"dit_aux_v_head_depth": self.aux_v_head_depth} if self.aux_v_head_depth else {}),
        }

    def init_weights(self) -> None:
        """Wan2.2 ``WanModel.init_weights`` (heads: both zero; modulations keep randn/sqrt(d))."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Wan initializes the flattened weight; the Conv3d bias keeps PyTorch's default init.
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for module in self.time_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)
        nn.init.zeros_(self.base_head.head.weight)
        # MeanFlow condition embeddings: first Linear like time_embedding, last Linear all zero, so
        # they add exactly 0 at initialization (a flow-matching model's weights compute the same).
        for embedding in self.condition_embeddings().values():
            nn.init.normal_(embedding[0].weight, std=0.02)
            nn.init.zeros_(embedding[2].weight)
            nn.init.zeros_(embedding[2].bias)
        if self.aux_v_head_depth:
            nn.init.zeros_(self.aux_v_head.head.weight)

    # ------------------------------------------------------------------- forward

    def _sinusoid(self, values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return sinusoidal_embedding_1d(self.freq_dim, TIME_SCALE * values).to(dtype)

    def _time_embeddings(
        self,
        time: torch.Tensor,
        frames: int,
        *,
        time_delta: torch.Tensor | None = None,
        guidance: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-frame ``e [B,T,d]`` and ``e0 [B,T,6,d]`` in fp32: frame 0 at t=0, the rest at t.

        ``time_delta`` (h [B]; frame 0 at h = 0) and ``guidance`` (omega, t_min, t_max, each [B]; all
        frames) add their embeddings to ``e`` before ``time_projection`` (deviations 6-7).
        """

        batch = time.shape[0]
        values = torch.stack((torch.zeros_like(time), time), dim=1).flatten()  # [2B]: (0, t) pairs
        with _fp32_region(time):
            # .to(time.dtype) is Wan's .float() (the forward makes t float32) unless t is float64.
            e = self.time_embedding(self._sinusoid(values, time.dtype)).unflatten(0, (batch, 2))
            if time_delta is not None:
                deltas = torch.stack((torch.zeros_like(time_delta), time_delta), dim=1).flatten()
                e = e + self.time_delta_embedding(self._sinusoid(deltas, time.dtype)).unflatten(
                    0, (batch, 2)
                )
            if guidance is not None:
                scale, start, end = guidance
                condition = (
                    self.guidance_scale_embedding(self._sinusoid(1 - 1 / scale, time.dtype))
                    + self.guidance_start_embedding(self._sinusoid(start, time.dtype))
                    + self.guidance_end_embedding(self._sinusoid(end, time.dtype))
                )
                e = e + condition[:, None]
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        index = torch.ones(frames, dtype=torch.long, device=time.device)
        index[0] = 0
        return _upcast(e)[:, index], _upcast(e0)[:, index]

    def _unpatchify(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        """``[B, T*P, ph*pw*C]`` -> ``[B, T, h*w, C]`` (row-major tokens)."""

        batch = x.shape[0]
        _, ph, pw = self.patch_size
        gh, gw = self.grid_size[0] // ph, self.grid_size[1] // pw
        channels = self.in_channels
        # Wan: view(f, h, w, pt, ph, pw, c) -> einsum('fhwpqrc->cfphqwr'); pt = 1 here.
        x = x.reshape(batch, frames, gh, gw, ph, pw, channels)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(batch, frames, gh * ph * gw * pw, channels)
        return x

    def forward(
        self,
        noisy_future: torch.Tensor,
        time: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        frame_features: torch.Tensor | None = None,
        condition_drop_mask: torch.Tensor | None = None,
        context_drop_mask: torch.Tensor | None = None,
        condition_generator: torch.Generator | None = None,
        return_base: bool = False,
        return_intermediate: bool = False,
        time_delta: torch.Tensor | float | None = None,
        guidance_scale: torch.Tensor | float | None = None,
        guidance_interval: Sequence[torch.Tensor | float] | None = None,
        return_aux_v: bool = False,
        return_dict: bool = False,
    ) -> Any:
        if frame_features is None:
            frame_features = context
        elif context is not None:
            raise ValueError("pass z0 as either context or frame_features")
        if frame_features is None:
            raise ValueError("the first-frame latent z0 is required")
        if condition_drop_mask is not None and context_drop_mask is not None:
            raise ValueError("pass only one of condition_drop_mask or context_drop_mask")
        drop_mask = condition_drop_mask if condition_drop_mask is not None else context_drop_mask
        expected = self.future_latent_shape
        if noisy_future.ndim != 4 or tuple(noisy_future.shape[1:]) != expected:
            raise ValueError(
                f"noisy_future must be [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(noisy_future.shape)}"
            )
        batch = noisy_future.shape[0]
        if tuple(frame_features.shape) != (batch, self.frame_tokens, self.frame_channels):
            raise ValueError(
                f"z0 must be [{batch},{self.frame_tokens},{self.frame_channels}], "
                f"got {tuple(frame_features.shape)}"
            )
        if frame_features.device != noisy_future.device:
            raise ValueError("z0 and noisy_future must be on the same device")
        if time.ndim == 0:
            time = time.expand(batch)
        if time.ndim != 1 or time.shape[0] != batch:
            raise ValueError(f"time must have shape [{batch}], got {tuple(time.shape)}")
        time_dtype = torch.float64 if noisy_future.dtype == torch.float64 else torch.float32
        time = time.to(device=noisy_future.device, dtype=time_dtype)

        def per_sample(value: torch.Tensor | float, name: str) -> torch.Tensor:
            value = torch.as_tensor(value, device=noisy_future.device, dtype=time_dtype)
            if value.ndim == 0:
                value = value.expand(batch)
            if value.ndim != 1 or value.shape[0] != batch:
                raise ValueError(f"{name} must be a scalar or have shape [{batch}], got {tuple(value.shape)}")
            return value

        if time_delta is not None and not self.time_delta_conditioning:
            raise ValueError("time_delta needs time_delta_conditioning=True")
        if self.time_delta_conditioning:
            time_delta = per_sample(0.0 if time_delta is None else time_delta, "time_delta")
        if (guidance_scale is not None or guidance_interval is not None) and not self.guidance_conditioning:
            raise ValueError("guidance_scale / guidance_interval need guidance_conditioning=True")
        guidance = None
        if self.guidance_conditioning:
            interval = (0.0, 1.0) if guidance_interval is None else tuple(guidance_interval)
            if len(interval) != 2:
                raise ValueError("guidance_interval must be (t_min, t_max)")
            guidance = (
                per_sample(1.0 if guidance_scale is None else guidance_scale, "guidance_scale"),
                per_sample(interval[0], "guidance_interval[0]"),
                per_sample(interval[1], "guidance_interval[1]"),
            )
        if return_aux_v and not self.aux_v_head_depth:
            raise ValueError("return_aux_v needs aux_v_head_depth > 0")
        if return_aux_v and not return_dict:
            raise ValueError("return_aux_v needs return_dict=True")

        mask = _resolve_drop_mask(
            batch_size=batch,
            device=noisy_future.device,
            probability=self.frame_dropout,
            training=self.training,
            drop_mask=drop_mask,
            generator=condition_generator,
        )
        condition = frame_features.to(noisy_future.dtype)
        condition = torch.where(
            mask[:, None, None], self.null_condition[0].to(condition.dtype), condition
        )
        frames = 1 + self.num_chunks
        latents = torch.cat((condition[:, None], noisy_future), dim=1)  # [B,T,h*w,C]
        gh, gw = self.grid_size
        video = latents.reshape(batch, frames, gh, gw, self.in_channels).permute(0, 4, 1, 2, 3)
        x = self.patch_embedding(video)  # [B,d,T,h/ph,w/pw]
        token_grid = tuple(int(value) for value in x.shape[2:])
        x = x.flatten(2).transpose(1, 2)  # [B,L,d], (frame,row,column) raster order

        e, e0 = self._time_embeddings(time, frames, time_delta=time_delta, guidance=guidance)
        cos, sin = rope_cos_sin(token_grid, self.dim // self.num_heads, self.rope_theta, x.device)

        def run(block: nn.Module, value: torch.Tensor) -> torch.Tensor:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                return checkpoint(block, value, e0, cos, sin, use_reentrant=False)
            return block(value, e0, cos, sin)

        hidden_base: torch.Tensor | None = None
        hidden_split: torch.Tensor | None = None
        split = self.num_layers - self.aux_v_head_depth  # deviation 8: the v branch forks here
        for index, block in enumerate(self.blocks, start=1):
            x = run(block, x)
            if index == self.base_depth:
                hidden_base = x
            if index == split:
                hidden_split = x
        assert hidden_base is not None
        full = self._unpatchify(self.head(x, e), frames)[:, 1:]
        base = self._unpatchify(self.base_head(hidden_base, e), frames)[:, 1:]
        result: Any
        if return_dict:
            result = {"full": full}
            if return_base:
                result["base"] = base
            if return_aux_v:
                assert hidden_split is not None
                value = hidden_split
                for block in self.aux_v_blocks:
                    value = run(block, value)
                result["aux_v"] = self._unpatchify(self.aux_v_head(value, e), frames)[:, 1:]
        else:
            result = (full, base) if return_base else full
        if return_intermediate:
            return result, hidden_base
        return result


@DIT_MODELS.decorator("wan_ti2v_dit")
def build_wan_ti2v_dit(*, config: Mapping[str, Any], **overrides: Any) -> WanTI2VDiT:
    return WanTI2VDiT(**_factory_parameters(config, overrides))


__all__ = [
    "WanTI2VDiT",
    "apply_rope",
    "build_wan_ti2v_dit",
    "rope_cos_sin",
    "rope_split",
    "sinusoidal_embedding_1d",
    "wan_base_depth",
]
