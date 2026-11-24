"""MeanFlow (MF) and improved MeanFlow (iMF) transports with u-prediction (average velocity).

Same interpolation as vrae.models.dit.transport: z_t = (1 - t) x + t e, t = 1 noise, t = 0 data, and
the conditional velocity v_t = e - x (no t_eps, no clamp). The network u_theta(z_t, t, h = t - r | ...)
outputs the average velocity over [r, t]; one step is z_r = z_t - (t - r) u_theta.

Training (MF: Geng et al. 2025, "Mean Flows for One-step Generative Modeling"; iMF:
github.com/Lyy-iiis/imeanflow imf.py; PyTorch JVP as in github.com/Gsunshine/py-meanflow)::

    dudt = d/dt u_theta(z, t, t - r)  along (dz, dt, dr) = (tangent, 1, 0)    # r != t samples only
    V    = u_theta(z, t, t - r) + (t - r) sg(dudt)
    loss = mean_i  L_i / sg((L_i + eps)^p),   L_i = sum over the sample's elements of (V - target)^2

MF: tangent = target = v_t. iMF: target = v_t, tangent = the model's own instantaneous velocity
V0(z, t) = u_theta(z, t, 0) (``tangent: boundary``) or an auxiliary v head at h = 0 (``aux_head``,
whose head is trained on the same target with ``aux_v_coeff``).

Classifier-free guidance (``cfg.mode``):

* ``fixed`` (MF, MF paper Eq. 18-19): target = tangent = omega v_t + (1 - omega) u_theta(z, t, 0 | null);
  omega is baked into the model, so sampling uses cfg_scale 1.
* ``conditioned`` (iMF): omega ~ p(omega) ~ omega^-beta on [1, omega_max] and a guidance interval
  [t_min, t_max] are network inputs; target v_g = v_t + (1 - 1/omega_eff)(v_c - v_u) with
  omega_eff = omega inside the interval and 1 outside, tangent v_c (imf.py ``guidance_fn``). Sampling
  feeds the guidance scale and interval to the network and runs a single branch.
* ``none``: no guidance in training (sampling may still apply naive CFG to u, a heuristic).

Samples whose condition is dropped (``frame_dropout``) get target v_t; with ``conditioned`` their
tangent is v_u (``dropped_tangent: unconditional``) or v_c (``conditional``, the official choice).

The JVP is computed with ``torch.func.jvp`` under ``torch.no_grad()`` on the unwrapped model
(``jvp_model``); only dudt is kept. u comes from one ordinary forward through ``model`` (possibly
DDP), so gradient synchronization, no_sync accumulation and gradient checkpointing are untouched.
The condition-dropout mask is drawn once per call and passed to every forward.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from vrae.models.dit.conditioning import sample_condition_dropout
from vrae.models.dit.guidance import GuidanceInterval, guided_model_forward
from vrae.models.dit.transport import FlowMatchingTransport, expand_time

VARIANTS = ("mf", "imf")
TR_SAMPLERS = ("max_min", "v1")
TANGENTS = {"mf": ("conditional",), "imf": ("boundary", "aux_head")}
DEFAULT_TANGENT = {"mf": "conditional", "imf": "boundary"}
CFG_MODES = {"mf": ("none", "fixed"), "imf": ("none", "conditioned")}
DROPPED_TANGENTS = ("unconditional", "conditional")
JVP_DTYPES = ("float32", "autocast")
JVP_ATTENTION_BACKENDS = ("math", "default")
TIME_GRIDS = ("uniform", "shifted")
DEFAULT_ADAPTIVE_WEIGHT = {"p": 1.0, "eps": 0.01}
DEFAULT_CFG = {
    "mode": "none",
    "omega": 2.0,
    "omega_max": 8.0,
    "beta": 1.0,
    "interval": True,
    "dropped_tangent": "unconditional",
}
DEFAULT_JVP = {"dtype": "float32", "attention_backend": "math", "neq_subset_only": True}
LOSS_TERMS = (
    "loss",
    "loss_full",
    "loss_base",
    "loss_aux_v",
    "mse_full",
    "mse_base",
    "mse_fm",
    "mse_mf",
    "dudt_rms",
)


def _options(value: Mapping[str, Any] | None, defaults: Mapping[str, Any], name: str) -> dict[str, Any]:
    if value is None:
        return dict(defaults)
    if not isinstance(value, Mapping):
        raise TypeError(f"transport.{name} must be a mapping")
    unknown = set(value) - set(defaults)
    if unknown:
        raise ValueError(f"unknown transport.{name} keys {sorted(unknown)}; expected {sorted(defaults)}")
    return {**defaults, **dict(value)}


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def adaptive_loss(per_sample: torch.Tensor, p: float, eps: float) -> torch.Tensor:
    """``L_i / sg((L_i + eps)^p)`` averaged over the batch (MF / iMF adaptive weighting)."""

    return (per_sample / (per_sample.detach() + eps).pow(p)).mean()


def _per_sample_sum(value: torch.Tensor) -> torch.Tensor:
    return value.flatten(1).sum(dim=1)


def _subset_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return value.new_zeros(())
    return value[mask].mean()


class MeanFlowTransport:
    """MF / iMF training objective and few-step sampler (see the module docstring)."""

    def __init__(
        self,
        *,
        type: str = "meanflow",
        variant: str = "imf",
        prediction: str = "u",
        time_dist_type: str = "logit-normal_0_1",
        time_dist_shift: float = 1.0,
        tr_sampler: str = "max_min",
        r_time_dist_type: str | None = None,
        r_time_dist_shift: float | None = None,
        ratio_r_neq_t: float = 0.5,
        adaptive_weight: Mapping[str, Any] | None = None,
        tangent: str | None = None,
        base_model_coeff: float = 1.0,
        aux_v_coeff: float = 1.0,
        cfg: Mapping[str, Any] | None = None,
        jvp: Mapping[str, Any] | None = None,
        t_eps: float | None = None,
    ) -> None:
        if str(type) != "meanflow":
            raise ValueError("MeanFlowTransport needs transport.type: meanflow")
        if t_eps is not None:
            raise ValueError("MeanFlow takes no t_eps (v_t = e - x is never divided by t); set t_eps: null")
        if variant not in VARIANTS:
            raise ValueError(f"transport.variant must be one of {VARIANTS}")
        if prediction != "u":
            raise ValueError("MeanFlow predicts the average velocity: transport.prediction must be 'u'")
        if tr_sampler not in TR_SAMPLERS:
            raise ValueError(f"transport.tr_sampler must be one of {TR_SAMPLERS}")
        if tr_sampler == "max_min" and (r_time_dist_type is not None or r_time_dist_shift is not None):
            raise ValueError("r_time_dist_type / r_time_dist_shift are used by tr_sampler v1 only")
        if not 0 <= float(ratio_r_neq_t) <= 1:
            raise ValueError("transport.ratio_r_neq_t must be in [0, 1]")
        weight = _options(adaptive_weight, DEFAULT_ADAPTIVE_WEIGHT, "adaptive_weight")
        if float(weight["p"]) < 0:
            raise ValueError("transport.adaptive_weight.p must be non-negative")
        if float(weight["eps"]) <= 0:
            raise ValueError("transport.adaptive_weight.eps must be positive")
        tangent = DEFAULT_TANGENT[variant] if tangent is None else str(tangent)
        if tangent not in TANGENTS[variant]:
            raise ValueError(f"transport.tangent of {variant} must be one of {TANGENTS[variant]}")
        if float(base_model_coeff) < 0 or float(aux_v_coeff) < 0:
            raise ValueError("base_model_coeff and aux_v_coeff must be non-negative")
        cfg = _options(cfg, DEFAULT_CFG, "cfg")
        if cfg["mode"] not in CFG_MODES[variant]:
            raise ValueError(f"transport.cfg.mode of {variant} must be one of {CFG_MODES[variant]}")
        if float(cfg["omega"]) < 1 or float(cfg["omega_max"]) < 1:
            raise ValueError("transport.cfg.omega and omega_max must be >= 1")
        if float(cfg["beta"]) < 0:
            raise ValueError("transport.cfg.beta must be non-negative")
        if cfg["dropped_tangent"] not in DROPPED_TANGENTS:
            raise ValueError(f"transport.cfg.dropped_tangent must be one of {DROPPED_TANGENTS}")
        jvp = _options(jvp, DEFAULT_JVP, "jvp")
        if jvp["dtype"] not in JVP_DTYPES:
            raise ValueError(f"transport.jvp.dtype must be one of {JVP_DTYPES}")
        if jvp["attention_backend"] not in JVP_ATTENTION_BACKENDS:
            raise ValueError(f"transport.jvp.attention_backend must be one of {JVP_ATTENTION_BACKENDS}")

        # Time distributions: FlowMatchingTransport's shifted logit-normal, reused as is.
        self._time = FlowMatchingTransport(
            prediction="velocity", time_dist_type=time_dist_type, time_dist_shift=time_dist_shift
        )
        self._r_time = FlowMatchingTransport(
            prediction="velocity",
            time_dist_type=time_dist_type if r_time_dist_type is None else r_time_dist_type,
            time_dist_shift=time_dist_shift if r_time_dist_shift is None else r_time_dist_shift,
        )
        self.variant = str(variant)
        self.prediction = "u"
        self.time_dist_type = self._time.time_dist_type
        self.time_dist_shift = self._time.time_dist_shift
        self.tr_sampler = str(tr_sampler)
        self.ratio_r_neq_t = float(ratio_r_neq_t)
        self.adaptive_p = float(weight["p"])
        self.adaptive_eps = float(weight["eps"])
        self.tangent = tangent
        self.base_model_coeff = float(base_model_coeff)
        self.aux_v_coeff = float(aux_v_coeff)
        self.cfg = {
            "mode": str(cfg["mode"]),
            "omega": float(cfg["omega"]),
            "omega_max": float(cfg["omega_max"]),
            "beta": float(cfg["beta"]),
            "interval": bool(cfg["interval"]),
            "dropped_tangent": str(cfg["dropped_tangent"]),
        }
        self.jvp = {
            "dtype": str(jvp["dtype"]),
            "attention_backend": str(jvp["attention_backend"]),
            "neq_subset_only": bool(jvp["neq_subset_only"]),
        }

    # ------------------------------------------------------------------ interface

    @property
    def cfg_mode(self) -> str:
        return self.cfg["mode"]

    @property
    def loss_terms(self) -> tuple[str, ...]:
        """The training-loss dict entries the trainer logs (vrae.training.nanovvm_video_pred.train)."""

        return LOSS_TERMS

    @property
    def requires_guidance_conditioning(self) -> bool:
        return self.cfg_mode == "conditioned"

    @property
    def requires_aux_v_head(self) -> bool:
        return self.tangent == "aux_head"

    def shift_time(self, time: torch.Tensor) -> torch.Tensor:
        return self._time.shift_time(time)

    def metadata(self) -> dict[str, Any]:
        """Algorithm fields for checkpoint metadata; the runtime ``jvp`` options are left out."""

        v1 = self.tr_sampler == "v1"
        return {
            "type": "meanflow",
            "variant": self.variant,
            "prediction": self.prediction,
            "time_dist_type": self.time_dist_type,
            "time_dist_shift": self.time_dist_shift,
            "tr_sampler": self.tr_sampler,
            "r_time_dist_type": self._r_time.time_dist_type if v1 else None,
            "r_time_dist_shift": self._r_time.time_dist_shift if v1 else None,
            "ratio_r_neq_t": self.ratio_r_neq_t,
            "adaptive_weight": {"p": self.adaptive_p, "eps": self.adaptive_eps},
            "tangent": self.tangent,
            "base_model_coeff": self.base_model_coeff,
            "aux_v_coeff": self.aux_v_coeff,
            "cfg": dict(self.cfg),
        }

    # ------------------------------------------------------------------ sampling of t, r

    def sample_t_r(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(t, r, neq)``: ``neq`` marks the round(ratio * B) randomly placed samples kept with r != t.

        ``max_min``: t, r = max / min of two draws of the time distribution. ``v1``: t from the time
        distribution (the flow-matching marginal), r from the r distribution, then r = min(t, r).
        Every other sample gets r = t. A fixed count keeps the JVP sub-batch size constant.
        """

        batch_size = int(batch_size)
        first = self._time.sample_time(batch_size, device=device, generator=generator)
        if self.tr_sampler == "max_min":
            second = self._time.sample_time(batch_size, device=device, generator=generator)
            t, r = torch.maximum(first, second), torch.minimum(first, second)
        else:
            t = first
            r = self._r_time.sample_time(batch_size, device=device, generator=generator)
        count = _round_half_up(self.ratio_r_neq_t * batch_size)
        neq = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if count:
            order = torch.randperm(batch_size, device=device, generator=generator)
            neq[order[:count]] = True
        r = torch.where(neq, r, t)
        if self.tr_sampler == "v1":
            r = torch.minimum(t, r)
        return t.clamp(0, 1), r.clamp(0, 1), neq

    def sample_guidance(
        self,
        batch_size: int,
        neq: torch.Tensor,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(omega, t_min, t_max)`` [B] for ``cfg.mode: conditioned`` (imf.py ``sample_cfg_scale`` and
        ``sample_cfg_interval``): p(omega) ~ omega^-beta on [1, omega_max]; t_min ~ U[0, 0.5],
        t_max ~ U[0.5, 1], and [0, 1] for the r = t rows or with ``interval: false``."""

        batch_size = int(batch_size)
        uniform = torch.rand(batch_size, device=device, generator=generator)
        log_max = math.log1p(self.cfg["omega_max"] - 1)  # log(omega_max)
        beta = self.cfg["beta"]
        if beta == 1:
            omega = torch.exp(uniform * log_max)
        else:
            omega = torch.exp(torch.log1p(uniform * math.expm1((1 - beta) * log_max)) / (1 - beta))
        neq = neq.to(device=device, dtype=torch.bool)
        if self.cfg["interval"]:
            start = 0.5 * torch.rand(batch_size, device=device, generator=generator)
            end = 0.5 + 0.5 * torch.rand(batch_size, device=device, generator=generator)
            start = torch.where(neq, start, torch.zeros_like(start))
            end = torch.where(neq, end, torch.ones_like(end))
        else:
            start, end = torch.zeros_like(omega), torch.ones_like(omega)
        return omega, start, end

    # ------------------------------------------------------------------ training

    @staticmethod
    def _forward(
        model: nn.Module,
        noisy: torch.Tensor,
        time: torch.Tensor,
        context: torch.Tensor,
        *,
        time_delta: torch.Tensor,
        drop: torch.Tensor,
        guidance: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        **flags: Any,
    ) -> Any:
        arguments: dict[str, Any] = {"time_delta": time_delta, "condition_drop_mask": drop, **flags}
        if guidance is not None:
            omega, start, end = guidance
            arguments.update(guidance_scale=omega, guidance_interval=(start, end))
        return model(noisy, time, context, **arguments)

    def instantaneous_velocity(
        self,
        model: nn.Module,
        noisy: torch.Tensor,
        time: torch.Tensor,
        context: torch.Tensor,
        *,
        drop: torch.Tensor,
        omega: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """V0(z, t | condition): u(z, t, h = 0), or the aux v head at h = 0, guidance interval [0, 1]."""

        guidance = None
        if omega is not None:
            guidance = (omega, torch.zeros_like(omega), torch.ones_like(omega))
        zeros = torch.zeros_like(time)
        if self.tangent == "aux_head":
            output = self._forward(
                model, noisy, time, context, time_delta=zeros, drop=drop, guidance=guidance,
                return_dict=True, return_aux_v=True,
            )
            return output["aux_v"].float()
        return self._forward(model, noisy, time, context, time_delta=zeros, drop=drop, guidance=guidance).float()

    def _target_and_tangent(
        self,
        model: nn.Module,
        noisy: torch.Tensor,
        time: torch.Tensor,
        context: torch.Tensor,
        velocity: torch.Tensor,
        drop: torch.Tensor,
        neq: torch.Tensor,
        guidance: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(target, tangent), both without gradients. Rows outside ``neq`` of an iMF tangent may be 0."""

        def rows(value: torch.Tensor) -> torch.Tensor:  # [B] -> [B,1,...] for broadcasting
            return expand_time(value, velocity)

        if self.variant == "mf":
            if self.cfg_mode != "fixed":
                return velocity, velocity
            unconditional = self._forward(
                model, noisy, time, context, time_delta=torch.zeros_like(time),
                drop=torch.ones_like(drop), guidance=None,
            ).float()
            omega = self.cfg["omega"]
            guided = omega * velocity + (1 - omega) * unconditional
            guided = torch.where(rows(drop), velocity, guided)
            return guided, guided
        if self.cfg_mode == "conditioned":
            assert guidance is not None
            omega, start, end = guidance
            batch = velocity.shape[0]
            inside = (time >= start) & (time <= end)
            # v_c with omega_eff = 1 is recomputed only where omega_eff != omega.
            redo = ~inside & (omega != 1)
            keep, null = torch.zeros_like(drop), torch.ones_like(drop)
            velocities = self.instantaneous_velocity(
                model,
                torch.cat((noisy, noisy, noisy[redo])),
                torch.cat((time, time, time[redo])),
                torch.cat((context, context, context[redo])),
                drop=torch.cat((keep, null, keep[redo])),
                omega=torch.cat((omega, torch.ones_like(omega), torch.ones_like(omega[redo]))),
            )
            conditional, unconditional = velocities[:batch].clone(), velocities[batch : 2 * batch]
            conditional[redo] = velocities[2 * batch :]
            effective = torch.where(inside, omega, torch.ones_like(omega))
            guided = velocity + rows(1 - 1 / effective) * (conditional - unconditional)
            target = torch.where(rows(drop), velocity, guided)
            dropped = unconditional if self.cfg["dropped_tangent"] == "unconditional" else conditional
            tangent = torch.where(rows(drop), dropped, conditional)
            return target, tangent
        # iMF without CFG: the instantaneous velocity under the sample's (dropped or kept) condition.
        tangent = torch.zeros_like(velocity)
        rows = neq if self.jvp["neq_subset_only"] else torch.ones_like(neq)
        if bool(rows.any()):
            tangent[rows] = self.instantaneous_velocity(
                model, noisy[rows], time[rows], context[rows], drop=drop[rows]
            )
        return velocity, tangent

    def _dudt(
        self,
        model: nn.Module,
        noisy: torch.Tensor,
        time: torch.Tensor,
        time_r: torch.Tensor,
        context: torch.Tensor,
        drop: torch.Tensor,
        neq: torch.Tensor,
        guidance: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        tangent: torch.Tensor,
        need_base: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """d u / d t along (tangent, 1, 0) by ``torch.func.jvp``; zero for the r = t samples."""

        dudt = torch.zeros_like(tangent, dtype=torch.float32)
        dudt_base = torch.zeros_like(dudt) if need_base else None
        rows = neq if self.jvp["neq_subset_only"] else torch.ones_like(neq)
        if not bool(rows.any()):
            return dudt, dudt_base
        index = rows.nonzero().squeeze(1)
        z, t, r, direction = noisy[index], time[index], time_r[index], tangent[index]
        condition, mask = context[index], drop[index]
        sub_guidance = None if guidance is None else tuple(value[index] for value in guidance)
        if self.jvp["dtype"] == "float32":
            precision = torch.autocast(device_type=noisy.device.type, enabled=False)
            z, t, r, direction, condition = (value.float() for value in (z, t, r, direction, condition))
            if sub_guidance is not None:
                sub_guidance = tuple(value.float() for value in sub_guidance)
        else:  # the surrounding autocast (bf16 training) stays on
            precision = nullcontext()
        attention = sdpa_kernel(SDPBackend.MATH) if self.jvp["attention_backend"] == "math" else nullcontext()

        def u_fn(z_: torch.Tensor, t_: torch.Tensor, r_: torch.Tensor) -> Any:
            # Deterministic: the drop mask and guidance are fixed inputs; dh/dt = 1 flows through t_ - r_.
            return self._forward(
                model, z_, t_, condition, time_delta=t_ - r_, drop=mask, guidance=sub_guidance,
                return_base=need_base,
            )

        with torch.no_grad(), precision, attention:
            _, derivative = torch.func.jvp(
                u_fn, (z, t, r), (direction, torch.ones_like(t), torch.zeros_like(r))
            )
        full, base = derivative if need_base else (derivative, None)
        dudt[index] = full.float()
        if dudt_base is not None:
            dudt_base[index] = base.float()
        if not self.jvp["neq_subset_only"]:
            keep = expand_time(neq.to(dudt.dtype), dudt)
            dudt = dudt * keep
            if dudt_base is not None:
                dudt_base = dudt_base * keep
        return dudt, dudt_base

    def training_losses(
        self,
        model: nn.Module,
        clean: torch.Tensor,
        context: torch.Tensor,
        *,
        jvp_model: nn.Module,
        generator: torch.Generator | None = None,
        time: torch.Tensor | None = None,
        time_r: torch.Tensor | None = None,
        neq: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        drop_mask: torch.Tensor | None = None,
        guidance: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """One micro-batch of the MF / iMF loss.

        ``model`` may be the DDP wrapper and is called exactly once (with gradients); ``jvp_model`` is
        the unwrapped DiT used for every no-grad evaluation (targets, tangents, JVP). ``time``,
        ``time_r``, ``neq``, ``noise``, ``drop_mask`` and ``guidance`` override the random draws (tests).
        """

        if not clean.is_floating_point() or clean.ndim < 2:
            raise ValueError("clean must be a batched floating-point tensor")
        batch = clean.shape[0]
        device = clean.device
        if noise is None:
            noise = torch.randn(clean.shape, device=device, dtype=clean.dtype, generator=generator)
        elif noise.shape != clean.shape:
            raise ValueError("noise must have the same shape as clean")
        if time is None:
            if time_r is not None or neq is not None:
                raise ValueError("time_r / neq need time")
            time, time_r, neq = self.sample_t_r(batch, device=device, generator=generator)
        else:
            time = time.to(device=device, dtype=torch.float32).expand(batch).contiguous()
            time_r = time.clone() if time_r is None else time_r.to(device=device, dtype=torch.float32)
            if bool((time_r > time).any()):
                raise ValueError("r must not exceed t")
            neq = (time_r < time) if neq is None else neq.to(device=device, dtype=torch.bool)
        if guidance is not None and self.cfg_mode != "conditioned":
            raise ValueError("guidance inputs exist only with cfg.mode: conditioned")
        if drop_mask is None:
            drop_mask = sample_condition_dropout(
                batch, float(jvp_model.frame_dropout), device=device, generator=generator
            )
        drop_mask = drop_mask.to(device=device, dtype=torch.bool)
        if self.cfg_mode == "conditioned":
            if guidance is None:
                guidance = self.sample_guidance(batch, neq, device=device, generator=generator)
            guidance = tuple(value.to(device=device, dtype=torch.float32) for value in guidance)

        expanded = expand_time(time.to(clean.dtype), clean)
        noisy = (1 - expanded) * clean + expanded * noise
        velocity = (noise - clean).float()
        need_base = self.base_model_coeff != 0
        need_aux = self.tangent == "aux_head"

        with torch.no_grad():
            target, tangent = self._target_and_tangent(
                jvp_model, noisy, time, context, velocity, drop_mask, neq, guidance
            )
            dudt, dudt_base = self._dudt(
                jvp_model, noisy, time, time_r, context, drop_mask, neq, guidance, tangent, need_base
            )

        output = self._forward(
            model, noisy, time, context, time_delta=time - time_r, drop=drop_mask, guidance=guidance,
            return_dict=True, return_base=need_base, return_aux_v=need_aux,
        )
        interval = expand_time(time - time_r, dudt)
        target = target.float()

        def objective(prediction: torch.Tensor, derivative: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
            value = prediction.float() if derivative is None else prediction.float() + interval * derivative
            error = (value - target).square()
            return adaptive_loss(_per_sample_sum(error), self.adaptive_p, self.adaptive_eps), error

        loss_full, error_full = objective(output["full"], dudt)
        zero = loss_full.new_zeros(())
        loss_base, mse_base = zero, zero
        if need_base:
            loss_base, error_base = objective(output["base"], dudt_base)
            mse_base = error_base.mean()
        loss_aux_v = zero
        if need_aux:
            loss_aux_v, _ = objective(output["aux_v"], None)
        loss = loss_full + self.base_model_coeff * loss_base + self.aux_v_coeff * loss_aux_v
        per_sample_error = error_full.detach().flatten(1).mean(dim=1)
        has_interval = time_r < time
        return {
            "loss": loss,
            "loss_full": loss_full,
            "loss_base": loss_base,
            "loss_aux_v": loss_aux_v,
            "mse_full": error_full.detach().mean(),
            "mse_base": mse_base.detach(),
            "mse_fm": _subset_mean(per_sample_error, ~has_interval),
            "mse_mf": _subset_mean(per_sample_error, has_interval),
            "dudt_rms": _subset_mean(dudt.flatten(1).square().mean(dim=1), neq).sqrt(),
            "t": time,
            "r": time_r,
            "neq": neq,
            "noise": noise,
            "noisy": noisy,
            "target": target,
            "tangent": tangent,
            "dudt": dudt,
            "dudt_base": dudt_base,
            "drop_mask": drop_mask,
            "guidance": guidance,
        }

    # ------------------------------------------------------------------ sampling

    _warned_naive_cfg = False

    @torch.no_grad()
    def euler_sample(
        self,
        model: nn.Module,
        noise: torch.Tensor,
        *,
        model_kwargs: Mapping[str, Any] | None = None,
        unconditional_kwargs: Mapping[str, Any] | None = None,
        num_steps: int = 1,
        steps: int | None = None,
        cfg_scale: float | torch.Tensor = 1.0,
        ig_scale: float | torch.Tensor = 1.0,
        ig_intervals: Sequence[GuidanceInterval | tuple[float, float]] | None = None,
        return_trajectory: bool = False,
        guidance_interval: Sequence[float] | None = None,
        time_grid: str | None = None,
    ) -> torch.Tensor:
        """Few-step MeanFlow sampling: ``z_r = z_t - (t - r) u(z_t, t, t - r)`` on a grid 1 -> 0.

        Same call as ``FlowMatchingTransport.euler_sample``, plus ``guidance_interval`` (t_min, t_max)
        and ``time_grid`` (``uniform`` linspace(1, 0, N + 1), the default and the official grid, or
        ``shifted`` by the time_dist_shift like flow matching). Guidance by ``cfg.mode``:
        ``conditioned``: ``cfg_scale`` is the network's omega input (scalar or [B]) together with the
        guidance interval (default [0, 1]), one forward per step, ``unconditional_kwargs`` unused;
        ``fixed``: omega is baked in, ``cfg_scale`` must be 1; ``none``: ``cfg_scale != 1`` applies
        naive CFG to u (two forwards per step, a heuristic). Internal guidance as flow matching.
        """

        if steps is not None:
            num_steps = int(steps)
        if int(num_steps) <= 0:
            raise ValueError("num_steps must be positive")
        if not noise.is_floating_point() or noise.ndim < 2:
            raise ValueError("noise must be a batched floating-point tensor")
        grid_kind = "uniform" if time_grid is None else str(time_grid)
        if grid_kind not in TIME_GRIDS:
            raise ValueError(f"time_grid must be one of {TIME_GRIDS}")
        interval = (0.0, 1.0) if guidance_interval is None else tuple(guidance_interval)
        if len(interval) != 2 or not 0 <= float(interval[0]) <= float(interval[1]) <= 1:
            raise ValueError("guidance_interval must be (t_min, t_max) with 0 <= t_min <= t_max <= 1")
        mode = self.cfg_mode
        if mode != "conditioned" and tuple(float(v) for v in interval) != (0.0, 1.0):
            raise ValueError("guidance_interval is a network input of cfg.mode: conditioned only")
        use_cfg = bool(torch.as_tensor(cfg_scale).ne(1).any())
        if mode == "fixed" and use_cfg:
            raise ValueError("cfg.mode fixed bakes omega into the model: sample with cfg_scale 1")
        if mode == "none" and use_cfg and not MeanFlowTransport._warned_naive_cfg:
            MeanFlowTransport._warned_naive_cfg = True
            warnings.warn(
                "MeanFlow trained without CFG (cfg.mode: none): cfg_scale != 1 extrapolates the average "
                "velocity u like flow-matching CFG, a heuristic with two forwards per step",
                stacklevel=2,
            )
        value = noise
        grid = torch.linspace(1, 0, int(num_steps) + 1, device=value.device, dtype=torch.float32)
        if grid_kind == "shifted":
            grid = self.shift_time(grid)
        trajectory = [value.clone()] if return_trajectory else None
        batch = value.shape[0]
        for index in range(int(num_steps)):
            current, following = grid[index], grid[index + 1]
            time = current.expand(batch)
            time_delta = (current - following).expand(batch)
            conditional = {**dict(model_kwargs or {}), "time_delta": time_delta}
            unconditional = None
            scale: float | torch.Tensor = 1.0
            if mode == "conditioned":
                conditional.update(guidance_scale=cfg_scale, guidance_interval=interval)
            elif use_cfg:
                if unconditional_kwargs is None:
                    raise ValueError("unconditional_kwargs are required when cfg_scale is not 1")
                unconditional = {**dict(unconditional_kwargs), "time_delta": time_delta}
                scale = cfg_scale
            velocity = guided_model_forward(
                model,
                value,
                time,
                condition_kwargs=conditional,
                unconditional_kwargs=unconditional,
                cfg_scale=scale,
                ig_scale=ig_scale,
                ig_intervals=ig_intervals,
            )
            value = value - (current - following).to(value.dtype) * velocity
            if trajectory is not None:
                trajectory.append(value.clone())
        if trajectory is not None:
            return torch.stack(trajectory)
        return value


__all__ = ["LOSS_TERMS", "MeanFlowTransport", "TIME_GRIDS", "adaptive_loss"]
