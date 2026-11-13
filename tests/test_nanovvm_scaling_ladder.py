"""NanoVVM-style scaling ladder for nanovvm_video_pred: CPU tests.

Covers the ratio LR schedule, gradient clipping, the NanoVVM-named training metrics,
W&B arguments, the preprocessing equality with NanoVVM, and the six ladder configs.
Run from the repository root:
    PYTHONPATH=src python -m unittest tests.test_nanovvm_scaling_ladder -v
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
import sys
import types
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

import torch
import torch.nn.functional as functional
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrae.config import load_config  # noqa: E402
from vrae.training.common.accumulation import clip_and_step, optimizer_step  # noqa: E402
from vrae.training.common.optim import build_scheduler, resolve_scheduler_steps  # noqa: E402
from vrae.training.common.wandb import WandbLogger  # noqa: E402
from vrae.training.nanovvm_video_pred import data as nv_data  # noqa: E402
from vrae.training.nanovvm_video_pred import train as nv_train  # noqa: E402

LADDER = ROOT / "configs/training/nanovvm_video_pred/scaling-ladder"
SIZES = ("tiny", "small", "medium", "large", "xl", "xxl")


def ladder_path(size: str) -> Path:
    stem = f"vrae_eupe_mix5_{size}_xpred_interval3_cond1gen16_bs64_lr2e-4_wd0_ep1"
    return LADDER / size / f"{stem}.yaml"


# ---------------------------------------------------------------- NanoVVM reference
# Verbatim copy of resize_video, resize_short_side, random_crop_video,
# validate_preprocessing and spatial_preprocess from nanovvm data/online.py at
# nanovvm@94cc92d8ba69a4ccc639faef5d3ab0c436845014 (file last changed in 34ed042).
# random_crop_video and validate_preprocessing are included because spatial_preprocess
# calls them. NanoVVM is not a dependency of V-RAE.
# fmt: off
# BEGIN nanovvm data/online.py
def resize_video(video: torch.Tensor, size: tuple, *, mode: str = "bilinear",
                 antialias: bool = True) -> torch.Tensor:
    """[T,3,H,W] resize preserving dtype; uint8 is rounded and clamped after interpolation."""
    if tuple(video.shape[-2:]) == tuple(size):
        return video.contiguous()
    working = video.float() if not video.is_floating_point() else video
    resized = functional.interpolate(working, size=size, mode=mode, align_corners=False,
                                     antialias=antialias)
    if video.dtype == torch.uint8:
        return resized.round().clamp_(0, 255).to(torch.uint8).contiguous()
    return resized.to(video.dtype).contiguous()


def resize_short_side(video: torch.Tensor, size: int, *, mode: str = "bilinear",
                      antialias: bool = True) -> torch.Tensor:
    height, width = video.shape[-2:]
    if height <= width:
        output = (size, max(size, round(width * size / height)))
    else:
        output = (max(size, round(height * size / width)), size)
    return resize_video(video, output, mode=mode, antialias=antialias)


def random_crop_video(video: torch.Tensor, size: int,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
    height, width = video.shape[-2:]
    if size > height or size > width:
        raise ValueError(f"crop {size} exceeds video spatial size {(height, width)}")
    top = int(torch.randint(height - size + 1, (), generator=generator).item())
    left = int(torch.randint(width - size + 1, (), generator=generator).item())
    return video[..., top:top + size, left:left + size].contiguous()


def validate_preprocessing(config):
    """Small, independent operations; image_size remains the output size."""
    if not isinstance(config, dict):
        raise ValueError("training.video.preprocessing must be a mapping")
    unknown = set(config) - {"horizontal_flip_prob", "resize", "crop"}
    if unknown:
        raise ValueError(f"Unknown preprocessing keys: {sorted(unknown)}")
    prob = config.get("horizontal_flip_prob", 0.5)
    if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not math.isfinite(prob) or not 0 <= prob <= 1:
        raise ValueError("preprocessing.horizontal_flip_prob must be between zero and one")
    resize = config.get("resize", {})
    if not isinstance(resize, dict) or set(resize) - {"mode", "interpolation", "antialias"}:
        raise ValueError("preprocessing.resize accepts mode, interpolation and antialias")
    if resize.get("mode", "short_side") not in ("short_side", "none"):
        raise ValueError("preprocessing.resize.mode must be short_side or none")
    if resize.get("interpolation", "bilinear") not in ("bilinear", "bicubic"):
        raise ValueError("preprocessing.resize.interpolation must be bilinear or bicubic")
    if not isinstance(resize.get("antialias", False), bool):
        raise ValueError("preprocessing.resize.antialias must be boolean")
    crop = config.get("crop", {})
    if not isinstance(crop, dict) or set(crop) - {"mode"} or crop.get("mode", "random") not in ("random", "center"):
        raise ValueError("preprocessing.crop.mode must be random or center")


def spatial_preprocess(video, image_size, *, preprocessing=None, generator=None):
    config = {} if preprocessing is None else preprocessing
    validate_preprocessing(config)
    prob = config.get("horizontal_flip_prob", 0.5)
    if prob > 0 and bool(torch.rand((), generator=generator) < prob):
        video = video.flip(-1)
    resize = config.get("resize", {})
    if resize.get("mode", "short_side") == "short_side":
        video = resize_short_side(video, image_size, mode=resize.get("interpolation", "bilinear"),
                                  antialias=resize.get("antialias", False))
    if config.get("crop", {}).get("mode", "random") == "random":
        return random_crop_video(video, image_size, generator=generator)
    height, width = video.shape[-2:]
    if image_size > height or image_size > width:
        raise ValueError(f"crop {image_size} exceeds video spatial size {(height, width)}")
    top, left = (height - image_size) // 2, (width - image_size) // 2
    return video[..., top:top + image_size, left:left + image_size].contiguous()

# END nanovvm data/online.py
# fmt: on


# The NanoVVM mix5 ladder's preprocessing (training.video.preprocessing of every bs64 file).
NANOVVM_PREPROCESSING = {
    "horizontal_flip_prob": 0.0,
    "resize": {"mode": "short_side", "interpolation": "bilinear", "antialias": False},
    "crop": {"mode": "center"},
}

# k, dit values, and parameter counts (millions, one decimal) measured on the meta device;
# nanovvm is NanoVVM's own count for the same size.
LADDER_TABLE = {
    "tiny": dict(k=3, hidden=[216, 384], depth=[6, 1], heads=[3, 3], encoder_depth=2,
                 total=9.5, encoder=5.1, decoder=2.7, nanovvm=9.9),
    "small": dict(k=7, hidden=[504, 896], depth=[13, 1], heads=[7, 7], encoder_depth=4,
                  total=80.0, encoder=59.6, decoder=14.5, nanovvm=78.8),
    "medium": dict(k=11, hidden=[792, 1408], depth=[19, 1], heads=[11, 11], encoder_depth=5,
                   total=262.7, encoder=214.8, decoder=35.7, nanovvm=265.7),
    "large": dict(k=14, hidden=[1008, 1792], depth=[27, 2], heads=[14, 14], encoder_depth=8,
                  total=628.1, encoder=494.3, decoder=115.7, nanovvm=629.5),
    "xl": dict(k=18, hidden=[1296, 2304], depth=[33, 2], heads=[18, 18], encoder_depth=9,
               total=1217.5, encoder=998.5, decoder=191.2, nanovvm=1229.2),
    "xxl": dict(k=22, hidden=[1584, 2816], depth=[37, 3], heads=[22, 22], encoder_depth=11,
                total=2140.0, encoder=1672.1, decoder=428.4, nanovvm=2124.1),
}

LADDER_TRAINING = {
    "resume": None,
    "init_from": None,
    "global_batch_size": 64,
    "gradient_accumulation_steps": 4,
    "epochs": 1,
    "checkpoint_interval": 5000,
    "precision": "bf16",
    "num_workers": 8,
    "optimizer": {"name": "adamw", "lr": 2.0e-4, "betas": [0.9, 0.95], "weight_decay": 0.0,
                  "eps": 1.0e-8},
    "scheduler": {"name": "cosine", "warmup_steps": None, "warmup_ratio": 0.02,
                  "decay_end_ratio": 1.0, "final_lr": 2.0e-5, "warmup_from_zero": True},
    "ema": {"decay": 0.9995},
    "clip_grad": 1.0,
}


def lr_trace(scheduler, steps: int) -> list[float]:
    """LR used by each of ``steps`` optimizer steps (read before the step, as training does)."""

    optimizer = scheduler.optimizer
    values = []
    for _ in range(steps):
        values.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return values


def make_optimizer(lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW([nn.Parameter(torch.zeros(2))], lr=lr)


@contextlib.contextmanager
def replaced_module(name: str, module: types.ModuleType):
    """Put ``module`` at ``sys.modules[name]`` and restore only that key afterwards.

    mock.patch.dict(sys.modules, ...) would also drop every module first imported inside
    the block, e.g. torch._dynamo by optimizer.step(); importing torch._dynamo a second time
    in this process fails (duplicate TORCH_LIBRARY registration).
    """

    missing = object()
    previous = sys.modules.get(name, missing)
    sys.modules[name] = module
    try:
        yield module
    finally:
        if previous is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class FakeRun:
    def __init__(self, module):
        self.module = module

    def log(self, values, step=None):
        self.module.logs.append((step, dict(values)))

    def finish(self):
        self.module.finished += 1


def fake_wandb_module() -> types.ModuleType:
    """Records wandb.init arguments and run.log calls; wandb itself is not installed here."""

    module = types.ModuleType("wandb")
    module.inits, module.logs, module.finished = [], [], 0
    module.api = types.SimpleNamespace(api_key=None)

    def init(**kwargs):
        module.inits.append(kwargs)
        return FakeRun(module)

    class Video:
        def __init__(self, data, fps=4, format=None, caption=None):
            self.shape, self.fps = tuple(data.shape), fps

    module.init, module.Video = init, Video
    return module


# ---------------------------------------------------------------- scheduler


class RatioSchedulerTest(unittest.TestCase):
    # nanovvm optim.build_scheduler, steps_per_epoch 25, epochs 2 (50 steps):
    # cosine, warmup_ratio 0.1 (5 steps, from zero), decay_end_ratio 0.9 (step 45),
    # lr 2e-4 -> final_lr 2e-5. Values printed by NanoVVM's own scheduler.
    NANOVVM_COSINE = [
        4e-05, 8e-05, 0.00012, 0.00016, 0.0002, 0.0002, 0.00019972256003598153,
        0.0001988919506535624, 0.0001975132928357909, 0.00019559508646656385,
        0.00019314915792601581, 0.0001901905871769531, 0.00018673761479186832,
        0.00018281152949374527, 0.00017843653690400278, 0.00017363961030678927,
        0.00016845032434971652, 0.0001629006727063226, 0.0001570248708244354,
        0.00015085914497655923, 0.00014444150891285807, 0.00013781152949374526,
        0.0001310100827470315, 0.00012407910185362078, 0.00011706131861550607,
        0.00011000000000000002, 0.00010293868138449399, 9.592089814637921e-05,
        8.898991725296855e-05, 8.218847050625476e-05, 7.555849108714192e-05,
        6.91408550234408e-05, 6.297512917556461e-05, 5.709932729367744e-05,
        5.154967565028348e-05, 4.6360389693210735e-05, 4.156346309599723e-05,
        3.718847050625475e-05, 3.326238520813171e-05, 2.98094128230469e-05,
        2.6850842073984196e-05, 2.4404913533436186e-05, 2.2486707164209112e-05,
        2.1108049346437613e-05, 2.0277439964018484e-05, 2e-05, 2e-05, 2e-05, 2e-05, 2e-05,
        2e-05, 2e-05,
    ]
    # The ladder recipe at steps_per_epoch 327: warmup round(6.54) = 7, decay end 327.
    NANOVVM_LADDER_327 = {
        0: 2.857142857142857e-05, 1: 5.714285714285714e-05, 6: 0.0002, 7: 0.0002,
        8: 0.00019999566280633953, 167: 0.00011000000000000002, 326: 2.0004337193660482e-05,
        327: 2e-05,
    }
    # linear, no warmup_from_zero, 3 epochs x 2 steps: round(1.5) = 2 and round(4.5) = 4
    # (round half to even, as in NanoVVM).
    NANOVVM_LINEAR = [0.001, 0.001, 0.001, 0.00055, 0.0001, 0.0001, 0.0001]

    def test_cosine_matches_nanovvm_every_step(self):
        config = {"name": "cosine", "warmup_ratio": 0.1, "decay_end_ratio": 0.9,
                  "final_lr": 2e-5, "warmup_from_zero": True}
        scheduler = build_scheduler(make_optimizer(2e-4), config, steps_per_epoch=25, epochs=2)
        trace = lr_trace(scheduler, len(self.NANOVVM_COSINE))
        self.assertEqual(trace, self.NANOVVM_COSINE)
        self.assertEqual(trace[0], 2e-4 / 5)  # step 0 with warmup_from_zero
        self.assertEqual(trace[4], 2e-4)  # last warmup step
        self.assertEqual(trace[25], 0.00011000000000000002)  # mid-decay
        self.assertEqual(trace[49], 2e-5)  # last step of the run
        resolved = resolve_scheduler_steps(config, peak_lr=2e-4, steps_per_epoch=25, epochs=2)
        self.assertEqual(
            (resolved["total_steps"], resolved["warmup_steps"], resolved["decay_end_steps"]),
            (50, 5, 45),
        )

    def test_ladder_recipe_matches_nanovvm(self):
        config = load_config(ladder_path("tiny"))["training"]["scheduler"]
        scheduler = build_scheduler(make_optimizer(2e-4), config, steps_per_epoch=327, epochs=1)
        trace = lr_trace(scheduler, 328)
        self.assertEqual({step: trace[step] for step in self.NANOVVM_LADDER_327},
                         self.NANOVVM_LADDER_327)

    def test_linear_rounding_matches_nanovvm(self):
        config = {"name": "linear", "warmup_ratio": 0.25, "decay_end_ratio": 0.75,
                  "final_lr": 1e-4, "warmup_from_zero": False}
        scheduler = build_scheduler(make_optimizer(1e-3), config, steps_per_epoch=2, epochs=3)
        self.assertEqual(lr_trace(scheduler, 7), self.NANOVVM_LINEAR)

    def test_mix5_schedule(self):
        # NanoVVM's mix5 train split: 20,978,653 clips -> 327,791 steps of batch 64.
        config = load_config(ladder_path("tiny"))["training"]["scheduler"]
        steps_per_epoch = 20_978_653 // 64
        resolved = resolve_scheduler_steps(config, peak_lr=2e-4, steps_per_epoch=steps_per_epoch,
                                           epochs=1)
        self.assertEqual(
            (resolved["total_steps"], resolved["warmup_steps"], resolved["decay_end_steps"]),
            (327_791, 6_556, 327_791),
        )

    def test_mixed_and_invalid_keys_are_rejected(self):
        mixed = [
            {"name": "cosine", "warmup_ratio": 0.02, "warmup_steps": 10},
            {"name": "cosine", "warmup_ratio": 0.02, "warmup_steps": 0},
            {"name": "linear", "warmup_ratio": 0.1, "warmup_epochs": 1},
            {"name": "cosine", "decay_end_ratio": 1.0, "decay_end_epoch": 3},
            {"name": "cosine", "warmup_ratio": 0.1, "decay_end_steps": 50},
        ]
        for config in mixed:
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "mixes"):
                build_scheduler(make_optimizer(2e-4), config, steps_per_epoch=10, epochs=1)
        invalid = [
            ({"name": "cosine", "warmup_ratio": 0.5, "decay_end_ratio": 0.4}, "warmup_ratio"),
            ({"name": "cosine", "warmup_ratio": 0.1, "decay_end_ratio": 1.5}, "warmup_ratio"),
            ({"name": "cosine", "warmup_ratio": -0.1}, "warmup_ratio"),
            ({"name": "cosine", "warmup_ratio": 0.1, "final_lr": 3e-4}, "final_lr"),
            ({"name": "cosine", "warmup_ratio": 0.1, "base_lr": 2e-4}, "Unknown"),
            ({"name": "constant", "warmup_ratio": 0.1}, "linear or cosine"),
            ({"name": "cosine", "warmup_ratio": 0.1, "warmup_from_zero": "yes"}, "boolean"),
        ]
        for config, message in invalid:
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, message):
                build_scheduler(make_optimizer(2e-4), config, steps_per_epoch=10, epochs=1)
        with self.assertRaisesRegex(ValueError, "epochs"):
            build_scheduler(make_optimizer(2e-4), {"name": "cosine", "warmup_ratio": 0.1},
                            steps_per_epoch=10)
        # An inherited step key switched off with null is not a mix.
        build_scheduler(make_optimizer(2e-4),
                        {"name": "cosine", "warmup_steps": None, "warmup_ratio": 0.1},
                        steps_per_epoch=10, epochs=1)

    def test_step_and_epoch_keys_unchanged(self):
        # Expected values from the implementation before the ratio keys were added (V-RAE 6a11aef).
        linear_epochs = [
            1.6666666666666667e-05, 3.3333333333333335e-05, 5e-05, 6.666666666666667e-05,
            8.333333333333334e-05, 0.0001, 0.0001, 9.583333333333334e-05, 9.166666666666667e-05,
            8.75e-05, 8.333333333333334e-05, 7.916666666666666e-05, 7.500000000000001e-05,
            7.083333333333332e-05, 6.666666666666668e-05, 6.25e-05, 5.8333333333333326e-05,
            5.416666666666668e-05, 5e-05, 5e-05,
        ]
        cosine_steps = [
            0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0002897237368890237, 0.0002604594154601839,
            0.00021666226336928708, 0.000165, 0.00011333773663071288, 6.954058453981609e-05,
            4.027626311097629e-05, 2.9999999999999997e-05, 2.9999999999999997e-05,
        ]
        scheduler = build_scheduler(
            make_optimizer(1e-4),
            {"name": "linear", "warmup_epochs": 2, "decay_end_epoch": 6, "base_lr": 1e-4,
             "final_lr": 5e-5, "warmup_from_zero": True},
            steps_per_epoch=3,
        )
        self.assertEqual(lr_trace(scheduler, 20), linear_epochs)
        scheduler = build_scheduler(
            make_optimizer(3e-4),
            {"name": "cosine", "warmup_steps": 4, "decay_end_steps": 12, "final_lr": 3e-5},
            steps_per_epoch=5,
        )
        self.assertEqual(lr_trace(scheduler, 14), cosine_steps)
        scheduler = build_scheduler(make_optimizer(3e-4), {"name": "constant", "warmup_steps": 0})
        self.assertEqual(lr_trace(scheduler, 3), [0.0003] * 3)
        with self.assertRaisesRegex(ValueError, "steps_per_epoch"):
            build_scheduler(make_optimizer(1e-4), {"name": "cosine", "decay_end_steps": 5})
        with self.assertRaisesRegex(ValueError, "no implicit warmup"):
            build_scheduler(make_optimizer(1e-4), {"name": "constant", "warmup_steps": 2})


# ---------------------------------------------------------------- clipping and metrics


class ClippingTest(unittest.TestCase):
    def test_large_gradient_is_clipped_and_norm_returned(self):
        parameter = nn.Parameter(torch.zeros(3, 4))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        parameter.grad = torch.full((3, 4), 10.0)
        stepped, norm = clip_and_step(optimizer, max_grad_norm=1.0, parameters=[parameter])
        self.assertTrue(stepped)
        self.assertTrue(torch.allclose(norm, torch.tensor(math.sqrt(1200.0))))
        self.assertTrue(torch.allclose(parameter.grad.norm(), torch.tensor(1.0), atol=1e-6))
        # SGD with lr 1 applied exactly the clipped gradient.
        self.assertTrue(torch.allclose(parameter.detach().norm(), torch.tensor(1.0), atol=1e-6))

    def test_without_clipping_and_old_api(self):
        parameter = nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        parameter.grad = torch.full((2,), 10.0)
        self.assertEqual(clip_and_step(optimizer), (True, None))
        self.assertTrue(torch.equal(parameter.detach(), torch.full((2,), -10.0)))
        parameter.grad = torch.full((2,), 10.0)
        self.assertIs(optimizer_step(optimizer, max_grad_norm=1.0, parameters=[parameter]), True)
        self.assertIs(optimizer_step(optimizer), True)


class TrainMetricsTest(unittest.TestCase):
    @staticmethod
    def micro(total, full, base):
        return {name: torch.tensor(value) for name, value in
                zip(("loss", "loss_full", "loss_base"), (total, full, base))}

    def test_window_and_epoch_means(self):
        metrics = nv_train.TrainMetrics(torch.device("cpu"), accumulation_steps=2)
        # Step 1: micro losses (3,2,1) and (5,3,2) -> step loss (4,2.5,1.5); norm 4 over 16 values.
        metrics.add_microbatch(self.micro(3.0, 2.0, 1.0))
        metrics.add_microbatch(self.micro(5.0, 3.0, 2.0))
        metrics.end_step(stepped=True, grad_norm=torch.tensor(4.0), grad_numel=16, clip_grad=1.0)
        # A skipped update (GradScaler overflow) does not enter the window.
        metrics.add_microbatch(self.micro(7.0, 4.0, 3.0))
        metrics.add_microbatch(self.micro(9.0, 5.0, 4.0))
        metrics.end_step(stepped=False, grad_norm=torch.tensor(99.0), grad_numel=16, clip_grad=1.0)
        # Step 2: (1,.5,.5) and (3,1.5,1.5) -> (2,1,1); norm 0.5 is not clipped.
        metrics.add_microbatch(self.micro(1.0, 0.5, 0.5))
        metrics.add_microbatch(self.micro(3.0, 1.5, 1.5))
        metrics.end_step(stepped=True, grad_norm=torch.tensor(0.5), grad_numel=16, clip_grad=1.0)
        self.assertEqual(
            metrics.log_values(clip_grad=1.0),
            {
                "loss/train": 3.0,
                "loss/train_full": 1.75,
                "loss/train_base": 1.25,
                "optimizer/grad_norm": 2.25,  # (4 + 0.5) / 2
                "optimizer/grad_rms": 0.5625,  # (4/4 + 0.5/4) / 2
                "optimizer/grad_clip_scale": 0.625,  # (1/4 + 1/1) / 2
            },
        )
        metrics.add_microbatch(self.micro(2.0, 1.0, 1.0))
        metrics.add_microbatch(self.micro(2.0, 1.0, 1.0))
        metrics.end_step(stepped=True, grad_norm=None, grad_numel=0, clip_grad=None)
        self.assertEqual(
            metrics.log_values(clip_grad=None),
            {"loss/train": 2.0, "loss/train_full": 1.0, "loss/train_base": 1.0},
        )
        # Every micro-batch of the epoch, the skipped update's included (as NanoVVM).
        self.assertEqual(metrics.end_epoch(), {"loss/train_epoch": 32.0 / 8})
        self.assertIsNone(metrics.end_epoch())

    def test_parameter_metrics(self):
        module = nn.Linear(3, 4, bias=False)
        frozen = nn.Linear(5, 5)
        frozen.requires_grad_(False)
        with torch.no_grad():
            module.weight.fill_(2.0)
        values = nv_train.parameter_metrics(nn.ModuleList([module, frozen]))
        self.assertAlmostEqual(values["parameters/norm"], math.sqrt(12 * 4.0), places=6)
        self.assertAlmostEqual(values["parameters/rms"], 2.0, places=6)


# ---------------------------------------------------------------- W&B


class WandbArgumentsTest(unittest.TestCase):
    def init_kwargs(self, config, env):
        module = fake_wandb_module()
        with (
            replaced_module("wandb", module),
            mock.patch.dict(os.environ, env),
        ):
            for key in ("WANDB_MODE", "WANDB_API_KEY"):
                if key not in env:
                    os.environ.pop(key, None)
            logger = WandbLogger(config, enabled=True, run_dir="unused", run_id="run-id")
            logger.log({"loss/train": 1.0}, step=5)
            logger.finish()
        self.assertEqual(module.logs, [(5, {"loss/train": 1.0})])
        return module.inits[0]

    def test_ladder_destination_reaches_wandb_init(self):
        from vrae.training.common.wandb import bind_wandb_run_name

        config = load_config(ladder_path("small"))
        wandb_config = bind_wandb_run_name(config["wandb"], config["run_name"])
        # The user exports ENTITY/PROJECT for NanoVVM; V-RAE must not read them.
        kwargs = self.init_kwargs(
            wandb_config,
            {"WANDB_API_KEY": "test-key", "ENTITY": "someone-else", "PROJECT": "video-gen"},
        )
        self.assertEqual(kwargs["entity"], "zhaosh5t5an")
        self.assertEqual(kwargs["project"], "nanovvm")
        self.assertEqual(kwargs["name"], ladder_path("small").stem)
        self.assertEqual(kwargs["mode"], "online")
        self.assertEqual(kwargs["resume"], "allow")
        self.assertEqual(kwargs["id"], "run-id")
        self.assertEqual(kwargs["group"], "nanovvm_video_prediction")
        self.assertEqual(kwargs["tags"][-3:], ["mix5", "scaling_ladder", "small"])

    def test_without_entity_unchanged_and_offline_fallback(self):
        config = load_config(ROOT / "configs/training/nanovvm_video_pred/eupe_mix5.yaml")["wandb"]
        kwargs = self.init_kwargs(dict(config, name="x"), {})
        self.assertNotIn("entity", kwargs)
        self.assertEqual((kwargs["project"], kwargs["mode"]), ("V-RAE", "offline"))
        ladder = load_config(ladder_path("tiny"))["wandb"]
        kwargs = self.init_kwargs(dict(ladder, name="x"), {})  # online without an API key
        self.assertEqual((kwargs["entity"], kwargs["mode"]), ("zhaosh5t5an", "offline"))

    def test_invalid_entity_is_rejected(self):
        from vrae.training.common.wandb import validate_wandb_config

        config = load_config(ladder_path("tiny"))["wandb"]
        for entity in ("", "  ", 3):
            with self.subTest(entity=entity), self.assertRaisesRegex(ValueError, "entity"):
                validate_wandb_config(dict(config, entity=entity))


# ---------------------------------------------------------------- data augmentation


class PreprocessingEqualityTest(unittest.TestCase):
    SHAPES = ((256, 256), (448, 448), (480, 640), (720, 1280), (640, 480), (181, 321))

    def test_ported_preprocessing_equals_nanovvm(self):
        self.assertEqual(nv_data.PREPROCESSING, NANOVVM_PREPROCESSING)
        generator = torch.Generator().manual_seed(0)
        for height, width in self.SHAPES:
            with self.subTest(size=(height, width)):
                video = torch.randint(0, 256, (17, 3, height, width), dtype=torch.uint8,
                                      generator=generator)
                expected = spatial_preprocess(video.clone(), 256,
                                              preprocessing=copy.deepcopy(NANOVVM_PREPROCESSING))
                actual = nv_data.spatial_preprocess(video.clone(), 256,
                                                    preprocessing=nv_data.PREPROCESSING)
                self.assertEqual(tuple(actual.shape), (17, 3, 256, 256))
                self.assertEqual(actual.dtype, torch.uint8)
                self.assertTrue(torch.equal(actual, expected))


# ---------------------------------------------------------------- ladder configs


class LadderConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = {size: load_config(ladder_path(size)) for size in SIZES}
        cls.mix5 = load_config(ROOT / "configs/training/nanovvm_video_pred/eupe_mix5.yaml")

    def test_layout(self):
        files = sorted(str(path.relative_to(LADDER)) for path in LADDER.rglob("*.yaml"))
        self.assertEqual(
            files, sorted(["_mix5_common.yaml"]
                          + [str(ladder_path(size).relative_to(LADDER)) for size in SIZES])
        )

    def test_sizes_and_parameter_counts(self):
        default = load_config(ROOT / "configs/models/dit_first_frame_prediction.yaml")["dit"]
        size_keys = {"hidden_size", "depth", "num_heads", "encoder_depth", "mlp_ratio"}
        stage1 = {"patch_size": 16, "temporal_compression_ratio": 4, "hidden_size": 768}
        for size, row in LADDER_TABLE.items():
            with self.subTest(size=size):
                config = self.configs[size]
                dit = config["dit"]
                k, depth = row["k"], row["depth"][0]
                self.assertEqual(config["run_name"], ladder_path(size).stem)
                self.assertEqual(dit["hidden_size"], [72 * k, 128 * k])
                self.assertEqual(dit["num_heads"], [k, k])
                self.assertEqual(dit["depth"], [depth, max(1, round(depth / 14))])
                self.assertEqual(dit["encoder_depth"], round(depth * 8 / 28))
                self.assertTrue(1 <= dit["encoder_depth"] <= depth)
                self.assertEqual(
                    (dit["hidden_size"], dit["depth"], dit["num_heads"], dit["encoder_depth"],
                     dit["mlp_ratio"]),
                    (row["hidden"], row["depth"], row["heads"], row["encoder_depth"], 4.0),
                )
                self.assertEqual({key: value for key, value in dit.items() if key not in size_keys},
                                 {key: value for key, value in self.mix5["dit"].items()
                                  if key not in size_keys})
                self.assertEqual(dit["frame_dropout"], default["frame_dropout"])
                with torch.device("meta"):
                    model = nv_train.build_nanovvm_dit(config, stage1)
                counts = [
                    sum(p.numel() for p in module.parameters())
                    for module in (model, model.encoder_blocks, model.decoder_blocks)
                ]
                self.assertEqual([round(count / 1e6, 1) for count in counts],
                                 [row["total"], row["encoder"], row["decoder"]])
                self.assertLess(abs(counts[0] / (row["nanovvm"] * 1e6) - 1), 0.04)

    def test_recipe(self):
        tiny = self.configs["tiny"]
        for size, config in self.configs.items():
            with self.subTest(size=size):
                nv_train.validate_build(config)
                self.assertEqual(config["training"], LADDER_TRAINING)
                self.assertEqual(
                    config["wandb"],
                    {
                        **self.mix5["wandb"],
                        "entity": "zhaosh5t5an",
                        "project": "nanovvm",
                        "mode": "online",
                        "resume": "allow",
                        "log_interval": 5,
                        "sample_interval": 500,
                        "sample_count": 4,
                        "tags": [*self.mix5["wandb"]["tags"], "mix5", "scaling_ladder", size],
                    },
                )
                # Data and the shared future-latent statistics are eupe_mix5.yaml's.
                self.assertEqual(config["data"], self.mix5["data"])
                self.assertEqual(config["data"]["preprocessing"], NANOVVM_PREPROCESSING)
                self.assertEqual(
                    (config["data"]["image_size"], config["data"]["rgb_seq_len"],
                     config["data"]["frame_interval"]),
                    (256, 17, 3),
                )
                self.assertEqual(config["latent_normalizer"], self.mix5["latent_normalizer"])
                # The six sizes differ only in run_name, the DiT size and the size tag.
                strip = copy.deepcopy(config)
                reference = copy.deepcopy(tiny)
                for item in (strip, reference):
                    item.pop("run_name")
                    item["wandb"]["tags"] = item["wandb"]["tags"][:-1]
                    for key in ("hidden_size", "depth", "num_heads", "encoder_depth"):
                        item["dit"].pop(key)
                self.assertEqual(strip, reference)
        from vrae.training.common.optim import build_optimizer

        optimizer = build_optimizer([nn.Parameter(torch.zeros(2))], tiny["training"]["optimizer"])
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        defaults = optimizer.defaults
        self.assertEqual(
            (defaults["lr"], defaults["betas"], defaults["weight_decay"], defaults["eps"]),
            (2e-4, (0.9, 0.95), 0.0, 1e-8),
        )


if __name__ == "__main__":
    unittest.main()
