"""First-frame conditioned NanoVVM future prediction: CPU tests on synthetic data and weights.

Run from the repository root:
    PYTHONPATH=src python -m unittest tests.test_nanovvm_video_pred -v
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrae.config import load_config  # noqa: E402
from vrae.models.dit.guidance import guided_model_forward  # noqa: E402
from vrae.models.dit.transport import FlowMatchingTransport  # noqa: E402
from vrae.paths import ProjectPaths  # noqa: E402
from vrae.registry import DIT_MODELS, register_builtin_models  # noqa: E402
from vrae.training.common.latent_norm import LatentNormalizer  # noqa: E402
from vrae.training.nanovvm_video_pred import data as nv_data  # noqa: E402
from vrae.training.nanovvm_video_pred.latents import (  # noqa: E402
    encode_condition_and_future,
    encode_first_frame,
)
from vrae.training.nanovvm_video_pred.manifest import load_clip_manifest  # noqa: E402

SPAN = 49
EUPE_ENCODER = load_config(ROOT / "configs/models/vrae_eupe_b_k7.yaml")["model"]["encoder"]
LADDER_TINY = (
    ROOT
    / "configs/training/nanovvm_video_pred/scaling-ladder/tiny"
    / "vrae_eupe_mix5_tiny_xpred_interval3_cond1gen16_bs64_lr2e-4_wd0_ep1.yaml"
)
# The scalar keys of one training log line, under NanoVVM's names (nanovvm docs/wandb_metrics.md).
TRAIN_LOG_KEYS = {
    "loss/train",
    "loss/train_full",
    "loss/train_base",
    "optimizer/lr",
    "optimizer/grad_norm",
    "optimizer/grad_rms",
    "optimizer/grad_clip_scale",
    "parameters/norm",
    "parameters/rms",
    "train/epoch",
    "train/global_batch_size",
    "data/wait_seconds_per_step",
}
SAMPLE_LOG_KEYS = (
    {"samples/future16_real_vs_predicted/video_000"},
    {"samples/display17_first_frame_plus_future/video_000"},
)


# ---------------------------------------------------------------- helpers


def write_manifest(
    directory: Path,
    records,
    *,
    split="train",
    rgb_seq_len=17,
    frame_interval=3,
    total=None,
    sha=None,
) -> tuple[Path, Path]:
    """Write a NanoVVM manifest.json/meta.yaml pair in the generator's byte format."""

    directory.mkdir(parents=True, exist_ok=True)
    keys = ("video_path", "start_frame", "end_frame_exclusive", "label")
    lines = [json.dumps({key: record[key] for key in keys}) for record in records]
    payload = ("[\n" + ",\n".join(lines) + "\n]\n").encode()
    span = 1 + (rgb_seq_len - 1) * frame_interval
    meta = {
        "schema_version": 1,
        "split": split,
        "sampling": {
            "rgb_seq_len": rgb_seq_len,
            "frame_interval": frame_interval,
            "source_frame_span": span,
            "clip_stride": span,
            "segmentation": "sequential_non_overlapping",
            "incomplete_clip_policy": "discard",
        },
        "statistics": {"total_clips": len(records) if total is None else total},
        "labels": {"rule": "top_level_directory", "classes": ["a"]},
        "manifest": {
            "filename": "manifest.json",
            "path_type": "relative",
            "start_frame_inclusive": True,
            "end_frame_exclusive": True,
            "sha256": hashlib.sha256(payload).hexdigest() if sha is None else sha,
        },
    }
    (directory / "manifest.json").write_bytes(payload)
    (directory / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    return directory / "manifest.json", directory / "meta.yaml"


def record(path: str, start: int, label: int = 0) -> dict:
    return {
        "video_path": path,
        "start_frame": start,
        "end_frame_exclusive": start + SPAN,
        "label": label,
    }


class FakeVideoReader:
    """decord.VideoReader stand-in: every pixel of frame i equals i (mod 256)."""

    videos: dict[str, tuple[int, int, int]] = {}

    def __init__(self, path, ctx=None, num_threads=1):
        del ctx, num_threads
        self.length, self.height, self.width = self.videos[str(path)]

    def __len__(self):
        return self.length

    def get_batch(self, indices):
        indices = np.asarray(indices)
        frames = np.empty((len(indices), self.height, self.width, 3), dtype=np.uint8)
        for row, index in enumerate(indices):
            frames[row] = int(index) % 256
        return types.SimpleNamespace(asnumpy=lambda: frames)


@contextmanager
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


@contextmanager
def fake_decord():
    module = types.SimpleNamespace(VideoReader=FakeVideoReader, cpu=lambda index: index)
    with replaced_module("decord", module):
        yield


def make_data_root(base: Path) -> Path:
    """Two cameras with identical file names; train/val holdout manifests (non-contiguous)."""

    root = base / "nanovvm"
    for camera in ("cam_a", "cam_b"):
        videos = root / "toy_video" / "data" / camera
        videos.mkdir(parents=True)
        for name in ("ep0.mp4", "ep1.mp4"):
            (videos / name).write_bytes(b"")  # decoded by FakeVideoReader
            FakeVideoReader.videos[str((videos / name).resolve())] = (160, 120, 200)
        manifests = root / "toy_video" / "manifests" / camera / "cond1gen16_interval3" / "split"
        # Train skips clip 1 of ep0 (held out); val holds exactly that clip.
        write_manifest(
            manifests / "train",
            [
                record("ep0.mp4", 0),
                record("ep0.mp4", 98),
                record("ep1.mp4", 0),
                record("ep1.mp4", 49),
            ],
        )
        write_manifest(manifests / "val", [record("ep0.mp4", 49)], split="val")
    return root


def toy_sources(split: str) -> list[dict]:
    return [
        {
            "name": f"toy/{camera}",
            "dataset": "toy",
            "root": f"toy_video/data/{camera}",
            "manifest": f"toy_video/manifests/{camera}/cond1gen16_interval3/split/{split}/manifest.json",
            "manifest_meta": f"toy_video/manifests/{camera}/cond1gen16_interval3/split/{split}/meta.yaml",
        }
        for camera in ("cam_a", "cam_b")
    ]


def task_config(**overrides) -> dict:
    config = load_config(ROOT / "configs/training/nanovvm_video_pred/eupe_bridgedatav2.yaml")
    config["data"]["train"] = {"sources": toy_sources("train")}
    config["data"]["val"] = {"sources": toy_sources("val")}
    config["dit"].update(
        {
            "hidden_size": [64, 64],
            "depth": [2, 1],
            "encoder_depth": 1,
            "num_heads": [4, 4],
            "gradient_checkpointing": False,
        }
    )
    config["training"].update(
        {
            "global_batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "precision": "fp32",
            "checkpoint_interval": 1,
        }
    )
    config["latent_normalizer"].update(
        {"path": "ckpts/stats/toy_future.pt", "fit_clips": 3, "batch_size": 2}
    )
    config["sampling"].update({"steps": 2, "cfg_scale": 1.5})
    config["wandb"].update(
        {"mode": "disabled", "log_interval": 1, "sample_interval": 1, "sample_count": 1}
    )
    for key, value in overrides.items():
        config[key] = value
    return config


class FakeEUPEBackbone(nn.Module):
    """Tiny EUPE-shaped ViT-B/16 (12 blocks, width 768) exposing get_intermediate_layers."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 768
        self.patch_size = 16
        self.patch_embed = nn.Conv2d(3, 768, 16, 16)
        self.blocks = nn.ModuleList(nn.Linear(768, 768) for _ in range(12))
        self.norm = nn.LayerNorm(768)

    def get_intermediate_layers(
        self, frames, n, reshape=False, return_class_token=False, norm=True
    ):
        hidden = self.patch_embed(frames).flatten(2).transpose(1, 2)
        outputs = []
        for index, block in enumerate(self.blocks):
            hidden = hidden + torch.tanh(block(hidden)) * 0.1
            if index in n:
                outputs.append(self.norm(hidden) if norm else hidden)
        return tuple(outputs)


def tiny_stage1():
    from vrae.models.adapter import VRAELatentAdapter
    from vrae.models.autoencoder import VRAE
    from vrae.models.decoder import VRAEDecoder
    from vrae.models.encoders.eupe import EUPEAdapter
    from vrae.models.pooling import TemporalAttentionPool

    torch.manual_seed(0)
    encoder = EUPEAdapter(dict(EUPE_ENCODER, runtime_image_size=256), backbone=FakeEUPEBackbone())
    pool = TemporalAttentionPool(768, 4, num_heads=16)
    decoder = VRAEDecoder(
        {
            "hidden_size": 64,
            "depth": 1,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "patch_size": 16,
            "tubelet_size": 4,
            "image_size": [256, 256],
            "attention_backend": "sdpa",
        },
        input_dim=768,
    )
    with torch.no_grad():
        pool.query.normal_(std=0.5)
        pool.time_bias.normal_(std=0.5)
    vrae = VRAE(encoder, pool, decoder).requires_grad_(False).eval()
    metadata = vrae.metadata()
    metadata["checkpoint_weight_source"] = "ema"
    return VRAELatentAdapter(vrae, metadata, precision="fp32")


def small_dit(**overrides) -> nn.Module:
    register_builtin_models()
    config = {
        "name": "vrae_first_frame_prediction_dit",
        "hidden_size": [32, 48],
        "depth": [3, 2],
        "encoder_depth": 2,
        "num_heads": [4, 4],
        "frame_dropout": 0.1,
    }
    arguments = {"grid_size": (4, 4), "in_channels": 8, "frame_channels": 12, "future_chunks": 4}
    arguments.update(overrides)
    return DIT_MODELS.build(config, **arguments)


class FakeWandbRun:
    def __init__(self, module):
        self.module = module

    def log(self, values, step=None):
        self.module.logs.append((step, dict(values)))

    def finish(self):
        pass


def fake_wandb_module() -> types.ModuleType:
    """Records wandb.init arguments and run.log calls (wandb is not installed in the test env)."""

    module = types.ModuleType("wandb")
    module.inits, module.logs = [], []
    module.api = types.SimpleNamespace(api_key=None)

    def init(**kwargs):
        module.inits.append(kwargs)
        return FakeWandbRun(module)

    class Video:
        def __init__(self, data, fps=4, format=None, caption=None):
            self.shape = tuple(data.shape)

    module.init, module.Video = init, Video
    return module


def assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-5) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    difference = (actual.float() - expected.float()).abs().max().item() if actual.numel() else 0.0
    if difference > atol:
        raise AssertionError(f"max abs difference {difference} > {atol}")


@torch.no_grad()
def randomize(module: nn.Module, std: float = 0.05, seed: int = 1) -> None:
    """Break the AdaLN-Zero / zero-head initialization so every branch reaches the output."""

    generator = torch.Generator().manual_seed(seed)
    for parameter in module.parameters():
        parameter.add_(torch.randn(parameter.shape, generator=generator) * std)


# ---------------------------------------------------------------- data contract


class ManifestContractTest(unittest.TestCase):
    def test_noncontiguous_holdout_loads_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            records = [record("a.mp4", 49), record("a.mp4", 147), record("b.mp4", 0)]
            manifest, meta = write_manifest(directory / "ok", records)
            clips = load_clip_manifest(manifest, meta, rgb_seq_len=17, frame_interval=3)
            self.assertEqual(clips.start_frames.tolist(), [49, 147, 0])
            self.assertEqual(clips.record(0)["end_frame_exclusive"], 98)

            bad_sha = write_manifest(directory / "sha", records, sha="0" * 64)
            with self.assertRaisesRegex(ValueError, "sha256"):
                load_clip_manifest(*bad_sha, rgb_seq_len=17, frame_interval=3)
            bad_count = write_manifest(directory / "count", records, total=4)
            with self.assertRaisesRegex(ValueError, "total_clips"):
                load_clip_manifest(*bad_count, rgb_seq_len=17, frame_interval=3)
            with self.assertRaisesRegex(ValueError, "rgb_seq_len=16"):
                load_clip_manifest(
                    *write_manifest(directory / "len", records, rgb_seq_len=16),
                    rgb_seq_len=17,
                    frame_interval=3,
                )
            off_grid = write_manifest(
                directory / "grid",
                [{**record("a.mp4", 49), "start_frame": 50, "end_frame_exclusive": 99}],
            )
            with self.assertRaisesRegex(ValueError, "multiple of 49"):
                load_clip_manifest(*off_grid, rgb_seq_len=17, frame_interval=3)
            unordered = write_manifest(
                directory / "order", [record("a.mp4", 98), record("a.mp4", 49)]
            )
            with self.assertRaisesRegex(ValueError, "chronological"):
                load_clip_manifest(*unordered, rgb_seq_len=17, frame_interval=3)


class FrameProtocolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_data_root(Path(self.tmp.name))
        self.paths = ProjectPaths(project_root=Path(self.tmp.name), datasets={"nanovvm": self.root})

    def tearDown(self):
        self.tmp.cleanup()

    def test_nonzero_start_frames(self):
        config = task_config()
        with fake_decord():
            dataset = nv_data.build_nanovvm_dataset(config, self.paths, split="val")
            item = dataset[0]
        self.assertEqual(item["metadata"]["start_frame"], 49)
        self.assertEqual(item["frame_indices"].tolist(), [49 + 3 * k for k in range(17)])
        self.assertEqual(tuple(item["condition"].shape), (3, 256, 256))
        self.assertEqual(tuple(item["future"].shape), (16, 3, 256, 256))
        self.assertEqual(item["condition"].dtype, torch.uint8)
        self.assertTrue(torch.all(item["condition"] == 49))
        future_frames = [int(frame.unique().item()) for frame in item["future"]]
        self.assertEqual(future_frames, list(range(52, 98, 3)))

    def test_sources_concat_and_distinct_ids(self):
        config = task_config()
        with fake_decord():
            dataset = nv_data.build_nanovvm_dataset(config, self.paths, split="train")
            ids = [dataset.clip_record(index)["sample_id"] for index in range(len(dataset))]
        self.assertEqual(len(dataset), 8)
        self.assertEqual(len(set(ids)), 8)
        self.assertIn("toy/cam_a:ep0.mp4:98", ids)
        self.assertIn("toy/cam_b:ep0.mp4:98", ids)
        self.assertEqual(dataset.locate(5)[0].name, "toy/cam_b")

    def test_single_source_form_and_path_errors(self):
        config = task_config()
        single = toy_sources("val")[0]
        config["data"]["val"] = {key: single[key] for key in ("root", "manifest", "manifest_meta")}
        with fake_decord():
            dataset = nv_data.build_nanovvm_dataset(config, self.paths, split="val")
        self.assertEqual(dataset.clip_record(0)["sample_id"], "default:ep0.mp4:49")
        interpolated = task_config()
        interpolated["data"]["train"]["sources"][0]["root"] = "${paths.data}/toy_video/data/cam_a"
        with self.assertRaisesRegex(ValueError, "interpolation"):
            nv_data.split_source_entries(interpolated, "train")
        missing = task_config()
        missing["data"]["train"]["sources"][1]["manifest"] = "toy_video/nope/manifest.json"
        with self.assertRaisesRegex(FileNotFoundError, "toy/cam_b: manifest is not a file"):
            nv_data.resolve_split_sources(missing, self.paths, "train")

    @unittest.skipUnless(
        importlib.util.find_spec("cv2") and importlib.util.find_spec("decord"),
        "needs cv2 and decord",
    )
    def test_real_decord_exact_frames(self):
        import cv2

        # Frame i shows i in binary as seven horizontal black/white bands, which survive
        # MJPG level shifts, the short-side resize, and the center crop (rows are kept).
        path = Path(self.tmp.name) / "real" / "index.avi"
        path.parent.mkdir()
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (200, 126))
        for index in range(120):
            frame = np.zeros((126, 200, 3), dtype=np.uint8)
            for bit in range(7):
                if index >> bit & 1:
                    frame[bit * 18 : (bit + 1) * 18] = 255
            writer.write(frame)
        writer.release()
        frames, indices = nv_data.decode_clip_frames(path, 49, where="test")
        clip = nv_data.spatial_preprocess(frames, 256, preprocessing=nv_data.PREPROCESSING)
        condition, future = nv_data.split_condition_future(clip)
        rows = [int((bit + 0.5) * 256 / 7) for bit in range(7)]
        decoded = [
            sum(
                1 << bit
                for bit, row in enumerate(rows)
                if frame[:, row, 100:156].float().mean() > 127
            )
            for frame in (condition, *future)
        ]
        self.assertEqual(indices.tolist(), list(range(49, 98, 3)))
        self.assertEqual(decoded, list(range(49, 98, 3)))
        self.assertEqual(tuple(clip.shape), (17, 3, 256, 256))


# ---------------------------------------------------------------- stage 1


class Stage1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage1 = tiny_stage1()

    def test_single_image_matches_video_frame(self):
        video = torch.randint(0, 256, (2, 4, 3, 256, 256), dtype=torch.uint8)
        encoder = self.stage1._autoencoder.encoder
        per_frame = encoder(video)  # [B,4,C,h,w]
        single = self.stage1.encode_image_grid(video[:, 2])
        assert_close(single, per_frame[:, 2])
        tokens = self.stage1.image_grid_to_tokens(single)
        reference = self.stage1.grid_to_tokens(per_frame[:, 2:3])[:, 0]
        assert_close(tokens, reference)

    def test_video_path_still_requires_four_frames(self):
        vrae = self.stage1._autoencoder
        for frames in (1, 5):
            with self.assertRaisesRegex(ValueError, "divisible by"):
                vrae.encode_frames(torch.zeros(1, frames, 3, 256, 256))

    def test_non_per_frame_backbone_is_rejected(self):
        vrae = self.stage1._autoencoder
        with mock.patch.object(type(vrae.encoder), "PER_FRAME_ENCODING", False):
            with self.assertRaisesRegex(NotImplementedError, "single-image"):
                vrae.encode_image_features(torch.zeros(1, 3, 256, 256))
        from vrae.models.encoders.vjepa2_1 import VJEPA21Adapter

        self.assertFalse(VJEPA21Adapter.PER_FRAME_ENCODING)

    def test_condition_and_target_are_independent(self):
        generator = torch.Generator().manual_seed(0)
        condition = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.uint8, generator=generator)
        future = torch.randint(0, 256, (2, 16, 3, 256, 256), dtype=torch.uint8, generator=generator)
        features, target = encode_condition_and_future(self.stage1, condition, future)
        self.assertEqual(tuple(features.shape), (2, 256, 768))
        self.assertEqual(tuple(target.shape), (2, 4, 256, 768))
        other_future = torch.randint(0, 256, future.shape, dtype=torch.uint8, generator=generator)
        other_condition = torch.randint(
            0, 256, condition.shape, dtype=torch.uint8, generator=generator
        )
        features_2, target_2 = encode_condition_and_future(self.stage1, condition, other_future)
        features_3, target_3 = encode_condition_and_future(self.stage1, other_condition, future)
        self.assertTrue(torch.equal(features_2, features))
        self.assertTrue(torch.equal(target_3, target))
        self.assertFalse(torch.equal(target_2, target))
        self.assertFalse(torch.equal(features_3, features))


# ---------------------------------------------------------------- DiT


class FirstFrameDiTTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dit = small_dit()
        randomize(self.dit)
        self.dit.eval()
        generator = torch.Generator().manual_seed(3)
        self.noisy = torch.randn(2, 4, 16, 8, generator=generator)
        self.time = torch.tensor([0.3, 0.7])
        self.frames = torch.randn(2, 16, 12, generator=generator)
        self.other_frames = torch.randn(2, 16, 12, generator=generator)

    def test_full_and_base_shapes(self):
        full, base = self.dit(self.noisy, self.time, context=self.frames, return_base=True)
        self.assertEqual(full.shape, self.noisy.shape)
        self.assertEqual(base.shape, self.noisy.shape)
        self.assertEqual(self.dit.future_latent_shape, (4, 16, 8))
        self.assertEqual(self.dit._encoder_positions[:, 0].unique().tolist(), [0, 1, 2, 3])

    def test_dropped_condition_ignores_first_frame(self):
        drop = torch.ones(2, dtype=torch.bool)
        first = self.dit(
            self.noisy,
            self.time,
            frame_features=self.frames,
            condition_drop_mask=drop,
            return_base=True,
        )
        second = self.dit(
            self.noisy,
            self.time,
            frame_features=self.other_frames,
            condition_drop_mask=drop,
            return_base=True,
        )
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_enabled_condition_changes_output(self):
        keep = torch.zeros(2, dtype=torch.bool)
        first = self.dit(
            self.noisy, self.time, frame_features=self.frames, condition_drop_mask=keep
        )
        second = self.dit(
            self.noisy, self.time, frame_features=self.other_frames, condition_drop_mask=keep
        )
        self.assertGreater((first - second).abs().max().item(), 1e-4)

    def test_adaln_condition_contract_and_spatial_alignment(self):
        adapter = self.dit.frame_projection
        condition = self.dit.adaln_condition(
            self.time, self.frames, drop_mask=torch.zeros(2, dtype=torch.bool)
        )
        expected = (self.dit.time_embedder(self.time) + adapter.project(self.frames))[:, None]
        expected = expected.expand(-1, 4, -1, -1).reshape(2, 4 * 16, 32)
        assert_close(condition, expected)
        perturbed = self.frames.clone()
        perturbed[:, 5] += 1.0  # one first-frame patch (row 1, col 1)
        changed = self.dit.adaln_condition(
            self.time, perturbed, drop_mask=torch.zeros(2, dtype=torch.bool)
        )
        moved = (changed - condition).abs().amax(dim=(0, 2)) > 0
        self.assertEqual(moved.nonzero().flatten().tolist(), [5, 21, 37, 53])

    def test_patch2_alignment(self):
        dit = DIT_MODELS.build(
            {
                "name": "vrae_first_frame_prediction_dit",
                "hidden_size": [32, 48],
                "depth": [2, 1],
                "encoder_depth": 1,
                "num_heads": [4, 4],
                "patch_size": 2,
            },
            grid_size=(4, 4),
            in_channels=8,
            frame_channels=12,
            future_chunks=4,
        )
        randomize(dit)
        projected = dit.frame_projection.project(self.frames)
        self.assertEqual(tuple(projected.shape), (2, 4, 32))
        perturbed = self.frames.clone()
        perturbed[:, 4 * 3 + 2] += 1.0  # frame patch (3,2) -> DiT patch (1,1) = index 3
        moved = (dit.frame_projection.project(perturbed) - projected).abs().amax(dim=(0, 2)) > 0
        self.assertEqual(moved.nonzero().flatten().tolist(), [3])

    def test_cfg_uses_explicit_null_branch(self):
        keep = torch.zeros(2, dtype=torch.bool)
        guided = guided_model_forward(
            self.dit,
            self.noisy,
            self.time,
            condition_kwargs={"frame_features": self.frames, "condition_drop_mask": keep},
            unconditional_kwargs={"frame_features": self.frames, "condition_drop_mask": ~keep},
            cfg_scale=2.0,
            ig_scale=1.5,
        )
        full_c, base_c = self.dit(
            self.noisy,
            self.time,
            frame_features=self.frames,
            condition_drop_mask=keep,
            return_base=True,
        )
        full_u, base_u = self.dit(
            self.noisy,
            self.time,
            frame_features=self.other_frames,
            condition_drop_mask=~keep,
            return_base=True,
        )
        conditional = base_c + 1.5 * (full_c - base_c)
        unconditional = base_u + 1.5 * (full_u - base_u)
        assert_close(guided, unconditional + 2.0 * (conditional - unconditional))

    def test_single_step_backward_reaches_condition_branch(self):
        self.dit.train()
        transport = FlowMatchingTransport(
            prediction="x", time_dist_shift=float(np.sqrt(4 * 16 * 8 / 4096))
        )
        clean = torch.randn(2, 4, 16, 8)
        drop = torch.tensor([False, True])
        losses = transport.training_losses(
            self.dit, clean, self.frames, model_kwargs={"condition_drop_mask": drop}
        )
        self.assertEqual(losses["loss_full"].ndim, 0)
        self.assertGreater(losses["loss_base"].item(), 0)
        losses["loss"].backward()
        adapter = self.dit.frame_projection
        for name in ("patch_embed.proj.weight", "projection.1.weight", "null_condition"):
            gradient = dict(adapter.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)
        self.assertGreater(self.dit.base_final_layer.linear.weight.grad.abs().sum().item(), 0)
        # Only the kept sample's frame reaches the projection; the dropped one uses the null.
        projection_grad = adapter.projection[1].weight.grad.clone()
        adapter.zero_grad()
        self.dit.zero_grad()
        transport.training_losses(
            self.dit,
            clean,
            self.frames,
            model_kwargs={"condition_drop_mask": torch.ones(2, dtype=torch.bool)},
        )["loss"].backward()
        self.assertEqual(adapter.projection[1].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(projection_grad.abs().sum().item(), 0)


# ---------------------------------------------------------------- sampling and training


class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage1 = tiny_stage1()

    def test_sample_from_first_frame_only(self):
        from vrae.training.nanovvm_video_pred.sample import (
            load_first_frame_image,
            sample_from_first_frame,
        )

        register_builtin_models()
        dit = DIT_MODELS.build(
            {
                "name": "vrae_first_frame_prediction_dit",
                "hidden_size": [64, 64],
                "depth": [2, 1],
                "encoder_depth": 1,
                "num_heads": [4, 4],
            },
            grid_size=(16, 16),
            in_channels=768,
            frame_channels=768,
            future_chunks=4,
        )
        randomize(dit, std=0.02)
        dit.eval()
        normalizer = LatentNormalizer(torch.zeros(768), torch.ones(768))
        transport = FlowMatchingTransport(prediction="x", time_dist_shift=float(np.sqrt(192)))
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image

            image = Image.fromarray(
                np.random.default_rng(0).integers(0, 255, (180, 320, 3), dtype=np.uint8)
            )
            image.save(Path(tmp) / "frame.png")
            first = load_first_frame_image(Path(tmp) / "frame.png")
        self.assertEqual(tuple(first.shape), (3, 256, 256))
        arguments = dict(base_seed=3407, steps=2, transport=transport, cfg_scale=1.5)
        video = sample_from_first_frame(
            self.stage1, dit, normalizer, first[None], sample_indices=[7], **arguments
        )
        again = sample_from_first_frame(
            self.stage1, dit, normalizer, first[None], sample_indices=[7], **arguments
        )
        other_seed = sample_from_first_frame(
            self.stage1, dit, normalizer, first[None], sample_indices=[8], **arguments
        )
        self.assertEqual(tuple(video.shape), (1, 16, 3, 256, 256))
        self.assertTrue(torch.equal(video, again))
        self.assertFalse(torch.equal(video, other_seed))
        self.assertEqual(tuple(encode_first_frame(self.stage1, first[None]).shape), (1, 256, 768))

    def test_train_resume_and_sample(self):
        from vrae.training.common.contracts import create_run_metadata, load_run_metadata
        from vrae.training.common.wandb import bind_wandb_run_name
        from vrae.config import save_resolved_config
        from vrae.training.nanovvm_video_pred import latent_stats, sample, train as train_module

        wandb_module = fake_wandb_module()
        with (
            tempfile.TemporaryDirectory() as tmp,
            fake_decord(),
            replaced_module("wandb", wandb_module),
            # ENTITY/PROJECT are exported for NanoVVM on this cluster; V-RAE must ignore them.
            mock.patch.dict(
                os.environ,
                {"WANDB_API_KEY": "test-key", "ENTITY": "someone-else", "PROJECT": "video-gen"},
            ),
        ):
            os.environ.pop("WANDB_MODE", None)
            base = Path(tmp)
            root = make_data_root(base)
            paths = ProjectPaths(project_root=base, datasets={"nanovvm": root})
            config = task_config()
            # The scaling ladder's training block (cosine with ratios, clip_grad 1, accumulation
            # 4, AdamW, EMA) and W&B destination. CPU-only overrides: global batch 4 (micro-batch
            # 1), fp32, no workers, and a checkpoint, log line and sample every step. 8 train
            # clips / 4 = 2 optimizer steps: warmup round(0.02 * 2) = 0, decay end 2.
            ladder = load_config(LADDER_TINY)
            config["training"] = {
                **copy.deepcopy(ladder["training"]),
                "global_batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
                "precision": "fp32",
                "checkpoint_interval": 1,
            }
            config["wandb"] = bind_wandb_run_name(
                {
                    **copy.deepcopy(ladder["wandb"]),
                    "log_interval": 1,
                    "sample_interval": 1,
                    "sample_count": 1,
                },
                config["run_name"],
            )
            train_module.validate_build(config)
            loader = mock.Mock(return_value=(self.stage1, {}))
            with (
                mock.patch.object(latent_stats, "load_frozen_stage1", loader),
                mock.patch.object(train_module, "load_frozen_stage1", loader),
                mock.patch.object(sample, "load_frozen_stage1", loader),
            ):
                stats = latent_stats.compute_nanovvm_latent_stats(config, paths)
                normalizer = LatentNormalizer.load(stats)
                self.assertEqual(normalizer.metadata["fit"]["clips"], 3)
                self.assertEqual(normalizer.metadata["future_latent_shape"], [4, 256, 768])
                self.assertEqual(normalizer.metadata["scope"], "future")
                self.assertEqual(len(normalizer.metadata["train_sources"]), 2)

                run = paths.training_run(config["task"], config["run_name"], create=True)
                for directory in ("checkpoints", "samples", "wandb"):
                    (run / directory).mkdir()
                save_resolved_config(create_run_metadata(config), run / "run_metadata.yaml")
                train_module.train(config, paths, run, max_steps=1)
                checkpoint = run / "checkpoints" / "step-00000001.pt"
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                self.assertEqual(payload["task"], "nanovvm_video_pred")
                self.assertEqual(payload["model_metadata"]["future_chunks"], 4)
                self.assertTrue(
                    any(key.startswith("condition_adapter.") for key in payload["model"])
                )
                self.assertTrue(
                    any(key.startswith("condition_adapter.") for key in payload["ema"]["shadow"])
                )
                samples = torch.load(run / "samples" / "step-00000001.pt")
                self.assertEqual(tuple(samples["real_future_rgb"].shape), (1, 16, 3, 256, 256))
                self.assertEqual(tuple(samples["predicted_future"].shape), (1, 16, 3, 256, 256))
                self.assertEqual(tuple(samples["condition_rgb"].shape), (1, 3, 256, 256))
                for parameter in self.stage1._autoencoder.parameters():
                    self.assertFalse(parameter.requires_grad)
                    self.assertIsNone(parameter.grad)

                # The resolved schedule is recorded, and step 1 left the LR one step into it.
                metadata = load_run_metadata(run)
                self.assertEqual(
                    metadata["lr_schedule"],
                    {
                        "name": "cosine",
                        "steps_per_epoch": 2,
                        "epochs": 1,
                        "total_steps": 2,
                        "warmup_steps": 0,
                        "decay_end_steps": 2,
                        "peak_lr": 2e-4,
                        "final_lr": 2e-5,
                        "warmup_from_zero": True,
                    },
                )
                # cosine at progress 1/2: 2e-5 + (2e-4 - 2e-5) / 2; then final_lr at step 2.
                lr_after = {1: 1.1e-4, 2: 2e-5}
                self.assertEqual(payload["scheduler"]["last_epoch"], 1)
                self.assertTrue(
                    math.isclose(
                        payload["optimizer"]["param_groups"][0]["lr"], lr_after[1], rel_tol=1e-12
                    )
                )

                # W&B: the ladder destination reaches wandb.init, with the run id of the run.
                first_init = wandb_module.inits[0]
                self.assertEqual(
                    (
                        first_init["entity"],
                        first_init["project"],
                        first_init["name"],
                        first_init["mode"],
                        first_init["resume"],
                        first_init["id"],
                    ),
                    (
                        "zhaosh5t5an",
                        "nanovvm",
                        config["run_name"],
                        "online",
                        "allow",
                        metadata["wandb_run_id"],
                    ),
                )
                # Exactly the NanoVVM-named keys: the training line, the two sample videos, and
                # the epoch loss (max_steps ends the epoch loop after step 1).
                logs = list(wandb_module.logs)
                self.assertEqual(
                    [(step, set(values)) for step, values in logs],
                    [
                        (1, TRAIN_LOG_KEYS),
                        (1, SAMPLE_LOG_KEYS[0]),
                        (1, SAMPLE_LOG_KEYS[1]),
                        (1, {"loss/train_epoch"}),
                    ],
                )
                values = logs[0][1]
                self.assertTrue(math.isclose(values["optimizer/lr"], lr_after[1], rel_tol=1e-12))
                self.assertEqual(values["train/global_batch_size"], 4)
                self.assertTrue(
                    math.isclose(
                        values["loss/train"],
                        values["loss/train_full"] + values["loss/train_base"],
                        rel_tol=1e-6,
                    )
                )
                # Pre-clip norm: the toy loss is large, so the update was clipped to norm 1.
                grad_norm = values["optimizer/grad_norm"]
                self.assertGreater(grad_norm, 1.0)
                self.assertTrue(
                    math.isclose(
                        values["optimizer/grad_clip_scale"], 1.0 / grad_norm, rel_tol=1e-9
                    )
                )
                parameter_count = sum(value.numel() for value in payload["model"].values())
                gradient_count = round((grad_norm / values["optimizer/grad_rms"]) ** 2)
                self.assertTrue(0 < gradient_count <= parameter_count)
                # parameters/* describe the weights after the update, i.e. the checkpoint.
                weight_norm = torch.linalg.vector_norm(
                    torch.stack([value.float().norm() for value in payload["model"].values()])
                ).item()
                self.assertTrue(
                    math.isclose(values["parameters/norm"], weight_norm, rel_tol=1e-5)
                )
                self.assertTrue(
                    math.isclose(
                        values["parameters/rms"],
                        values["parameters/norm"] / math.sqrt(parameter_count),
                        rel_tol=1e-9,
                    )
                )

                resumed = copy.deepcopy(config)
                resumed["training"]["resume"] = str(checkpoint.relative_to(base))
                train_module.train(resumed, paths, run, max_steps=2)
                self.assertTrue((run / "checkpoints" / "step-00000002.pt").is_file())
                payload = torch.load(
                    run / "checkpoints" / "step-00000002.pt", map_location="cpu", weights_only=False
                )
                # The resumed run continues the schedule: step 2 ends at final_lr.
                self.assertEqual(payload["scheduler"]["last_epoch"], 2)
                self.assertTrue(
                    math.isclose(
                        payload["optimizer"]["param_groups"][0]["lr"], lr_after[2], rel_tol=1e-12
                    )
                )
                resumed_init = wandb_module.inits[1]
                self.assertEqual(
                    (resumed_init["id"], resumed_init["resume"]),
                    (metadata["wandb_run_id"], "must"),
                )
                resumed_logs = wandb_module.logs[len(logs) :]
                self.assertEqual(
                    [(step, set(values)) for step, values in resumed_logs],
                    [
                        (2, TRAIN_LOG_KEYS),
                        (2, SAMPLE_LOG_KEYS[0]),
                        (2, SAMPLE_LOG_KEYS[1]),
                        (2, {"loss/train_epoch"}),
                    ],
                )
                self.assertTrue(
                    math.isclose(
                        resumed_logs[0][1]["optimizer/lr"], lr_after[2], rel_tol=1e-12
                    )
                )

                stage1, dit, loaded = sample.load_sampling_stack(
                    config, paths, run / "checkpoints" / "step-00000002.pt", torch.device("cpu")
                )
                video = sample.sample_from_first_frame(
                    stage1,
                    dit,
                    loaded,
                    torch.zeros(1, 3, 256, 256, dtype=torch.uint8),
                    sample_indices=[0],
                    base_seed=1,
                    steps=2,
                    transport=FlowMatchingTransport(**config["transport"]),
                    cfg_scale=2.0,
                )
                self.assertEqual(tuple(video.shape), (1, 16, 3, 256, 256))

                # A normalizer fit on different train manifests is rejected.
                write_manifest(
                    root / "toy_video/manifests/cam_a/cond1gen16_interval3/split/train",
                    [record("ep0.mp4", 0)],
                )
                with self.assertRaisesRegex(ValueError, "train_sources_sha256"):
                    train_module.load_task_normalizer(
                        config,
                        paths,
                        stage1_metadata=self.stage1.metadata(),
                        device=torch.device("cpu"),
                    )


class WorldSizeResumeTest(unittest.TestCase):
    def test_mid_epoch_resume_on_fewer_gpus_keeps_the_global_batches(self):
        from vrae.training.common.sampler import StatefulDistributedBatchSampler

        def samplers(world_size):
            # global batch 64, accumulation 4: local micro-batch 64 / (world_size * 4)
            return [
                StatefulDistributedBatchSampler(
                    1000, 64 // (world_size * 4), rank=rank, world_size=world_size, seed=3,
                    gradient_accumulation_steps=4,
                )
                for rank in range(world_size)
            ]

        def global_micro_batches(ranks):
            per_rank = [list(sampler) for sampler in ranks]
            return [sorted(sum(batches, [])) for batches in zip(*per_rank, strict=True)]

        four = samplers(4)
        iterators = [iter(sampler) for sampler in four]
        for _ in range(8):  # two optimizer steps, then checkpoint mid-epoch
            for sampler, iterator in zip(four, iterators):
                next(iterator)
                sampler.commit_batch()
        del iterators
        state = four[0].state_dict()
        self.assertGreater(state["sampler_cursor"], 0)
        expected = global_micro_batches(four)
        two = samplers(2)
        with self.assertRaisesRegex(ValueError, "world_size"):
            two[0].load_state_dict(state)
        for sampler in two:
            sampler.load_state_dict(state, allow_world_size_change=True)
        self.assertEqual(global_micro_batches(two), expected)
        with self.assertRaisesRegex(ValueError, "global_micro_batch_size"):
            StatefulDistributedBatchSampler(
                1000, 4, rank=0, world_size=2, seed=3, gradient_accumulation_steps=4
            ).load_state_dict(state, allow_world_size_change=True)


class BatchLayoutResumeTest(unittest.TestCase):
    def test_mid_epoch_accumulation_change_keeps_every_optimizer_step(self):
        from vrae.training.common.sampler import StatefulDistributedBatchSampler

        def ranks(world_size, accumulation):
            return [
                StatefulDistributedBatchSampler(
                    1000, 64 // (world_size * accumulation), rank=rank, world_size=world_size,
                    seed=5, gradient_accumulation_steps=accumulation,
                )
                for rank in range(world_size)
            ]

        def optimizer_steps(samplers):
            accumulation = samplers[0].gradient_accumulation_steps
            micro = [sorted(sum(batches, [])) for batches in zip(*[list(s) for s in samplers])]
            return [sorted(sum(micro[i : i + accumulation], [])) for i in range(0, len(micro), accumulation)]

        before = ranks(2, 4)
        iterators = [iter(sampler) for sampler in before]
        for _ in range(12):  # three optimizer steps, then checkpoint mid-epoch
            for sampler, iterator in zip(before, iterators):
                next(iterator)
                sampler.commit_batch()
        del iterators
        state = before[0].state_dict()
        expected = optimizer_steps(before)
        for world_size, accumulation in ((2, 1), (4, 2)):
            with self.subTest(world_size=world_size, accumulation=accumulation):
                after = ranks(world_size, accumulation)
                with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
                    after[0].load_state_dict(state, allow_world_size_change=True)
                for sampler in after:
                    sampler.load_state_dict(state, allow_accumulation_change=True)
                self.assertEqual(after[0].next_batch_index, state["sampler_cursor"] // (64 // accumulation))
                self.assertEqual(optimizer_steps(after), expected)
        with self.assertRaisesRegex(ValueError, "global_batch_size"):
            StatefulDistributedBatchSampler(
                1000, 16, rank=0, world_size=2, seed=5, gradient_accumulation_steps=1
            ).load_state_dict(state, allow_accumulation_change=True)

    def test_stop_file_checkpoints_and_resume_changes_accumulation(self):
        from vrae.config import save_resolved_config
        from vrae.training.common.contracts import load_run_metadata
        from vrae.training.common.engine import prepare_run
        from vrae.training.nanovvm_video_pred import latent_stats, train as train_module

        stage1 = tiny_stage1()
        wandb_module = fake_wandb_module()
        loader = mock.Mock(return_value=(stage1, {}))
        with (
            tempfile.TemporaryDirectory() as tmp,
            fake_decord(),
            replaced_module("wandb", wandb_module),
            mock.patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
            mock.patch.object(latent_stats, "load_frozen_stage1", loader),
            mock.patch.object(train_module, "load_frozen_stage1", loader),
        ):
            os.environ.pop("VRAE_ALLOW_BATCH_LAYOUT_CHANGE_ON_RESUME", None)
            base = Path(tmp)
            root = make_data_root(base)
            config = task_config()
            config["training"].update({"global_batch_size": 4, "gradient_accumulation_steps": 2,
                                       "checkpoint_interval": 1000})
            config["wandb"].update({"mode": "online", "sample_interval": 1000})
            paths_file = base / "paths.yaml"
            save_resolved_config(
                {"project_root": str(base), "datasets": {"nanovvm": str(root)},
                 "checkpoint_write_root": str(base / "ckpts")},
                paths_file,
            )
            first = base / "first.yaml"
            save_resolved_config(config, first)
            config, paths, run = prepare_run(first, paths_file)
            latent_stats.compute_nanovvm_latent_stats(config, paths)
            (run / "STOP").touch()
            train_module.train(config, paths, run)  # stops at the first log step, step 1
            self.assertTrue((run / "STOP.done").is_file())
            self.assertFalse((run / "STOP").exists())
            self.assertEqual(sorted(p.name for p in (run / "checkpoints").glob("step-*.pt")),
                             ["step-00000001.pt"])

            resumed = copy.deepcopy(config)
            resumed["training"].update({"gradient_accumulation_steps": 1,
                                        "resume": f"ckpts/nanovvm_video_pred/{config['run_name']}/checkpoints/step-00000001.pt"})
            second = base / "second.yaml"
            save_resolved_config(resumed, second)
            with self.assertRaisesRegex(Exception, "gradient_accumulation_steps"):
                prepare_run(second, paths_file)
            os.environ["VRAE_ALLOW_BATCH_LAYOUT_CHANGE_ON_RESUME"] = "1"
            try:
                resumed, paths, run = prepare_run(second, paths_file)
                train_module.train(resumed, paths, run, max_steps=2)
            finally:
                os.environ.pop("VRAE_ALLOW_BATCH_LAYOUT_CHANGE_ON_RESUME", None)
            self.assertTrue((run / "checkpoints" / "step-00000002.pt").is_file())
            metadata = load_run_metadata(run)
            self.assertEqual(metadata["gradient_accumulation_steps"], 1)
            self.assertEqual(metadata["local_micro_batch_size"], 4)
            self.assertEqual(metadata["batch_layout_changes"],
                             [{"step": 1, "gradient_accumulation_steps": [2, 1]}])
            self.assertEqual(load_config(run / "resolved_config.yaml")["training"]
                             ["gradient_accumulation_steps"], 1)
            self.assertTrue((run / "resolved_config.accumulation2.yaml").is_file())
            # Later resumes need no opt-in: the run now records the new split.
            later = copy.deepcopy(resumed)
            later["training"]["resume"] = f"ckpts/nanovvm_video_pred/{config['run_name']}/checkpoints/step-00000002.pt"
            third = base / "third.yaml"
            save_resolved_config(later, third)
            prepare_run(third, paths_file)


class CityscapesRegressionTest(unittest.TestCase):
    def test_cityscapes_build_only_and_dit(self):
        from vrae.training.cityscapes_video_pred.train import build_cityscapes_dit, validate_build

        config = load_config(ROOT / "configs/training/cityscapes_video_pred/cityscapes.yaml")
        summary = validate_build(config)
        self.assertEqual(summary["latent_grid"], [27, 48])
        small = copy.deepcopy(config)
        small["dit"].update(
            {
                "hidden_size": [32, 32],
                "depth": [1, 1],
                "encoder_depth": 1,
                "num_heads": [4, 4],
                "gradient_checkpointing": False,
            }
        )
        dit = build_cityscapes_dit(small, {"patch_size": 16, "hidden_size": 768})
        context = torch.randn(1, 3, 27 * 48, 768)
        full, base = dit(
            torch.randn(1, 3, 27 * 48, 768), torch.tensor([0.5]), context, return_base=True
        )
        self.assertEqual(full.shape, context.shape)
        self.assertEqual(base.shape, context.shape)
        self.assertEqual(type(dit).__name__, "VRAEVideoPredictionDiT")


if __name__ == "__main__":
    unittest.main()
