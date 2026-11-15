"""NanoVVM-protocol FVD for nanovvm_video_pred: config checks, the engine end to end on CPU (toy
encoder, real MJPG videos, NanoVVM manifests from two datasets), caches, RNG, and resume identity.

Run from the repository root:
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m unittest tests.test_nanovvm_fvd -v
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from vrae.config import load_config  # noqa: E402
from vrae.evaluation.nanovvm_fvd import fvd as engine  # noqa: E402
from vrae.paths import ProjectPaths  # noqa: E402
from vrae.storage.backend import configure_storage  # noqa: E402
from vrae.training.nanovvm_video_pred.fvd import (  # noqa: E402
    resolve_fvd_config,
    validate_fvd_section,
)

LADDER_TINY = (
    ROOT
    / "configs/training/nanovvm_video_pred/scaling-ladder/tiny"
    / "vrae_eupe_mix5_tiny_xpred_interval3_cond1gen16_bs64_lr2e-4_wd0_ep1.yaml"
)


class ToyEncoder:
    """Stands in for I3D / VideoMAE-v2: per-channel mean and spread of a [B,3,16,H,W] clip."""

    calls = 0

    def __init__(self, name, checkpoint, device, source_path=None):
        self.name, self.feature_dim = name, 6
        assert Path(checkpoint).is_file()

    def __call__(self, clips):
        ToyEncoder.calls += 1
        if clips.ndim != 5 or clips.shape[1] != 3 or clips.shape[2] != 16:
            raise ValueError(f"expected [B,3,16,H,W], got {tuple(clips.shape)}")
        flat = clips.flatten(2).double()
        return torch.cat((flat.mean(2), flat.std(2)), dim=1).numpy()


class FakePredictor:
    """Identity V-RAE; the "model" repeats the conditioning frame (a static prediction)."""

    def __init__(self):
        self.reconstructions = 0
        self.generated_indices = []

    def identity(self):
        return {"fake": 1}

    def encode_condition(self, rgb):
        return rgb.clone()

    def generate(self, condition, sample_indices):
        self.generated_indices.extend(sample_indices)
        return condition[:, None].expand(-1, 16, -1, -1, -1).clone()

    def reconstruct_future(self, future):
        self.reconstructions += 1
        return future.float().clone()


def write_video(path: Path, length: int, seed: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (3,), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (96, 64))
    for index in range(length):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:] = base
        frame[:, (index * 2) % 96 :] = 255 - base  # a moving edge: real futures differ from frame 0
        writer.write(frame)
    writer.release()


def fvd_data_root(base: Path) -> Path:
    """Two datasets, NanoVVM layout, val/train_eval holdout manifests (cond1gen16, interval 3)."""

    from tests.test_nanovvm_video_pred import record, write_manifest

    root = base / "nanovvm"
    for k, dataset in enumerate(("alpha", "beta")):
        for j in range(3):
            write_video(root / f"{dataset}_video" / "data" / f"ep{j}.avi", 120, 10 * k + j)
        manifests = root / f"{dataset}_video" / "manifests" / "cond1gen16_interval3" / "split"
        write_manifest(manifests / "val", [record("ep0.avi", 0), record("ep1.avi", 49)], split="val")
        write_manifest(manifests / "train_eval", [record("ep2.avi", 0), record("ep2.avi", 49)])
    return root


def fvd_section(**overrides) -> dict:
    def sources(split):
        return [
            {
                "name": dataset,
                "dataset": dataset,
                "root": f"{dataset}_video/data",
                "manifest": f"{dataset}_video/manifests/cond1gen16_interval3/split/{split}/manifest.json",
                "manifest_meta": f"{dataset}_video/manifests/cond1gen16_interval3/split/{split}/meta.yaml",
            }
            for dataset in ("alpha", "beta")
        ]

    section = {
        "interval": 2,
        "frame_interval": 3,
        "seed": 42,
        "batch_size": 3,
        "image_size": 256,
        "sample_fps": 8,
        "use_ema": False,
        "baselines": ["ae_reconstruction", "static"],
        "drop_conditioning_frames": 1,
        "start_mode": "random",
        "mp4_quality": 9,
        "splits": {
            "val": {"num_videos": "all", "sources": sources("val")},
            "train": {"num_videos": "all", "sources": sources("train_eval")},
        },
        "encoders": {
            "i3d": {"checkpoint": "ckpts/fvd/i3d.pt", "batch_size": 2},
            "videomaev2": {"checkpoint": "ckpts/fvd/videomae.pt", "batch_size": 4},
        },
    }
    section.update(overrides)
    return section


def write_encoder_files(project: Path) -> None:
    for name in ("i3d.pt", "videomae.pt"):
        path = project / "ckpts" / "fvd" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())


class FVDConfigTest(unittest.TestCase):
    def test_ladder_section_matches_nanovvm_protocol(self):
        config = load_config(LADDER_TINY)
        section = validate_fvd_section(config)
        self.assertEqual(section["interval"], 10000)
        self.assertEqual(set(section["splits"]), {"val", "train"})
        self.assertEqual(set(section["encoders"]), {"i3d", "videomaev2", "vjepa2_1"})
        self.assertEqual(section["baselines"], ["ae_reconstruction", "static"])
        self.assertEqual((section["drop_conditioning_frames"], section["use_ema"]), (1, False))
        val = section["splits"]["val"]["sources"]
        train = section["splits"]["train"]["sources"]
        self.assertEqual([s["name"] for s in val], [s["name"] for s in config["data"]["val"]["sources"]])
        self.assertEqual([s["manifest"] for s in val], [s["manifest"] for s in config["data"]["val"]["sources"]])
        for source, holdout in zip(train, val, strict=True):
            for key in ("manifest", "manifest_meta"):
                head, tail = holdout[key].rsplit("/val/", 1)  # only the holdout component changes
                self.assertEqual(source[key], f"{head}/train_eval/{tail}")
            self.assertEqual(source["root"], holdout["root"])
        self.assertIsNone(validate_fvd_section({**config, "fvd": None}))

    def test_rejected_settings(self):
        cases = [
            ({"drop_conditioning_frames": 0}, "must be 1"),
            ({"baselines": ["constant_velocity_2frame"]}, "subset"),
            ({"colour": 1}, "Unknown fvd keys"),
            ({"interval": 0}, "positive integer"),
            ({"image_size": 128}, "must be 256"),
            ({"frame_interval": 1}, "frame interval"),
            ({"encoders": {"inception": {"checkpoint": "x"}}}, "Unknown FVD encoder"),
            ({"splits": {"holdout": {"sources": []}}}, "must be named"),
        ]
        for override, match in cases:
            with self.subTest(match=match), self.assertRaisesRegex(ValueError, match):
                validate_fvd_section({"fvd": fvd_section(**override)})

    def test_resume_ignores_the_fvd_section(self):
        from vrae.training.common.contracts import compare_resolved_configs

        config = load_config(LADDER_TINY)
        without = copy.deepcopy(config)
        without.pop("fvd")
        compare_resolved_configs(without, config)  # a run started before FVD existed resumes with it
        changed = copy.deepcopy(config)
        changed["fvd"]["interval"] = 5000
        compare_resolved_configs(config, changed)


class FVDEngineTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.root = fvd_data_root(self.tmp)
        write_encoder_files(self.tmp)
        self.paths = ProjectPaths(project_root=self.tmp, datasets={"nanovvm": self.root})
        ToyEncoder.calls = 0

    def tearDown(self):
        configure_storage(None)
        self._tmp.cleanup()

    def config(self):
        from tests.test_nanovvm_video_pred import task_config

        config = task_config()
        config["fvd"] = fvd_section()
        return config

    def run_engine(self, fvd_config, predictor, step):
        with mock.patch.object(engine, "VideoEncoder", ToyEncoder):
            return engine.run_fvd_evaluation(
                predictor=predictor, rank=0, world_size=1, device=torch.device("cpu"),
                run_dir=self.tmp / "run", global_step=step, fvd_config=fvd_config,
            )

    def test_paths_resolve_below_the_data_root_and_checkpoint_root(self):
        resolved = resolve_fvd_config(self.config(), self.paths)
        source = resolved["splits"]["train"]["sources"][1]
        self.assertEqual(source["root"], str((self.root / "beta_video/data").resolve()))
        self.assertTrue(Path(source["manifest"]).is_file())
        self.assertEqual(resolved["encoders"]["i3d"]["checkpoint"], str(self.tmp / "ckpts/fvd/i3d.pt"))
        broken = self.config()
        broken["fvd"]["encoders"]["i3d"]["checkpoint"] = "ckpts/fvd/missing.pt"
        with self.assertRaises(FileNotFoundError):
            resolve_fvd_config(broken, self.paths)

    def test_engine_scores_pooled_and_per_dataset_with_nanovvm_keys(self):
        fvd_config = resolve_fvd_config(self.config(), self.paths)
        predictor = FakePredictor()
        torch.manual_seed(7)
        before = torch.rand(3)
        torch.manual_seed(7)
        metrics = self.run_engine(fvd_config, predictor, step=2)
        self.assertTrue(torch.equal(torch.rand(3), before))  # the caller's RNG is untouched
        expected = {
            f"{prefix}/{split}/{encoder}/{source}"
            for prefix in ("eval_fvd", "eval_fvd_alpha", "eval_fvd_beta")
            for split in ("val", "train")
            for encoder in ("i3d", "videomaev2")
            for source in ("model", "ae_reconstruction", "static")
        }
        self.assertEqual(set(metrics), expected)
        for split in ("val", "train"):
            # The model repeats frame 0, like the static control (both through an identity V-RAE);
            # the real future through the identity V-RAE is the closest to the real clips.
            model = metrics[f"eval_fvd/{split}/i3d/model"]
            self.assertAlmostEqual(model, metrics[f"eval_fvd/{split}/i3d/static"], delta=1e-3 + 0.05 * model)
            self.assertLess(metrics[f"eval_fvd/{split}/i3d/ae_reconstruction"], model)
        out = self.tmp / "run" / "fvd_eval" / "step_0000002" / "val"
        score = json.loads((out / "fvd_score.json").read_text())
        self.assertEqual((score["num_videos"], score["scored_frames"]), (4, 16))
        self.assertEqual(score["datasets"]["alpha"]["num_videos"], 2)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(len(manifest), 4)
        import imageio.v2 as imageio

        frames = imageio.mimread(manifest[0]["gen_path"])
        self.assertEqual((len(frames), *frames[0].shape), (17, 256, 256, 3))
        self.assertEqual(predictor.reconstructions, 2 * 8)  # two controls per clip, 8 clips
        self.assertEqual(len(set(predictor.generated_indices)), 8)

        # A later step: controls and real features come from the cache; noise indices repeat.
        calls = ToyEncoder.calls
        again = FakePredictor()
        later = self.run_engine(fvd_config, again, step=4)
        self.assertEqual(again.reconstructions, 0)
        self.assertEqual(sorted(again.generated_indices), sorted(predictor.generated_indices))
        self.assertLess(ToyEncoder.calls - calls, calls)  # only the model videos are encoded again
        for key in ("eval_fvd/val/i3d/static", "eval_fvd_beta/train/videomaev2/ae_reconstruction"):
            self.assertAlmostEqual(later[key], metrics[key], places=9)


if __name__ == "__main__":
    unittest.main()
