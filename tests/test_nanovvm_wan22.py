"""Wan2.2 VAE + Wan2.2-TI2V DiT path of nanovvm_video_pred: CPU tests, no real weights.

A randomly initialized, narrow ``WanVAE_`` (the vendored Wan2.2 code with small channel widths)
stands in for Wan2.2_VAE.pth at 32x32. Run from the repository root:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONPATH=src python -m unittest tests.test_nanovvm_wan22 -v
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from vrae.config import load_config  # noqa: E402
from vrae.models.dit.transport import FlowMatchingTransport  # noqa: E402
from vrae.models.dit.wan_ti2v_dit import (  # noqa: E402
    WanTI2VDiT,
    rope_split,
    sinusoidal_embedding_1d,
    wan_base_depth,
)
from vrae.models.video_vae import wan22  # noqa: E402
from vrae.models.video_vae.backend import (  # noqa: E402
    WAN22_REVISION,
    Wan22LatentBackend,
    load_wan22_backend,
)
from vrae.paths import ProjectPaths  # noqa: E402
from vrae.training.common.latent_norm import LatentNormalizer  # noqa: E402
from vrae.training.nanovvm_video_pred import latents as nv_latents  # noqa: E402
from vrae.training.nanovvm_video_pred import train as nv_train  # noqa: E402
from tests.test_nanovvm_video_pred import (  # noqa: E402
    assert_close,
    fake_decord,
    fake_wandb_module,
    make_data_root,
    randomize,
    replaced_module,
    toy_sources,
)

NV = ROOT / "configs/training/nanovvm_video_pred"
LADDER = NV / "scaling-ladder"
SIZES = ("tiny", "small", "medium", "large", "xl", "xxl")
# size: (num_heads, dim, num_layers, ffn_dim, base_depth, parameters incl. IG head + null z0, NanoVVM M)
WAN_LADDER = {
    "tiny": (3, 384, 4, 1792, 1, 9_085_408, 9.9),
    "small": (6, 768, 9, 3584, 3, 75_350_112, 78.8),
    "medium": (10, 1280, 12, 6016, 3, 275_727_200, 265.7),
    "large": (13, 1664, 16, 7808, 5, 613_513_440, 629.5),
    "xl": (16, 2048, 21, 9600, 6, 1_209_044_960, 1229.2),
    "xxl": (20, 2560, 24, 11904, 7, 2_139_929_184, 2124.1),
}
# The plan's table (millions, rounded).
PLAN_MILLIONS = {"tiny": 9.1, "small": 75.3, "medium": 275.7, "large": 613.5, "xl": 1209.0, "xxl": 2139.9}
WAN_STAGE1 = {
    "type": "wan22_vae",
    "spatial_compression": 16,
    "temporal_compression": 4,
    "latent_channels": 48,
    "latent_grid": [8, 8],
}


def wan_ladder_path(size: str) -> Path:
    return LADDER / size / f"wan22_ti2v_mix5_{size}_xpred_interval3_cond1gen16_res128_bs32_lr2e-4_wd0_ep1.yaml"


def eupe_res128_path(size: str) -> Path:
    return LADDER / size / f"vrae_eupe_mix5_{size}_xpred_interval3_cond1gen16_res128_bs32_lr2e-4_wd0_ep1.yaml"


def tiny_wan_vae(seed: int = 0) -> torch.nn.Module:
    """The Wan2.2 VAE architecture (z_dim 48, dim_mult [1,2,4,4], temporal [F,T,T]) at width 8."""

    torch.manual_seed(seed)
    return wan22.WanVAE_(
        dim=8,
        dec_dim=8,
        z_dim=48,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True],
    ).eval()


def load_tiny_backend(
    weights: Path, *, image_size: int = 32, seed: int = 0, precision: str = "fp32"
) -> Wan22LatentBackend:
    """load_wan22_backend with the official Wan2_2_VAE wrapper; only _video_vae is replaced."""

    model = tiny_wan_vae(seed)
    calls = []

    def video_vae(**kwargs):
        calls.append(kwargs)
        return model

    with mock.patch.object(wan22, "_video_vae", video_vae):
        backend = load_wan22_backend(
            weights, image_size=image_size, revision=WAN22_REVISION, precision=precision
        )
    assert calls and calls[0]["pretrained_path"] == str(weights)
    backend.loader_call = calls[0]
    return backend


@contextlib.contextmanager
def tiny_backend_file(image_size: int = 32):
    with tempfile.TemporaryDirectory() as tmp:
        weights = Path(tmp) / "Wan2.2_VAE.pth"
        weights.write_bytes(b"not real weights")
        yield load_tiny_backend(weights, image_size=image_size), weights


def small_wan_dit(**overrides) -> WanTI2VDiT:
    arguments = dict(
        grid_size=(4, 4),
        in_channels=48,
        dim=64,
        ffn_dim=96,
        num_heads=2,
        num_layers=3,
        base_depth=1,
        patch_size=(1, 1, 1),
        frame_dropout=0.1,
    )
    arguments.update(overrides)
    return WanTI2VDiT(**arguments)


def parameter_formula(d, f, layers, *, p=1, heads=2, null=8 * 8 * 48, cross=False, text=False):
    """Plan section 3.4. cross/text add Wan2.2's text branch; p is the spatial patch."""

    block = 4 * (d * d + d) + 2 * d + 2 * d * f + f + d + 6 * d
    if cross:
        block += 4 * (d * d + d) + 4 * d  # cross-attention q/k/v/o, its q/k RMSNorm, affine norm3
    shared = (48 * p * p * d + d) + (256 * d + d + d * d + d) + (6 * d * d + 6 * d)
    shared += heads * (d * 48 * p * p + 48 * p * p + 2 * d) + null
    if text:
        shared += 4096 * d + d + d * d + d
    return layers * block + shared


# ---------------------------------------------------------------- Wan2.2 reference


def _rope_params(max_seq_len, dim, theta=10000):  # wan/modules/model.py rope_params
    freqs = torch.outer(
        torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    return torch.polar(torch.ones_like(freqs), freqs)


@torch.no_grad()
def wan_reference_forward(model: WanTI2VDiT, noisy_future, time, z0):
    """WanModel.forward (wan/modules/model.py @ 1ea34ff) without text, written out literally:
    per-token timesteps [B, L] with 0 on latent frame 0 (TI2V's mask2 * timestep), per-token
    e0 [B, L, 6, d], float64 complex RoPE, Head and unpatchify einsum."""

    batch, channels = noisy_future.shape[0], model.in_channels
    frames = 1 + model.num_chunks
    gh, gw = model.grid_size
    video = torch.cat([z0[:, None], noisy_future], 1).reshape(batch, frames, gh, gw, channels)
    x = model.patch_embedding(video.permute(0, 4, 1, 2, 3))
    f, h, w = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)
    length, per_frame = f * h * w, h * w
    t = torch.cat([torch.zeros(batch, per_frame), time[:, None].expand(batch, length - per_frame)], 1)
    e = model.time_embedding(
        sinusoidal_embedding_1d(model.freq_dim, (1000 * t).flatten()).unflatten(0, (batch, length)).float()
    )
    e0 = model.time_projection(e).unflatten(2, (6, model.dim))
    d = model.dim // model.num_heads
    freqs = torch.cat(
        [_rope_params(1024, d - 4 * (d // 6)), _rope_params(1024, 2 * (d // 6)), _rope_params(1024, 2 * (d // 6))],
        dim=1,
    )

    def rope_apply(value):
        n, c = value.size(2), value.size(3) // 2
        parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        output = []
        for i in range(batch):
            x_i = torch.view_as_complex(value[i].to(torch.float64).reshape(length, n, -1, 2))
            freqs_i = torch.cat(
                [
                    parts[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(length, 1, -1)
            output.append(torch.view_as_real(x_i * freqs_i).flatten(2))
        return torch.stack(output).float()

    for block in model.blocks:
        m = (block.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
        attention = block.self_attn
        heads = attention.num_heads

        def attend(value):
            q = attention.norm_q(attention.q(value)).view(batch, length, heads, d)
            k = attention.norm_k(attention.k(value)).view(batch, length, heads, d)
            v = attention.v(value).view(batch, length, heads, d)
            out = F.scaled_dot_product_attention(
                rope_apply(q).transpose(1, 2), rope_apply(k).transpose(1, 2), v.transpose(1, 2)
            )
            return attention.o(out.transpose(1, 2).flatten(2))

        y = attend(block.norm1(x).float() * (1 + m[1].squeeze(2)) + m[0].squeeze(2))
        x = x + y * m[2].squeeze(2)
        y = block.ffn(block.norm2(x).float() * (1 + m[4].squeeze(2)) + m[3].squeeze(2))
        x = x + y * m[5].squeeze(2)
    hm = (model.head.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
    out = model.head.head(model.head.norm(x) * (1 + hm[1].squeeze(2)) + hm[0].squeeze(2))
    videos = []
    for u in out:
        u = u[:length].view(f, h, w, *model.patch_size, channels)
        u = torch.einsum("fhwpqrc->cfphqwr", u)
        videos.append(u.reshape(channels, *[i * j for i, j in zip((f, h, w), model.patch_size)]))
    tokens = torch.stack(videos).permute(0, 2, 3, 4, 1).reshape(batch, frames, gh * gw, channels)
    return tokens[:, 1:]


# ---------------------------------------------------------------- VAE backend


class Wan22BackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.weights = Path(cls._tmp.name) / "Wan2.2_VAE.pth"
        cls.weights.write_bytes(b"not real weights")
        cls.backend = load_tiny_backend(cls.weights)
        generator = torch.Generator().manual_seed(0)
        cls.condition = torch.rand(2, 3, 32, 32, generator=generator)
        cls.future = torch.rand(2, 16, 3, 32, 32, generator=generator)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_loader_uses_the_ti2v_vae_and_official_normalization(self):
        call = self.backend.loader_call
        self.assertEqual(
            (call["z_dim"], call["dim"], call["dim_mult"], call["temperal_downsample"]),
            (48, 160, [1, 2, 4, 4], [False, True, True]),
        )
        self.assertAlmostEqual(self.backend.latent_mean[0].item(), -0.2289, places=6)
        self.assertAlmostEqual(self.backend.latent_inv_std[0].item(), 1 / 0.4765, places=4)
        self.assertAlmostEqual(self.backend.latent_mean[-1].item(), -0.0667, places=6)
        self.assertEqual(
            self.backend.metadata(),
            {
                "type": "wan22_vae",
                "revision": WAN22_REVISION,
                "weights_sha256": hashlib.sha256(b"not real weights").hexdigest(),
                "latent_channels": 48,
                "spatial_compression": 16,
                "temporal_compression": 4,
                "latent_grid": [2, 2],
                "normalization": "official",
                "posterior": "mean",
                "dtype": "float32",
            },
        )
        with self.assertRaisesRegex(ValueError, "revision"):
            load_wan22_backend(self.weights, image_size=32, revision="main")
        with self.assertRaisesRegex(ValueError, "fp32 or bf16"):
            load_wan22_backend(self.weights, image_size=32, precision="fp16")

    def test_z0_is_causal_and_the_future_sees_frame_0(self):
        z0, future = self.backend.encode_clip(self.condition, self.future)
        self.assertEqual((tuple(z0.shape), tuple(future.shape)), ((2, 4, 48), (2, 4, 4, 48)))
        # Latent 0 of the joint 17-frame encoding is frame 0 encoded alone.
        assert_close(z0, self.backend.encode_condition(self.condition), atol=1e-5)
        other_z0, other_future = self.backend.encode_clip(self.condition, torch.rand_like(self.future))
        self.assertTrue(torch.equal(z0, other_z0))
        self.assertFalse(torch.allclose(future, other_future))
        # Unlike V-RAE, the target depends on the pixels of frame 0 (as in Wan2.2-TI2V).
        _, shifted = self.backend.encode_clip(self.condition.flip(-1), self.future)
        self.assertFalse(torch.allclose(future[:, 0], shifted[:, 0]))
        # The latents.py entry points route to the joint encoding.
        routed = nv_latents.encode_condition_and_future(self.backend, self.condition, self.future)
        self.assertTrue(torch.equal(routed[0], z0) and torch.equal(routed[1], future))
        self.assertTrue(torch.equal(nv_latents.encode_first_frame(self.backend, self.condition), z0))

    def test_only_the_whole_17_frame_clip_is_encoded(self):
        for frames in (15, 12, 17):
            with self.subTest(future_frames=frames), self.assertRaisesRegex(ValueError, "16 frames"):
                self.backend.encode_clip(self.condition, torch.rand(2, frames, 3, 32, 32))
        for frames in (16, 13, 5):
            with self.subTest(clip_frames=frames), self.assertRaisesRegex(ValueError, "17-frame clip"):
                self.backend._encode(torch.rand(1, frames, 3, 32, 32))
        with self.assertRaisesRegex(ValueError, "together with frame 0"):
            nv_latents.encode_future(self.backend, self.future)
        with self.assertRaisesRegex(ValueError, "32x32"):
            self.backend.encode_condition(torch.rand(1, 3, 48, 48))

    def test_decode_is_causal_with_z0(self):
        z0, future = self.backend.encode_clip(self.condition, self.future)
        video = self.backend.decode_future(z0, future)
        self.assertEqual(tuple(video.shape), (2, 16, 3, 32, 32))
        self.assertTrue(0 <= video.min().item() <= video.max().item() <= 1)
        # Frames 1..16 of decoding all five latents with Wan's decoder.
        latents = torch.cat((z0[:, None], future), 1).reshape(2, 5, 2, 2, 48).permute(0, 4, 1, 2, 3)
        direct = self.backend.model.decode(latents, [self.backend.latent_mean, self.backend.latent_inv_std])
        expected = direct.clamp(-1, 1).add(1).mul(0.5).permute(0, 2, 1, 3, 4)[:, 1:]
        assert_close(video, expected, atol=1e-5)
        changed = self.backend.decode_future(z0 + 1.0, future)
        self.assertFalse(torch.allclose(video[:, 0], changed[:, 0]))
        # latents.decode_future_tokens requires the raw z0 for this backend.
        with self.assertRaisesRegex(ValueError, "condition="):
            nv_latents.decode_future_tokens(self.backend, future, grid_size=(2, 2))
        routed = nv_latents.decode_future_tokens(self.backend, future, grid_size=(2, 2), condition=z0)
        self.assertTrue(torch.equal(routed, video))

    def test_uint8_input_and_outer_autocast(self):
        rgb = (self.condition * 255).round().to(torch.uint8)
        assert_close(
            self.backend.encode_condition(rgb), self.backend.encode_condition(rgb.float() / 255), atol=0
        )
        plain = self.backend.encode_clip(self.condition, self.future)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            inside = self.backend.encode_clip(self.condition, self.future)
            decoded = self.backend.decode_future(*inside)
        self.assertTrue(torch.equal(plain[0], inside[0]) and torch.equal(plain[1], inside[1]))
        self.assertEqual(decoded.dtype, torch.float32)
        self.assertTrue(torch.equal(decoded, self.backend.decode_future(*plain)))

    def test_bf16_precision(self):
        bf16 = load_tiny_backend(self.weights, precision="bf16")
        self.assertEqual(bf16.metadata()["dtype"], "bfloat16")
        self.assertEqual(bf16.decoder_execution_metadata()["autocast"], "bf16_cuda")
        self.assertEqual(next(bf16.model.parameters()).dtype, torch.float32)  # fp32 weights
        # bf16 autocast applies on CUDA only; on the CPU it computes exactly the fp32 latents.
        z0, future = bf16.encode_clip(self.condition, self.future)
        reference = self.backend.encode_clip(self.condition, self.future)
        self.assertTrue(torch.equal(z0, reference[0]) and torch.equal(future, reference[1]))
        if not torch.cuda.is_available():
            return
        fp32, bf16 = self.backend.cuda(), bf16.cuda()
        condition, clip = self.condition.cuda(), self.future.cuda()
        exact = fp32.encode_clip(condition, clip)
        with torch.autocast("cuda", dtype=torch.float16):  # the caller's autocast is replaced
            approx = bf16.encode_clip(condition, clip)
        self.assertEqual(approx[1].dtype, torch.float32)
        relative = (approx[1] - exact[1]).norm() / exact[1].norm()
        self.assertTrue(0 < relative.item() < 0.05)
        self.backend.cpu()

    def test_condition_features_use_the_task_normalizer(self):
        z0, _ = self.backend.encode_clip(self.condition, self.future)
        normalizer = LatentNormalizer(torch.linspace(-1, 1, 48), torch.linspace(0.5, 2, 48))
        features = nv_latents.condition_features(self.backend, z0, normalizer)
        assert_close(features, (z0 - normalizer.mean) / normalizer.std, atol=1e-6)


# ---------------------------------------------------------------- DiT


class WanTI2VDiTTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dit = small_wan_dit()
        randomize(self.dit, std=0.05)
        self.dit.eval()
        generator = torch.Generator().manual_seed(1)
        self.noisy = torch.randn(2, 4, 16, 48, generator=generator)
        self.z0 = torch.randn(2, 16, 48, generator=generator)
        self.time = torch.tensor([0.3, 0.8])

    def test_constants(self):
        self.assertEqual(rope_split(128), (44, 42, 42))
        self.assertEqual({size: wan_base_depth(row[2]) for size, row in WAN_LADDER.items()},
                         {size: row[4] for size, row in WAN_LADDER.items()})
        default = WanTI2VDiT.__init__.__kwdefaults__
        self.assertEqual(
            (default["dim"], default["ffn_dim"], default["num_heads"], default["num_layers"],
             default["freq_dim"], default["eps"], default["rope_theta"]),
            (3072, 14336, 24, 30, 256, 1e-6, 10000.0),
        )

    def test_matches_the_wan_reference_forward(self):
        for patch, grid in (((1, 1, 1), (4, 4)), ((1, 2, 2), (4, 4))):
            with self.subTest(patch=patch):
                torch.manual_seed(0)
                dit = small_wan_dit(patch_size=patch, grid_size=grid)
                randomize(dit, std=0.05)
                dit.eval()
                actual = dit(self.noisy, self.time, context=self.z0)
                expected = wan_reference_forward(dit, self.noisy, self.time, self.z0)
                assert_close(actual, expected, atol=2e-5)

    def test_shapes_aliases_and_base(self):
        full = self.dit(self.noisy, self.time, context=self.z0)
        self.assertEqual(full.shape, self.noisy.shape)
        alias = self.dit(self.noisy, self.time, frame_features=self.z0)
        self.assertTrue(torch.equal(full, alias))
        pair = self.dit(self.noisy, self.time, frame_features=self.z0, return_base=True)
        self.assertIsInstance(pair, tuple)
        self.assertEqual(len(pair), 2)
        self.assertTrue(torch.equal(pair[0], full))
        self.assertEqual(pair[1].shape, self.noisy.shape)
        self.assertFalse(torch.allclose(pair[0], pair[1]))
        (result, hidden) = self.dit(self.noisy, self.time, context=self.z0, return_intermediate=True)
        self.assertEqual(tuple(hidden.shape), (2, 5 * 16, 64))
        self.assertTrue(torch.equal(result, full))
        with self.assertRaisesRegex(ValueError, "either context or frame_features"):
            self.dit(self.noisy, self.time, context=self.z0, frame_features=self.z0)
        with self.assertRaisesRegex(ValueError, "noisy_future"):
            self.dit(self.noisy[:, :3], self.time, context=self.z0)
        self.assertEqual(
            (self.dit.frame_tokens, self.dit.frame_channels, self.dit.in_channels, self.dit.future_chunks,
             self.dit.future_latent_shape, self.dit.grid_size, self.dit.patch_size),
            (16, 48, 48, 4, (4, 16, 48), (4, 4), (1, 1, 1)),
        )

    def test_z0_tokens_are_embedded_at_t0(self):
        e, e0 = self.dit._time_embeddings(self.time, 5)
        self.assertEqual((tuple(e.shape), tuple(e0.shape)), ((2, 5, 64), (2, 5, 6, 64)))

        def embed(value):
            sinusoid = sinusoidal_embedding_1d(256, torch.full((2,), 1000.0 * value) if isinstance(value, float)
                                               else 1000 * value).float()
            return self.dit.time_embedding(sinusoid)

        assert_close(e[:, 0], embed(0.0), atol=1e-6)
        for frame in range(1, 5):
            assert_close(e[:, frame], embed(self.time), atol=1e-6)
        assert_close(e0[:, 0], self.dit.time_projection(embed(0.0)).unflatten(1, (6, 64)), atol=1e-6)
        # The z0 tokens see t = 0 whatever the future's t: changing t alone changes the output.
        moved = self.dit(self.noisy, torch.tensor([0.31, 0.8]), context=self.z0)
        base = self.dit(self.noisy, self.time, context=self.z0)
        self.assertFalse(torch.allclose(moved[0], base[0]))
        self.assertTrue(torch.allclose(moved[1], base[1]))

    def test_condition_dropout(self):
        other = torch.randn_like(self.z0)
        drop = torch.ones(2, dtype=torch.bool)
        keep = torch.zeros(2, dtype=torch.bool)
        dropped = self.dit(self.noisy, self.time, context=self.z0, condition_drop_mask=drop)
        dropped_other = self.dit(self.noisy, self.time, context=other, condition_drop_mask=drop)
        self.assertTrue(torch.equal(dropped, dropped_other))
        kept = self.dit(self.noisy, self.time, context=self.z0, condition_drop_mask=keep)
        kept_other = self.dit(self.noisy, self.time, context=other, condition_drop_mask=keep)
        self.assertFalse(torch.allclose(kept, kept_other))
        # The null z0 is the learned [1,1,h*w,C] parameter, zero at initialization.
        fresh = small_wan_dit()
        self.assertEqual(tuple(fresh.null_condition.shape), (1, 1, 16, 48))
        self.assertEqual(fresh.null_condition.abs().sum().item(), 0)
        null = self.dit.null_condition[0].expand(2, -1, -1)
        assert_close(dropped, self.dit(self.noisy, self.time, context=null, condition_drop_mask=keep), atol=0)
        # Training draws the mask with frame_dropout; at probability 1 it is the null branch.
        always = small_wan_dit(frame_dropout=1.0)
        always.load_state_dict(self.dit.state_dict())
        always.train()
        assert_close(always(self.noisy, self.time, context=self.z0), dropped, atol=1e-6)
        output = always(self.noisy, self.time, context=self.z0, return_base=True)
        (output[0].sum() + output[1].sum()).backward()
        self.assertGreater(always.null_condition.grad.abs().sum().item(), 0)

    def test_gradient_checkpointing_is_exact(self):
        results = []
        for enabled in (False, True):
            dit = small_wan_dit(gradient_checkpointing=enabled, frame_dropout=0.0)
            dit.load_state_dict(self.dit.state_dict())
            dit.train()
            full, base = dit(self.noisy, self.time, context=self.z0, return_base=True)
            (full.square().mean() + base.square().mean()).backward()
            grads = {name: p.grad.clone() for name, p in dit.named_parameters() if p.grad is not None}
            results.append((full.detach(), base.detach(), grads))
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        self.assertEqual(set(results[0][2]), set(results[1][2]))
        for name in results[0][2]:
            assert_close(results[0][2][name], results[1][2][name], atol=1e-7)
        toggled = small_wan_dit()
        toggled.set_gradient_checkpointing(True)
        self.assertTrue(toggled.gradient_checkpointing)

    def test_initialization_follows_wan(self):
        torch.manual_seed(0)
        dit = small_wan_dit()
        self.assertEqual(dit.head.head.weight.abs().sum().item(), 0)
        self.assertEqual(dit.base_head.head.weight.abs().sum().item(), 0)
        self.assertEqual(dit.blocks[0].self_attn.q.bias.abs().sum().item(), 0)
        self.assertLess(abs(dit.time_embedding[0].weight.std().item() - 0.02), 0.004)
        self.assertTrue(torch.equal(dit.blocks[0].self_attn.norm_q.weight, torch.ones(64)))
        self.assertEqual(tuple(dit.blocks[0].modulation.shape), (1, 6, 64))
        self.assertEqual(tuple(dit.head.modulation.shape), (1, 2, 64))

    def test_transport_trains_full_and_base_on_the_future_only(self):
        dit = small_wan_dit(frame_dropout=0.5)
        randomize(dit, std=0.02)  # the zero-initialized heads would block every upstream gradient
        dit.train()
        transport = FlowMatchingTransport(prediction="x", time_dist_shift=math.sqrt(3), base_model_coeff=1.0)
        future = torch.randn(4, 4, 16, 48)
        z0 = torch.randn(4, 16, 48)
        losses = transport.training_losses(dit, future, z0, generator=torch.Generator().manual_seed(0))
        self.assertEqual(losses["noisy"].shape, future.shape)  # only the future is noised
        self.assertGreater(losses["loss_base"].item(), 0)
        self.assertTrue(math.isclose(losses["loss"].item(),
                                     losses["loss_full"].item() + losses["loss_base"].item(), rel_tol=1e-6))
        losses["loss"].backward()
        self.assertGreater(dit.head.head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(dit.base_head.head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(dit.patch_embedding.weight.grad.abs().sum().item(), 0)
        # Euler sampling with CFG uses the explicit null branch.
        dit.eval()
        sample = transport.euler_sample(
            dit, torch.randn(1, 4, 16, 48), num_steps=2, cfg_scale=2.0, ig_scale=1.5,
            model_kwargs={"frame_features": z0[:1], "condition_drop_mask": torch.zeros(1, dtype=torch.bool)},
            unconditional_kwargs={"frame_features": z0[:1], "condition_drop_mask": torch.ones(1, dtype=torch.bool)},
        )
        self.assertEqual(tuple(sample.shape), (1, 4, 16, 48))


# ---------------------------------------------------------------- parameters and configs


class ParameterCountTest(unittest.TestCase):
    def test_ladder_parameter_counts(self):
        for size, (heads, dim, layers, ffn, base, count, nanovvm) in WAN_LADDER.items():
            with self.subTest(size=size):
                self.assertEqual(dim, 128 * heads)
                self.assertEqual(ffn, round(14 / 3 * dim / 128) * 128)
                self.assertTrue(85 <= dim / layers <= 107)
                self.assertEqual(base, max(1, round(2 * layers / 7)))
                with torch.device("meta"):
                    model = nv_train.build_nanovvm_dit(load_config(wan_ladder_path(size)), WAN_STAGE1)
                self.assertIsInstance(model, WanTI2VDiT)
                self.assertEqual((model.dim, model.num_heads, model.num_layers, model.ffn_dim, model.base_depth),
                                 (dim, heads, layers, ffn, base))
                total = sum(p.numel() for p in model.parameters())
                self.assertEqual(total, count)
                self.assertEqual(total, parameter_formula(dim, ffn, layers))
                # Within 0.1% of the plan's table, or within its 0.1M resolution (tiny: 9.085M vs 9.1M).
                table = PLAN_MILLIONS[size]
                self.assertLess(abs(total / 1e6 - table), max(1e-3 * table, 0.1))
                self.assertLess(abs(total / (nanovvm * 1e6) - 1), 0.09)

    def test_formula_reproduces_wan22_ti2v_5b(self):
        # Text branch back, patch (1,2,2), one head, no null latent: Wan2.2-TI2V-5B has 5.000B.
        total = parameter_formula(3072, 14336, 30, p=2, heads=1, null=0, cross=True, text=True)
        self.assertEqual(round(total / 1e9, 3), 5.000)


class WanConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = {size: load_config(wan_ladder_path(size)) for size in SIZES}
        cls.eupe = {size: load_config(eupe_res128_path(size)) for size in SIZES}

    def test_build_only_cli(self):
        for size in SIZES:
            with self.subTest(size=size):
                argv = ["train", "--config", str(wan_ladder_path(size)), "--build-only"]
                stdout = io.StringIO()
                with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
                    nv_train.main()
                summary = json.loads(stdout.getvalue())
                self.assertEqual(summary["transport"]["time_dist_shift"], math.sqrt(3))
                self.assertEqual(summary["expected_time_dist_shift"], math.sqrt(3))
                self.assertEqual((summary["dit"], summary["stage1_type"]), ("wan_ti2v_dit", "wan22_vae"))
                self.assertEqual(summary["latent_grid"], [8, 8])
                self.assertEqual(summary["shapes"]["dit_sequence"], ["B", 320, WAN_LADDER[size][1]])
                self.assertEqual((len(summary["train_sources"]), len(summary["val_sources"])), (18, 18))

    def test_ladder_sizes(self):
        for size, config in self.configs.items():
            heads, dim, layers, ffn, base = WAN_LADDER[size][:5]
            with self.subTest(size=size):
                self.assertEqual(config["run_name"], wan_ladder_path(size).stem)
                self.assertEqual(
                    {key: config["dit"][key] for key in ("dim", "num_heads", "num_layers", "ffn_dim", "base_depth")},
                    {"dim": dim, "num_heads": heads, "num_layers": layers, "ffn_dim": ffn, "base_depth": base},
                )
                self.assertEqual(
                    {key: value for key, value in config["dit"].items()
                     if key not in ("dim", "num_heads", "num_layers", "ffn_dim", "base_depth")},
                    {"name": "wan_ti2v_dit", "freq_dim": 256, "eps": 1e-6, "qk_norm": "rms_norm_across_heads",
                     "rope_theta": 10000.0, "patch_size": [1, 1, 1], "frame_dropout": 0.1,
                     "gradient_checkpointing": True},
                )
                self.assertEqual(
                    config["wandb"]["tags"],
                    ["nanovvm", "video_prediction", "first_frame", "wan22_vae", "wan_ti2v_dit", "mix5",
                     "scaling_ladder", size, "res128", "bs32"],
                )

    def test_recipe_equals_the_eupe_res128_ladder(self):
        for size, config in self.configs.items():
            eupe = self.eupe[size]
            with self.subTest(size=size):
                self.assertNotIn("model", config)
                self.assertEqual(
                    config["stage1"],
                    {"type": "wan22_vae", "checkpoint": "ckpts/pretrained/wan22_vae/Wan2.2_VAE.pth",
                     "revision": WAN22_REVISION, "precision": "bf16"},
                )
                for section in ("training", "data", "sampling", "fvd"):
                    self.assertEqual(config[section], eupe[section], section)
                self.assertEqual(config["task"], eupe["task"])
                transport = dict(config["transport"])
                self.assertEqual(transport.pop("time_dist_shift"), math.sqrt(3))
                expected = dict(eupe["transport"])
                expected.pop("time_dist_shift")
                self.assertEqual(transport, expected)
                self.assertEqual({k: v for k, v in config["wandb"].items() if k != "tags"},
                                 {k: v for k, v in eupe["wandb"].items() if k != "tags"})
                self.assertEqual(config["dit"]["frame_dropout"], eupe["dit"]["frame_dropout"])
                self.assertEqual(
                    config["latent_normalizer"],
                    {"path": "ckpts/nanovvm_video_pred/latent_stats/wan22_mix5_cond1gen16_res128_future_bf16.pt",
                     "fit_clips": 20000, "fit_seed": 0, "batch_size": 16},
                )
                self.assertEqual(
                    (config["training"]["global_batch_size"], config["training"]["gradient_accumulation_steps"],
                     config["data"]["image_size"], config["fvd"]["image_size"]),
                    (32, 1, 128, 128),
                )
                geometry = nv_train.task_geometry(config)
                self.assertEqual(geometry["time_dist_shift"], math.sqrt(3))
                self.assertEqual((geometry["latent_channels"], geometry["future_chunks"]), (48, 4))
        # The six sizes differ only in run_name, the DiT size and the size tag.
        reference = copy.deepcopy(self.configs["tiny"])
        for size, config in self.configs.items():
            strip = copy.deepcopy(config)
            for item in (strip, reference):
                item.pop("run_name", None)
                item["wandb"]["tags"] = [tag for tag in item["wandb"]["tags"] if tag not in SIZES]
                for key in ("dim", "num_heads", "num_layers", "ffn_dim", "base_depth"):
                    item["dit"].pop(key, None)
            self.assertEqual(strip, reference, size)

    def test_mix5_sources_equal_eupe(self):
        wan = load_config(NV / "wan22_mix5.yaml")
        eupe = load_config(NV / "eupe_mix5.yaml")
        self.assertEqual(wan["data"]["train"], eupe["data"]["train"])
        self.assertEqual(wan["data"]["val"], eupe["data"]["val"])
        self.assertEqual(len(wan["data"]["train"]["sources"]), 18)
        nv_train.validate_build(wan)

    def test_validation(self):
        config = copy.deepcopy(self.configs["tiny"])
        for mutate, match in (
            (lambda c: c["transport"].update(time_dist_shift=6.928203230275509), "time_dist_shift"),
            (lambda c: c["data"].update(image_size=120), "16x"),
            (lambda c: c["stage1"].update(precision="fp16"), "fp32 or bf16"),
            (lambda c: c["stage1"].update(revision="main"), "revision"),
            (lambda c: c.update(model={"encoder": {"name": "eupe"}}), "no V-RAE model section"),
            (lambda c: c["dit"].update(patch_size=[2, 1, 1]), r"\[1, p_h, p_w\]"),
            (lambda c: c["dit"].update(name="vrae_video_dit"), "dit.name"),
            (lambda c: c["stage1"].update(type="sd_vae"), "stage1.type"),
        ):
            broken = copy.deepcopy(config)
            mutate(broken)
            with self.subTest(match=match), self.assertRaisesRegex(ValueError, match):
                nv_train.validate_build(broken)
        # V-RAE configs cannot select the Wan DiT.
        eupe = copy.deepcopy(self.eupe["tiny"])
        eupe["dit"]["name"] = "wan_ti2v_dit"
        with self.assertRaisesRegex(ValueError, "vrae_first_frame_prediction_dit"):
            nv_train.validate_build(eupe)


# ---------------------------------------------------------------- end to end


def wan_task_config() -> dict:
    """wan22_mix5.yaml on toy data at 32x32 (2x2 latent grid) with a 2-block DiT, CPU settings."""

    config = load_config(NV / "wan22_mix5.yaml")
    ladder = load_config(wan_ladder_path("tiny"))
    config["data"]["train"] = {"sources": toy_sources("train")}
    config["data"]["val"] = {"sources": toy_sources("val")}
    config["data"]["image_size"] = 32
    config["transport"]["time_dist_shift"] = math.sqrt(4 * 2 * 2 * 48 / 4096)
    config["dit"].update(
        {"dim": 64, "num_heads": 2, "num_layers": 2, "ffn_dim": 96, "base_depth": 1, "gradient_checkpointing": False}
    )
    config["training"] = {
        **copy.deepcopy(ladder["training"]),
        "global_batch_size": 4,
        "num_workers": 0,
        "pin_memory": False,
        "precision": "fp32",
        "checkpoint_interval": 1,
    }
    config["latent_normalizer"].update({"path": "ckpts/stats/toy_wan_future.pt", "fit_clips": 3, "batch_size": 2})
    config["sampling"].update({"steps": 2, "cfg_scale": 1.5})
    config["wandb"].update({"mode": "disabled", "log_interval": 1, "sample_interval": 1, "sample_count": 1})
    return config


class WanEndToEndTest(unittest.TestCase):
    def test_stats_train_resume_sample_and_controls(self):
        from vrae.config import save_resolved_config
        from vrae.training.common.contracts import create_run_metadata
        from vrae.training.common.wandb import bind_wandb_run_name
        from vrae.training.nanovvm_video_pred import latent_stats, sample

        with (
            tempfile.TemporaryDirectory() as tmp,
            fake_decord(),
            replaced_module("wandb", fake_wandb_module()),
        ):
            base = Path(tmp)
            root = make_data_root(base)
            weights = base / "ckpts/pretrained/wan22_vae/Wan2.2_VAE.pth"
            weights.parent.mkdir(parents=True)
            (base / "Wan2.2_VAE.pth").write_bytes(b"not real weights")
            weights.symlink_to(base / "Wan2.2_VAE.pth")  # the weights are a symlink, as on the cluster
            paths = ProjectPaths(project_root=base, datasets={"nanovvm": root})
            config = wan_task_config()
            config["wandb"] = bind_wandb_run_name(config["wandb"], config["run_name"])
            nv_train.validate_build(config)
            backend = load_tiny_backend(
                paths.checkpoint(config["stage1"]["checkpoint"]), precision=config["stage1"]["precision"]
            )
            self.assertEqual(backend.metadata()["dtype"], "bfloat16")
            loader = mock.Mock(return_value=(backend, None))
            with (
                mock.patch.object(latent_stats, "load_frozen_latent_backend", loader),
                mock.patch.object(nv_train, "load_frozen_latent_backend", loader),
                mock.patch.object(sample, "load_frozen_latent_backend", loader),
                mock.patch.object(nv_train, "load_frozen_stage1", side_effect=AssertionError("V-RAE loader")),
            ):
                stats = latent_stats.compute_nanovvm_latent_stats(config, paths)
                normalizer = LatentNormalizer.load(stats)
                self.assertEqual(normalizer.metadata["future_latent_shape"], [4, 4, 48])
                self.assertEqual(normalizer.metadata["stage1"], backend.metadata())
                self.assertEqual(normalizer.metadata["stage1_weights"], "official")
                condition = normalizer.metadata["condition_latent"]
                self.assertEqual((len(condition["mean"]), len(condition["std"])), (48, 48))
                self.assertEqual(condition["scope"], "first_frame_z0")
                nv_train.load_task_normalizer(
                    config, paths, stage1_metadata=backend.metadata(), device=torch.device("cpu")
                )
                # A normalizer of another Wan checkpoint (different weights hash) is rejected.
                with self.assertRaisesRegex(ValueError, "weights_sha256"):
                    nv_train.load_task_normalizer(
                        config, paths, stage1_metadata={**backend.metadata(), "weights_sha256": "0" * 64},
                        device=torch.device("cpu"),
                    )

                run = paths.training_run(config["task"], config["run_name"], create=True)
                for directory in ("checkpoints", "samples", "wandb"):
                    (run / directory).mkdir()
                save_resolved_config(create_run_metadata(config), run / "run_metadata.yaml")
                nv_train.train(config, paths, run, max_steps=1)
                checkpoint = run / "checkpoints" / "step-00000001.pt"
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                metadata = payload["model_metadata"]
                self.assertEqual(metadata["stage1"], backend.metadata())
                self.assertEqual(metadata["dit_name"], "wan_ti2v_dit")
                self.assertEqual(metadata["first_frame_condition"], "wan22_z0_latent_frame_t0")
                self.assertEqual((metadata["dit_dim"], metadata["dit_num_layers"], metadata["dit_base_depth"]),
                                 (64, 2, 1))
                self.assertIn("null_condition", payload["model"])
                samples = torch.load(run / "samples" / "step-00000001.pt")
                self.assertEqual(tuple(samples["predicted_future"].shape), (1, 16, 3, 32, 32))
                self.assertEqual(tuple(samples["real_future_rgb"].shape), (1, 16, 3, 32, 32))

                resumed = copy.deepcopy(config)
                resumed["training"]["resume"] = str(checkpoint.relative_to(base))
                nv_train.train(resumed, paths, run, max_steps=2)
                self.assertTrue((run / "checkpoints" / "step-00000002.pt").is_file())

                stage1, dit, loaded = sample.load_sampling_stack(
                    config, paths, run / "checkpoints" / "step-00000002.pt", torch.device("cpu")
                )
                arguments = dict(sample_indices=[0], base_seed=1, steps=2,
                                 transport=FlowMatchingTransport(**config["transport"]), cfg_scale=2.0)
                first = torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8)
                video = sample.sample_from_first_frame(stage1, dit, loaded, first, **arguments)
                self.assertEqual(tuple(video.shape), (1, 16, 3, 32, 32))
                self.assertTrue(torch.equal(video, sample.sample_from_first_frame(stage1, dit, loaded, first,
                                                                                  **arguments)))
                # sample_from_first_frame = encode frame 0 alone, DiT on the normalized z0, decode with raw z0.
                z0 = backend.encode_condition(first)
                tokens = sample.predict_future_tokens(
                    dit, nv_latents.condition_features(backend, z0, loaded),
                    loaded, **{key: value for key, value in arguments.items()},
                )
                self.assertTrue(torch.equal(video, backend.decode_future(z0, tokens)))


class WanControlsTest(unittest.TestCase):
    def test_controls_only_fvd_with_the_wan_backend(self):
        from tests.test_nanovvm_fvd import ToyEncoder, fvd_data_root, fvd_section, write_encoder_files
        from vrae.evaluation.nanovvm_fvd import fvd as engine
        from vrae.storage.backend import configure_storage
        from vrae.training.nanovvm_video_pred import fvd as task_fvd

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = fvd_data_root(base)
            write_encoder_files(base)
            weights = base / "ckpts/pretrained/wan22_vae/Wan2.2_VAE.pth"
            weights.parent.mkdir(parents=True)
            (base / "Wan2.2_VAE.pth").write_bytes(b"not real weights")
            weights.symlink_to(base / "Wan2.2_VAE.pth")
            paths = ProjectPaths(project_root=base, datasets={"nanovvm": root})
            config = wan_task_config()
            config["fvd"] = fvd_section(image_size=32)
            backend = load_tiny_backend(weights)
            fvd_config = task_fvd.resolve_fvd_config(config, paths)
            recorded = []
            original = task_fvd.VRAEFuturePredictor.reconstruct_future

            def record(self, future_rgb, condition_rgb=None):
                recorded.append((future_rgb.clone(), condition_rgb.clone()))
                return original(self, future_rgb, condition_rgb)

            try:
                with (
                    mock.patch.object(engine, "VideoEncoder", ToyEncoder),
                    mock.patch.object(task_fvd.VRAEFuturePredictor, "reconstruct_future", record),
                    mock.patch.object(task_fvd.VRAEFuturePredictor, "generate",
                                      side_effect=AssertionError("no model in controls-only")),
                ):
                    metrics = task_fvd.evaluate(
                        config=config, paths=paths, fvd_config=fvd_config, stage1=backend, dit=None,
                        normalizer=None, transport=None, autocast=contextlib.nullcontext, rank=0,
                        world_size=1, device=torch.device("cpu"), run_dir=base / "controls", step=0,
                        model_name="none", controls_only=True,
                    )
            finally:
                configure_storage(None)
            self.assertEqual(
                set(metrics),
                {f"{prefix}/{split}/{encoder}/{source}"
                 for prefix in ("eval_fvd", "eval_fvd_alpha", "eval_fvd_beta")
                 for split in ("val", "train") for encoder in ("i3d", "videomaev2")
                 for source in ("ae_reconstruction", "static")},
            )
            # Every control got its conditioning frame; static is frame 0 repeated (17 identical frames).
            self.assertEqual(len(recorded), 2 * 8)
            for future, condition in recorded:
                self.assertEqual((tuple(future.shape), tuple(condition.shape)), ((1, 16, 3, 32, 32), (1, 3, 32, 32)))
            static = [item for item in recorded if torch.equal(item[0], item[1][:, None].expand_as(item[0]))]
            self.assertEqual(len(static), 8)
            score = json.loads((base / "controls/fvd_eval/step_0000000/val/fvd_score.json").read_text())
            self.assertTrue(score["controls_only"])
            self.assertEqual(score["protocol"]["ae"]["backend"], "wan22_vae")
            self.assertEqual(score["protocol"]["ae"]["structure"], backend.metadata())
            self.assertNotIn("model", score["scores"]["i3d"]["values"])
            # The Wan identity differs from a V-RAE one, so the control caches never mix.
            self.assertNotIn("vrae_stage1", score["protocol"]["ae"])


if __name__ == "__main__":
    unittest.main()
