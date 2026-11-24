"""MeanFlow (MF) and improved MeanFlow (iMF) for nanovvm_video_pred with wan_ti2v_dit: CPU tests.

Covers the DiT's MeanFlow inputs (time_delta, guidance, aux v branch), MeanFlowTransport (t/r
sampling, JVP, losses, CFG, sampler), the engine / train / sample wiring, the configs and an end-to-end
run on the tiny Wan2.2 VAE stand-in of tests/test_nanovvm_wan22.py. Run from the repository root:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONPATH=src python -m unittest tests.test_nanovvm_meanflow -v
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import os
import socket
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from vrae.config import load_config  # noqa: E402
from vrae.models.dit.meanflow import MeanFlowTransport, adaptive_loss  # noqa: E402
from vrae.models.dit.transport import FlowMatchingTransport  # noqa: E402
from vrae.models.dit.wan_ti2v_dit import WanTI2VDiT  # noqa: E402
from vrae.training.common.engine import flow_transport_metadata  # noqa: E402
from torch import nn  # noqa: E402
from tests.test_nanovvm_video_pred import assert_close, randomize  # noqa: E402
from tests.test_nanovvm_wan22 import small_wan_dit, wan_ladder_path  # noqa: E402

MEANFLOW_FLAGS = dict(time_delta_conditioning=True, guidance_conditioning=True)
NEW_EMBEDDINGS = ("time_delta_embedding", "guidance_scale_embedding", "guidance_start_embedding",
                  "guidance_end_embedding")


def expected_default_keys(layers: int) -> list[str]:
    """state_dict keys of the flow-matching wan_ti2v_dit, in order (before the MeanFlow change)."""

    keys = ["null_condition", "patch_embedding.weight", "patch_embedding.bias"]
    keys += [f"time_embedding.{i}.{p}" for i in (0, 2) for p in ("weight", "bias")]
    keys += ["time_projection.1.weight", "time_projection.1.bias"]
    for index in range(layers):
        block = f"blocks.{index}"
        keys.append(f"{block}.modulation")
        keys += [f"{block}.self_attn.{n}.{p}" for n in ("q", "k", "v", "o") for p in ("weight", "bias")]
        keys += [f"{block}.self_attn.norm_q.weight", f"{block}.self_attn.norm_k.weight"]
        keys += [f"{block}.ffn.{i}.{p}" for i in (0, 2) for p in ("weight", "bias")]
    for head in ("head", "base_head"):
        keys += [f"{head}.modulation", f"{head}.head.weight", f"{head}.head.bias"]
    return keys


def jvp_math(function, primals, tangents):
    """torch.func.jvp with the math SDPA backend (flash / efficient have no forward-mode AD)."""

    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        return torch.func.jvp(function, primals, tangents)


# ---------------------------------------------------------------- step 1: the DiT


class FlowMatchingRegressionTest(unittest.TestCase):
    def test_default_model_is_unchanged(self):
        dit = small_wan_dit()
        state = dit.state_dict()
        self.assertEqual(list(state), expected_default_keys(3))
        self.assertEqual(sum(p.numel() for p in dit.parameters()), 144_768)  # wan22-ti2v @ e42a83f
        for name in (*NEW_EMBEDDINGS, "aux_v_blocks", "aux_v_head"):
            self.assertFalse(hasattr(dit, name), name)
        self.assertEqual(dit.condition_embeddings(), {})
        self.assertEqual(
            dit.architecture_metadata(),
            {"dit_family": "wan2.2_ti2v", "dit_wan_revision": "1ea34ff48f87168174e12956e200b1d908b1c5ff",
             "dit_dim": 64, "dit_ffn_dim": 96, "dit_num_heads": 2, "dit_num_layers": 3, "dit_base_depth": 1,
             "dit_patch_size": [1, 1, 1], "dit_freq_dim": 256, "dit_eps": 1e-6,
             "dit_qk_norm": "rms_norm_across_heads", "dit_rope_theta": 10000.0, "dit_rope_split": [12, 10, 10],
             "dit_time_scale": 1000.0, "dit_sequence": "z0_t0_then_future"},
        )
        # The default model rejects the MeanFlow inputs instead of ignoring them.
        noisy, z0, time = torch.randn(2, 4, 16, 48), torch.randn(2, 16, 48), torch.tensor([0.2, 0.7])
        with self.assertRaisesRegex(ValueError, "time_delta_conditioning"):
            dit(noisy, time, z0, time_delta=torch.zeros(2))
        with self.assertRaisesRegex(ValueError, "guidance_conditioning"):
            dit(noisy, time, z0, guidance_scale=2.0)
        with self.assertRaisesRegex(ValueError, "guidance_conditioning"):
            dit(noisy, time, z0, guidance_interval=(0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "aux_v_head_depth"):
            dit(noisy, time, z0, return_aux_v=True, return_dict=True)
        # return_dict only repackages the outputs.
        dit.eval()
        full, base = dit(noisy, time, z0, return_base=True)
        packed = dit(noisy, time, z0, return_base=True, return_dict=True)
        self.assertEqual(set(packed), {"full", "base"})
        self.assertTrue(torch.equal(packed["full"], full) and torch.equal(packed["base"], base))
        self.assertEqual(set(dit(noisy, time, z0, return_dict=True)), {"full"})

    def test_flow_matching_transport_metadata_snapshot(self):
        config = load_config(wan_ladder_path("tiny"))
        self.assertEqual(
            flow_transport_metadata(config),
            {"prediction": "x", "time_dist_type": "logit-normal_0_1", "time_dist_shift": math.sqrt(3),
             "t_eps": 0.05, "base_model_coeff": 1.0},
        )


class MeanFlowDiTTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        generator = torch.Generator().manual_seed(1)
        self.noisy = torch.randn(3, 4, 16, 48, generator=generator)
        self.z0 = torch.randn(3, 16, 48, generator=generator)
        self.time = torch.tensor([0.3, 0.8, 0.55])

    def test_new_layers_and_metadata(self):
        dit = small_wan_dit(**MEANFLOW_FLAGS, aux_v_head_depth=1)
        default_keys = expected_default_keys(3)
        keys = list(dit.state_dict())
        self.assertEqual(keys[: len(default_keys)], default_keys)  # the new layers come last
        extra = {key.split(".")[0] for key in keys[len(default_keys):]}
        self.assertEqual(extra, {*NEW_EMBEDDINGS, "aux_v_blocks", "aux_v_head"})
        self.assertEqual(set(dit.condition_embeddings()), set(NEW_EMBEDDINGS))
        for name in NEW_EMBEDDINGS:
            embedding = getattr(dit, name)
            self.assertEqual([type(m).__name__ for m in embedding], ["Linear", "SiLU", "Linear"])
            self.assertEqual((embedding[0].in_features, embedding[2].out_features), (256, 64))
            self.assertEqual(embedding[2].weight.abs().sum().item(), 0)
            self.assertEqual(embedding[2].bias.abs().sum().item(), 0)
            self.assertLess(abs(embedding[0].weight.std().item() - 0.02), 0.004)
            self.assertEqual(embedding[0].bias.abs().sum().item(), 0)
        self.assertEqual(len(dit.aux_v_blocks), 1)
        self.assertEqual(dit.aux_v_head.head.weight.abs().sum().item(), 0)
        metadata = dit.architecture_metadata()
        self.assertEqual(
            {key: metadata[key] for key in metadata if key not in small_wan_dit().architecture_metadata()},
            {"dit_time_delta_conditioning": True, "dit_guidance_conditioning": True, "dit_aux_v_head_depth": 1},
        )
        only_delta = small_wan_dit(time_delta_conditioning=True).architecture_metadata()
        self.assertNotIn("dit_guidance_conditioning", only_delta)
        self.assertNotIn("dit_aux_v_head_depth", only_delta)
        with self.assertRaisesRegex(ValueError, "base_depth"):
            small_wan_dit(aux_v_head_depth=3)  # base_depth 1 > 3 - 3
        with self.assertRaisesRegex(ValueError, "base_depth"):
            small_wan_dit(aux_v_head_depth=2, base_depth=2)
        small_wan_dit(aux_v_head_depth=2, base_depth=1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            small_wan_dit(aux_v_head_depth=-1)

    def test_zero_initialized_conditions_match_the_default_model(self):
        """Test 2: default weights + zero-initialized new layers -> bit-identical outputs."""

        reference = small_wan_dit()
        randomize(reference, std=0.05)
        reference.eval()
        drop = torch.tensor([False, True, False])
        expected_full, expected_base = reference(
            self.noisy, self.time, self.z0, condition_drop_mask=drop, return_base=True
        )
        for flags in ({"time_delta_conditioning": True}, {"guidance_conditioning": True}, MEANFLOW_FLAGS):
            model = small_wan_dit(**flags)
            missing, unexpected = model.load_state_dict(reference.state_dict(), strict=False)
            self.assertEqual(unexpected, [])
            self.assertEqual({key.split(".")[0] for key in missing},
                             set(model.condition_embeddings()))
            model.eval()
            for delta, scale, interval in (
                (None, None, None),
                (torch.tensor([0.0, 0.5, 0.25]), torch.tensor([1.0, 3.0, 7.5]), (torch.tensor([0.1, 0.0, 0.4]),
                                                                                  torch.tensor([0.9, 1.0, 0.6]))),
                (0.7, 2.0, (0.2, 0.8)),
            ):
                kwargs = {}
                if model.time_delta_conditioning and delta is not None:
                    kwargs["time_delta"] = delta
                if model.guidance_conditioning and scale is not None:
                    kwargs.update(guidance_scale=scale, guidance_interval=interval)
                with self.subTest(flags=flags, delta=delta):
                    full, base = model(self.noisy, self.time, self.z0, condition_drop_mask=drop,
                                       return_base=True, **kwargs)
                    self.assertTrue(torch.equal(full, expected_full))
                    self.assertTrue(torch.equal(base, expected_base))
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        low = model(self.noisy, self.time, self.z0, condition_drop_mask=drop, **kwargs)
                        low_reference = reference(self.noisy, self.time, self.z0, condition_drop_mask=drop)
                    self.assertTrue(torch.equal(low, low_reference))

    def test_conditions_reach_the_embedding(self):
        model = small_wan_dit(**MEANFLOW_FLAGS)
        randomize(model, std=0.05)  # the zero last layers would hide every condition
        model.eval()
        e, e0 = model._time_embeddings(self.time, 5)
        defaults = dict(time_delta=torch.zeros(3), guidance=(torch.ones(3), torch.zeros(3), torch.ones(3)))
        e_default, _ = model._time_embeddings(self.time, 5, **defaults)

        def embed(module, values):
            return module(model._sinusoid(values, torch.float32))

        delta = torch.tensor([0.1, 0.4, 0.0])
        scale, start, end = torch.tensor([2.0, 1.0, 5.0]), torch.tensor([0.1, 0.0, 0.3]), torch.tensor([0.6, 1.0, 0.9])
        e_cond, e0_cond = model._time_embeddings(self.time, 5, time_delta=delta, guidance=(scale, start, end))
        guidance = (embed(model.guidance_scale_embedding, 1 - 1 / scale)
                    + embed(model.guidance_start_embedding, start) + embed(model.guidance_end_embedding, end))
        # Frame 0: h = 0; frames 1..4: h; the guidance terms on every frame.
        assert_close(e_cond[:, 0], e[:, 0] + embed(model.time_delta_embedding, torch.zeros(3)) + guidance, atol=1e-6)
        for frame in range(1, 5):
            assert_close(e_cond[:, frame], e[:, frame] + embed(model.time_delta_embedding, delta) + guidance,
                         atol=1e-6)
        assert_close(e0_cond, model.time_projection(e_cond).unflatten(2, (6, 64)), atol=1e-6)
        # Omitted inputs mean h = 0, omega = 1, interval [0, 1].
        out_default = model(self.noisy, self.time, self.z0)
        out_explicit = model(self.noisy, self.time, self.z0, time_delta=0.0, guidance_scale=1.0,
                             guidance_interval=(0.0, 1.0))
        self.assertTrue(torch.equal(out_default, out_explicit))
        self.assertFalse(torch.allclose(e_default, e))
        # Each input changes the output of its own sample only.
        for kwargs in ({"time_delta": delta}, {"guidance_scale": scale}, {"guidance_interval": (start, end)}):
            moved = model(self.noisy, self.time, self.z0, **kwargs)
            with self.subTest(kwargs=list(kwargs)):
                self.assertFalse(torch.allclose(moved[0], out_default[0]))
        moved = model(self.noisy, self.time, self.z0, time_delta=torch.tensor([0.0, 0.0, 0.3]))
        self.assertTrue(torch.equal(moved[:2], out_default[:2]))
        with self.assertRaisesRegex(ValueError, "time_delta"):
            model(self.noisy, self.time, self.z0, time_delta=torch.zeros(2))
        with self.assertRaisesRegex(ValueError, r"\(t_min, t_max\)"):
            model(self.noisy, self.time, self.z0, guidance_interval=(0.0, 0.5, 1.0))

    def test_aux_v_branch(self):
        torch.manual_seed(0)
        model = small_wan_dit(time_delta_conditioning=True, aux_v_head_depth=1)
        randomize(model, std=0.05)
        model.eval()
        delta = torch.tensor([0.1, 0.2, 0.0])
        out = model(self.noisy, self.time, self.z0, time_delta=delta, return_dict=True, return_base=True,
                    return_aux_v=True)
        self.assertEqual(set(out), {"full", "base", "aux_v"})
        self.assertEqual(out["aux_v"].shape, self.noisy.shape)
        self.assertTrue(torch.equal(out["full"], model(self.noisy, self.time, self.z0, time_delta=delta)))
        self.assertFalse(torch.allclose(out["aux_v"], out["full"]))
        with self.assertRaisesRegex(ValueError, "return_dict"):
            model(self.noisy, self.time, self.z0, return_aux_v=True)
        # The v branch forks after block L - K = 2: its blocks and head see that hidden state.
        recorded = {}
        handle = model.blocks[1].register_forward_hook(lambda m, i, o: recorded.__setitem__("fork", o))
        model(self.noisy, self.time, self.z0, time_delta=delta)
        handle.remove()
        e, e0 = model._time_embeddings(self.time, 5, time_delta=delta)
        from vrae.models.dit.wan_ti2v_dit import rope_cos_sin

        cos, sin = rope_cos_sin((5, 4, 4), 32, 10000.0, self.noisy.device)
        value = model.aux_v_blocks[0](recorded["fork"], e0, cos, sin)
        expected = model._unpatchify(model.aux_v_head(value, e), 5)[:, 1:]
        assert_close(out["aux_v"], expected, atol=1e-6)
        # Gradient checkpointing covers the v branch and is exact; the v branch trains its own weights.
        grads = []
        for enabled in (False, True):
            twin = small_wan_dit(time_delta_conditioning=True, aux_v_head_depth=1, gradient_checkpointing=enabled,
                                 frame_dropout=0.0)
            twin.load_state_dict(model.state_dict())
            twin.train()
            calls = []
            if enabled:
                import vrae.models.dit.wan_ti2v_dit as wan_module

                original = wan_module.checkpoint

                def counting(*args, **kwargs):
                    calls.append(args[0])
                    return original(*args, **kwargs)

                wan_module.checkpoint = counting
            try:
                result = twin(self.noisy, self.time, self.z0, time_delta=delta, return_dict=True, return_aux_v=True)
            finally:
                if enabled:
                    wan_module.checkpoint = original
            if enabled:
                self.assertEqual(len(calls), 4)  # 3 trunk/u blocks + 1 v block
                self.assertIs(calls[-1], twin.aux_v_blocks[0])
            result["aux_v"].square().mean().backward()
            grads.append({n: p.grad.clone() for n, p in twin.named_parameters() if p.grad is not None})
            self.assertGreater(twin.aux_v_head.head.weight.grad.abs().sum().item(), 0)
            self.assertIsNone(twin.head.head.weight.grad)  # only the v output entered the loss
        self.assertEqual(set(grads[0]), set(grads[1]))
        for name in grads[0]:
            assert_close(grads[0][name], grads[1][name], atol=1e-7)


class JVPTest(unittest.TestCase):
    """Test 3 and 4: torch.func.jvp through the DiT against central differences, and under no_grad."""

    def test_jvp_matches_central_differences_in_float64(self):
        generator = torch.Generator().manual_seed(3)
        z = torch.randn(3, 4, 16, 48, generator=generator, dtype=torch.float64)
        tangent = torch.randn(3, 4, 16, 48, generator=generator, dtype=torch.float64)
        z0 = torch.randn(3, 16, 48, generator=generator, dtype=torch.float64)
        t = torch.tensor([0.35, 0.9, 0.6], dtype=torch.float64)
        r = torch.tensor([0.1, 0.25, 0.6], dtype=torch.float64)
        ones, zeros = torch.ones_like(t), torch.zeros_like(r)
        epsilon = 1e-6  # the 1000 t sinusoid makes the O(eps^2) term large for bigger steps
        for guidance in (False, True):
            torch.manual_seed(0)
            model = small_wan_dit(time_delta_conditioning=True, guidance_conditioning=guidance).double()
            randomize(model, std=0.05)
            model.train()  # an explicit drop mask: no randomness even in training mode
            for drop in (torch.zeros(3, dtype=torch.bool), torch.tensor([True, False, True])):
                for with_base in (False, True):
                    extra = {}
                    if guidance:
                        extra = dict(guidance_scale=torch.tensor([1.0, 2.5, 6.0], dtype=torch.float64),
                                     guidance_interval=(torch.tensor([0.0, 0.2, 0.3], dtype=torch.float64),
                                                        torch.tensor([1.0, 0.7, 0.9], dtype=torch.float64)))

                    def u_fn(z_, t_, r_):
                        return model(z_, t_, z0, time_delta=t_ - r_, condition_drop_mask=drop,
                                     return_base=with_base, **extra)

                    _, dudt = jvp_math(u_fn, (z, t, r), (tangent, ones, zeros))
                    with torch.no_grad():
                        plus = u_fn(z + epsilon * tangent, t + epsilon, r)
                        minus = u_fn(z - epsilon * tangent, t - epsilon, r)
                    pairs = zip(dudt, plus, minus) if with_base else [(dudt, plus, minus)]
                    for derivative, high, low in pairs:
                        difference = (high - low) / (2 * epsilon)
                        relative = ((derivative - difference).norm() / derivative.norm()).item()
                        with self.subTest(guidance=guidance, drop=drop.tolist(), base=with_base):
                            self.assertEqual(derivative.dtype, torch.float64)
                            self.assertLess(relative, 1e-4)

    def test_no_grad_jvp_in_training_mode_with_checkpointing(self):
        torch.manual_seed(0)
        model = small_wan_dit(time_delta_conditioning=True, gradient_checkpointing=True)
        randomize(model, std=0.05)
        model.train()
        z = torch.randn(2, 4, 16, 48)
        z0 = torch.randn(2, 16, 48)
        t, r = torch.tensor([0.6, 0.9]), torch.tensor([0.2, 0.4])
        drop = torch.zeros(2, dtype=torch.bool)
        import vrae.models.dit.wan_ti2v_dit as wan_module

        original = wan_module.checkpoint
        wan_module.checkpoint = lambda *a, **k: (_ for _ in ()).throw(AssertionError("checkpoint under no_grad"))
        try:
            output, dudt = jvp_math(
                lambda z_, t_, r_: model(z_, t_, z0, time_delta=t_ - r_, condition_drop_mask=drop, return_base=True),
                (z, t, r), (torch.randn_like(z), torch.ones(2), torch.zeros(2)),
            )
        finally:
            wan_module.checkpoint = original
        for value in (*output, *dudt):
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
            self.assertTrue(torch.isfinite(value).all())
        self.assertGreater(dudt[0].abs().sum().item(), 0)
        # With grad enabled the same model checkpoints its blocks (the training forward is unchanged).
        self.assertTrue(isinstance(model, WanTI2VDiT) and model.gradient_checkpointing and model.training)


# ---------------------------------------------------------------- step 2: MeanFlowTransport


def meanflow_dit(**overrides) -> WanTI2VDiT:
    torch.manual_seed(0)
    arguments = dict(time_delta_conditioning=True, frame_dropout=0.3)
    arguments.update(overrides)
    dit = small_wan_dit(**arguments)
    randomize(dit, std=0.02)  # the zero heads / zero condition layers would block every gradient
    return dit


def meanflow(**overrides) -> MeanFlowTransport:
    arguments = dict(variant="imf", time_dist_shift=math.sqrt(3), ratio_r_neq_t=0.5)
    arguments.update(overrides)
    return MeanFlowTransport(**arguments)


def toy_batch(batch: int = 4, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 4, 16, 48, generator=generator), torch.randn(batch, 16, 48, generator=generator)


class Spy(nn.Module):
    """Records every call: batch size, grad mode and the condition-drop mask."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.calls: list[dict] = []

    @property
    def frame_dropout(self) -> float:
        return self.inner.frame_dropout

    def forward(self, noisy, time, context=None, **kwargs):
        mask = kwargs.get("condition_drop_mask")
        self.calls.append({
            "batch": noisy.shape[0],
            "grad": torch.is_grad_enabled(),
            "mask": None if mask is None else mask.clone(),
            "omega": kwargs.get("guidance_scale"),
            "keys": sorted(kwargs),
        })
        return self.inner(noisy, time, context, **kwargs)


def no_internal_dropout():
    """Patch the DiT's mask resolution so a missing explicit mask fails the test."""

    import vrae.models.dit.wan_ti2v_dit as wan_module

    original = wan_module._resolve_drop_mask

    def strict(**kwargs):
        if kwargs["drop_mask"] is None:
            raise AssertionError("the DiT drew its own condition-dropout mask")
        return original(**kwargs)

    return unittest.mock.patch.object(wan_module, "_resolve_drop_mask", strict)


class TimeSamplingTest(unittest.TestCase):
    """Test 8."""

    def test_max_min(self):
        transport = meanflow(ratio_r_neq_t=0.25)
        for batch in (4, 7, 32):
            t, r, neq = transport.sample_t_r(batch, device="cpu", generator=torch.Generator().manual_seed(batch))
            self.assertTrue(bool((t >= r).all()))
            self.assertEqual(int(neq.sum()), math.floor(0.25 * batch + 0.5))
            self.assertTrue(torch.equal(t[~neq], r[~neq]))
            self.assertTrue(bool((r[neq] < t[neq]).all()))
            self.assertTrue(bool(((t >= 0) & (t <= 1) & (r >= 0) & (r <= 1)).all()))
        # Positions are random: different draws select different rows.
        selections = {tuple(meanflow(ratio_r_neq_t=0.5).sample_t_r(16, device="cpu", generator=torch.Generator()
                                                                 .manual_seed(seed))[2].tolist())
                      for seed in range(8)}
        self.assertGreater(len(selections), 1)
        for ratio, count in ((0.0, 0), (1.0, 9)):
            _, _, neq = meanflow(ratio_r_neq_t=ratio).sample_t_r(9, device="cpu")
            self.assertEqual(int(neq.sum()), count)
        # t, r are the max / min of two draws of FlowMatchingTransport's distribution, shift included.
        reference = FlowMatchingTransport(prediction="velocity", time_dist_shift=math.sqrt(3))
        generator = torch.Generator().manual_seed(5)
        first = reference.sample_time(64, device="cpu", generator=generator)
        second = reference.sample_time(64, device="cpu", generator=generator)
        t, r, neq = meanflow(ratio_r_neq_t=1.0).sample_t_r(64, device="cpu", generator=torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(t, torch.maximum(first, second)))
        self.assertTrue(torch.equal(r, torch.minimum(first, second)))
        unshifted, _, _ = meanflow(ratio_r_neq_t=1.0, time_dist_shift=1.0).sample_t_r(
            64, device="cpu", generator=torch.Generator().manual_seed(5))
        assert_close(t, reference.shift_time(unshifted), atol=1e-6)
        self.assertGreater(t.mean().item(), unshifted.mean().item())

    def test_v1_keeps_the_flow_matching_marginal(self):
        transport = meanflow(tr_sampler="v1", ratio_r_neq_t=0.5)
        reference = FlowMatchingTransport(prediction="velocity", time_dist_shift=math.sqrt(3))
        t, r, neq = transport.sample_t_r(32, device="cpu", generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(t, reference.sample_time(32, device="cpu",
                                                             generator=torch.Generator().manual_seed(3))))
        self.assertEqual(int(neq.sum()), 16)
        self.assertTrue(bool((r <= t).all()))
        self.assertTrue(torch.equal(r[~neq], t[~neq]))
        # A separate r distribution (r_time_dist_*) only moves r.
        other = meanflow(tr_sampler="v1", r_time_dist_type="logit-normal_-1_1", r_time_dist_shift=1.0)
        t2, r2, _ = other.sample_t_r(32, device="cpu", generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(t2, t))
        self.assertEqual(other.metadata()["r_time_dist_type"], "logit-normal_-1_1")
        self.assertEqual(transport.metadata()["r_time_dist_type"], "logit-normal_0_1")

    def test_invalid_settings(self):
        for kwargs, match in (
            ({"variant": "pmf"}, "variant"),
            ({"prediction": "x"}, "prediction"),
            ({"variant": "mf", "tangent": "boundary"}, "tangent"),
            ({"variant": "imf", "tangent": "conditional"}, "tangent"),
            ({"ratio_r_neq_t": 1.5}, "ratio_r_neq_t"),
            ({"adaptive_weight": {"p": -1.0}}, "adaptive_weight.p"),
            ({"adaptive_weight": {"eps": 0.0}}, "adaptive_weight.eps"),
            ({"adaptive_weight": {"q": 1.0}}, "unknown"),
            ({"t_eps": 0.05}, "t_eps"),
            ({"tr_sampler": "v0"}, "tr_sampler"),
            ({"r_time_dist_type": "logit-normal_0_1"}, "v1 only"),
            ({"jvp": {"dtype": "float16"}}, "jvp.dtype"),
            ({"jvp": {"attention_backend": "flash"}}, "attention_backend"),
            ({"jvp": {"compile": True}}, "unknown"),
            ({"cfg": {"mode": "sometimes"}}, "cfg.mode"),
            ({"type": "flow_matching"}, "meanflow"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, match):
                meanflow(**kwargs)
        meanflow(t_eps=None)  # the null that overrides the flow-matching t_eps of an included file


class MeanFlowLossTest(unittest.TestCase):
    """Test 5."""

    def test_reduces_to_velocity_flow_matching(self):
        dit = meanflow_dit()
        dit.eval()
        clean, z0 = toy_batch()
        noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(9))
        time = torch.tensor([0.2, 0.5, 0.7, 0.95])  # >= t_eps: FM's target is not clamped
        flow = FlowMatchingTransport(prediction="velocity", time_dist_shift=math.sqrt(3), base_model_coeff=1.0)
        expected = flow.training_losses(dit, clean, z0, time=time, noise=noise)
        for variant in ("mf", "imf"):
            transport = meanflow(variant=variant, ratio_r_neq_t=0.0, adaptive_weight={"p": 0.0, "eps": 0.01})
            losses = transport.training_losses(dit, clean, z0, jvp_model=dit, time=time, noise=noise,
                                               drop_mask=torch.zeros(4, dtype=torch.bool))
            dimension = clean[0].numel()
            with self.subTest(variant=variant):
                self.assertTrue(math.isclose(losses["loss_full"].item(), dimension * expected["loss_full"].item(),
                                             rel_tol=1e-5))
                self.assertTrue(math.isclose(losses["loss_base"].item(), dimension * expected["loss_base"].item(),
                                             rel_tol=1e-5))
                self.assertTrue(math.isclose(losses["mse_full"].item(), expected["loss_full"].item(), rel_tol=1e-5))
                self.assertTrue(math.isclose(losses["mse_fm"].item(), expected["loss_full"].item(), rel_tol=1e-5))
                self.assertEqual(losses["mse_mf"].item(), 0.0)
                self.assertEqual(losses["dudt"].abs().sum().item(), 0.0)
        # With p = 1 every sample's term is L_i / (L_i + eps) (adaptive weighting, sum over elements).
        per_sample = torch.tensor([3.0, 0.5, 100.0])
        self.assertTrue(torch.allclose(adaptive_loss(per_sample, 1.0, 0.01), (per_sample / (per_sample + 0.01)).mean()))
        self.assertTrue(torch.allclose(adaptive_loss(per_sample, 0.5, 0.01),
                                       (per_sample / (per_sample + 0.01).sqrt()).mean()))

    def test_tangents_and_jvp(self):
        dit = meanflow_dit()
        dit.train()
        clean, z0 = toy_batch(6)
        noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(4))
        time = torch.tensor([0.9, 0.6, 0.8, 0.3, 0.7, 0.5])
        time_r = torch.tensor([0.2, 0.6, 0.1, 0.3, 0.65, 0.0])
        drop = torch.tensor([False, True, False, False, True, False])
        neq = time_r < time
        common = dict(jvp_model=dit, time=time, time_r=time_r, noise=noise, drop_mask=drop)
        mf = meanflow(variant="mf").training_losses(dit, clean, z0, **common)
        self.assertTrue(torch.equal(mf["tangent"], noise - clean))  # MF: tangent = target = e - x
        self.assertTrue(torch.equal(mf["target"], noise - clean))
        imf = meanflow(variant="imf").training_losses(dit, clean, z0, **common)
        noisy = imf["noisy"]
        with torch.no_grad():
            boundary = dit(noisy, time, z0, time_delta=torch.zeros(6), condition_drop_mask=drop)
        assert_close(imf["tangent"][neq], boundary[neq], atol=1e-5)  # iMF: u(z, t, 0), drop-applied
        self.assertTrue(torch.equal(imf["target"], noise - clean))
        # dudt = the JVP of u(z, t, t - r) along (tangent, 1, 0) on the r != t rows, 0 elsewhere.
        for losses in (mf, imf):
            _, (expected_full, expected_base) = jvp_math(
                lambda z_, t_, r_: dit(z_[neq], t_[neq], z0[neq], time_delta=(t_ - r_)[neq],
                                       condition_drop_mask=drop[neq], return_base=True),
                (noisy, time, time_r), (losses["tangent"], torch.ones(6), torch.zeros(6)),
            )
            assert_close(losses["dudt"][neq], expected_full, atol=1e-4)
            assert_close(losses["dudt_base"][neq], expected_base, atol=1e-4)
            self.assertEqual(losses["dudt"][~neq].abs().sum().item(), 0.0)
            self.assertEqual(losses["dudt_base"][~neq].abs().sum().item(), 0.0)
            self.assertGreater(losses["dudt"][neq].abs().min().item(), 0.0)
        # Sub-batch JVP == full-batch JVP masked to the r != t rows.
        for variant in ("mf", "imf"):
            subset = meanflow(variant=variant).training_losses(dit, clean, z0, **common)
            full = meanflow(variant=variant, jvp={"neq_subset_only": False}).training_losses(dit, clean, z0, **common)
            with self.subTest(variant=variant):
                assert_close(subset["dudt"], full["dudt"], atol=1e-5)
                assert_close(subset["dudt_base"], full["dudt_base"], atol=1e-5)
                self.assertEqual(full["dudt"][~neq].abs().sum().item(), 0.0)
                assert_close(subset["loss"], full["loss"], atol=1e-5)
        # The jvp precision option: autocast (no fp32 override) still gives the same derivative in fp32.
        autocast = meanflow(variant="mf", jvp={"dtype": "autocast"}).training_losses(dit, clean, z0, **common)
        assert_close(autocast["dudt"], mf["dudt"], atol=1e-5)
        # The logged statistics.
        self.assertTrue(math.isclose(mf["dudt_rms"].item(), mf["dudt"][neq].square().mean().sqrt().item(),
                                     rel_tol=1e-5))
        error = ((dit(noisy, time, z0, time_delta=time - time_r, condition_drop_mask=drop).detach()
                  + (time - time_r)[:, None, None, None] * mf["dudt"]) - mf["target"]).square()
        self.assertTrue(math.isclose(mf["mse_mf"].item(), error[neq].mean().item(), rel_tol=1e-4))
        self.assertTrue(math.isclose(mf["mse_fm"].item(), error[~neq].mean().item(), rel_tol=1e-4))
        self.assertTrue(math.isclose(mf["mse_full"].item(), error.mean().item(), rel_tol=1e-4))

    def test_drop_mask_is_drawn_once(self):
        """Test 6: every forward of one call gets the same mask; the DiT never draws its own."""

        for variant, ratio in (("mf", 1.0), ("imf", 1.0), ("imf", 0.5)):
            dit = meanflow_dit(frame_dropout=0.5)
            dit.train()
            spy = Spy(dit)
            clean, z0 = toy_batch(8)
            with no_internal_dropout():
                losses = meanflow(variant=variant, ratio_r_neq_t=ratio).training_losses(
                    spy, clean, z0, jvp_model=spy, generator=torch.Generator().manual_seed(1))
            drop, neq = losses["drop_mask"], losses["neq"]
            with self.subTest(variant=variant, ratio=ratio):
                self.assertTrue(bool(drop.any()) and not bool(drop.all()))
                self.assertEqual(len(spy.calls), 2 if variant == "mf" else 3)
                for call in spy.calls:
                    expected = drop if call["batch"] == 8 else drop[neq]
                    self.assertTrue(torch.equal(call["mask"], expected))
                self.assertEqual([call["grad"] for call in spy.calls][-1], True)
                self.assertEqual(sum(call["grad"] for call in spy.calls), 1)


class MeanFlowGradientTest(unittest.TestCase):
    """Test 7."""

    def test_dudt_carries_no_gradient(self):
        for variant in ("mf", "imf"):
            dit = meanflow_dit(gradient_checkpointing=True)
            dit.train()
            clean, z0 = toy_batch(6)
            transport = meanflow(variant=variant)
            losses = transport.training_losses(dit, clean, z0, jvp_model=dit,
                                               generator=torch.Generator().manual_seed(2))
            self.assertFalse(losses["dudt"].requires_grad)
            losses["loss"].backward()
            grads = {name: p.grad.clone() for name, p in dit.named_parameters()}
            dit.zero_grad(set_to_none=True)
            # The same objective with dudt as a constant from the start.
            t, r = losses["t"], losses["r"]
            h = (t - r)[:, None, None, None]
            out = dit(losses["noisy"], t, z0, time_delta=t - r, condition_drop_mask=losses["drop_mask"],
                      return_base=True, return_dict=True)
            target = losses["target"]
            full = adaptive_loss(((out["full"] + h * losses["dudt"]) - target).square().flatten(1).sum(1), 1.0, 0.01)
            base = adaptive_loss(((out["base"] + h * losses["dudt_base"]) - target).square().flatten(1).sum(1),
                                 1.0, 0.01)
            (full + base).backward()
            with self.subTest(variant=variant):
                assert_close(full + base, losses["loss"].detach(), atol=1e-5)
                for name, parameter in dit.named_parameters():
                    assert_close(parameter.grad, grads[name], atol=1e-6)

    def test_every_parameter_gets_a_gradient_and_model_runs_once(self):
        for name, dit_flags, transport_flags in (
            ("mf", {}, {"variant": "mf"}),
            ("imf", {}, {"variant": "imf"}),
        ):
            dit = meanflow_dit(gradient_checkpointing=True, **dit_flags)
            dit.train()
            wrapper = Spy(dit)  # stands in for DDP: the trainer passes model=DDP(dit), jvp_model=dit
            clean, z0 = toy_batch(6)
            with no_internal_dropout():
                losses = meanflow(**transport_flags).training_losses(
                    wrapper, clean, z0, jvp_model=dit, generator=torch.Generator().manual_seed(3))
            losses["loss"].backward()
            with self.subTest(name=name):
                self.assertEqual(len(wrapper.calls), 1)
                self.assertTrue(wrapper.calls[0]["grad"])
                missing = [n for n, p in dit.named_parameters() if p.requires_grad and p.grad is None]
                self.assertEqual(missing, [])
                for key in ("loss", "loss_full", "loss_base", "loss_aux_v", "mse_full", "mse_base", "mse_fm",
                            "mse_mf", "dudt_rms"):
                    self.assertEqual(losses[key].ndim, 0, key)
                    self.assertTrue(torch.isfinite(losses[key]), key)


# ---------------------------------------------------------------- step 3: CFG and the aux v head


def mock_u(z, t, h, drop, omega, start, end, context, scale):
    """A smooth closed-form stand-in for u_theta (every input changes the output)."""

    def col(value):
        return value.reshape(-1, 1, 1, 1)

    condition = torch.where(drop[:, None, None], -0.5 * context, context)[:, None]
    return (scale * z * (1 + col(t)) + col(h) * col(t) ** 2 + 0.1 * col(omega) + 0.01 * col(start)
            + 0.001 * col(end) + condition)


class MockDiT(nn.Module):
    """The wan_ti2v_dit call interface over mock_u; base = 0.9 full, aux v = 0.5 full + 0.3."""

    def __init__(self, frame_dropout: float = 0.5) -> None:
        super().__init__()
        self.frame_dropout = frame_dropout
        self.scale = nn.Parameter(torch.tensor(1.5))

    def forward(self, noisy, time, context=None, *, time_delta=None, condition_drop_mask=None,
                guidance_scale=None, guidance_interval=None, return_base=False, return_aux_v=False,
                return_dict=False):
        batch = noisy.shape[0]

        def per_sample(value, default):
            return torch.as_tensor(default if value is None else value, dtype=noisy.dtype).expand(batch)

        start, end = (0.0, 1.0) if guidance_interval is None else guidance_interval
        full = mock_u(noisy, per_sample(time, 0.0), per_sample(time_delta, 0.0), condition_drop_mask,
                      per_sample(guidance_scale, 1.0), per_sample(start, 0.0), per_sample(end, 1.0), context,
                      self.scale)
        if not return_dict:
            return (full, 0.9 * full) if return_base else full
        result = {"full": full}
        if return_base:
            result["base"] = 0.9 * full
        if return_aux_v:
            result["aux_v"] = 0.5 * full + 0.3
        return result


class GuidanceSamplingTest(unittest.TestCase):
    def test_omega_distribution(self):
        count = 40_000
        neq = torch.ones(count, dtype=torch.bool)
        for beta in (1.0, 2.0):
            transport = meanflow(cfg={"mode": "conditioned", "beta": beta, "omega_max": 8.0})
            omega, _, _ = transport.sample_guidance(count, neq, device="cpu", generator=torch.Generator().manual_seed(0))
            self.assertTrue(bool(((omega >= 1) & (omega <= 8)).all()))
            for point in (1.5, 2.0, 4.0, 6.0):
                expected = (math.log(point) / math.log(8) if beta == 1
                            else (1 - point ** (1 - beta)) / (1 - 8 ** (1 - beta)))
                empirical = (omega <= point).double().mean().item()
                with self.subTest(beta=beta, point=point):
                    self.assertLess(abs(empirical - expected), 0.01)
        # The closed form against imf.py's sample_cfg_scale for one uniform draw.
        u = torch.tensor([0.0, 0.3, 0.999])
        transport = meanflow(cfg={"mode": "conditioned", "beta": 2.0, "omega_max": 8.0})
        with unittest.mock.patch.object(torch, "rand", lambda *a, **k: u):
            omega, _, _ = transport.sample_guidance(3, torch.ones(3, dtype=torch.bool), device="cpu")
        log_base = (1 - 2.0) * math.log1p(7.0)
        expected = torch.exp(torch.log1p(u * math.expm1(log_base)) / (1 - 2.0))
        assert_close(omega, expected, atol=1e-6)

    def test_interval(self):
        neq = torch.tensor([True, False] * 500)
        transport = meanflow(cfg={"mode": "conditioned"})
        _, start, end = transport.sample_guidance(1000, neq, device="cpu", generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(start[~neq], torch.zeros(500)) and torch.equal(end[~neq], torch.ones(500)))
        self.assertTrue(bool(((start[neq] >= 0) & (start[neq] <= 0.5)).all()))
        self.assertTrue(bool(((end[neq] >= 0.5) & (end[neq] <= 1)).all()))
        self.assertGreater(start[neq].std().item(), 0.1)
        _, start, end = meanflow(cfg={"mode": "conditioned", "interval": False}).sample_guidance(
            10, torch.ones(10, dtype=torch.bool), device="cpu")
        self.assertTrue(torch.equal(start, torch.zeros(10)) and torch.equal(end, torch.ones(10)))

    def test_mode_validation(self):
        with self.assertRaisesRegex(ValueError, "cfg.mode"):
            meanflow(variant="mf", cfg={"mode": "conditioned"})
        with self.assertRaisesRegex(ValueError, "cfg.mode"):
            meanflow(variant="imf", cfg={"mode": "fixed"})
        with self.assertRaisesRegex(ValueError, "omega"):
            meanflow(variant="mf", cfg={"mode": "fixed", "omega": 0.5})
        with self.assertRaisesRegex(ValueError, "dropped_tangent"):
            meanflow(cfg={"mode": "conditioned", "dropped_tangent": "zero"})
        with self.assertRaisesRegex(ValueError, "cfg.mode: conditioned"):
            meanflow().training_losses(MockDiT(), *toy_batch(2), jvp_model=MockDiT(),
                                       guidance=(torch.ones(2), torch.zeros(2), torch.ones(2)))


class GuidedTargetTest(unittest.TestCase):
    def setUp(self):
        self.clean, self.z0 = toy_batch(6)
        self.noise = torch.randn(self.clean.shape, generator=torch.Generator().manual_seed(7))
        self.time = torch.tensor([0.9, 0.6, 0.8, 0.3, 0.7, 0.45])
        self.time_r = torch.tensor([0.2, 0.6, 0.1, 0.3, 0.65, 0.0])
        self.drop = torch.tensor([False, True, False, False, True, False])
        # omega; interval; row 2 is outside [t_min, t_max] (0.8 > 0.75), row 5 below (0.45 < 0.5).
        self.omega = torch.tensor([3.0, 2.0, 5.0, 1.0, 4.0, 2.5])
        self.start = torch.tensor([0.1, 0.0, 0.2, 0.0, 0.3, 0.5])
        self.end = torch.tensor([0.95, 1.0, 0.75, 1.0, 0.9, 0.8])

    def expected(self, model, transport):
        scale = model.scale.detach()
        z = (1 - self.time[:, None, None, None]) * self.clean + self.time[:, None, None, None] * self.noise
        v_t = self.noise - self.clean
        zeros, ones = torch.zeros(6), torch.ones(6)
        keep, null = torch.zeros(6, dtype=torch.bool), torch.ones(6, dtype=torch.bool)

        def velocity(drop, omega):  # V0: h = 0, interval [0, 1]; the aux head is 0.5 u + 0.3
            u = mock_u(z, self.time, zeros, drop, omega, zeros, ones, self.z0, scale)
            return 0.5 * u + 0.3 if transport.tangent == "aux_head" else u

        inside = (self.time >= self.start) & (self.time <= self.end)
        effective = torch.where(inside, self.omega, ones)
        v_c = velocity(keep, effective)
        v_u = velocity(null, ones)
        v_g = v_t + (1 - 1 / effective)[:, None, None, None] * (v_c - v_u)
        drop = self.drop[:, None, None, None]
        target = torch.where(drop, v_t, v_g)
        dropped = v_u if transport.cfg["dropped_tangent"] == "unconditional" else v_c
        tangent = torch.where(drop, dropped, v_c)
        return target, tangent, inside

    def run_transport(self, transport, model):
        return transport.training_losses(
            model, self.clean, self.z0, jvp_model=model, time=self.time, time_r=self.time_r, noise=self.noise,
            drop_mask=self.drop, guidance=(self.omega, self.start, self.end))

    def test_conditioned_target_and_tangent(self):
        for tangent in ("boundary", "aux_head"):
            for dropped in ("unconditional", "conditional"):
                model = MockDiT()
                transport = meanflow(tangent=tangent, cfg={"mode": "conditioned", "dropped_tangent": dropped})
                losses = self.run_transport(transport, model)
                target, expected_tangent, inside = self.expected(model, transport)
                with self.subTest(tangent=tangent, dropped=dropped):
                    self.assertFalse(bool(inside[2]) or bool(inside[5]))
                    assert_close(losses["target"], target, atol=1e-5)
                    assert_close(losses["tangent"], expected_tangent, atol=1e-5)
                    v_t = self.noise - self.clean
                    self.assertTrue(torch.equal(losses["target"][self.drop], v_t[self.drop]))
                    # Outside the interval omega_eff = 1: the target is v_t (no guidance).
                    assert_close(losses["target"][2], v_t[2], atol=1e-6)
                    # dudt is the JVP of u(z, t, t - r | dropped z0, sampled omega and interval).
                    neq = self.time_r < self.time
                    scale = model.scale.detach()
                    _, expected_dudt = jvp_math(
                        lambda z_, t_, r_: mock_u(z_, t_, t_ - r_, self.drop, self.omega, self.start, self.end,
                                                  self.z0, scale),
                        (losses["noisy"], self.time, self.time_r), (expected_tangent, torch.ones(6), torch.zeros(6)),
                    )
                    assert_close(losses["dudt"][neq], expected_dudt[neq], atol=1e-4)
                    self.assertEqual(losses["dudt"][~neq].abs().sum().item(), 0.0)

    def test_conditioned_forwards(self):
        model = MockDiT()
        spy = Spy(model)
        transport = meanflow(tangent="aux_head", cfg={"mode": "conditioned"})
        losses = transport.training_losses(
            spy, self.clean, self.z0, jvp_model=spy, time=self.time, time_r=self.time_r, noise=self.noise,
            drop_mask=self.drop, guidance=(self.omega, self.start, self.end))
        # One target forward (cond + uncond + the omega_eff = 1 recomputations), the JVP, the grad forward.
        self.assertEqual([call["batch"] for call in spy.calls], [6 + 6 + 2, 4, 6])
        target_call, jvp_call, grad_call = spy.calls
        self.assertTrue(torch.equal(target_call["mask"], torch.tensor([False] * 6 + [True] * 6 + [False] * 2)))
        neq = self.time_r < self.time
        self.assertTrue(torch.equal(jvp_call["mask"], self.drop[neq]))
        self.assertTrue(torch.equal(grad_call["mask"], self.drop))
        self.assertTrue(torch.equal(grad_call["omega"], self.omega))  # the sampled omega, not omega_eff
        self.assertTrue(grad_call["grad"] and not jvp_call["grad"] and not target_call["grad"])
        self.assertIn("return_aux_v", grad_call["keys"])
        # The aux v loss: the grad forward's v output (h = t - r, sampled guidance) against the target.
        t, r = self.time, self.time_r
        u = mock_u(losses["noisy"], t, t - r, self.drop, self.omega, self.start, self.end, self.z0,
                   model.scale.detach())
        aux = adaptive_loss((0.5 * u + 0.3 - losses["target"]).square().flatten(1).sum(1), 1.0, 0.01)
        assert_close(losses["loss_aux_v"], aux, atol=1e-5)
        assert_close(losses["loss"], losses["loss_full"] + losses["loss_base"] + losses["loss_aux_v"], atol=1e-5)

    def test_mf_fixed_guidance(self):
        model = MockDiT()
        transport = meanflow(variant="mf", cfg={"mode": "fixed", "omega": 2.5})
        spy = Spy(model)
        losses = transport.training_losses(spy, self.clean, self.z0, jvp_model=spy, time=self.time,
                                           time_r=self.time_r, noise=self.noise, drop_mask=self.drop)
        z = losses["noisy"]
        v_t = self.noise - self.clean
        unconditional = mock_u(z, self.time, torch.zeros(6), torch.ones(6, dtype=torch.bool), torch.ones(6),
                               torch.zeros(6), torch.ones(6), self.z0, model.scale.detach())
        guided = 2.5 * v_t + (1 - 2.5) * unconditional
        guided[self.drop] = v_t[self.drop]
        assert_close(losses["target"], guided, atol=1e-5)
        assert_close(losses["tangent"], guided, atol=1e-5)
        self.assertTrue(torch.equal(spy.calls[0]["mask"], torch.ones(6, dtype=torch.bool)))
        self.assertTrue(torch.equal(spy.calls[-1]["mask"], self.drop))


class GuidedGradientTest(unittest.TestCase):
    def test_every_parameter_gets_a_gradient(self):
        for name, dit_flags, transport_flags in (
            ("imf+aux", {"aux_v_head_depth": 1}, {"tangent": "aux_head"}),
            ("imf+cfg", {"guidance_conditioning": True}, {"cfg": {"mode": "conditioned"}}),
            ("imf+cfg+aux", {"guidance_conditioning": True, "aux_v_head_depth": 2},
             {"tangent": "aux_head", "cfg": {"mode": "conditioned"}}),
            ("mf+fixed", {}, {"variant": "mf", "cfg": {"mode": "fixed"}}),
        ):
            dit = meanflow_dit(gradient_checkpointing=True, **dit_flags)
            dit.train()
            wrapper = Spy(dit)
            clean, z0 = toy_batch(6)
            with no_internal_dropout():
                losses = meanflow(**transport_flags).training_losses(
                    wrapper, clean, z0, jvp_model=dit, generator=torch.Generator().manual_seed(3))
            losses["loss"].backward()
            with self.subTest(name=name):
                self.assertEqual(len(wrapper.calls), 1)
                missing = [n for n, p in dit.named_parameters() if p.requires_grad and p.grad is None]
                self.assertEqual(missing, [])
                if "aux" in name:
                    self.assertGreater(losses["loss_aux_v"].item(), 0)
                    self.assertGreater(dit.aux_v_head.head.weight.grad.abs().sum().item(), 0)
                if "cfg" in name:
                    self.assertGreater(dit.guidance_scale_embedding[2].weight.grad.abs().sum().item(), 0)


# ---------------------------------------------------------------- step 4: the sampler


class MeanFlowSamplerTest(unittest.TestCase):
    """Test 10."""

    def setUp(self):
        self.dit = meanflow_dit(guidance_conditioning=True)
        self.dit.eval()
        self.noise, self.z0 = toy_batch(2, seed=5)
        self.keep = torch.zeros(2, dtype=torch.bool)
        self.kwargs = {"frame_features": self.z0, "condition_drop_mask": self.keep}
        self.unconditional = {"frame_features": self.z0, "condition_drop_mask": torch.ones(2, dtype=torch.bool)}

    def test_one_and_n_steps(self):
        dit = meanflow_dit()
        dit.eval()
        transport = meanflow(variant="mf")
        one = transport.euler_sample(dit, self.noise, model_kwargs=self.kwargs)
        with torch.no_grad():
            expected = self.noise - dit(self.noise, torch.ones(2), self.z0, time_delta=torch.ones(2),
                                        condition_drop_mask=self.keep)
        self.assertTrue(torch.equal(one, expected))
        for grid_kind in ("uniform", "shifted"):
            sample = transport.euler_sample(dit, self.noise, model_kwargs=self.kwargs, num_steps=3, time_grid=grid_kind,
                                            return_trajectory=True)
            grid = torch.linspace(1, 0, 4)
            if grid_kind == "shifted":
                grid = transport.shift_time(grid)
            value = self.noise
            with torch.no_grad():
                for t, r in zip(grid[:-1], grid[1:]):
                    u = dit(value, t.expand(2), self.z0, time_delta=(t - r).expand(2), condition_drop_mask=self.keep)
                    value = value - (t - r) * u
            with self.subTest(grid=grid_kind):
                self.assertEqual(tuple(sample.shape), (4, 2, 4, 16, 48))
                self.assertTrue(torch.equal(sample[-1], value))
        with self.assertRaisesRegex(ValueError, "time_grid"):
            transport.euler_sample(dit, self.noise, model_kwargs=self.kwargs, time_grid="cosine")
        # Internal guidance as in flow matching, from the same single forward.
        spy = Spy(dit)
        guided = transport.euler_sample(spy, self.noise, model_kwargs=self.kwargs, ig_scale=1.5)
        self.assertEqual(len(spy.calls), 1)
        self.assertIn("return_base", spy.calls[0]["keys"])
        with torch.no_grad():
            full, base = dit(self.noise, torch.ones(2), self.z0, time_delta=torch.ones(2),
                             condition_drop_mask=self.keep, return_base=True)
        assert_close(guided, self.noise - (base + 1.5 * (full - base)), atol=1e-6)

    def test_conditioned_guidance_is_a_network_input(self):
        transport = meanflow(cfg={"mode": "conditioned"})
        spy = Spy(self.dit)
        sample = transport.euler_sample(spy, self.noise, model_kwargs=self.kwargs, unconditional_kwargs=self.unconditional,
                                        num_steps=2, cfg_scale=3.0, guidance_interval=(0.2, 0.7))
        self.assertEqual(len(spy.calls), 2)  # one forward per step, no unconditional branch
        for call in spy.calls:
            self.assertEqual(call["omega"], 3.0)
            self.assertTrue(torch.equal(call["mask"], self.keep))
            self.assertIn("guidance_interval", call["keys"])
        value = self.noise
        with torch.no_grad():
            for t, r in ((1.0, 0.5), (0.5, 0.0)):
                u = self.dit(value, torch.full((2,), t), self.z0, time_delta=torch.full((2,), t - r),
                             condition_drop_mask=self.keep, guidance_scale=3.0, guidance_interval=(0.2, 0.7))
                value = value - (t - r) * u
        self.assertTrue(torch.equal(sample, value))
        # Per-sample omega; the default interval is [0, 1].
        spy.calls.clear()
        transport.euler_sample(spy, self.noise, model_kwargs=self.kwargs, cfg_scale=torch.tensor([1.0, 4.0]))
        self.assertTrue(torch.equal(spy.calls[0]["omega"], torch.tensor([1.0, 4.0])))
        with self.assertRaisesRegex(ValueError, "guidance_interval"):
            transport.euler_sample(self.dit, self.noise, model_kwargs=self.kwargs, guidance_interval=(0.8, 0.2))

    def test_fixed_and_none_modes(self):
        dit = meanflow_dit()
        dit.eval()
        fixed = meanflow(variant="mf", cfg={"mode": "fixed"})
        with self.assertRaisesRegex(ValueError, "cfg_scale 1"):
            fixed.euler_sample(dit, self.noise, model_kwargs=self.kwargs, unconditional_kwargs=self.unconditional,
                               cfg_scale=2.0)
        with self.assertRaisesRegex(ValueError, "conditioned only"):
            fixed.euler_sample(dit, self.noise, model_kwargs=self.kwargs, guidance_interval=(0.1, 0.9))
        fixed.euler_sample(dit, self.noise, model_kwargs=self.kwargs, guidance_interval=(0.0, 1.0))  # the default
        plain = meanflow(variant="imf")
        spy = Spy(dit)
        MeanFlowTransport._warned_naive_cfg = False
        with self.assertWarnsRegex(UserWarning, "heuristic"):
            sample = plain.euler_sample(spy, self.noise, model_kwargs=self.kwargs,
                                        unconditional_kwargs=self.unconditional, num_steps=2, cfg_scale=2.0)
        self.assertEqual(len(spy.calls), 4)  # two forwards per step
        self.assertEqual([bool(call["mask"].all()) for call in spy.calls], [False, True, False, True])
        value = self.noise
        with torch.no_grad():
            for t, r in ((1.0, 0.5), (0.5, 0.0)):
                arguments = dict(time_delta=torch.full((2,), t - r))
                conditional = dit(value, torch.full((2,), t), self.z0, condition_drop_mask=self.keep, **arguments)
                unconditional = dit(value, torch.full((2,), t), self.z0, condition_drop_mask=~self.keep, **arguments)
                value = value - (t - r) * (unconditional + 2.0 * (conditional - unconditional))
        assert_close(sample, value, atol=1e-6)
        with unittest.mock.patch("warnings.warn") as warn:  # warned once per process
            plain.euler_sample(dit, self.noise, model_kwargs=self.kwargs, unconditional_kwargs=self.unconditional,
                               cfg_scale=2.0)
        warn.assert_not_called()

    def test_predict_future_tokens_passes_meanflow_options(self):
        from vrae.training.common.latent_norm import LatentNormalizer
        from vrae.training.nanovvm_video_pred import sample as nv_sample

        normalizer = LatentNormalizer(torch.zeros(48), torch.ones(48))
        recorded = []

        class Recorder:
            def euler_sample(self, model, noise, **kwargs):
                recorded.append(kwargs)
                return noise

        arguments = dict(sample_indices=[0, 1], base_seed=1, steps=1, transport=Recorder())
        nv_sample.predict_future_tokens(self.dit, self.z0, normalizer, **arguments)
        self.assertNotIn("guidance_interval", recorded[-1])  # flow-matching calls are unchanged
        self.assertNotIn("time_grid", recorded[-1])
        nv_sample.predict_future_tokens(self.dit, self.z0, normalizer, guidance_interval=[0.1, 0.8],
                                        time_grid="shifted", cfg_scale=2.0, **arguments)
        self.assertEqual((recorded[-1]["guidance_interval"], recorded[-1]["time_grid"]), ((0.1, 0.8), "shifted"))
        # The CLI options override sampling.* of the config.
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--steps", type=int, default=None)
        parser.add_argument("--cfg-scale", type=float, default=None)
        nv_sample.add_meanflow_sampling_arguments(parser)
        config = {"sampling": {"steps": 1, "cfg_scale": 2.0, "guidance_interval": [0.0, 1.0], "time_grid": "uniform"}}
        settings = nv_sample._sampling_settings(config, parser.parse_args([]))
        self.assertEqual((settings["guidance_interval"], settings["time_grid"]), ([0.0, 1.0], "uniform"))
        settings = nv_sample._sampling_settings(
            config, parser.parse_args(["--guidance-interval", "0.2", "0.6", "--time-grid", "shifted"]))
        self.assertEqual((settings["guidance_interval"], settings["time_grid"]), ([0.2, 0.6], "shifted"))
        flow = nv_sample._sampling_settings({"sampling": {"steps": 100}}, parser.parse_args([]))
        self.assertEqual((flow["guidance_interval"], flow["time_grid"]), (None, None))


# ---------------------------------------------------------------- step 5: engine, train, metadata


def to_meanflow(config: dict, *, variant: str = "imf", cfg_mode: str = "none", tangent: str | None = None) -> dict:
    """A flow-matching nanovvm_video_pred config switched to MeanFlow (the meanflow/_*.yaml settings)."""

    config = copy.deepcopy(config)
    tangent = tangent or ("conditional" if variant == "mf" else "boundary")
    config["transport"] = {
        "type": "meanflow", "variant": variant, "prediction": "u", "time_dist_type": "logit-normal_0_1",
        "time_dist_shift": config["transport"]["time_dist_shift"], "tr_sampler": "max_min",
        "r_time_dist_type": None, "r_time_dist_shift": None, "ratio_r_neq_t": 0.25 if variant == "mf" else 0.5,
        "adaptive_weight": {"p": 1.0, "eps": 0.01}, "tangent": tangent, "base_model_coeff": 1.0,
        "aux_v_coeff": 1.0,
        "cfg": {"mode": cfg_mode, "omega": 2.0, "omega_max": 8.0, "beta": 1.0, "interval": True,
                "dropped_tangent": "unconditional"},
        "jvp": {"dtype": "float32", "attention_backend": "math", "neq_subset_only": True},
    }
    config["dit"].update(time_delta_conditioning=True, guidance_conditioning=cfg_mode == "conditioned",
                         aux_v_head_depth=1 if tangent == "aux_head" else 0)
    config["sampling"].update(steps=1, cfg_scale=2.0 if cfg_mode == "conditioned" else 1.0,
                              internal_guidance_scale=1.0, guidance_interval=[0.0, 1.0], time_grid="uniform")
    return config


def free_port() -> int:
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def _ddp_worker(rank: int, world: int, port: int, output: str) -> None:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from vrae.training.nanovvm_video_pred.train import check_gradient_sync

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        torch.set_num_threads(2)
        dit = meanflow_dit(guidance_conditioning=True, aux_v_head_depth=1, gradient_checkpointing=True)
        dit.train()
        model = DistributedDataParallel(dit, broadcast_buffers=False)
        transport = meanflow(tangent="aux_head", cfg={"mode": "conditioned"})
        for step in range(2):  # a second step catches unused parameters (DDP reduction errors)
            dit.zero_grad(set_to_none=True)
            clean, z0 = toy_batch(4, seed=10 * step + rank)  # different data on every rank
            losses = transport.training_losses(model, clean, z0, jvp_model=dit,
                                               generator=torch.Generator().manual_seed(100 * step + rank))
            losses["loss"].backward()
            names = check_gradient_sync(dit)
        flat = torch.cat([p.grad.flatten() for p in dit.parameters()])
        gathered = [torch.zeros_like(flat) for _ in range(world)]
        dist.all_gather(gathered, flat)
        # A rank whose gradients were not synchronized is caught.
        largest = max(dit.parameters(), key=lambda p: p.numel())
        if rank == 1:
            largest.grad.mul_(2)
        try:
            check_gradient_sync(dit)
            caught = False
        except RuntimeError:
            caught = True
        if rank == 0:
            torch.save({"equal": all(torch.equal(gathered[0], g) for g in gathered), "names": names,
                        "caught": caught, "loss": losses["loss"].item()}, output)
    finally:
        dist.destroy_process_group()


class EngineAndValidationTest(unittest.TestCase):
    """Test 11 (code side; the YAML files are tested in LadderMeanFlowConfigTest)."""

    @classmethod
    def setUpClass(cls):
        cls.flow = load_config(wan_ladder_path("tiny"))

    def test_build_and_metadata(self):
        from vrae.training.common.engine import build_flow_transport
        from vrae.training.nanovvm_video_pred import train as nv_train

        self.assertIsInstance(build_flow_transport(self.flow), FlowMatchingTransport)
        explicit = copy.deepcopy(self.flow)
        explicit["transport"]["type"] = "flow_matching"
        self.assertIsInstance(build_flow_transport(explicit), FlowMatchingTransport)
        self.assertEqual(flow_transport_metadata(explicit), flow_transport_metadata(self.flow))
        broken = copy.deepcopy(self.flow)
        broken["transport"]["type"] = "rectified"
        with self.assertRaisesRegex(ValueError, "transport.type"):
            build_flow_transport(broken)
        for variant, cfg_mode, tangent in (("mf", "none", None), ("mf", "fixed", None), ("imf", "none", None),
                                           ("imf", "conditioned", "aux_head")):
            config = to_meanflow(self.flow, variant=variant, cfg_mode=cfg_mode, tangent=tangent)
            if cfg_mode == "fixed":
                config["sampling"]["cfg_scale"] = 1.0
            transport = build_flow_transport(config)
            self.assertIsInstance(transport, MeanFlowTransport)
            metadata = flow_transport_metadata(config)
            with self.subTest(variant=variant, cfg_mode=cfg_mode):
                self.assertEqual(
                    set(metadata),
                    {"type", "variant", "prediction", "time_dist_type", "time_dist_shift", "tr_sampler",
                     "r_time_dist_type", "r_time_dist_shift", "ratio_r_neq_t", "adaptive_weight", "tangent",
                     "base_model_coeff", "aux_v_coeff", "cfg"},
                )
                self.assertNotIn("jvp", json.dumps(metadata))
                self.assertEqual(metadata["cfg"]["mode"], cfg_mode)
                summary = nv_train.validate_build(config)
                self.assertEqual(summary["transport"], metadata)

    def test_validation(self):
        from vrae.training.nanovvm_video_pred import train as nv_train

        base = to_meanflow(self.flow)
        cfg = to_meanflow(self.flow, cfg_mode="conditioned")
        aux = to_meanflow(self.flow, tangent="aux_head")
        fixed = to_meanflow(self.flow, variant="mf", cfg_mode="fixed")
        fixed["sampling"]["cfg_scale"] = 1.0
        for config in (base, cfg, aux, fixed):
            nv_train.validate_build(config)
        for source, mutate, match in (
            (base, lambda c: c["transport"].update(prediction="x"), "prediction"),
            (base, lambda c: c["dit"].pop("time_delta_conditioning"), "time_delta_conditioning"),
            (base, lambda c: c["dit"].update(guidance_conditioning=True), "guidance_conditioning"),
            (cfg, lambda c: c["dit"].update(guidance_conditioning=False), "guidance_conditioning"),
            (base, lambda c: c["dit"].update(aux_v_head_depth=1), "aux_v_head_depth"),
            (aux, lambda c: c["dit"].update(aux_v_head_depth=0), "aux_v_head_depth"),
            (aux, lambda c: c["dit"].update(aux_v_head_depth=4), "aux_v_head_depth"),  # tiny: 4 layers, base 1
            (cfg, lambda c: c["dit"].update(frame_dropout=0.0), "frame_dropout"),
            (fixed, lambda c: c["sampling"].update(cfg_scale=2.0), "cfg_scale"),
            (base, lambda c: c["sampling"].update(steps=0), "sampling.steps"),
            (cfg, lambda c: c["sampling"].update(guidance_interval=[0.7, 0.2]), "guidance_interval"),
            (cfg, lambda c: c["sampling"].update(guidance_interval=[0.0, 1.5]), "guidance_interval"),
            (base, lambda c: c["sampling"].update(guidance_interval=[0.1, 0.9]), "conditioned only"),
            (base, lambda c: c["sampling"].update(time_grid="cosine"), "time_grid"),
            (base, lambda c: c["dit"].update(name="vrae_first_frame_prediction_dit"), "wan_ti2v_dit"),
            (base, lambda c: c["transport"].update(time_dist_shift=2.0), "time_dist_shift"),
            (base, lambda c: c["transport"].update(t_eps=0.05), "t_eps"),
            (base, lambda c: c["transport"].update(variant="mf"), "tangent"),
            (base, lambda c: c["transport"]["cfg"].update(mode="fixed"), "cfg.mode"),
            (base, lambda c: c["training"].update(grad_sync_check_steps=-1), "grad_sync_check_steps"),
            # Flow-matching configs reject the MeanFlow-only keys.
            (self.flow, lambda c: c["dit"].update(time_delta_conditioning=True), "MeanFlow setting"),
            (self.flow, lambda c: c["sampling"].update(time_grid="uniform"), "MeanFlow setting"),
        ):
            broken = copy.deepcopy(source)
            mutate(broken)
            with self.subTest(match=match), self.assertRaisesRegex(ValueError, match):
                nv_train.validate_build(broken)

    def test_metadata_and_resume_rules(self):
        """Test 12 (unit level; the end-to-end test resumes real checkpoints)."""

        from vrae.checkpoint import CheckpointError, compare_metadata
        from vrae.training.common.contracts import canonical_resume_config, compare_resolved_configs
        from vrae.training.nanovvm_video_pred import train as nv_train

        stage1 = {"type": "wan22_vae", "spatial_compression": 16, "temporal_compression": 4,
                  "latent_channels": 48, "latent_grid": [8, 8]}
        config = to_meanflow(self.flow, cfg_mode="conditioned", tangent="aux_head")
        with torch.device("meta"):
            dit = nv_train.build_nanovvm_dit(config, stage1)
        self.assertEqual((dit.time_delta_conditioning, dit.guidance_conditioning, dit.aux_v_head_depth),
                         (True, True, 1))
        metadata = nv_train.nanovvm_model_metadata(config, stage1, dit)
        self.assertEqual(metadata["transport"]["type"], "meanflow")
        self.assertEqual((metadata["dit_time_delta_conditioning"], metadata["dit_guidance_conditioning"],
                          metadata["dit_aux_v_head_depth"]), (True, True, 1))
        runtime = copy.deepcopy(config)
        runtime["transport"]["jvp"] = {"dtype": "autocast", "attention_backend": "default", "neq_subset_only": False}
        runtime["training"]["grad_sync_check_steps"] = 5
        compare_metadata(metadata, nv_train.nanovvm_model_metadata(runtime, stage1, dit), tuple(metadata))
        compare_resolved_configs(config, runtime)
        self.assertNotIn("jvp", canonical_resume_config(config)["transport"])
        changed = copy.deepcopy(config)
        changed["transport"]["ratio_r_neq_t"] = 0.25
        with self.assertRaisesRegex(CheckpointError, "ratio_r_neq_t"):
            compare_metadata(metadata, nv_train.nanovvm_model_metadata(changed, stage1, dit), tuple(metadata))
        with self.assertRaisesRegex(CheckpointError, "ratio_r_neq_t"):
            compare_resolved_configs(config, changed)
        # A flow-matching config resolves exactly as before (no transport.jvp to drop).
        self.assertEqual(canonical_resume_config(self.flow)["transport"], self.flow["transport"])


class TrainLoopPiecesTest(unittest.TestCase):
    def test_metrics_follow_the_transport_terms(self):
        from vrae.models.dit.meanflow import LOSS_TERMS
        from vrae.training.nanovvm_video_pred.train import TrainMetrics

        metrics = TrainMetrics(torch.device("cpu"), accumulation_steps=1, terms=LOSS_TERMS)
        metrics.add_microbatch({name: torch.tensor(float(index + 1)) for index, name in enumerate(LOSS_TERMS)})
        metrics.end_step(stepped=True, grad_norm=torch.tensor(3.0), grad_numel=9, clip_grad=1.0)
        values = metrics.log_values(clip_grad=1.0)
        self.assertEqual(
            {key: value for key, value in values.items() if not key.startswith("optimizer/")},
            {"loss/train": 1.0, "loss/train_full": 2.0, "loss/train_base": 3.0, "loss/train_aux_v": 4.0,
             "mse/train_full": 5.0, "mse/train_base": 6.0, "mse/train_fm": 7.0, "mse/train_mf": 8.0,
             "stats/dudt_rms": 9.0},
        )
        self.assertEqual(values["optimizer/grad_norm"], 3.0)  # the pre-clip norm
        self.assertEqual(TrainMetrics(torch.device("cpu"), 1).terms, ("loss", "loss_full", "loss_base"))
        with self.assertRaisesRegex(ValueError, "loss terms"):
            TrainMetrics(torch.device("cpu"), 1, terms=("loss", "something"))

    def test_gradient_sync_check_under_ddp(self):
        """Two gloo ranks, DDP(dit) with the JVP on the bare dit: gradients stay synchronized."""

        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / "result.pt")
            mp.spawn(_ddp_worker, args=(2, free_port(), output), nprocs=2, join=True)
            result = torch.load(output)
        self.assertTrue(result["equal"])
        self.assertTrue(result["caught"])
        self.assertEqual(len(result["names"]), 4)
        self.assertTrue(math.isfinite(result["loss"]))


class MeanFlowEndToEndTest(unittest.TestCase):
    """Test 13: tiny Wan2.2 VAE stand-in; stats, train, resume, 1- and 2-step sampling, FVD."""

    def run_variant(self, base: Path, paths, backend, config: dict, name: str) -> None:
        from vrae.checkpoint import CheckpointError
        from vrae.config import save_resolved_config
        from vrae.training.common.contracts import create_run_metadata
        from vrae.training.common.engine import build_flow_transport
        from vrae.training.common.wandb import bind_wandb_run_name
        from vrae.training.nanovvm_video_pred import sample
        from vrae.training.nanovvm_video_pred import train as nv_train

        config = copy.deepcopy(config)
        config["run_name"] = f"meanflow_{name}"
        config["wandb"] = bind_wandb_run_name(config["wandb"], config["run_name"])
        nv_train.validate_build(config)
        run = paths.training_run(config["task"], config["run_name"], create=True)
        for directory in ("checkpoints", "samples", "wandb"):
            (run / directory).mkdir()
        save_resolved_config(create_run_metadata(config), run / "run_metadata.yaml")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            nv_train.train(config, paths, run, max_steps=1)
        logged = [json.loads(line) for line in stdout.getvalue().splitlines() if line.startswith("{")]
        step_log = next(item for item in logged if "loss/train" in item)
        for key in ("loss/train", "loss/train_full", "loss/train_base", "loss/train_aux_v", "mse/train_full",
                    "mse/train_base", "mse/train_fm", "mse/train_mf", "stats/dudt_rms", "optimizer/grad_norm"):
            self.assertIn(key, step_log)
            self.assertTrue(math.isfinite(step_log[key]), key)
        checkpoint = run / "checkpoints" / "step-00000001.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = payload["model_metadata"]
        self.assertEqual(metadata["transport"], build_flow_transport(config).metadata())
        self.assertEqual(metadata["transport"]["type"], "meanflow")
        self.assertNotIn("jvp", metadata["transport"])
        self.assertTrue(metadata["dit_time_delta_conditioning"])
        self.assertIn("time_delta_embedding.2.weight", payload["model"])
        samples = torch.load(run / "samples" / "step-00000001.pt")
        self.assertEqual(tuple(samples["predicted_future"].shape), (1, 16, 3, 32, 32))

        # Resume: a different JVP precision / backend is accepted, a different ratio is not.
        resumed = copy.deepcopy(config)
        resumed["training"]["resume"] = str(checkpoint.relative_to(base))
        resumed["transport"]["jvp"] = {"dtype": "autocast", "attention_backend": "math", "neq_subset_only": False}
        rejected = copy.deepcopy(resumed)
        rejected["transport"]["ratio_r_neq_t"] = 0.75
        with self.assertRaisesRegex(CheckpointError, "ratio_r_neq_t"):
            nv_train.train(rejected, paths, run, max_steps=2)
        with contextlib.redirect_stdout(io.StringIO()):
            nv_train.train(resumed, paths, run, max_steps=2)
        self.assertTrue((run / "checkpoints" / "step-00000002.pt").is_file())

        # 1- and 2-step sampling from the checkpoint, deterministic per seed.
        stage1, dit, loaded = sample.load_sampling_stack(
            config, paths, run / "checkpoints" / "step-00000002.pt", torch.device("cpu"))
        transport = build_flow_transport(config)
        first = torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
        settings = dict(sample_indices=[0], base_seed=1, transport=transport,
                        cfg_scale=float(config["sampling"]["cfg_scale"]),
                        guidance_interval=config["sampling"]["guidance_interval"],
                        time_grid=config["sampling"]["time_grid"])
        videos = {}
        for steps in (1, 2):
            videos[steps] = sample.sample_from_first_frame(stage1, dit, loaded, first, steps=steps, **settings)
            self.assertEqual(tuple(videos[steps].shape), (1, 16, 3, 32, 32))
            self.assertTrue(torch.equal(videos[steps], sample.sample_from_first_frame(
                stage1, dit, loaded, first, steps=steps, **settings)))
        self.assertFalse(torch.equal(videos[1], videos[2]))
        return dit, loaded, transport

    def run_fvd(self, base: Path, backend, config: dict, name: str, dit, loaded, transport) -> None:
        """The FVD path: model generation through MeanFlowTransport plus the controls (real decord)."""

        from tests.test_nanovvm_fvd import ToyEncoder, fvd_data_root, fvd_section, write_encoder_files
        from vrae.evaluation.nanovvm_fvd import fvd as engine
        from vrae.paths import ProjectPaths
        from vrae.storage.backend import configure_storage
        from vrae.training.nanovvm_video_pred import fvd as task_fvd

        fvd_root = fvd_data_root(base / f"fvd_{name}")
        write_encoder_files(base)
        fvd_paths = ProjectPaths(project_root=base, datasets={"nanovvm": fvd_root})
        fvd_config_source = copy.deepcopy(config)
        fvd_config_source["fvd"] = fvd_section(image_size=32)
        fvd_config = task_fvd.resolve_fvd_config(fvd_config_source, fvd_paths)
        generated = []
        original = task_fvd.VRAEFuturePredictor.generate

        def record(self_, condition, indices):
            output = original(self_, condition, indices)
            generated.append(output.shape)
            return output

        try:
            with mock_patch(engine, "VideoEncoder", ToyEncoder), \
                    mock_patch(task_fvd.VRAEFuturePredictor, "generate", record):
                metrics = task_fvd.evaluate(
                    config=fvd_config_source, paths=fvd_paths, fvd_config=fvd_config, stage1=backend, dit=dit,
                    normalizer=loaded, transport=transport, autocast=contextlib.nullcontext, rank=0,
                    world_size=1, device=torch.device("cpu"), run_dir=base / f"fvd_run_{name}", step=2,
                    model_name="model",
                )
        finally:
            configure_storage(None)
        self.assertTrue(generated)
        self.assertTrue(any(key.endswith("/model") for key in metrics))
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_mf_and_imf(self):
        from tests.test_nanovvm_video_pred import fake_decord, fake_wandb_module, replaced_module

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with fake_decord(), replaced_module("wandb", fake_wandb_module()):
                trained = self.train_variants(base)
            # The FVD fixture writes real videos, so it runs after the toy decoder is gone.
            for name, (backend, config, dit, loaded, transport) in trained.items():
                with self.subTest(variant=name, part="fvd"):
                    self.run_fvd(base, backend, config, name, dit, loaded, transport)

    def train_variants(self, base: Path) -> dict:
        from tests.test_nanovvm_video_pred import make_data_root
        from tests.test_nanovvm_wan22 import load_tiny_backend, wan_task_config
        from vrae.paths import ProjectPaths
        from vrae.training.common.latent_norm import LatentNormalizer
        from vrae.training.nanovvm_video_pred import latent_stats, sample
        from vrae.training.nanovvm_video_pred import train as nv_train

        root = make_data_root(base)
        weights = base / "ckpts/pretrained/wan22_vae/Wan2.2_VAE.pth"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"not real weights")
        paths = ProjectPaths(project_root=base, datasets={"nanovvm": root})
        flow = wan_task_config()
        backend = load_tiny_backend(paths.checkpoint(flow["stage1"]["checkpoint"]),
                                    precision=flow["stage1"]["precision"])
        loader = unittest.mock.Mock(return_value=(backend, None))
        variants = {
            "mf": to_meanflow(flow, variant="mf"),
            "imf_cfg_aux": to_meanflow(flow, variant="imf", cfg_mode="conditioned", tangent="aux_head"),
        }
        trained = {}
        with (
            mock_patch(latent_stats, "load_frozen_latent_backend", loader),
            mock_patch(nv_train, "load_frozen_latent_backend", loader),
            mock_patch(sample, "load_frozen_latent_backend", loader),
        ):
            LatentNormalizer.load(latent_stats.compute_nanovvm_latent_stats(flow, paths))
            for name, config in variants.items():
                with self.subTest(variant=name):
                    trained[name] = (backend, config, *self.run_variant(base, paths, backend, config, name))
            with self.subTest(part="init_from_flow_matching"):
                self.init_from_flow_matching(base, paths, backend, flow)
        return trained

    def init_from_flow_matching(self, base: Path, paths, backend, flow: dict) -> None:
        """Step 7: a MeanFlow run starts from a flow-matching checkpoint (only the zero layers are new)."""

        from vrae.checkpoint import CheckpointError
        from vrae.config import save_resolved_config
        from vrae.training.common.contracts import create_run_metadata
        from vrae.training.common.wandb import bind_wandb_run_name
        from vrae.training.nanovvm_video_pred import sample
        from vrae.training.nanovvm_video_pred import train as nv_train

        def prepare(config: dict, name: str) -> tuple[dict, Path]:
            config = copy.deepcopy(config)
            config["run_name"] = name
            config["wandb"] = bind_wandb_run_name(config["wandb"], name)
            run = paths.training_run(config["task"], name, create=True)
            for directory in ("checkpoints", "samples", "wandb"):
                (run / directory).mkdir()
            save_resolved_config(create_run_metadata(config), run / "run_metadata.yaml")
            return config, run

        flow_config, flow_run = prepare(flow, "flow_source")
        with contextlib.redirect_stdout(io.StringIO()):
            nv_train.train(flow_config, paths, flow_run, max_steps=1)
        source = flow_run / "checkpoints" / "step-00000001.pt"
        _, flow_dit, normalizer = sample.load_sampling_stack(flow_config, paths, source, torch.device("cpu"),
                                                            use_ema=False)
        stage1 = backend.metadata()
        for cfg_mode in ("none", "conditioned"):
            config = to_meanflow(flow, cfg_mode=cfg_mode)
            config["training"]["init_from_flow_matching"] = str(source.relative_to(base))
            nv_train.validate_build(config)
            dit = nv_train.build_nanovvm_dit(config, stage1)
            metadata = nv_train.nanovvm_model_metadata(config, stage1, dit, normalizer)
            missing = nv_train.load_flow_matching_init(source, dit, expected_metadata=metadata)
            self.assertEqual({key.split(".")[0] for key in missing}, set(dit.condition_embeddings()))
            dit.eval()
            generator = torch.Generator().manual_seed(0)
            noisy = torch.randn(2, 4, 4, 48, generator=generator)
            z0 = torch.randn(2, 4, 48, generator=generator)
            extra = {"guidance_scale": torch.tensor([1.0, 4.0]), "guidance_interval": (0.2, 0.8)} \
                if cfg_mode == "conditioned" else {}
            with torch.no_grad():
                self.assertTrue(torch.equal(
                    dit(noisy, torch.tensor([0.3, 0.9]), z0, time_delta=torch.tensor([0.2, 0.6]), **extra),
                    flow_dit(noisy, torch.tensor([0.3, 0.9]), z0)))
        # Only the zero-initialized embeddings may be missing; the metadata must match otherwise.
        aux = to_meanflow(flow, tangent="aux_head")
        dit = nv_train.build_nanovvm_dit(aux, stage1)
        with self.assertRaisesRegex(ValueError, "aux_v_blocks"):
            nv_train.load_flow_matching_init(
                source, dit, expected_metadata=nv_train.nanovvm_model_metadata(aux, stage1, dit, normalizer))
        wider = to_meanflow(flow)
        wider["dit"].update(dim=128, num_heads=2, ffn_dim=192)
        dit = nv_train.build_nanovvm_dit(wider, stage1)
        with self.assertRaisesRegex(CheckpointError, "dit_dim"):
            nv_train.load_flow_matching_init(
                source, dit, expected_metadata=nv_train.nanovvm_model_metadata(wider, stage1, dit, normalizer))
        meanflow_source = paths.training_run("nanovvm_video_pred", "meanflow_mf") / "checkpoints" / "step-00000001.pt"
        dit = nv_train.build_nanovvm_dit(to_meanflow(flow), stage1)
        with self.assertRaisesRegex(ValueError, "not a flow-matching checkpoint"):
            nv_train.load_flow_matching_init(meanflow_source, dit, expected_metadata={})
        for mutate, match in (
            (lambda c: c["training"].update(resume="ckpts/x.pt"), "excludes"),
            (lambda c: c["training"].update(init_from="ckpts/x.pt"), "excludes"),
            (lambda c: c["training"].update(init_from_flow_matching="/abs/x.pt"), "ckpts/"),
        ):
            broken = to_meanflow(flow)
            broken["training"]["init_from_flow_matching"] = "ckpts/x.pt"
            mutate(broken)
            with self.assertRaisesRegex(ValueError, match):
                nv_train.validate_build(broken)
        flow_with_init = copy.deepcopy(flow)
        flow_with_init["training"]["init_from_flow_matching"] = "ckpts/x.pt"
        with self.assertRaisesRegex(ValueError, "transport.type: meanflow"):
            nv_train.validate_build(flow_with_init)
        # The trainer path: load, rebuild the EMA from the loaded weights, train.
        config = to_meanflow(flow)
        config["training"]["init_from_flow_matching"] = str(source.relative_to(base))
        config, run = prepare(config, "meanflow_from_flow")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            nv_train.train(config, paths, run, max_steps=1)
        self.assertIn("[init] flow-matching weights", stdout.getvalue())
        payload = torch.load(run / "checkpoints" / "step-00000001.pt", map_location="cpu", weights_only=False)
        flow_state = torch.load(source, map_location="cpu", weights_only=False)["model"]
        shadow = payload["ema"]["shadow"]
        # One EMA step away from the flow-matching weights (decay 0.9995), not from a random init.
        name = "blocks.0.self_attn.q.weight"
        self.assertLess((shadow[name] - flow_state[name]).abs().max().item(),
                        (payload["model"][name] - flow_state[name]).abs().max().item() + 1e-6)


# ---------------------------------------------------------------- step 6: the configs

LADDER_SIZES = ("tiny", "small", "medium", "large", "xl", "xxl")
MEANFLOW_VARIANTS = {  # name: (sizes, to_meanflow arguments, ratio, extra tags)
    "mf": (LADDER_SIZES, {"variant": "mf"}, 0.25, ["meanflow", "mf"]),
    "imf": (LADDER_SIZES, {"variant": "imf"}, 0.5, ["meanflow", "imf"]),
    "imfcfg": (("tiny", "small", "medium"), {"variant": "imf", "cfg_mode": "conditioned"}, 0.5,
               ["meanflow", "imf", "cfg"]),
    "fmctrl": (("tiny", "small", "medium"), {"variant": "mf"}, 0.0, ["meanflow", "fmctrl"]),
}


def meanflow_ladder_path(size: str, variant: str) -> Path:
    return (wan_ladder_path(size).parent
            / f"wan22_ti2v_mix5_{size}_{variant}_upred_interval3_cond1gen16_res128_bs32_lr2e-4_wd0_ep1.yaml")


class LadderMeanFlowConfigTest(unittest.TestCase):
    """Test 11 on the YAML files; tests/test_nanovvm_scaling_ladder.py::test_layout lists them."""

    @classmethod
    def setUpClass(cls):
        cls.configs = {(size, variant): load_config(meanflow_ladder_path(size, variant))
                       for variant, (sizes, _, _, _) in MEANFLOW_VARIANTS.items() for size in sizes}
        cls.xpred = {size: load_config(wan_ladder_path(size)) for size in LADDER_SIZES}

    def test_build_only_cli(self):
        from unittest import mock

        from vrae.training.nanovvm_video_pred import train as nv_train

        for (size, variant), config in self.configs.items():
            with self.subTest(size=size, variant=variant):
                argv = ["train", "--config", str(meanflow_ladder_path(size, variant)), "--build-only"]
                stdout = io.StringIO()
                with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
                    nv_train.main()
                summary = json.loads(stdout.getvalue())
                self.assertEqual(summary["transport"]["type"], "meanflow")
                self.assertEqual(summary["transport"]["time_dist_shift"], math.sqrt(3))
                self.assertEqual(summary["transport"]["ratio_r_neq_t"], MEANFLOW_VARIANTS[variant][2])

    def test_equal_to_the_xpred_config_except_the_meanflow_keys(self):
        for (size, variant), config in self.configs.items():
            _, arguments, ratio, tags = MEANFLOW_VARIANTS[variant]
            xpred = self.xpred[size]
            with self.subTest(size=size, variant=variant):
                self.assertEqual(config["run_name"], meanflow_ladder_path(size, variant).stem)
                self.assertEqual(config["wandb"]["tags"], xpred["wandb"]["tags"] + tags)
                transport = dict(config["transport"])
                self.assertIsNone(transport.pop("t_eps"))  # overrides _wan22_common.yaml's flow-matching t_eps
                expected = to_meanflow(xpred, **arguments)
                expected["transport"]["ratio_r_neq_t"] = ratio
                self.assertEqual(transport, expected["transport"])
                self.assertEqual(config["sampling"], expected["sampling"])
                self.assertEqual(config["dit"], expected["dit"])
                strip, reference = copy.deepcopy(config), copy.deepcopy(xpred)
                for item in (strip, reference):
                    for key in ("transport", "sampling", "run_name"):
                        item.pop(key)
                    item["wandb"].pop("tags")
                    for key in ("time_delta_conditioning", "guidance_conditioning", "aux_v_head_depth"):
                        item["dit"].pop(key, None)
                self.assertEqual(strip, reference)
        cfg = self.configs[("tiny", "imfcfg")]
        self.assertEqual((cfg["transport"]["cfg"]["mode"], cfg["dit"]["guidance_conditioning"],
                          cfg["sampling"]["cfg_scale"]), ("conditioned", True, 2.0))
        self.assertEqual(self.configs[("tiny", "mf")]["transport"]["tangent"], "conditional")
        self.assertEqual(self.configs[("tiny", "imf")]["transport"]["tangent"], "boundary")
        self.assertEqual(self.configs[("tiny", "fmctrl")]["transport"]["variant"], "mf")

    def test_parameter_counts(self):
        from tests.test_nanovvm_wan22 import WAN_LADDER, WAN_STAGE1
        from vrae.training.nanovvm_video_pred import train as nv_train

        for (size, variant), config in self.configs.items():
            dim = WAN_LADDER[size][1]
            embedding = 256 * dim + dim + dim * dim + dim
            embeddings = 4 if variant == "imfcfg" else 1
            with self.subTest(size=size, variant=variant), torch.device("meta"):
                model = nv_train.build_nanovvm_dit(config, WAN_STAGE1)
                self.assertEqual(sum(p.numel() for p in model.parameters()),
                                 WAN_LADDER[size][5] + embeddings * embedding)


def mock_patch(target, attribute, value):
    return unittest.mock.patch.object(target, attribute, value)


if __name__ == "__main__":
    unittest.main()
