"""MeanFlow (MF) and improved MeanFlow (iMF) for nanovvm_video_pred with wan_ti2v_dit: CPU tests.

Covers the DiT's MeanFlow inputs (time_delta, guidance, aux v branch), MeanFlowTransport (t/r
sampling, JVP, losses, CFG, sampler), the engine / train / sample wiring, the configs and an end-to-end
run on the tiny Wan2.2 VAE stand-in of tests/test_nanovvm_wan22.py. Run from the repository root:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONPATH=src python -m unittest tests.test_nanovvm_meanflow -v
"""

from __future__ import annotations

import math
import sys
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


if __name__ == "__main__":
    unittest.main()
