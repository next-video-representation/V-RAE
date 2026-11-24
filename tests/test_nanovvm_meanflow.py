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
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from vrae.config import load_config  # noqa: E402
from vrae.models.dit.wan_ti2v_dit import WanTI2VDiT  # noqa: E402
from vrae.training.common.engine import flow_transport_metadata  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
