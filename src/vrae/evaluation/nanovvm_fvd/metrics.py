"""Shared video metric protocol: RGB clips, fixed encoder adapters, and FVD math.

Vendored from NanoVVM (eval/metrics.py, commit ca80730) with imports adapted, so that V-RAE's FVD
reads, crops and encodes clips exactly as NanoVVM's does. Keep the two copies in sync.
"""
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import random
import time
from numbers import Integral

import numpy as np
from PIL import Image
import scipy.linalg
import torch
import torch.distributed as dist

from vrae.training.nanovvm_video_pred.manifest import frame_indices

UPSTREAM_REVISION = "a1e037ab7cb087debd2221d14ae4a001ec054201"
I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
VIDEOMAE_URL = "https://huggingface.co/OpenGVLab/InternVideoMAE_models/resolve/706cc172d65ebd4dedbee3f9c0183a93df9fa125/mae-g/vit_g_hybrid_pt_1200e_ssv2_ft.pth"
VJEPA21_REVISION = "204698b45b3712590f06245fbfba32d3be539812"
VJEPA21_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
VJEPA21_SOURCE_URL = f"https://codeload.github.com/facebookresearch/vjepa2/tar.gz/{VJEPA21_REVISION}"
PROTOCOL_VERSION = 2
BASELINES = ("ae_reconstruction", "static", "constant_velocity_2frame")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_identity(path):
    """Identity of a file for cache keys: path, size and mtime; a remote-backed video (data/storage.py)
    reports its remote identity instead (URL, ETag, offset, size), without downloading it."""
    from vrae.storage.backend import active_storage
    remote = active_storage().identity(path)
    if remote is not None:
        return remote
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


_HASH_CACHE = {}


def weight_identity(path):
    info = file_identity(path)
    key = tuple(info.values())
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with Path(path).expanduser().open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        _HASH_CACHE[key] = digest.hexdigest()
    return {**info, "sha256": _HASH_CACHE[key]}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


@contextmanager
def preserve_rng_state():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def fvd_sync_timeout(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("fvd.sync_timeout_seconds must be a positive integer")
    return int(value)


def fvd_log(message):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    print(f"[FVD {time.strftime('%Y-%m-%d %H:%M:%S')} rank={rank}] {message}", flush=True)


@contextmanager
def fvd_control_group(timeout_seconds=7200):
    """Keep slow CPU evaluation phases off the training process group's watchdog."""
    timeout_seconds = fvd_sync_timeout(timeout_seconds)
    group = None
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        group = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_seconds))
    try:
        yield group
    finally:
        if group is not None:
            dist.destroy_process_group(group)


@contextmanager
def distributed_phase(name, group=None, log=False):
    """Every rank enters each phase; propagate Python errors before the next collective."""
    error = None
    started = time.monotonic()
    if log:
        fvd_log(f"{name}: starting")
    try:
        yield
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [error]
    if dist.is_available() and dist.is_initialized():
        errors = [None] * dist.get_world_size(group)
        if log:
            fvd_log(f"{name}: local work {'failed' if error else 'done'} after "
                    f"{time.monotonic() - started:.1f}s; synchronizing")
        try:
            dist.all_gather_object(errors, error, group=group)
        except Exception as exc:
            raise RuntimeError(f"FVD {name} synchronization failed on rank {dist.get_rank()} "
                               f"after {time.monotonic() - started:.1f}s: {exc}") from exc
    failures = [f"rank {rank}: {value}" for rank, value in enumerate(errors) if value]
    if failures:
        raise RuntimeError(f"FVD {name} failed: " + "; ".join(failures))
    if log:
        fvd_log(f"{name}: complete after {time.monotonic() - started:.1f}s")


def center_crop(frame, image_size):
    # Matches the preprocessing used by data_processing/tools/forward.py.
    pil = Image.fromarray(frame)
    while min(pil.size) >= 2 * image_size:
        pil = pil.resize(tuple(x // 2 for x in pil.size), resample=Image.Resampling.BOX)
    scale = image_size / min(pil.size)
    pil = pil.resize(tuple(round(x * scale) for x in pil.size), resample=Image.Resampling.BICUBIC)
    arr = np.asarray(pil)
    top, left = (arr.shape[0] - image_size) // 2, (arr.shape[1] - image_size) // 2
    return np.array(arr[top:top + image_size, left:left + image_size], copy=True)


def load_clip(path, num_frames, image_size, frame_interval=1, start=0):
    """Only decode selected source frames; return float32 [C,T,H,W] in [0,1]."""
    from decord import VideoReader, cpu
    from vrae.storage.backend import open_video
    with open_video(path) as local:  # a remote-backed video is leased from the node cache while decoding
        reader = VideoReader(local, ctx=cpu(0), num_threads=1)
        indices = frame_indices(len(reader), num_frames, frame_interval, start)
        frames = reader.get_batch(indices.tolist()).asnumpy()
        del reader
    frames = np.stack([center_crop(frame, image_size) for frame in frames])
    return torch.from_numpy(frames).permute(3, 0, 1, 2).float().div_(255)


def baseline_latents(z):
    """z is [T,C,H,W]; the two-frame baseline uses the second *sampled* frame."""
    if z.shape[0] < 2:
        raise ValueError("Baselines require at least two frames")
    k = torch.arange(z.shape[0], device=z.device, dtype=z.dtype).view(-1, 1, 1, 1)
    return {"ae_reconstruction": z,
            "static": z[:1].expand_as(z),
            "constant_velocity_2frame": z[:1] + k * (z[1:2] - z[:1])}


def encoder_configs(config):
    encoders = config.get("encoders")
    if encoders is None:
        encoders = {"i3d": {"checkpoint": config.get("i3d_ckpt_path",
                    str(Path.home() / ".cache/zst-project/i3d_torchscript.pt")),
                    "batch_size": config.get("fvd_batch_size", 16)}}
    if not encoders:
        raise ValueError("At least one FVD encoder must be configured")
    for name, params in encoders.items():
        if name not in {"i3d", "videomaev2", "vjepa2_1"}:
            raise ValueError(f"Unknown FVD encoder {name!r}")
        if not params.get("checkpoint") or not Path(params["checkpoint"]).expanduser().is_file():
            raise FileNotFoundError(f"Missing {name} checkpoint: {params.get('checkpoint')}. "
                                    "Prepare it with NanoVVM's data_processing/tools/prepare_fvd_models.py "
                                    "(or restore $S3/assets/nanovvm/models/fvd.tar and vjepa2_1.tar).")
        if name == "vjepa2_1":
            vjepa21_source_path(params["checkpoint"], params.get("source_path"))
        if int(params.get("batch_size", 1 if name == "vjepa2_1" else 2 if name == "videomaev2" else 16)) < 1:
            raise ValueError("Encoder batch_size must be positive")
    return encoders


def vjepa21_source_path(checkpoint, source_path=None):
    source = Path(source_path).expanduser() if source_path else Path(checkpoint).expanduser().parent / f"vjepa2-{VJEPA21_REVISION}"
    marker = source / ".fvd_source.json"
    if not marker.is_file() or not (source / "hubconf.py").is_file():
        raise FileNotFoundError(f"Missing pinned V-JEPA 2.1 source: {source}. Prepare it with NanoVVM's data_processing/tools/prepare_fvd_models.py --models vjepa2_1.")
    if json.loads(marker.read_text()).get("revision") != VJEPA21_REVISION:
        raise ValueError(f"V-JEPA 2.1 source revision mismatch: {source}")
    return source


class VideoEncoder:
    def __init__(self, name, checkpoint, device, source_path=None):
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Prepare the {name} weights before evaluation: {checkpoint}")
        self.name, self.device = name, torch.device(device)
        if name == "i3d":
            self.model = torch.jit.load(str(checkpoint), map_location=self.device)
            self.feature_dim = 400
        elif name == "videomaev2":
            from vrae.evaluation.nanovvm_fvd.vendor.videomaev2.utils import load_videomae_model
            self.model = load_videomae_model(self.device, str(checkpoint))
            self.feature_dim = self.model.num_features
        elif name == "vjepa2_1":
            source = vjepa21_source_path(checkpoint, source_path)
            self.model, predictor = torch.hub.load(str(source), "vjepa2_1_vit_base_384", source="local", pretrained=False)
            del predictor
            self.preprocessor = torch.hub.load(str(source), "vjepa2_preprocessor", source="local", crop_size=384)
            state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
            if not isinstance(state, dict) or "ema_encoder" not in state:
                raise ValueError("V-JEPA 2.1 checkpoint has no ema_encoder")
            weights = {key.removeprefix("module.").removeprefix("backbone."): value
                       for key, value in state["ema_encoder"].items()}
            self.model.load_state_dict(weights, strict=True)
            self.model = self.model.to(self.device)
            self.feature_dim = self.model.embed_dim
            del state, weights
        else:
            raise ValueError(f"Unknown encoder: {name}")
        self.model = self.model.float().eval()
        # ScriptModule does not implement Module.requires_grad_.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, clips):
        if clips.ndim != 5 or clips.shape[1] != 3:
            raise ValueError("Expected RGB [B,3,T,H,W] clips")
        # Metric precision does not inherit training autocast or TF32 settings.
        with torch.autocast(device_type=self.device.type, enabled=False):
            if self.name == "i3d":
                result = self.model(clips.to(self.device).float() * 255,
                                    rescale=True, resize=True, return_features=True)
            elif self.name == "videomaev2":
                from vrae.evaluation.nanovvm_fvd.vendor.videomaev2.utils import preprocess_videomae
                if clips.shape[2] != 16:
                    raise ValueError("The pinned VideoMAE-v2 protocol requires exactly 16 frames")
                videos = clips.permute(0, 2, 3, 4, 1).mul(255).round().clamp(0, 255).byte().cpu().numpy()
                inputs = preprocess_videomae(videos).to(self.device, dtype=torch.float32)
                result = self.model.forward_features(inputs)
            else:
                if clips.shape[2] != 16:
                    raise ValueError("The pinned V-JEPA 2.1 FVD protocol requires exactly 16 frames")
                videos = clips.permute(0, 2, 3, 4, 1).mul(255).round().clamp(0, 255).byte().cpu().numpy()
                inputs = torch.stack([self.preprocessor(list(video))[0] for video in videos]).to(self.device, dtype=torch.float32)
                tokens = self.model(inputs)
                result = tokens.mean(dim=1)
        result = result.detach().double().cpu().numpy()
        if result.shape != (len(clips), self.feature_dim) or not np.isfinite(result).all():
            raise ValueError(f"Invalid {self.name} features: {result.shape}")
        return result


class StreamingStats:
    def __init__(self, dim):
        self.n = 0
        self.sum = np.zeros(dim, np.float64)
        self.sumsq = np.zeros((dim, dim), np.float64)

    def update(self, features):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(self.sum) or not np.isfinite(features).all():
            raise ValueError("Invalid metric features")
        self.n += len(features)
        self.sum += features.sum(0)
        self.sumsq += features.T @ features

    def reduce(self, device):
        if dist.is_available() and dist.is_initialized():
            device = device if dist.get_backend() == "nccl" else torch.device("cpu")
            for name in ("sum", "sumsq"):
                tensor = torch.as_tensor(getattr(self, name), device=device)
                dist.all_reduce(tensor)
                setattr(self, name, tensor.cpu().numpy())
            count = torch.tensor(self.n, device=device, dtype=torch.int64)
            dist.all_reduce(count)
            self.n = int(count.item())
        return self

    @classmethod
    def combine(cls, parts):
        """Statistics of the union of the parts' populations (sums add)."""
        parts = list(parts)
        combined = cls(len(parts[0].sum))
        for part in parts:
            combined.n += part.n
            combined.sum = combined.sum + part.sum
            combined.sumsq = combined.sumsq + part.sumsq
        return combined

    def finalize(self):
        if self.n < 2:
            raise ValueError("FVD requires at least two videos")
        mean = self.sum / self.n
        covariance = self.sumsq / self.n - np.outer(mean, mean)
        return mean, (covariance + covariance.T) * 0.5


def frechet_distance(mean_r, cov_r, mean_g, cov_g):
    """Symmetric PSD formulation, including singular covariances when N < dimension."""
    for value in (mean_r, cov_r, mean_g, cov_g):
        if not np.isfinite(value).all():
            raise ValueError("FVD statistics must be finite")
    eigenvalues, eigenvectors = scipy.linalg.eigh((cov_r + cov_r.T) * 0.5)
    root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
    sandwich = root @ cov_g @ root
    product_eigenvalues = scipy.linalg.eigvalsh((sandwich + sandwich.T) * 0.5)
    score = (np.square(mean_r - mean_g).sum() + np.trace(cov_r) + np.trace(cov_g)
             - 2 * np.sqrt(np.maximum(product_eigenvalues, 0)).sum())
    if not np.isfinite(score):
        raise ValueError("Non-finite FVD")
    return float(max(score, 0))
