"""NanoVVM's in-training FVD protocol for first-frame conditioned V-RAE prediction.

Ported from NanoVVM (eval/fvd.py, commit ca80730). Populations, clip loading and cropping
(metrics.load_clip), the mp4 round trip, the encoders (I3D, VideoMAE-v2, V-JEPA 2.1), the
statistics, the feature/score caches, the per-dataset split and the W&B keys
(``eval_fvd/<split>/<encoder>/<source>`` and ``eval_fvd_<dataset>/<split>/<encoder>/<source>``)
are NanoVVM's, so the real-video statistics, and therefore the FVDs, are comparable across the two
codebases. What differs is only how the compared videos are made:

* model: frame 0 of each clip (NanoVVM's FVD crop) is encoded by the frozen V-RAE as the condition,
  the DiT samples the 16 future frames (per-clip noise seeded by fvd.seed and the clip id, so a clip
  gets the same noise at every checkpoint and on any rank), and the V-RAE decoder renders them;
* ae_reconstruction: the 16 real future frames, encoded and decoded by the frozen V-RAE;
* static: the conditioning frame repeated 16 times, encoded and decoded by the frozen V-RAE
  (NanoVVM: the AE-decoded first frame repeated).

Every video is saved as 17 frames (the real conditioning frame first); with
drop_conditioning_frames: 1 only frames 1..16 are scored, as in NanoVVM. Paths in the config are
absolute here; vrae.training.nanovvm_video_pred.fvd resolves V-RAE's relative ones.
"""
from __future__ import annotations

import gc
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist

from vrae.evaluation.nanovvm_fvd.metrics import (
    PROTOCOL_VERSION,
    UPSTREAM_REVISION,
    VJEPA21_REVISION,
    StreamingStats,
    VideoEncoder,
    atomic_json,
    distributed_phase,
    encoder_configs,
    file_identity,
    fingerprint,
    frechet_distance,
    fvd_control_group,
    fvd_log,
    fvd_sync_timeout,
    load_clip,
    preserve_rng_state,
    vjepa21_source_path,
    weight_identity,
)
from vrae.storage.backend import active_storage, is_storage_error
from vrae.training.nanovvm_video_pred.manifest import (
    clip_span,
    frame_indices,
    load_clip_manifest,
    read_meta,
    validate_meta,
    validate_sources,
)

# V-RAE decodes 4-frame chunks, so only controls defined on whole clips are available
# (NanoVVM's constant_velocity_2frame extrapolates per-frame AE latents).
BASELINES = ("ae_reconstruction", "static")
START_MODES = ("zero", "random")
SPLIT_NAMES = ("train", "val", "test")
CONFIG_KEYS = {
    "interval", "frame_interval", "seed", "batch_size", "image_size", "sample_fps", "use_ema",
    "baselines", "drop_conditioning_frames", "start_mode", "mp4_quality", "splits", "encoders",
    "cache_dir", "sync_timeout_seconds", "num_videos",
}


def eval_options(config):
    """NanoVVM's optional protocol knobs (eval/fvd.py)."""
    config = dict(config)
    drop = config.get("drop_conditioning_frames", 0)
    if isinstance(drop, bool) or not isinstance(drop, int) or drop < 0:
        raise ValueError("fvd.drop_conditioning_frames must be a non-negative integer")
    start_mode = config.get("start_mode", "zero")
    if start_mode not in START_MODES:
        raise ValueError(f"fvd.start_mode must be one of {START_MODES}")
    quality = config.get("mp4_quality")
    if quality is not None and (isinstance(quality, bool) or not isinstance(quality, (int, float))
                                or not 0 <= quality <= 10):
        raise ValueError("fvd.mp4_quality must be in [0, 10]")
    return {"drop_conditioning_frames": drop, "start_mode": start_mode, "mp4_quality": quality}


def _protocol_options(options):
    values = {}
    if options["drop_conditioning_frames"]:
        values["drop_conditioning_frames"] = options["drop_conditioning_frames"]
    if options["start_mode"] != "zero":
        values["start_mode"] = options["start_mode"]
    if options["mp4_quality"] is not None:
        values["video_encoding"] = f"imageio_ffmpeg_quality{options['mp4_quality']:g}_mp4_round_uint8"
    return values


def _clip_start(video_id, length, span, protocol, seed):
    if protocol.get("start_mode", "zero") == "zero":
        return 0
    digest = hashlib.sha256(f"{seed}:{video_id}".encode()).hexdigest()
    return int(digest[:16], 16) % (length - span + 1)


def _scored_frames(protocol):
    return protocol["num_frames"] - protocol.get("drop_conditioning_frames", 0)


def split_configs(config):
    """{name: split} for name in train / val / test (NanoVVM's split forms)."""
    splits = config.get("splits")
    if not splits or set(splits) - set(SPLIT_NAMES):
        raise ValueError(f"fvd.splits must be named {', '.join(SPLIT_NAMES)}")
    return splits


def _split_num_videos(split, config):
    if split.get("sources") is not None:
        return split.get("num_videos", "all")
    return split.get("num_videos", config.get("num_videos", 512))


def validate_fvd_settings(config, *, frame_interval, num_frames):
    """File-free checks of an fvd section (shapes, names, values)."""
    if not isinstance(config, dict):
        raise ValueError("fvd must be a mapping")
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown fvd keys {sorted(unknown)}")
    interval = config.get("interval")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError("fvd.interval must be a positive integer (optimizer steps)")
    fvd_sync_timeout(config.get("sync_timeout_seconds", 7200))
    if config.get("frame_interval", frame_interval) != frame_interval:
        raise ValueError("fvd.frame_interval must match the training frame interval")
    options = eval_options(config)
    if options["drop_conditioning_frames"] != 1:
        raise ValueError("fvd.drop_conditioning_frames must be 1: the model is given frame 0 and "
                         "generates frames 1..16, which are scored")
    if num_frames - options["drop_conditioning_frames"] != 16:
        raise ValueError("FVD scores exactly 16 frames (VideoMAE-v2 and V-JEPA 2.1 need 16)")
    if not isinstance(config.get("use_ema", False), bool):
        raise ValueError("fvd.use_ema must be true or false")
    for name, split in split_configs(config).items():
        if not isinstance(split, dict):
            raise ValueError(f"fvd.splits.{name} must be a mapping")
        if (split.get("sources") is None) == (split.get("raw_video_root") is None):
            raise ValueError(f"fvd.splits.{name}: set exactly one of sources or raw_video_root")
        if split.get("sources") is not None:
            validate_sources(split["sources"], f"fvd.splits.{name}.sources")
        count = _split_num_videos(split, config)
        if count != "all" and (isinstance(count, bool) or not isinstance(count, int) or count < 2):
            raise ValueError("num_videos must be an integer >=2 or 'all'")
    baselines = config.get("baselines", list(BASELINES))
    if len(set(baselines)) != len(baselines) or set(baselines) - set(BASELINES):
        raise ValueError(f"fvd.baselines must be a subset of {BASELINES}, got {baselines}")
    for key in ("batch_size", "sample_fps", "image_size"):
        if key in config and (isinstance(config[key], bool) or int(config[key]) < 1):
            raise ValueError(f"fvd.{key} must be positive")
    encoders = config.get("encoders")
    if not isinstance(encoders, dict) or not encoders:
        raise ValueError("fvd.encoders must name at least one of i3d, videomaev2, vjepa2_1")
    for name, params in encoders.items():
        if name not in {"i3d", "videomaev2", "vjepa2_1"}:
            raise ValueError(f"Unknown FVD encoder {name!r}")
        if not isinstance(params, dict) or not params.get("checkpoint"):
            raise ValueError(f"fvd.encoders.{name}.checkpoint is required")
    return config


def validate_fvd_config(config, frame_interval, num_frames):
    """validate_fvd_settings plus the files: manifests, meta.yaml, video roots, encoder weights."""
    config = validate_fvd_settings(dict(config), frame_interval=frame_interval, num_frames=num_frames)
    for name, split in split_configs(config).items():
        if split.get("sources") is not None:
            for source in split["sources"]:
                # A remote-backed root (vrae.storage) needs no local video tree; manifests are local.
                if not active_storage().is_dir(source["root"]):
                    raise FileNotFoundError(f"Missing {name} source {source['name']} root: {source['root']}")
                if not Path(source["manifest"]).expanduser().is_file():
                    raise FileNotFoundError(f"Missing {name} source {source['name']} manifest: {source['manifest']}")
                validate_meta(read_meta(source["manifest_meta"]), rgb_seq_len=num_frames,
                              frame_interval=frame_interval, source=source["manifest_meta"])
        elif not active_storage().is_dir(split["raw_video_root"]):
            raise FileNotFoundError(f"Missing {name} raw_video_root: {split['raw_video_root']}")
    encoder_configs(config)
    return config


def _broadcast(value, rank, group=None):
    if dist.is_available() and dist.is_initialized():
        box = [value if rank == 0 else None]
        dist.broadcast_object_list(box, src=0, group=group)
        return box[0]
    return value


def _progress(items, name):
    started = last_report = time.monotonic()
    count = 0
    fvd_log(f"{name}: starting")
    for item in items:
        now = time.monotonic()
        if now - last_report >= 30:
            fvd_log(f"{name}: processed={count}, elapsed={now - started:.1f}s")
            last_report = now
        yield item
        count += 1
    fvd_log(f"{name}: processed={count}, complete after {time.monotonic() - started:.1f}s")


def _population(split, protocol, num_videos, seed, cache_root):
    """Rank zero: one deterministic clip per video of a raw video root (NanoVVM's raw split)."""
    import os

    from decord import VideoReader, cpu
    storage = active_storage()
    if storage.root_for(split["raw_video_root"]) is not None:
        raw_root = Path(os.path.normpath(os.path.abspath(os.path.expanduser(split["raw_video_root"]))))
        paths = list(_progress(storage.list_videos(raw_root), f"population inventory {raw_root}"))
    else:
        raw_root = Path(split["raw_video_root"]).expanduser().resolve()
        paths = sorted(_progress(raw_root.rglob("*"), f"population inventory {raw_root}"))
    candidates = []
    for video in _progress(paths, "population file identities"):
        if video.suffix.lower() not in {".avi", ".mp4", ".mov", ".mkv"}:
            continue
        relative = video.relative_to(raw_root)
        candidates.append({"video_id": relative.as_posix(), "avi_path": str(video),
                           "raw_identity": file_identity(video), "class_idx": 0})
    signature = fingerprint({"candidates": candidates, "protocol": protocol, "count": num_videos, "seed": seed})
    path = cache_root / "populations" / f"{signature}.json"
    if path.is_file():
        population = json.loads(path.read_text())
        fvd_log(f"population cache hit: candidates={len(candidates)}, "
                f"selected={len(population['entries'])}, path={path}")
        return population
    fvd_log(f"population cache miss: candidates={len(candidates)}, path={path}")
    eligible, rejected = [], []
    span = clip_span(protocol["num_frames"], protocol["frame_interval"])
    with storage.prefetch_along([entry["avi_path"] for entry in candidates]) as advance:
        for position, entry in enumerate(_progress(candidates, "population length checks")):
            advance(position)
            try:
                with storage.open(entry["avi_path"]) as local:
                    length = len(VideoReader(local, ctx=cpu(0), num_threads=1))
                if length < span:
                    rejected.append({"video_id": entry["video_id"], "reason": "too_short", "frames": length})
                    continue
                eligible.append({**entry, "source_num_frames": length})
            except Exception as exc:
                if is_storage_error(exc):
                    raise  # a fetch failure must not silently shrink the population
                rejected.append({"video_id": entry["video_id"], "reason": str(exc)})
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(eligible))
    selected = []
    limit = len(eligible) if num_videos == "all" else min(num_videos, len(eligible))
    for index in _progress(order, "population clip validation"):
        entry = eligible[int(index)]
        start = _clip_start(entry["video_id"], entry["source_num_frames"], span, protocol, seed)
        try:
            load_clip(entry["avi_path"], protocol["num_frames"], protocol["image_size"], protocol["frame_interval"], start)
        except Exception as exc:
            if is_storage_error(exc):
                raise
            rejected.append({"video_id": entry["video_id"], "reason": str(exc)})
            continue
        entry["frame_indices"] = frame_indices(entry["source_num_frames"], protocol["num_frames"],
                                               protocol["frame_interval"], start).tolist()
        selected.append(entry)
        if len(selected) == limit:
            break
    selected.sort(key=lambda entry: entry["video_id"])
    if len(selected) < 2:
        raise ValueError("Fewer than two decodable, sufficiently long videos in split")
    population = {"signature": signature, "entries": selected, "rejected": rejected,
                  "num_candidates": len(candidates), "num_selected": len(selected),
                  "requested": num_videos, "seed": seed, "protocol": protocol}
    atomic_json(path, population)
    return population


def _manifest_population(split, protocol, num_videos, seed, cache_root):
    """Rank zero: the clips listed by the split's manifests, each tagged with its dataset."""
    candidates = []
    for source in split["sources"]:
        clips = load_clip_manifest(source["manifest"], source["manifest_meta"],
                                   rgb_seq_len=protocol["num_frames"], frame_interval=protocol["frame_interval"])
        root = Path(source["root"]).expanduser().resolve()
        identities = {}
        for index in range(len(clips)):
            record = clips.record(index)
            path = root / record["video_path"]
            if path not in identities:
                identities[path] = file_identity(path)
            candidates.append({
                "video_id": f"{source['name']}/{record['video_path']}@{record['start_frame']}",
                "avi_path": str(path), "raw_identity": identities[path], "class_idx": record["label"],
                "dataset": source["dataset"], "source": source["name"],
                "frame_indices": frame_indices(record["end_frame_exclusive"], protocol["num_frames"],
                                               protocol["frame_interval"], record["start_frame"]).tolist()})
    signature = fingerprint({"candidates": candidates, "protocol": protocol, "count": num_videos, "seed": seed})
    path = cache_root / "populations" / f"{signature}.json"
    if path.is_file():
        population = json.loads(path.read_text())
        fvd_log(f"population cache hit: candidates={len(candidates)}, "
                f"selected={len(population['entries'])}, path={path}")
        return population
    fvd_log(f"population cache miss: candidates={len(candidates)}, path={path}")
    order = (range(len(candidates)) if num_videos == "all"
             else np.random.RandomState(seed).permutation(len(candidates)))
    limit = len(candidates) if num_videos == "all" else min(num_videos, len(candidates))
    selected, rejected = [], []
    order = [int(index) for index in order]
    with active_storage().prefetch_along([candidates[index]["avi_path"] for index in order[:limit]]) as advance:
        for position, index in enumerate(_progress(order, "population clip validation")):
            if position < limit:
                advance(position)
            entry = candidates[index]
            try:
                load_clip(entry["avi_path"], protocol["num_frames"], protocol["image_size"],
                          protocol["frame_interval"], entry["frame_indices"][0])
            except Exception as exc:
                if is_storage_error(exc):
                    raise  # a fetch failure must not silently shrink the population
                rejected.append({"video_id": entry["video_id"], "reason": str(exc)})
                continue
            selected.append(entry)
            if len(selected) == limit:
                break
    selected.sort(key=lambda entry: entry["video_id"])
    if len(selected) < 2:
        raise ValueError("Fewer than two decodable clips in the split's manifests")
    population = {"signature": signature, "entries": selected, "rejected": rejected,
                  "num_candidates": len(candidates), "num_selected": len(selected),
                  "requested": num_videos, "seed": seed, "protocol": protocol}
    atomic_json(path, population)
    fvd_log(f"population cached: candidates={len(candidates)}, selected={len(selected)}, "
            f"rejected={len(rejected)}, path={path}")
    return population


def _save_video(frames, path, fps, quality=None):
    """[T,3,H,W] in [0,1] -> mp4 (NanoVVM's encoding)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.mp4")
    frames = frames.permute(0, 2, 3, 1).mul(255).round().clamp(0, 255).byte().cpu().numpy()
    options = {} if quality is None else {"quality": quality, "macro_block_size": 1}
    with imageio.get_writer(str(temporary), fps=fps, **options) as writer:
        for frame in frames:
            writer.append_data(frame)
    temporary.replace(path)


def sample_index(video_id: str) -> int:
    """Stable per-clip noise index: the same clip gets the same noise on any rank and checkpoint."""
    return int(fingerprint(video_id)[:12], 16)


@torch.no_grad()
def _materialize(entry, protocol, predictor, cache_root, baselines, mp4_quality=None):
    """(conditioning frame [3,H,W] in [0,1], {baseline: cached mp4 path})."""
    identity = fingerprint({"source": entry, "protocol": protocol})
    start = entry["frame_indices"][0]
    directory = cache_root / "clips" / identity
    paths = {name: directory / f"{name}.mp4" for name in baselines}
    missing = [name for name, path in paths.items() if not path.is_file()]
    real = load_clip(entry["avi_path"], protocol["num_frames"] if missing else 1,
                     protocol["image_size"], protocol["frame_interval"], start)
    frames = real.permute(1, 0, 2, 3)  # [T,3,H,W]
    condition = frames[0]
    for name in missing:
        future = frames[1:] if name == "ae_reconstruction" else condition[None].expand(
            protocol["num_frames"] - 1, -1, -1, -1)
        reconstructed = predictor.reconstruct_future(future[None])[0].float().clamp(0, 1).cpu()
        _save_video(torch.cat((condition[None], reconstructed)), paths[name], protocol["sample_fps"], mp4_quality)
    return condition, {name: str(path) for name, path in paths.items()}


def _cached_feature(path):
    return np.load(path, allow_pickle=False) if path.is_file() else None


def _save_feature(path, feature):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, feature, allow_pickle=False)
    temporary.replace(path)


def _encoder_stats(shard, protocol, name, params, cache_root, device, baselines, groups=(None,)):
    """{group: {source: StreamingStats}} over this rank's shard (NanoVVM's implementation)."""
    if name == "vjepa2_1":
        encoder = VideoEncoder(name, params["checkpoint"], device, source_path=params.get("source_path"))
    else:
        encoder = VideoEncoder(name, params["checkpoint"], device)
    sources = ("real", "model", *baselines)
    stats = {group: {source: StreamingStats(encoder.feature_dim) for source in sources} for group in groups}
    identity = {"name": name, "weights": weight_identity(params["checkpoint"]),
                "upstream": UPSTREAM_REVISION if name == "videomaev2" else VJEPA21_REVISION if name == "vjepa2_1" else "legacy-i3d",
                "precision": "fp32", "protocol": protocol}
    if name == "vjepa2_1":
        identity["source"] = file_identity(vjepa21_source_path(params["checkpoint"], params.get("source_path")) / "hubconf.py")
        identity["features"] = "final_norm_all_video_tokens_mean"
        identity["preprocessing"] = "official_eval_crop_384_imagenet_normalize_uint8"
    batch_size = int(params.get("batch_size", 1 if name == "vjepa2_1" else 2 if name == "videomaev2" else 16))
    try:
        for source in sources:
            pending = []

            def flush():
                if not pending:
                    return
                features = encoder(torch.stack([clip for clip, _, _ in pending]))  # noqa: F821 (del in finally)
                for group in dict.fromkeys(group for _, _, group in pending):
                    rows = [i for i, (_, _, member) in enumerate(pending) if member == group]
                    stats[group][source].update(features[rows])
                for feature, (_, path, _) in zip(features, pending):
                    if path is not None:
                        _save_feature(path, feature)
                pending.clear()

            for entry in shard:
                group = entry.get("dataset")
                key = fingerprint({"encoder": identity, "source": source,
                                   "raw": entry["raw_identity"],
                                   "indices": entry["frame_indices"]})
                cache_path = cache_root / "features" / name / f"{key}.npy" if source != "model" else None
                feature = _cached_feature(cache_path) if cache_path is not None else None
                if feature is not None:
                    stats[group][source].update(feature[None])
                    continue
                path = entry["avi_path"] if source == "real" else entry["gen_path"] if source == "model" else entry["baselines"][source]
                # A generated/control frame already spans frame_interval source frames.
                interval = protocol["frame_interval"] if source == "real" else 1
                drop = protocol.get("drop_conditioning_frames", 0)
                start = entry["frame_indices"][0] + drop * interval if source == "real" else drop
                clip = load_clip(path, _scored_frames(protocol), protocol["image_size"], interval, start)
                pending.append((clip, cache_path, group))
                if len(pending) == batch_size:
                    flush()
            flush()
        return stats, identity
    finally:
        del encoder
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


@torch.no_grad()
def run_fvd_evaluation(*, predictor, rank, world_size, device, run_dir, global_step, fvd_config,
                       num_frames=17, frame_interval=3, model_name="model"):
    """Return flat W&B metric keys on rank zero, None elsewhere.

    predictor: identity() -> dict (the frozen V-RAE, for cache keys); encode_condition(rgb [B,3,H,W])
    -> condition; generate(condition, sample_indices) -> future RGB [B,16,3,H,W] in [0,1];
    reconstruct_future(rgb [B,16,3,H,W]) -> RGB [B,16,3,H,W]. The caller's RNG and TF32 state are
    restored afterwards.
    """
    device = torch.device(device)
    metrics = {}
    generation_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    with preserve_rng_state(), fvd_control_group(
            dict(fvd_config).get("sync_timeout_seconds", 7200)) as control_group:
        phase = partial(distributed_phase, group=control_group, log=True)
        config = None
        with phase("configuration"):
            config = validate_fvd_config(fvd_config, frame_interval, num_frames)
            options = eval_options(config)
        baselines = tuple(config.get("baselines", BASELINES))
        cache_root = Path(config.get("cache_dir", Path(run_dir) / "fvd_cache"))
        output_root = Path(run_dir) / "fvd_eval" / f"step_{global_step:07d}"
        ae_info = None
        with phase("AE identity"):
            if rank == 0:
                ae_info = predictor.identity()
        ae_info = _broadcast(ae_info, rank, group=control_group)
        protocol = {"version": PROTOCOL_VERSION, "num_frames": num_frames, "frame_interval": frame_interval,
                    "frame_start": 0, "image_size": int(config.get("image_size", 256)),
                    "sample_fps": int(config.get("sample_fps", 8)), "ae": ae_info,
                    "spatial_preprocessing": "forward_center_crop_bicubic",
                    "video_encoding": "imageio_ffmpeg_default_mp4_round_uint8",
                    "covariance": "population",
                    "generator": "vrae_first_frame_prediction",
                    **_protocol_options(options)}
        seed = int(config.get("seed", 42))
        for split_name, split in split_configs(config).items():
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = generation_tf32
            population = None
            with phase(f"{split_name} population"):
                if rank == 0:
                    build = _manifest_population if split.get("sources") is not None else _population
                    population = build(split, protocol, _split_num_videos(split, config), seed, cache_root)
            population = _broadcast(population, rank, group=control_group)
            entries = population["entries"]
            datasets = sorted({entry["dataset"] for entry in entries if entry.get("dataset") is not None})
            dataset_counts = {name: sum(entry.get("dataset") == name for entry in entries) for name in datasets}
            shard = entries[rank::world_size]
            local_manifest = []
            batch_size = int(config.get("batch_size", 8))
            if rank == 0:
                print(f"[FVD] {split_name}: {len(entries)} videos, interval={frame_interval}, baselines={baselines}"
                      + (f", per dataset {dataset_counts}" if datasets else ""), flush=True)
            with phase(f"{split_name} generation and controls"), \
                    active_storage().prefetch_along([entry["avi_path"] for entry in shard]) as advance:
                for start in range(0, len(shard), batch_size):
                    advance(start + batch_size - 1)
                    batch = shard[start:start + batch_size]
                    conditions, controls = zip(*[_materialize(e, protocol, predictor, cache_root, baselines,
                                                              options["mp4_quality"]) for e in batch])
                    condition = torch.stack(conditions)
                    future = predictor.generate(predictor.encode_condition(condition.to(device)),
                                                [sample_index(e["video_id"]) for e in batch])
                    if tuple(future.shape) != (len(batch), num_frames - 1, 3, *condition.shape[-2:]):
                        raise ValueError(f"Unexpected generated shape {tuple(future.shape)}")
                    future = future.float().clamp(0, 1).cpu()
                    for entry, first, frames, baseline_paths in zip(batch, condition, future, controls):
                        path = output_root / split_name / "generated_videos" / (fingerprint(entry["video_id"])[:20] + ".mp4")
                        _save_video(torch.cat((first[None], frames)), path, protocol["sample_fps"],
                                    options["mp4_quality"])
                        drop = options["drop_conditioning_frames"]
                        local_manifest.append({**entry, "gen_path": str(path), "baselines": baseline_paths,
                                               "frame_interval": frame_interval,
                                               "frame_start": entry["frame_indices"][0] + drop * frame_interval,
                                               "gen_frame_start": drop, "num_frames": num_frames - drop,
                                               "conditioning_frames": 1})
            merged = [None] * world_size
            if dist.is_available() and dist.is_initialized():
                dist.all_gather_object(merged, local_manifest, group=control_group)
            else:
                merged = [local_manifest]
            manifest = sorted([entry for part in merged for entry in part], key=lambda e: e["video_id"])
            with phase(f"{split_name} manifest"):
                if rank == 0:
                    atomic_json(output_root / split_name / "manifest.json", manifest)
                    atomic_json(output_root / split_name / "population.json", population)
            scores, dataset_scores = {}, {name: {} for name in datasets}
            # Generation uses the model's autocast; feature extraction below is strictly FP32.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            for name, params in encoder_configs(config).items():
                result = None
                with phase(f"{split_name}/{name} feature extraction"):
                    result = _encoder_stats(local_manifest, protocol, name, params, cache_root, device, baselines,
                                            groups=tuple(datasets) or (None,))
                group_stats, encoder_identity = result
                for per_source in group_stats.values():
                    for statistic in per_source.values():
                        statistic.reduce(device)
                stats = {source: StreamingStats.combine([per_source[source] for per_source in group_stats.values()])
                         for source in next(iter(group_stats.values()))}
                with phase(f"{split_name}/{name} distance"):
                    if rank == 0:
                        for source, statistic in stats.items():
                            if statistic.n != len(entries):
                                raise ValueError(f"Population mismatch for {source}: {statistic.n} != {len(entries)}")

                        def distances(statistics, key_extra, prefix):
                            reference = statistics["real"].finalize()
                            control_key = fingerprint({"population": population["signature"],
                                                       "encoder": encoder_identity, "baselines": baselines,
                                                       **key_extra})
                            control_path = cache_root / "scores" / f"{control_key}.json"
                            cached = json.loads(control_path.read_text()) if control_path.is_file() else {}
                            values = {}
                            for source in ("model", *baselines):
                                value = (cached[source] if source in cached
                                         else frechet_distance(*reference, *statistics[source].finalize()))
                                values[source] = value
                                metrics[f"{prefix}/{split_name}/{name}/{source}"] = value
                            atomic_json(control_path, {key: values[key] for key in baselines})
                            return values

                        values = distances(stats, {}, "eval_fvd")
                        scores[name] = {"values": values, "encoder": encoder_identity}
                        print(f"[FVD] step={global_step} {split_name}/{name}: {values}", flush=True)
                        for dataset in datasets:
                            for source, statistic in group_stats[dataset].items():
                                if statistic.n != dataset_counts[dataset]:
                                    raise ValueError(f"Population mismatch for {dataset}/{source}: "
                                                     f"{statistic.n} != {dataset_counts[dataset]}")
                            if dataset_counts[dataset] < 2:
                                print(f"[FVD] step={global_step} {split_name}/{dataset}/{name}: skipped, "
                                      f"{dataset_counts[dataset]} clip(s); FVD needs at least two", flush=True)
                                continue
                            values = distances(group_stats[dataset], {"dataset": dataset}, f"eval_fvd_{dataset}")
                            dataset_scores[dataset][name] = values
                            print(f"[FVD] step={global_step} {split_name}/{dataset}/{name} "
                                  f"({dataset_counts[dataset]} clips): {values}", flush=True)
            with phase(f"{split_name} results"):
                if rank == 0:
                    atomic_json(output_root / split_name / "fvd_score.json",
                                {"step": global_step, "split": split_name, "protocol": protocol,
                                 "scored_frames": _scored_frames(protocol),
                                 "num_videos": len(entries), "num_filtered": len(population["rejected"]),
                                 "model_weights": model_name, "scores": scores,
                                 "datasets": {dataset: {"num_videos": dataset_counts[dataset],
                                                        "scores": dataset_scores[dataset]}
                                              for dataset in datasets},
                                 "conditioning_frames": {"model": 1, "static": 1,
                                                         "ae_reconstruction": num_frames}})
        with phase("evaluation complete"):
            pass
    return metrics if rank == 0 else None


__all__ = [
    "BASELINES",
    "run_fvd_evaluation",
    "sample_index",
    "validate_fvd_config",
    "validate_fvd_settings",
]
