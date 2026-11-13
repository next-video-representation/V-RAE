# First-Frame to 16-Frame Prediction on NanoVVM Clips

This task (`nanovvm_video_pred`) generates the next 16 RGB frames of a video from one
observed RGB frame. It trains on the raw videos and clip manifests prepared for
[NanoVVM](https://github.com/next-video-representation/nanovvm), and it uses the V-RAE
stage-1 model, flow-matching objective, and sampler. NanoVVM supplies the data, the
spatial preprocessing, and the frame protocol; it is not a dependency.

The general V-RAE setup (installation, encoder weights, path file) is in the main
[README](README.md). This page covers only this task.

## Frame Protocol

A manifest record with `start_frame = s` covers source frames `[s, s+49)`. The loader
decodes exactly the source frames `s + 3k` for `k = 0..16` with decord, applies one
spatial transform to all 17 frames, and splits the clip:

| Step | Setting |
| --- | --- |
| Resize | short side to 256, bilinear, `antialias=false` |
| Crop | 256x256 center crop |
| Flip | probability 0 |
| Pixel values | uint8; the encoder applies its own normalization |

```text
condition_rgb = clip[:, 0]       # [B,3,256,256]     source frame s, the only observed frame
future_rgb    = clip[:, 1:17]    # [B,16,3,256,256]  source frames s+3 ... s+48
```

Each manifest record is one training clip. Several sources are concatenated clip by clip
and shuffled together, so each source contributes in proportion to its clip count. The
sample id is `<source name>:<video_path>:<start_frame>`, which keeps identical file names
and frame numbers from different cameras or sources apart. Labels are kept as metadata.

## Model

**Frozen stage 1.** The first frame and the future are encoded by separate calls, so
neither sees the other's pixels:

- The future frames go through `encode_grid` and `grid_to_tokens`, giving the
  prediction target `z_future`. Only these latents are normalized, with statistics fit
  on the train split.
- The first frame goes through a single-image encoder entry (`encode_images`). It
  returns the fused multi-layer patch features before temporal pooling and keeps the
  spatial grid. This entry exists for per-frame image backbones (EUPE, DINOv3, SigLIP2).
  V-JEPA2.1 encodes frames jointly and raises an error. The video entry still requires a
  frame count divisible by four.

**DiT** (`vrae_first_frame_prediction_dit`). Only the four noisy future chunks enter
self-attention; the RoPE time coordinates are 0..3. The first frame enters only through
AdaLN. A trainable projection inside the DiT maps each first-frame patch to the DiT
width. Here `P` is the number of DiT patches per chunk and `D` the DiT encoder width:

```text
frame_cond   = frame_projection(frame_features)                  # [B,P,D]
t_cond       = time_embedder(t)                                  # [B,1,D]
cond_grid    = (t_cond + frame_cond)[:, None].expand(-1,4,-1,-1) # [B,4,P,D]
adaln_cond   = cond_grid.reshape(B, 4*P, D)                      # chunk-major
noisy_tokens = patch_embed(noisy_future)                         # [B,4*P,D]
decoder_cond = encoder_to_decoder(silu(adaln_cond + final_encoder_states))
base_cond    = silu(adaln_cond + base_encoder_states)
```

Each future patch is therefore modulated by the first-frame patch at the same spatial
position. The projection applies a non-affine LayerNorm, a patch convolution with the
DiT patch size, and SiLU + Linear.

**Condition dropout and guidance.** Dropped samples receive a learned per-patch null
condition, so their output carries no first-frame information. Classifier-free
guidance uses the same null condition in its unconditional branch, and internal
guidance (full vs. base head) is kept.

**Objective.** `FlowMatchingTransport` with x-prediction, velocity MSE, and the full and
base losses. Noise and losses apply to the four future chunks only. The timestep shift
follows the future latent size, `sqrt(4*h*w*C/4096)`; for EUPE-B/16 at 256x256 that is
`sqrt(192) = 13.856406460551018`.

### Shapes (EUPE-B/16, 256x256, DiT patch size 1)

| Tensor | Shape |
| --- | --- |
| `condition_rgb` / `future_rgb` | `[B,3,256,256]` / `[B,16,3,256,256]` |
| `frame_features` | `[B,256,768]` |
| `z_future` and sampling noise | `[B,4,256,768]` |
| `frame_cond` / `t_cond` | `[B,256,1152]` / `[B,1,1152]` |
| `adaln_cond` and DiT sequence | `[B,1024,1152]` |
| Decoded prediction | `[B,16,3,256,256]` |

## Configuration

| File | Content |
| --- | --- |
| `configs/training/nanovvm_video_pred/eupe_mix5.yaml` | 18 train and 18 val sources of NanoVVM's mix5 config (`split_val0.0001_min100` holdout) |
| `configs/training/nanovvm_video_pred/eupe_bridgedatav2.yaml` | single-source example |
| `configs/training/nanovvm_video_pred/_eupe_common.yaml` | shared settings; not runnable alone |
| `configs/training/nanovvm_video_pred/scaling-ladder/` | six mix5 sizes matched to NanoVVM's ladder (see [Scaling Ladder](#scaling-ladder)) |
| `configs/models/dit_first_frame_prediction.yaml` | DiT architecture |

The mix5 sources come from NanoVVM's
`configs/scaling-ladder/tiny/mix5_tiny_vpred_interval3_l1_cond1gen16_bs64_lr2e-4_wd0_ep1.yaml`.
The train sources are its `training.video.sources`, and the val sources are its
`fvd.splits.val.sources`.

### Data Root

V-RAE reads configs with plain PyYAML, so NanoVVM's `${paths.data}` and `${oc.env:...}`
interpolations are not resolved. Source paths here are written relative to the NanoVVM
data root, and a path that still contains `${` is rejected. Set the root in the path
file:

```yaml
datasets:
  nanovvm: /path/to/NANOVVM_DATA   # holds bridgedatav2_video/, droid_video/, fineart_video/, ...
```

Alternatively, set `data.data_root` in the config. An absolute path (or one relative to
the repository root) overrides the path file.

### Sources

A split is either a single source or a list of sources:

```yaml
data:
  train:                      # single source; name and dataset are optional
    root: bridgedatav2_video/data/raw
    manifest: bridgedatav2_video/manifests/cond1gen16_interval3/split_val0.0001_min100/train/manifest.json
    manifest_meta: bridgedatav2_video/manifests/cond1gen16_interval3/split_val0.0001_min100/train/meta.yaml
  val:
    sources:                  # several sources, each with its own video root
      - name: droid/wrist_image_left
        dataset: droid
        root: droid_video/data/wrist_image_left
        manifest: droid_video/manifests/wrist_image_left/cond1gen16_interval3/split_val0.0001_min100/val/manifest.json
        manifest_meta: droid_video/manifests/wrist_image_left/cond1gen16_interval3/split_val0.0001_min100/val/meta.yaml
```

Manifests are checked with NanoVVM's rules:

- schema version, sampling parameters (`rgb_seq_len=17`, `frame_interval=3`), and segmentation;
- frame ranges on the 49-frame grid, chronological within each video;
- clip count and SHA-256 against `meta.yaml`.

Holdout subsets that skip clips are accepted. Existing manifests are used as they are;
there is nothing to rebuild or convert.

## Usage

Run the commands from the repository root with `PYTHONPATH=src` (or after installing the
package). `configs/paths.local.yaml` is the path file described above.

**1. Check the configuration.** `--build-only` prints the derived shapes and reads no
files. `check` resolves every source and confirms that each video root, manifest, and
`meta.yaml` exists. With `--load-manifests` it also validates every manifest, and with
`--decode N` it decodes the first clip of N sources.

```bash
python -m vrae.training.nanovvm_video_pred.train \
  --config configs/training/nanovvm_video_pred/eupe_mix5.yaml --build-only
python -m vrae.training.nanovvm_video_pred.check \
  --config configs/training/nanovvm_video_pred/eupe_mix5.yaml \
  --paths configs/paths.local.yaml --load-manifests --decode 18
```

**2. Fit the future latent statistics.** The fit uses `latent_normalizer.fit_clips`
seeded train clips (or `all`) and only their future latents. The file records the
stage-1 checkpoint and structure, the identity of every train manifest (path, SHA-256,
clip count), the image size, the preprocessing, and the frame protocol. Training reuses
an existing file only if all of these match the config, and fits a new one when the file
is missing.

```bash
torchrun --standalone --nproc_per_node=8 \
  -m vrae.training.nanovvm_video_pred.latent_stats \
  --config configs/training/nanovvm_video_pred/eupe_mix5.yaml \
  --paths configs/paths.local.yaml
```

**3. Train.** `--max-steps 1` runs a single step; drop it to train. Checkpoints,
samples, and the resolved config go to
`ckpts/nanovvm_video_pred/<run_name>/`. Exact resume works through `training.resume`, as
in the other V-RAE tasks.

```bash
torchrun --standalone --nproc_per_node=8 \
  -m vrae.training.nanovvm_video_pred.train \
  --config configs/training/nanovvm_video_pred/eupe_mix5.yaml \
  --paths configs/paths.local.yaml --max-steps 1
```

**4. Sample.** Give one first frame and a seed: an image, or frame `S` of a video. To
sample the val manifests instead, use `--split val`.

```bash
python -m vrae.training.nanovvm_video_pred.sample \
  --config configs/training/nanovvm_video_pred/eupe_mix5.yaml --paths configs/paths.local.yaml \
  --checkpoint ckpts/nanovvm_video_pred/eupe_mix5_cond1gen16/checkpoints/latest.pt \
  --output outputs/nanovvm_samples --first-frame frame.png --seed 0
# or: --video path/to/video.mp4 --start-frame 49 --seed 0
# or: --split val --count 64
```

The sampler extracts the first-frame features once, outside the Euler loop, and both
CFG branches reuse them. Each sample starts from independent Gaussian noise
`[4,h*w,C]`. Each output `.pt` file holds:

| Key | Shape | Meaning |
| --- | --- | --- |
| `predicted_future` | `[16,3,256,256]` | decoded prediction of source frames `s+3 ... s+48` |
| `condition_rgb` | `[3,256,256]` | the observed first frame |
| `real_future_rgb` | `[16,3,256,256]` | `--split` only: the RGB target `clip[:, 1:17]` for metrics |
| `display17` | `[17,3,256,256]` | visualization only: the first frame followed by the prediction |

Training logs the same comparison against the RGB target every
`wandb.sample_interval` steps. The 17-frame first-frame-plus-prediction video is logged
under a separate name.

## Scaling Ladder

Six DiT sizes whose parameter counts match NanoVVM's six mix5 ladder sizes. The reference
is NanoVVM's batch-64 ladder at nanovvm@94cc92d,
`configs/scaling-ladder/<size>/mix5_<size>_vpred_interval3_l1_cond1gen16_bs64_lr2e-4_wd0_ep1.yaml`.
The six runs share NanoVVM's data, augmentation, optimization recipe and W&B destination.
Only the DiT size differs between them.

```text
configs/training/nanovvm_video_pred/scaling-ladder/
  _mix5_common.yaml     include: ../eupe_mix5.yaml + every shared setting (not runnable alone)
  <size>/vrae_eupe_mix5_<size>_xpred_interval3_cond1gen16_bs64_lr2e-4_wd0_ep1.yaml
                        include: ../_mix5_common.yaml + run_name (= file stem), dit, wandb.tags
```

Config includes merge mappings but replace lists, so each size file writes out the full
`[encoder, decoder]` lists and the full tag list. The `vrae_` prefix keeps these runs apart
from NanoVVM's `mix5_<size>_vpred_...` runs in the shared W&B project.

### Scaling Rule

The DiT is a DDT: an AdaLN encoder (the trunk) followed by a shallow, wide decoder (the
DDT head). NanoVVM's `n_layer/n_embd/n_head` are not copied. Instead, the default DiT
(`configs/models/dit_first_frame_prediction.yaml`, 843.3M parameters) is scaled with all
of its ratios kept:

| | encoder | decoder (DDT head) |
| --- | --- | --- |
| width | 1152 | 2048 = 16/9 of the encoder width |
| heads x head_dim | 16 x 72 | 16 x 128 |
| SwiGLU FFN hidden, `int(2/3 * 4.0 * width)` | 3072 | 5461 |
| depth | 28 (width/depth 41.1) | 2 = encoder depth / 14 |
| block parameters | 669.5M | 151.1M (17.9% of the total) |

There is one knob, the head count `k` (16 in the default):

- encoder: width `72k`, `k` heads, depth `L`;
- decoder: width `128k`, `k` heads, depth `max(1, round(L / 14))`;
- `encoder_depth = round(L * 8 / 28)`, the block after which the base head reads out;
- `mlp_ratio: 4.0`; every other `dit` key is the default's.

For each size, `k` and `L` are chosen so that the total DiT parameter count matches
NanoVVM's model of that size, with the encoder's width/depth near the default's 41.
NanoVVM's own ladder uses n_embd/n_layer = 40. The counts below were measured on the meta
device for both repositories:

| size | k | `dit.hidden_size` | `dit.depth` | `dit.num_heads` | `dit.encoder_depth` | width/depth | DiT params: total / encoder blocks / decoder blocks | NanoVVM | diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | 3 | [216, 384] | [6, 1] | [3, 3] | 2 | 36.0 | 9.5M / 5.1M / 2.7M | 9.9M | -3.6% |
| small | 7 | [504, 896] | [13, 1] | [7, 7] | 4 | 38.8 | 80.0M / 59.6M / 14.5M | 78.8M | +1.6% |
| medium | 11 | [792, 1408] | [19, 1] | [11, 11] | 5 | 41.7 | 262.7M / 214.8M / 35.7M | 265.7M | -1.1% |
| large | 14 | [1008, 1792] | [27, 2] | [14, 14] | 8 | 37.3 | 628.1M / 494.3M / 115.7M | 629.5M | -0.2% |
| xl | 18 | [1296, 2304] | [33, 2] | [18, 18] | 9 | 39.3 | 1217.5M / 998.5M / 191.2M | 1229.2M | -0.9% |
| xxl | 22 | [1584, 2816] | [37, 3] | [22, 22] | 11 | 42.8 | 2140.0M / 1672.1M / 428.4M | 2124.1M | +0.8% |

The 3D RoPE accepts any even head_dim, so 72 and 128 work at every size.

### Aligned with NanoVVM

| | NanoVVM bs64 ladder | Here |
| --- | --- | --- |
| Data | 18 mix5 sources, `split_val0.0001_min100` holdout | inherited unchanged from `eupe_mix5.yaml` |
| Augmentation | 256x256; short-side bilinear resize, `antialias: false`; center crop; flip 0; 17 frames at interval 3 from each record's fixed start; sources concatenated and reshuffled every epoch | the same port (`data.py`), checked by an exact equality test against a verbatim copy of NanoVVM's `data/online.py` functions |
| Optimizer | AdamW, lr 2e-4, betas (0.9, 0.95), weight decay 0, eps 1e-8 | the same |
| Schedule | cosine, `warmup_ratio: 0.02` (from zero), `decay_end_ratio: 1.0`, final lr 2e-5, as fractions of the total optimizer steps | the same keys, with NanoVVM's LR formula, validation and `round()` |
| Batch | global 64, accumulation 4, 1 epoch | the same |
| Clipping | `clip_grad: 1.0` | the same |
| EMA decay | 0.9995 | the same |
| Precision | bf16 autocast (`scripts/train.sh` default) | bf16 |
| Checkpoints | every 5000 optimizer steps | every 5000 optimizer steps |
| W&B destination | `--wandb`, online, `ENTITY`/`PROJECT` from the environment (`zhaosh5t5an`/`nanovvm`) | `wandb.entity: zhaosh5t5an`, `wandb.project: nanovvm`, `mode: online`, written in the config |
| W&B intervals | metrics every 5 steps, up to 4 sample clips every 500 steps | `log_interval: 5`, `sample_interval: 500`, `sample_count: 4` |
| W&B metric names | `loss/train`, `loss/train_epoch`, `optimizer/{lr,grad_norm,grad_rms,grad_clip_scale}`, `parameters/{norm,rms}` | the same names, plus `loss/train_full` and `loss/train_base` for the two flow-matching terms |

Notes on the alignment:

- V-RAE does not read the `ENTITY`/`PROJECT` environment variables. They are exported
  globally on this cluster, so reading them would send every other V-RAE task to the
  NanoVVM project. Configs without `wandb.entity` call `wandb.init` as before.
- The W&B run name is `run_name`, i.e. the file stem. Exact resume continues the same W&B
  run through the uuid stored in `run_metadata.yaml`, not NanoVVM's name hash. Without an
  API key (`WANDB_API_KEY` or a saved login), the run falls back to offline logging.
- `loss/train` is the mean, over the optimizer steps since the previous log, of each step's
  mean micro-batch loss, averaged over ranks. NanoVVM logs rank 0 and divides by
  `log_interval`; the two agree whenever the window is full. Gradient statistics are
  window means of the norm before clipping, its RMS over the gradient elements and
  `clip_grad / max(norm, clip_grad)`. `parameters/*` cover the trainable DiT parameters
  after the update. `train/epoch`, `train/global_batch_size`, the `samples/...` videos
  and the JSON line on stdout remain. NanoVVM's model diagnostics (`velocity/*`,
  `qk_norm/*`, `block_*`) have no counterpart here.
- NanoVVM's `weight_decay_split: gpt` has no effect at weight decay 0 and is not ported.
- `latent_normalizer` is inherited from `eupe_mix5.yaml`. All six sizes share one
  future-latent statistics file, because it depends only on stage 1 and the train data.

With NanoVVM's 20,978,653 mix5 train clips, one epoch at batch 64 is 327,791 optimizer
steps. Warmup is `round(0.02 * 327,791) = 6,556` steps, and the cosine ends at step
327,791. The trainer prints the resolved schedule at start-up and records it in
`run_metadata.yaml` under `lr_schedule`.

### Shared Code Changes

These changes are needed by the ladder and leave the other tasks' behavior unchanged:

- `common/optim.py`: `build_scheduler` also accepts `warmup_ratio` and `decay_end_ratio`
  with an `epochs` argument. A ratio schedule may not be combined with
  `warmup_steps/warmup_epochs/decay_end_steps/decay_end_epoch`, except as `null`. The
  ladder writes `warmup_steps: null` to switch off the `warmup_steps: 0` inherited from
  `_eupe_common.yaml`. The existing keys produce the same LR at every step as before.
- `common/accumulation.py`: `clip_and_step` returns the gradient norm before clipping.
  `optimizer_step` still returns only the step flag.
- `common/wandb.py`: optional `wandb.entity`.
- `nanovvm_video_pred/train.py`: the scheduler is built after the loader, with
  `steps_per_epoch = len(loader) // gradient_accumulation_steps`. A linear or cosine
  schedule therefore works for this task; before this change it failed. `clip_grad` is
  applied, the metrics above are logged, and the resolved config is passed to W&B.

### Launch

```bash
CONFIG=configs/training/nanovvm_video_pred/scaling-ladder/tiny/vrae_eupe_mix5_tiny_xpred_interval3_cond1gen16_bs64_lr2e-4_wd0_ep1.yaml
python -m vrae.training.nanovvm_video_pred.train --config $CONFIG --build-only
python -m vrae.training.nanovvm_video_pred.check --config $CONFIG --paths configs/paths.local.yaml

# Once for all six sizes: the shared future-latent statistics.
torchrun --standalone --nproc_per_node=8 -m vrae.training.nanovvm_video_pred.latent_stats \
  --config $CONFIG --paths configs/paths.local.yaml

torchrun --standalone --nproc_per_node=8 -m vrae.training.nanovvm_video_pred.train \
  --config $CONFIG --paths configs/paths.local.yaml
```

The trainer fits the statistics file when it is missing. Fit it once before starting
several sizes at the same time, so that the runs do not all fit and write the same file.
With 8 processes, the global batch of 64 with accumulation 4 is a micro-batch of 2 per
GPU. To resume, use a config that includes the size file and sets `training.resume` to
`ckpts/nanovvm_video_pred/<run_name>/checkpoints/latest.pt`. The LR schedule, data
position and W&B run all continue from the checkpoint.

### Differences Kept on Purpose

- **No FVD in the training loop.** NanoVVM evaluates `fvd.interval: 10000` on val and
  train_eval with I3D, VideoMAEv2 and V-JEPA2.1. Here, val sampling (`sample --split val`)
  saves prediction/target pairs for external metrics.
- **Objective.** V-RAE uses x-prediction flow matching with first-frame condition dropout
  `dit.frame_dropout: 0.1` and classifier-free guidance. NanoVVM uses autoregressive
  v-prediction with an L1 loss. `frame_dropout` is a model setting, not data augmentation,
  and keeps its default.
- **EMA.** NanoVVM's `scripts/slurm/job.sh` exports `DISABLE_EMA=1`, and its FVD uses
  `use_ema: false`. V-RAE keeps EMA at 0.9995 and samples with it.
- **Checkpoint pruning.** NanoVVM keeps the last 10 checkpoints (`keep_last: 10`). V-RAE
  never prunes. One epoch writes 66 checkpoints: 65 at the 5000-step interval and the
  final one at step 327,791. Each holds about 16 bytes per DiT parameter (weights, EMA,
  AdamW m and v):

| size | per checkpoint | 66 checkpoints |
| --- | --- | --- |
| tiny | 0.15 GB | 10 GB |
| small | 1.3 GB | 84 GB |
| medium | 4.2 GB | 277 GB |
| large | 10.0 GB | 663 GB |
| xl | 19.5 GB | 1.29 TB |
| xxl | 34.2 GB | 2.26 TB |

## Tests

```bash
PYTHONPATH=src python -m unittest tests.test_nanovvm_video_pred tests.test_nanovvm_scaling_ladder -v
```

The tests run on CPU with synthetic videos and small weights. They cover:

- the manifest checks, and the frames decoded for a non-zero start (`s=49` gives
  condition frame 49 and targets 52, 55, ..., 97);
- the single-image encoder entry, and that the condition and target encodings do not
  depend on each other;
- the DiT: condition dropout, spatial alignment, CFG, and gradients;
- sampling from a first frame alone, and a train/resume/sample round trip with the
  ladder's training block (cosine with ratios, `clip_grad`, accumulation 4). The round trip
  checks that the LR continues the schedule after resume and that W&B receives exactly the
  NanoVVM-named keys;
- a Cityscapes regression check;
- for the scaling ladder (`test_nanovvm_scaling_ladder.py`):
  - the ratio LR at every step against values from NanoVVM's scheduler;
  - rejection of mixed ratio and step keys, and unchanged step/epoch schedules;
  - clipping and the returned pre-clip norm, and the training-metric formulas;
  - the `wandb.init` arguments;
  - the exact preprocessing equality with NanoVVM;
  - each ladder config against the size table and the recipe.

The repository's `.gitignore` excludes `/tests/`, so these files were added with `git add -f`.

## Code

| Path | Role |
| --- | --- |
| `src/vrae/training/nanovvm_video_pred/manifest.py` | NanoVVM manifest schema and validation |
| `src/vrae/training/nanovvm_video_pred/data.py` | frame decoding, preprocessing, sources, loader |
| `src/vrae/training/nanovvm_video_pred/latents.py` | separate first-frame and future encoding, decoding |
| `src/vrae/training/nanovvm_video_pred/latent_stats.py` | future normalizer and its identity check |
| `src/vrae/training/nanovvm_video_pred/train.py` | build checks, training loop, checkpoints |
| `src/vrae/training/nanovvm_video_pred/sample.py` | sampling from a first frame or a split |
| `src/vrae/training/nanovvm_video_pred/check.py` | data path and manifest check |
| `src/vrae/models/dit/first_frame_prediction_dit.py` | the DiT (a subclass of `VRAEVideoDiT`) |
| `src/vrae/models/dit/conditioning.py` | `FirstFrameConditionAdapter` |
| `src/vrae/models/encoders/base.py` | `encode_images` single-image entry |

`VRAEVideoDiT` now runs its encoder, decoder, and heads through a shared `_denoise`
method, and its UCF101/Kinetics-600 outputs are unchanged. The Cityscapes 12-to-12 task
and its checkpoints are not affected.

## Notes

- Every configured manifest must exist. `check` lists any missing file. For mix5, this
  includes the `split_val0.0001_min100` holdout of all four `egopro448_video` views.
- Loading the 18 mix5 train manifests (about 21M clips) takes about a minute and a few
  GB of memory per rank, as in NanoVVM.
- The default DiT has about 843M parameters. A training checkpoint with EMA and AdamW
  state is about 13.5 GB, and old checkpoints are not pruned (see the ladder's
  checkpoint sizes above).
- The encoder AdaLN condition is per token. This costs more than one shared condition
  vector per sample.
- There is no FVD protocol for this task yet. Val sampling saves prediction/target pairs
  for external metrics.
