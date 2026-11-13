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

## Tests

```bash
PYTHONPATH=src python -m unittest tests.test_nanovvm_video_pred -v
```

The tests run on CPU with synthetic videos and small weights. They cover:

- the manifest checks, and the frames decoded for a non-zero start (`s=49` gives
  condition frame 49 and targets 52, 55, ..., 97);
- the single-image encoder entry, and that the condition and target encodings do not
  depend on each other;
- the DiT: condition dropout, spatial alignment, CFG, and gradients;
- sampling from a first frame alone, and a train/resume/sample round trip;
- a Cityscapes regression check.

The repository's `.gitignore` excludes `/tests/`, so this file was added with `git add -f`.

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
  state is about 13.5 GB, and old checkpoints are not pruned.
- The encoder AdaLN condition is per token. This costs more than one shared condition
  vector per sample.
- There is no FVD protocol for this task yet. Val sampling saves prediction/target pairs
  for external metrics.
