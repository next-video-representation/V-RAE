# First-Frame to 16-Frame Prediction on NanoVVM Clips

This task (`nanovvm_video_pred`) generates the next 16 RGB frames of a video from one
observed RGB frame. It trains on the raw videos and clip manifests prepared for
[NanoVVM](https://github.com/next-video-representation/nanovvm), and it uses the V-RAE
stage-1 model, flow-matching objective, and sampler. NanoVVM supplies the data, the
spatial preprocessing, and the frame protocol; it is not a dependency.

The general V-RAE setup (installation, encoder weights, path file) is in the main
[README](README.md). This page covers only this task.

## Prompt You Should Give to The Agent to Directly Run this Task

Open a coding agent with shell access (for example Claude Code) at the repository
root on your cluster login node. Replace every `<...>` field, then paste the prompt.
It mirrors NanoVVM's own prompt, and the mix5 data is the same NanoVVM data:

```text
Run the nanovvm_video_pred scaling ladder of this repository end to end by following
README_nanovvm_video_pred.md, "Usage" steps 1-3 and "Scaling Ladder".

My settings:
- W&B API key: <your-wandb-api-key>
- W&B entity / project: zhaosh5t5an / nanovvm (already set in the ladder configs; V-RAE does not read ENTITY/PROJECT)
- Storage: DATA='/home/sz128/scratch_sz128/datasets' (NanoVVM's NANOVVM_DATA), VRAE_CKPTS='/home/sz128/scratch_sz128/ckpts/V-RAE'
- Existing weights: stage 1 '/home/sz128/scratch_sz128/models/V-RAE-Models/vrae/vrae_eupe.pt', EUPE-ViT-B '/home/sz128/projects/V-RAE-ori/ckpts/pretrained/encoders/eupe/model.pt'
- NanoVVM checkout, for its data tools: '/home/sz128/projects/nanovvm'
- Dataset: mix5 = BridgeDataV2 + DROID + FineART + EgoPro + Ego-Exo4D (NanoVVM README step 3, "Mixed data (mix5)")
- Base config: configs/training/nanovvm_video_pred/scaling-ladder/tiny/vrae_eupe_mix5_tiny_xpred_interval3_cond1gen16_bs64_lr2e-4_wd0_ep1.yaml
- Run name: 'vrae_eupe_mix5_tiny_xpred_interval3_cond1gen16_bs64_lr2e-4_wd0_ep1'
- Config changes: 'none'
- Compute: Slurm: account cw220, partition commons, 2 x H200 (--gres=gpu:h200:2)

Do these steps in order. Skip anything that already exists. If a step fails twice, stop and ask me.
1. Install as in README.md, "Installation" (conda env vrae). Then, on the login node, run
   `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONPATH=src python -m unittest tests.test_nanovvm_video_pred tests.test_nanovvm_scaling_ladder tests.test_nanovvm_fvd`.
2. Run `wandb login` with my key. Never write the key into the repository.
3. Write configs/paths.local.yaml (git-ignored): project_root = this checkout,
   datasets.nanovvm = $DATA, checkpoint_write_root = $VRAE_CKPTS. Runs, latent statistics and
   checkpoints then go under $VRAE_CKPTS. Pass `--paths configs/paths.local.yaml` to every command.
   Weights: $VRAE_CKPTS/vrae/vrae_eupe.pt
   (`hf download Guomh0707/V-RAE-Models vrae/vrae_eupe.pt --local-dir $VRAE_CKPTS`) and
   $VRAE_CKPTS/pretrained/encoders/eupe/model.pt (third_party/README.md, "EUPE ViT-B/16").
   FVD encoders: NanoVVM's models/fvd and models/vjepa2_1 at $VRAE_CKPTS/pretrained/fvd and
   $VRAE_CKPTS/pretrained/vjepa2_1 ("Usage" step 5; NanoVVM's prepare_fvd_models.py makes them).
   Symlink my existing weights there instead of downloading them again.
4. Data. Run `python -m vrae.training.nanovvm_video_pred.check --config <base config>`: every
   video root, manifest and meta.yaml it names must exist. Build missing split_val0.0001_min100
   holdout manifests in the NanoVVM checkout, with its .venv and data_processing/split_clip_manifest.py,
   as NanoVVM README step 3 shows. The ladder reads EgoPro from egopro448_video, so for EgoPro run
   `(umask 077; python data_processing/split_clip_manifest.py --val-fraction 0.0001 --min-val-clips 100 $DATA/egopro448_video/manifests/{head_left,head_right,left_wrist,right_wrist}/cond1gen16_interval3)`.
   Never download, copy or upload EgoPro or Ego-Exo4D: if either is missing, stop and ask me.
   Then rerun check with `--load-manifests --decode 18`.
5. Configs, next to the base config. If my config changes are 'none' and the run name equals the
   base config's run_name, the run config is the base config. Otherwise write <run name>.yaml,
   which includes the base config and sets run_name and my config changes (includes replace
   lists, so write lists in full). Also write:
   - <run name>-smoke.yaml: includes the run config and sets run_name: <run name>-smoke;
   - <run name>-resume.yaml: includes the run config and sets training.resume:
     ckpts/nanovvm_video_pred/<run name>/checkpoints/latest.pt.
   Check each with `python -m vrae.training.nanovvm_video_pred.train --config <config> --build-only`,
   and check that training.global_batch_size is divisible by GPUs x training.gradient_accumulation_steps.
6. Train on Slurm. Write scripts/slurm/train_<run name>.slurm: job name <run name>, one node, the
   GPUs above, 32 CPUs, 256 GB, 24 hours. It activates the env, changes to this checkout and runs
   `torchrun --standalone --nproc_per_node=<GPUs> -m vrae.training.nanovvm_video_pred.train --config <config> --paths configs/paths.local.yaml`,
   with <config> = the resume config if $VRAE_CKPTS/nanovvm_video_pred/<run name>/checkpoints/latest.pt
   exists, and the run config otherwise.
   a. Smoke test first, as a separate short job with the same GPUs: the smoke config,
      `--max-steps 20`, WANDB_MODE=disabled. If the shared future-latent statistics file is
      missing, this job fits it first ("Usage" step 2). Wait for it to finish and read its log.
   b. Submit the training job with sbatch. Wait until the first training logs appear, and check
      that they reach W&B. One epoch is 327,791 optimizer steps, likely longer than one 24-hour
      job; each job resumes from latest.pt (a checkpoint every 5,000 steps). Give me the command
      that queues follow-up jobs, `sbatch --dependency=singleton scripts/slurm/train_<run name>.slurm`
      (once per extra 24 hours), but do not run it.
7. Stop there. Do not wait for training to finish.

Rules: change no model, training or evaluation code; only add configs/paths.local.yaml, the
configs of step 5 and the Slurm script. Don't delete data, checkpoints or runs, and don't commit
or push. A fresh run refuses to start in a run directory that already holds files; if that
happens, stop and ask me.

Report at the end: the commands you ran, the data, weight and run directories, the Slurm job IDs,
the W&B run URL, and anything you skipped or changed.
```

To keep the API key out of the agent's transcript, run `wandb login` yourself
first, or export `WANDB_API_KEY` before you start the agent. Then replace the
key line with "W&B is already logged in."

## Prompt for AD

This version is for Adobe AI Platform (Pluto) GPU nodes in `epic-intern-pool`.
There is no Slurm: the agent trains on the node it runs on. Raw videos are not staged: they stay in
uncompressed tars on S3 and are read on demand, whole video by whole video, through a bounded cache on
the node SSD ("Remote Video Storage" below). The tars, indexes and manifests are NanoVVM's, and so is
the cache format: V-RAE and NanoVVM runs on one node share one cache and one budget. Storage follows
NanoVVM's "Prompt for AD":

| What | Where |
|---|---|
| Code: this checkout and the new configs | Sensei FS (`/sensei-fs-3/...`) |
| uv cache, Python and venv; weights, runs, statistics, caches | node-local SSD `/mnt/localssd/<user>/vrae`, erased when the job stops |
| mix5 manifests and tar indexes (no videos) | NanoVVM's data directory on the node, `/mnt/localssd/<user>/nanovvm/data` |
| Video cache (disposable, shared with NanoVVM) | `/mnt/localssd/<user>/nanovvm/video_cache` |
| Durable copies: env, weights, statistics, runs; NanoVVM's dataset tars and indexes | `s3://phidias/<user>/checkpoint/` |

Get a node first (P2 only, `request_node.sh --name auto` from the `aip-node` skill),
open the agent on that node at the repository root, replace every `<...>` field, then
paste the prompt:

```text
Run the nanovvm_video_pred scaling ladder of this repository (the checkout you are in, on Sensei FS)
end to end on this Adobe AI Platform node by following README_nanovvm_video_pred.md, "Usage"
steps 1-3 and "Scaling Ladder", adapted as below. There is no Slurm. Videos are read from S3 on demand
through the node video cache (README_nanovvm_video_pred.md, "Remote Video Storage"); never download or
extract whole datasets.

My settings:
- W&B: WANDB_API_KEY comes from the secrets file; never print it or write it anywhere else
- W&B entity / project: zhaosh5t5an / nanovvm (already set in the ladder configs; V-RAE does not read ENTITY/PROJECT)
- Local root: LOCAL=/mnt/localssd/yidongh/vrae
- Data root: DATA=/mnt/localssd/yidongh/nanovvm/data (NanoVVM's NANOVVM_DATA on this node: manifests and tar indexes)
- S3 root: S3=s3://phidias/yidongh/checkpoint (mix5 tars and indexes in $S3/assets/nanovvm, V-RAE env, weights and statistics in $S3/assets/vrae, runs in $S3/runs/vrae)
- NanoVVM checkout, for its data tools: /sensei-fs-3/users/yidongh/yf/stzhao/nanovvm
- Dataset: mix5 = BridgeDataV2 + DROID + FineART + EgoPro + Ego-Exo4D (NanoVVM README step 3, "Mixed data (mix5)")
- Base config: configs/training/nanovvm_video_pred/scaling-ladder/tiny/vrae_eupe_mix5_tiny_xpred_interval3_cond1gen16_bs64_lr2e-4_wd0_ep1.yaml
- Run name: vrae_eupe_mix5_tiny_xpred_interval3_cond1gen16_bs64_lr2e-4_wd0_ep1
- Config changes: none
- Video storage: VRAE_STORAGE_CONFIG=configs/storage/adobe_s3.yaml, VRAE_VIDEO_CACHE=/mnt/localssd/yidongh/nanovvm/video_cache,
  VRAE_VIDEO_CACHE_GIB=300, VRAE_VIDEO_CACHE_RESERVE_GIB=100 (the same budget as NanoVVM runs on this node)
- Compute: this node, A100 80 GB, 4 GPUs (NPROC_PER_NODE = count from `nvidia-smi -L`), P2, preemptible

Storage rules:
- The repository on Sensei FS holds code only. Write nothing into it except the new configs:
  no venv, egg-info, paths file, data, weights, runs, wandb/, __pycache__ or other caches.
- Everything large goes under $LOCAL or $DATA on the node SSD, which is erased when the job
  stops. Durable copies go to $S3. The video cache ($VRAE_VIDEO_CACHE) is disposable: never back it up.
- Reach S3 from the shell only through this wrapper, which uses the node's own EC2 role:
    s5() { env -u AWS_ROLE_ARN -u AWS_WEB_IDENTITY_TOKEN_FILE -u AWS_ACCESS_KEY_ID \
             -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN -u AWS_PROFILE \
             AWS_DEFAULT_REGION=us-west-2 s5cmd "$@"; }
  (Training reads S3 the same way, through the instance role: storage.s3.credentials=instance_role.)
  phidias is shared by about 140 users. Touch only s3://phidias/yidongh/, use only ls, du, cat, cp, pipe
  and sync, and never rm, mv or sync --delete. New files on S3 go only to $S3/runs/vrae/<run name>/,
  $S3/assets/vrae/ and, create-only, $S3/assets/nanovvm/indexes/v1/; never overwrite an existing asset.
- EgoPro and Ego-Exo4D videos are licensed: never download, copy or upload them. Their tars are read from
  $S3/assets/nanovvm only; if a tar is missing there, stop and ask me.

Do these steps in order. Skip anything that already exists. If a step fails twice, stop and ask me.
0. Check the node: `nvidia-smi -L` lists GPUs (if none, stop and ask me for a node), `s5 ls $S3/` works,
   and `df -h /mnt/localssd` shows at least VRAE_VIDEO_CACHE_GIB + VRAE_VIDEO_CACHE_RESERVE_GIB + 50 GB
   free (about 450 GB with my settings; lower VRAE_VIDEO_CACHE_GIB if not, never below 50; a NanoVVM run
   sharing the cache counts once), plus the checkpoints: V-RAE never prunes them, and one epoch writes 66
   (README_nanovvm_video_pred.md, "Differences Kept on Purpose": 10 GB in total for tiny). Never create,
   stop or change AI Platform jobs.
1. Write $LOCAL/env.sh and source it in every shell, including the training session.
   $HOME is on the container root file system, so do not use ~/.bashrc. It sets:
     PATH=$LOCAL/bin:$PATH (uv lives there: `curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$LOCAL/bin UV_NO_MODIFY_PATH=1 sh`)
     UV_CACHE_DIR=$LOCAL/uv/cache UV_PYTHON_INSTALL_DIR=$LOCAL/uv/python VENV=$LOCAL/uv/venv
     PYTHONPATH=<this checkout>/src PYTHONPYCACHEPREFIX=$LOCAL/cache/pycache
     HF_HOME=$LOCAL/hf_home TORCH_HOME=$LOCAL/torch_home TMPDIR=$LOCAL/tmp
     TRITON_CACHE_DIR=$LOCAL/cache/triton TORCHINDUCTOR_CACHE_DIR=$LOCAL/cache/inductor
     VRAE_STORAGE_CONFIG=configs/storage/adobe_s3.yaml VRAE_VIDEO_CACHE=... VRAE_VIDEO_CACHE_GIB=... VRAE_VIDEO_CACHE_RESERVE_GIB=...
   plus LOCAL, DATA, S3, the secrets from the secrets file, and the s5 function.
2. Environment: if $S3/assets/vrae/uv.tar exists, restore it with
   `s5 cat $S3/assets/vrae/uv.tar | tar -C $LOCAL -xf -`. Otherwise run
   `uv venv $VENV --python 3.10`. Then `source $VENV/bin/activate` and
   `uv pip install -r pyproject.toml --extra s3` (botocore for the S3 video backend).
   This installs the dependencies but not the package, so nothing is written into the checkout;
   PYTHONPATH stands in for the install. FFmpeg is not needed: this task decodes with decord.
   With the venv active, run the CPU tests (GPUs hidden):
   `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 python -m unittest tests.test_nanovvm_video_pred tests.test_nanovvm_scaling_ladder tests.test_remote_video_storage tests.test_nanovvm_fvd`.
   After a fresh install, upload the env with `tar -C $LOCAL -cf - uv | s5 pipe $S3/assets/vrae/uv.tar`.
3. Paths, weights and small assets.
   a. Write $LOCAL/paths.yaml: project_root = this checkout, datasets.nanovvm = $DATA,
      checkpoint_write_root = $LOCAL/ckpts. Pass `--paths $LOCAL/paths.yaml` to every command.
   b. Weights: if $S3/assets/vrae/weights.tar exists, stream it into $LOCAL. Otherwise download
      $LOCAL/ckpts/vrae/vrae_eupe.pt
      (`hf download Guomh0707/V-RAE-Models vrae/vrae_eupe.pt --local-dir $LOCAL/ckpts`) and
      $LOCAL/ckpts/pretrained/encoders/eupe/model.pt (third_party/README.md, "EUPE ViT-B/16"),
      then upload them with `tar -C $LOCAL -cf - ckpts/vrae ckpts/pretrained/encoders | s5 pipe $S3/assets/vrae/weights.tar`.
      FVD encoders ("Usage" step 5), NanoVVM's: for <a> in fvd and vjepa2_1,
      `s5 cat $S3/assets/nanovvm/models/<a>.tar | tar -C $LOCAL/ckpts/pretrained --strip-components=1 -xf -`
      (if a tar is missing, stop and ask me).
   c. For each dataset tar the base config reads (configs/storage/adobe_s3.yaml maps them; $DATA/<d>_video for
      bridgedatav2, droid, fineart, egopro448, egoexo4d): if $S3/assets/nanovvm/indexes/v1/<d>_video.tar.index.sqlite
      exists, copy it to $DATA/remote_index/ with `s5 cp`. If you need a run checkpoint (step 6a), it comes later.
4. Remote video sources, one dataset at a time (T="python -m vrae.storage.remote_video"):
   a. `s5 ls $S3/assets/nanovvm/data/<d>_video.tar` must exist. For EgoPro or Ego-Exo4D, if it does not,
      stop and ask me. For a public dataset, stop and ask me before preparing it (NanoVVM README,
      "Prompt for AD", step 4a: a one-time download, then one uncompressed tar indexed while it uploads).
   b. Index: if $DATA/remote_index/<d>_video.tar.index.sqlite is still missing, build it from the
      headers of the S3 tar: `$T index --archive $S3/assets/nanovvm/data/<d>_video.tar --output
      $DATA/remote_index/<d>_video.tar.index.sqlite` (one ranged GET per video, about 30 ms each,
      no payloads; resumable: rerun the same command after an interruption). Then
      `$T upload-index --index <it> --dest $S3/assets/nanovvm/indexes/v1/<d>_video.tar.index.sqlite`.
      Check its member prefix with `$T list --index <it> --limit 5` (expected data/<d>_video/...).
   c. Manifests: fetch only them from the tar, never the videos:
      `$T extract --index <it> --pattern 'data/<d>_video/manifests/*' --strip-prefix data --dest $DATA`.
      Then build the missing split_val0.0001_min100 holdout manifests in the NanoVVM checkout, with this
      venv and data_processing/split_clip_manifest.py, as NanoVVM README step 3 shows (about 3 minutes;
      deterministic, so they match the cluster's). For EgoPro, use egopro448_video:
      `(umask 077; python data_processing/split_clip_manifest.py --val-fraction 0.0001 --min-val-clips 100 $DATA/egopro448_video/manifests/{head_left,head_right,left_wrist,right_wrist}/cond1gen16_interval3)`.
   d. `python -m vrae.training.nanovvm_video_pred.check --config <base config> --paths $LOCAL/paths.yaml
      --load-manifests --decode 18`: credentials, archive identities, the cache budget, every manifest,
      every train and val manifest video resolved in its index, and 18 clips decoded through the cache.
      It must end with "storage check passed".
5. Configs, next to the base config in this checkout. If my config changes are 'none' and the run
   name equals the base config's run_name, the run config is the base config. Otherwise write
   <run name>.yaml, which includes the base config and sets run_name and my config changes
   (includes replace lists, so write lists in full). Also write:
   - <run name>-smoke.yaml: includes the run config and sets run_name: <run name>-smoke;
   - <run name>-resume.yaml: includes the run config and sets training.resume:
     ckpts/nanovvm_video_pred/<run name>/checkpoints/latest.pt.
   Check each with `python -m vrae.training.nanovvm_video_pred.train --config <config> --build-only`.
   Check that training.global_batch_size is divisible by NPROC_PER_NODE x
   training.gradient_accumulation_steps. If it is not, stop and ask me.
6. Train. The run directory is $LOCAL/ckpts/nanovvm_video_pred/<run name>, and the shared
   future-latent statistics file is latent_normalizer.path of the config, under $LOCAL/ckpts.
   a. Resume: if $S3/runs/vrae/<run name>/checkpoints/ holds a step-*.pt, copy that run directory
      into the run directory, but of checkpoints/ only the newest step-*.pt, and make
      checkpoints/latest.pt a symlink to it. If $S3/assets/vrae/latent_stats/ holds the
      statistics file, copy it to its place.
   b. Smoke run, unless you are resuming:
      `WANDB_MODE=disabled torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m vrae.training.nanovvm_video_pred.train --config <smoke config> --paths $LOCAL/paths.yaml --max-steps 20`.
      It first fits the statistics file if it is missing ("Usage" step 2; its 20,000 clips come through the
      cache too). Its log must show "[storage] Remote video storage" lines and cache/* metrics. Then copy
      the statistics file to $S3/assets/vrae/latent_stats/.
   c. Start `torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m vrae.training.nanovvm_video_pred.train --config <config> --paths $LOCAL/paths.yaml`,
      with <config> = the resume config if checkpoints/latest.pt exists and the run config
      otherwise, in a detached tmux session named vrae-train (screen, or setsid nohup, if tmux
      is missing), logging to $LOCAL/logs/<run name>.log.
   d. Start a second detached session, vrae-sync, that every 15 minutes runs
      `s5 sync --exclude "*latest.pt" --exclude "fvd_cache/*" --exclude "*.tmp" --exclude "*.part" <run dir>/ $S3/runs/vrae/<run name>/`
      (latest.pt is a symlink; s5cmd would upload it as a second full copy. The video cache lives outside
      the run directory and is never synced.)
   Wait until the first training logs appear, they reach W&B, and the first sync has finished.
   The log line "[fvd] every 10000 steps ..." confirms the in-training FVD; its eval_fvd/* values reach W&B
   at step 10000. Report data/wait_seconds_per_step and cache/hit_rate from the first log lines (with a shared cache,
   the cache/* counters are node-wide).
7. Stop there. Do not wait for training to finish. A P2 node can be preempted at any time;
   only what is on S3 survives it. A new node starts cold: the first steps download their videos.

Rules: change no model, training or evaluation code; only add the configs of step 5 and files
under $LOCAL and $DATA (and the shared video cache). Don't delete data, checkpoints or runs, locally or
on S3, and don't commit or push. A fresh run refuses to start in a run directory that already holds
files; if that happens, stop and ask me.

Report at the end: the commands you ran, the $LOCAL, $DATA and $S3 paths of the env, indexes,
manifests, weights, statistics, video cache and run, the tmux sessions and process IDs, the W&B run
URL, and anything you skipped or changed.
```

For another user, replace `yidongh` in the secrets file, `LOCAL`, `DATA`, `S3` and `VRAE_VIDEO_CACHE` (and set
`VRAE_S3_ASSETS` if the NanoVVM tars live elsewhere). If you have no secrets file, run `wandb login`
yourself, export `WANDB_API_KEY` before you start the agent, and replace the W&B line with "W&B is
already logged in; WANDB_API_KEY is exported."

To train from fully local data instead (an offline node, or to compare), restore the dataset tars into
the data root's parent (`s5 cat $S3/assets/nanovvm/data/<d>_video.tar | tar -C /mnt/localssd/yidongh/nanovvm -xf -`,
about 3 TB for mix5) and set `VRAE_STORAGE_CONFIG=local`: paths, manifests and checkpoints stay the same.

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

## Remote Video Storage

On nodes without the mix5 videos (about 3 TB), the videos can stay on S3 in NanoVVM's uncompressed
tars and be read on demand. When a clip needs a video, the whole video is fetched with one ranged GET
into a bounded cache on the node SSD and decoded from there. Prefetch threads warm the cache along each
rank's sampler order. The clips, their order, the RNG and the decoded frames are the same as with local
files. Manifests, tar indexes, weights and checkpoints stay local. Local files remain the default.

Select a storage file per machine with an environment variable. The training configs do not change, so
a run resumes the same way with local or remote videos:

```bash
uv pip install -r pyproject.toml --extra s3          # botocore; local training never imports it
export VRAE_STORAGE_CONFIG=configs/storage/adobe_s3.yaml   # unset or "local": local files
export VRAE_VIDEO_CACHE=/mnt/localssd/<user>/nanovvm/video_cache VRAE_VIDEO_CACHE_GIB=300 VRAE_VIDEO_CACHE_RESERVE_GIB=100
```

[`configs/storage/adobe_s3.yaml`](configs/storage/adobe_s3.yaml) maps each mix5 video directory below
the data root (`bridgedatav2_video`, `droid_video`, `fineart_video`, `egopro448_video`,
`egoexo4d_video`) to `$VRAE_S3_ASSETS/data/<dataset>_video.tar` (default
`s3://phidias/yidongh/checkpoint/assets/nanovvm`) and to its index,
`<data root>/remote_index/<dataset>_video.tar.index.sqlite`. Relative `local` and `index` paths are
below the data root. `${NAME:-default}` values are taken from the environment. S3 is reached only
through the node's EC2 instance role (`credentials: instance_role`). Other credentials in the
environment are ignored, and `os.environ` is not modified.

- **Same assets as NanoVVM.** The tars, indexes and manifests are the ones NanoVVM's "Prompt for AD"
  creates. `python -m vrae.storage.remote_video` builds or checks indexes from the tar headers (`index`,
  `bind`, `verify`, `upload-index`, create-only), lists members, extracts the manifests alone
  (`extract --pattern 'data/<d>_video/manifests/*' --strip-prefix data --dest <data root>`) and
  maintains the cache (`cache --stats | --recover | --clear`).
- **One cache per node.** The cache's byte budget covers every rank and DataLoader worker. A video is
  downloaded once and never evicted while it is being decoded. `vrae/storage/` is vendored from NanoVVM
  (`data/`, commit ca80730), and the cache format is the same, so NanoVVM and V-RAE runs on one node can
  share a cache directory. Use the same capacity and reserve in both.
- **Checks before weights load.** `check` (Usage step 1) and training test the S3 credentials, each
  archive's identity against its index (a changed archive is reported as stale), and the cache budget.
  Every manifest video must resolve in its index. Each read is pinned to the indexed ETag, and a
  per-member header CRC rejects a stale or misapplied index. With remote storage, `check` ends with
  "storage check passed".
- **Logs.** Training prints `[storage] Remote video storage: ...` lines. Every log line has
  `data/wait_seconds_per_step` (rank 0's wait for batches, per optimizer step). With remote storage it
  also has NanoVVM's `cache/*` metrics: `hit_rate`, `demand_downloads`, `prefetch_downloads`,
  `prefetch_used`, `fetched_gib`, `requests`, `retries`, `evictions`, `demand_wait_seconds`,
  `used_gib` and `inflight_downloads`. The cache's counters are node-wide: with a shared cache they
  include the other runs.
- **Cold starts.** A new node starts with an empty cache, and its first steps wait for downloads. The
  20-step smoke run of the tiny config on one A100 waited 0.9-2.9 s per step on a cold cache.
- Compressed archives cannot be read by ranged GETs and are refused. Repack them as uncompressed tars.

## Usage

Run the commands from the repository root with `PYTHONPATH=src` (or after installing the
package). `configs/paths.local.yaml` is the path file described above.

**1. Check the configuration.** `--build-only` prints the derived shapes and reads no
files. `check` resolves every source and confirms that each video root, manifest, and
`meta.yaml` exists. With `--load-manifests` it also validates every manifest, and with
`--decode N` it decodes the first clip of N sources. With `VRAE_STORAGE_CONFIG` set, a video root
exists when its tar index has it, and the remote checks of "Remote Video Storage" run too.

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

**5. FVD.** The scaling-ladder configs evaluate FVD during training with NanoVVM's protocol, the
`fvd` section of `scaling-ladder/_mix5_common.yaml`:

- **Schedule and data.** Every `fvd.interval: 10000` optimizer steps, on the val holdout of all 18
  sources (`fvd.splits.val`, 2,204 clips) and as many train clips (`train`, the `train_eval`
  manifests).
- **Encoders.** I3D, VideoMAE-v2 and V-JEPA 2.1, scoring frames 1..16; frame 0 is the condition.
- **Controls.** `ae_reconstruction` (the real future through the frozen V-RAE) and `static`
  (frame 0 repeated, through the frozen V-RAE).
- **W&B keys.** NanoVVM's: `eval_fvd/<split>/<encoder>/<source>` pooled and
  `eval_fvd_<dataset>/<split>/<encoder>/<source>` per dataset, with `<source>` = `model`,
  `ae_reconstruction` or `static`.
- **Shared with NanoVVM.** Populations, the clip crop, the mp4 round trip, the encoder adapters
  and the statistics are NanoVVM's code (`src/vrae/evaluation/nanovvm_fvd/`, vendored from
  NanoVVM's `eval/`). The real-video statistics are therefore NanoVVM's, and the FVDs of the two
  codebases are comparable.
- **Generation.** The model samples from frame 0 with `sampling.*` (100 Euler steps). Each clip
  gets the same noise at every checkpoint, so successive evaluations differ only by the weights.
  Live weights are used, like NanoVVM (`fvd.use_ema: false`).
- **Encoder weights.** NanoVVM's `models/fvd` and `models/vjepa2_1`, placed at
  `<checkpoint root>/pretrained/fvd/` and `<checkpoint root>/pretrained/vjepa2_1/`. Their source is
  NanoVVM's `data_processing/tools/prepare_fvd_models.py`, or
  `s5 cat $S3/assets/nanovvm/models/<fvd|vjepa2_1>.tar | tar -C <checkpoint root>/pretrained --strip-components=1 -xf -`.
  Training checks them, and every FVD manifest, before it starts.
- **Outputs.** Results go to `<run>/fvd_eval/step_<step>/<split>/`: `fvd_score.json`,
  `manifest.json`, `population.json` and the generated mp4s. Real features and control videos are
  cached once in `<run>/fvd_cache/`; do not sync that directory.
- **Cost.** The first evaluation also encodes the real clips and both controls with the three
  encoders, so it is the slowest. Later ones generate and encode only the model's videos.
- **Resume.** The `fvd` section is not part of a run's resume identity, so adding or changing it
  does not block resuming a run.

Offline, for any checkpoint (`--num-videos N` for a seeded subset, `--splits val` for one split,
`--ema on|off` to override `fvd.use_ema`):

```bash
torchrun --standalone --nproc_per_node=2 -m vrae.training.nanovvm_video_pred.fvd \
  --config configs/training/nanovvm_video_pred/scaling-ladder/tiny/vrae_eupe_mix5_tiny_xpred_interval3_cond1gen16_bs64_lr2e-4_wd0_ep1.yaml \
  --paths configs/paths.local.yaml \
  --checkpoint ckpts/nanovvm_video_pred/vrae_eupe_mix5_tiny_xpred_interval3_cond1gen16_bs64_lr2e-4_wd0_ep1/checkpoints/step-00010000.pt
```

**6. Sample.** Give one first frame and a seed: an image, or frame `S` of a video. To
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
- `common/sampler.py`: an optional `prefetcher` attribute is told the rank's upcoming order
  (computed lazily) and how far the DataLoader has drawn. The yielded batches are unchanged, and
  without a prefetcher nothing runs.
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

- **FVD controls.** The in-training FVD is NanoVVM's protocol ("Usage" step 5). Its controls are
  `ae_reconstruction` and `static`, both through the frozen V-RAE. NanoVVM's
  `constant_velocity_2frame` extrapolates per-frame AE latents, which a 4-frame-chunk tokenizer
  does not have. NanoVVM's `decode_batch_size` has no counterpart.
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

## Wan2.2 VAE + TI2V DiT

An ablation of the latent space and the DiT on the same task (cond1gen16, mix5, 128x128): the
frozen V-RAE is replaced by the Wan2.2 VAE and the DDT by a DiT with Wan2.2-TI2V-5B's architecture
and first-frame injection. Data, transport, optimization, EMA, CFG dropout, internal guidance,
sampling, FVD and the training loop are the ones above; `stage1.type` and `dit.name` select the
new path, and V-RAE configs (no `stage1.type`, i.e. `vrae`) are unchanged.

### Protocol

- **Latents.** Wan2.2 VAE of Wan-AI/Wan2.2-TI2V-5B (`Wan2.2_VAE.pth`; source vendored from
  Wan-Video/Wan2.2@1ea34ff, `src/vrae/models/video_vae/wan22.py`): 16x spatial, 4x temporal, 48
  channels, posterior mean, Wan's official per-channel normalization, `stage1.precision: bf16`
  (bf16 autocast with fp32 weights, as the V-RAE configs; `fp32` runs it in fp32 with TF32 off,
  about 11x slower on an A100). 128x128 gives an 8x8 latent grid.
- **Joint, causal encoding.** Frame 0 and the 16 future frames are encoded as one 17-frame clip
  (the encoder silently drops frames beyond 4n+1, so any other length is rejected): latent 0 is
  the condition `z0`, latents 1..4 are the target `[B,4,64,48]`. The VAE is causal: `z0` equals
  frame 0 encoded alone and does not depend on the future, but the target does depend on frame
  0's pixels (as in Wan2.2-TI2V; with V-RAE neither depends on the other). Decoding is causal too:
  the raw `z0` is decoded with the 4 latents and frame 0 of the 17 outputs is dropped.
- **First-frame injection (Wan2.2-TI2V `i2v`).** The DiT sequence is `[z0, x_t^1..x_t^4]`
  (5 latent frames x 64 tokens = 320 tokens, RoPE time 0..4, full attention); the `z0` tokens use
  t = 0 and the future tokens the transport's t (TI2V's per-token `mask2 * timestep`). The
  transport noises and supervises only the future; `z0` is passed clean at every training and
  sampling step (TI2V re-inserts the clean latent after every step); the DiT returns the future.
- **Normalization.** The future latents get V-RAE's per-channel train statistics (20000 clips);
  `z0` is normalized with the same statistics before the DiT. The statistics file also records
  `z0`'s own mean/std (`metadata["condition_latent"]`) for comparison.
- **CFG.** With probability `frame_dropout = 0.1`, `z0` is replaced by a learned null latent
  `[1,1,64,48]` (zero-initialized; its tokens keep t = 0); `condition_drop_mask=True` is the
  unconditional branch.
- **Shift.** `sqrt(future_chunks*h*w*C/4096) = sqrt(4*8*8*48/4096) = sqrt(3)`.

### DiT

`wan_ti2v_dit` (`src/vrae/models/dit/wan_ti2v_dit.py`) is Wan2.2's `WanModel` (patch-embedding
Conv3d, sinusoidal embedding of `1000 t` -> Linear-SiLU-Linear, shared `SiLU -> Linear(d, 6d)`,
per-block `[1,6,d]` modulation in fp32, LayerNorm without affine, q/k RMSNorm across heads, 3D RoPE
split 44/42/42 for head_dim 128, GELU(tanh) FFN, fp32 output head modulated by `e`, Wan's
`init_weights`). Deliberate deviations: no text cross-attention / `norm3` / `text_embedding`;
`patch_size` [1,1,1] at 128 px (64 tokens per frame, like Wan's [1,2,2] at 256 px); an independent
internal-guidance head after block `max(1, round(2L/7))`; the null `z0`. A test checks the module
against a literal port of `WanModel.forward` with per-token timesteps.

Ladder (head_dim 128, `ffn_dim = round(14/3 * dim / 128) * 128`, dim/L near 3072/30):

| size | heads | dim | layers | ffn_dim | base_depth | parameters | NanoVVM | TFLOP/step | V-RAE res128 TFLOP/step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | 3 | 384 | 4 | 1792 | 1 | 9.09M | 9.9M | 0.49 | 0.44 |
| small | 6 | 768 | 9 | 3584 | 3 | 75.35M | 78.8M | 4.36 | 3.85 |
| medium | 10 | 1280 | 12 | 6016 | 3 | 275.73M | 265.7M | 16.2 | 12.8 |
| large | 13 | 1664 | 16 | 7808 | 5 | 613.51M | 629.5M | 36.5 | 30.7 |
| xl | 16 | 2048 | 21 | 9600 | 6 | 1209.04M | 1229.2M | 72.4 | 59.5 |
| xxl | 20 | 2560 | 24 | 11904 | 7 | 2139.93M | 2124.1M | 128.6 | 104.8 |

Parameters include the internal-guidance head and the null `z0`. TFLOP/step is forward + backward
at global batch 32 (`torch.utils.flop_counter`), without the extra forward of gradient
checkpointing. The Wan DiT reads 320 tokens (64 of them `z0`), the V-RAE DDT 256.

### Configuration

```text
configs/models/wan22_vae.yaml                stage1: {type: wan22_vae, checkpoint, revision, precision: bf16}
configs/models/dit_wan_ti2v.yaml             dit: {name: wan_ti2v_dit, ...}
configs/training/nanovvm_video_pred/
  _wan22_common.yaml                         stage 1, DiT, transport (shift sqrt(3)), data at 128, sampling
  wan22_mix5.yaml                            the 18 + 18 mix5 sources of eupe_mix5.yaml, latent_normalizer
  scaling-ladder/_mix5_recipe.yaml           training, wandb, fvd of the ladder (also included by _mix5_common.yaml)
  scaling-ladder/<size>/wan22_ti2v_mix5_<size>_xpred_interval3_cond1gen16_res128_bs32_lr2e-4_wd0_ep1.yaml
```

Every section except `stage1`, `dit`, `transport.time_dist_shift`, `latent_normalizer` and the
W&B tags equals the size's `vrae_eupe_mix5_<size>_..._res128_bs32_...` config (a test checks it).
The weights are a symlink:

```bash
mkdir -p $CKPTS/pretrained/wan22_vae   # $CKPTS: checkpoint_write_root (or ckpts/)
ln -s /path/to/Wan2.2-TI2V-5B/Wan2.2_VAE.pth $CKPTS/pretrained/wan22_vae/Wan2.2_VAE.pth
```

### Commands

```bash
CONFIG=configs/training/nanovvm_video_pred/scaling-ladder/tiny/wan22_ti2v_mix5_tiny_xpred_interval3_cond1gen16_res128_bs32_lr2e-4_wd0_ep1.yaml
python -m vrae.training.nanovvm_video_pred.train --config $CONFIG --build-only
python -m vrae.training.nanovvm_video_pred.check --config $CONFIG --paths configs/paths.local.yaml
# Once for all six sizes: the future-latent statistics (and z0's, recorded).
torchrun --standalone --nproc_per_node=2 -m vrae.training.nanovvm_video_pred.latent_stats \
  --config $CONFIG --paths configs/paths.local.yaml
# The Wan2.2 VAE's reconstruction ceiling: ae_reconstruction and static vs real, no DiT.
torchrun --standalone --nproc_per_node=2 -m vrae.training.nanovvm_video_pred.fvd \
  --config $CONFIG --paths configs/paths.local.yaml --controls-only --splits val --output <directory>
torchrun --standalone --nproc_per_node=2 -m vrae.training.nanovvm_video_pred.train \
  --config $CONFIG --paths configs/paths.local.yaml
```

The FVD controls receive frame 0 as well: `reconstruct_future(future, condition_rgb)` encodes and
decodes the 17-frame clip with the Wan2.2 VAE (static: 17 copies of frame 0); the V-RAE predictor
ignores it. The predictor identity names the backend and the weights' hash, so Wan and V-RAE
control caches never mix.

## MeanFlow and improved MeanFlow (Wan path)

MeanFlow (MF; Geng et al. 2025) and improved MeanFlow (iMF; github.com/Lyy-iiis/imeanflow) train
`wan_ti2v_dit` for one- or few-step generation on the Wan2.2 path. `transport.type: meanflow` selects
`MeanFlowTransport` (`src/vrae/models/dit/meanflow.py`); without it the flow-matching transport and
every existing config, model, checkpoint and sample are unchanged (a test compares them bit for bit).

### Objective

Same interpolation as flow matching, `z_t = (1 - t) x + t e` (t = 1 noise), with `v_t = e - x` (no
`t_eps`). The DiT predicts the average velocity `u(z_t, t, h = t - r)` over `[r, t]` (u-prediction)
and one step is `z_r = z_t - (t - r) u`.

```text
dudt = JVP of u(z, t, t - r) along (dz, dt, dr) = (tangent, 1, 0)     # r != t rows only, no gradient
V    = u(z, t, t - r) + (t - r) dudt
loss = mean_i  L_i / sg((L_i + eps)^p),   L_i = sum over the sample's 4*64*48 elements of (V - target)^2
```

- **MF**: target = tangent = `v_t`.
- **iMF**: target = `v_t`, tangent = the model's own instantaneous velocity `V0(z, t)`:
  `u(z, t, 0)` (`tangent: boundary`, default) or an auxiliary v head (`tangent: aux_head`, trained on the
  same target with `aux_v_coeff`).
- **(t, r)**: `max_min` takes the max / min of two draws of the flow-matching shifted logit-normal (shift
  sqrt(3)); `v1` keeps t's flow-matching marginal and draws r separately, then `r = min(t, r)`. Exactly
  `round(ratio_r_neq_t * B)` randomly placed rows of each micro-batch keep r != t, the others get r = t
  (plain u = v flow matching).
- **Internal guidance**: the base head gets the same loss (`base_model_coeff`); one JVP call returns
  both heads' dudt. Sampling applies IG to u as before, from the same forward.
- **CFG** (`cfg.mode`): `none`; `fixed` (MF only, MF Eq. 18-19: target = tangent =
  `omega v_t + (1 - omega) u(z, t, 0 | null)`, sample with `cfg_scale: 1`); `conditioned` (iMF only):
  omega ~ omega^-beta on `[1, omega_max]` and an interval `t_min ~ U[0, 0.5]`, `t_max ~ U[0.5, 1]`
  (`[0, 1]` on r = t rows) are network inputs, target `v_g = v_t + (1 - 1/omega_eff)(v_c - v_u)` with
  `omega_eff = 1` outside the interval, tangent `v_c`. Rows whose `z0` is dropped (`frame_dropout`) get
  target `v_t` and tangent `v_u` (`dropped_tangent: unconditional`) or `v_c` (`conditional`).

**Implementation.** The JVP is `torch.func.jvp` under `torch.no_grad()` on the unwrapped DiT, in fp32
with autocast off and the math SDPA backend (flash / efficient attention have no forward-mode AD); only
dudt is kept. u comes from the one forward per micro-step through the (DDP) model, so gradient
synchronization, `no_sync` accumulation and gradient checkpointing are untouched (the DiT skips
checkpointing under no_grad). The drop mask is drawn once per micro-step and passed to every forward.
For the first `training.grad_sync_check_steps` optimizer steps (MeanFlow default 2) the trainer
all-gathers the largest gradients' norms and stops if the ranks differ.

### DiT inputs

`dit.time_delta_conditioning: true` adds `time_delta_embedding` (Wan's `time_embedding` form,
`sinusoid(1000 h)`; frame 0 at h = 0, the future at h); `dit.guidance_conditioning: true` adds
embeddings of `1 - 1/omega`, `t_min` and `t_max` on every frame. All are summed into the time embedding
before `time_projection`, and their last Linear is zero-initialized, so a new model computes exactly
what the same weights compute without them. `dit.aux_v_head_depth: K` adds K blocks and a head that
fork after block L - K (for `tangent: aux_head`; `base_depth <= L - K`); sampling never runs them.
Extra parameters: `256 d + d^2 + 2 d` per embedding (tiny +0.25M, medium +1.97M; iMF-CFG four times).

### Configuration

```text
configs/training/nanovvm_video_pred/meanflow/
  _imf.yaml       iMF, boundary tangent, ratio 0.5, no CFG; all keys and defaults below
  _mf.yaml        MF: variant mf, tangent conditional, ratio 0.25 (cfg.mode fixed optional)
  _imf_cfg.yaml   iMF with cfg.mode conditioned, dit.guidance_conditioning, sampling.cfg_scale 2.0 (a placeholder)
  _fmctrl.yaml    control: MF with ratio_r_neq_t 0 = u (= v) flow matching with MF's loss weighting
scaling-ladder/<size>/wan22_ti2v_mix5_<size>_{mf,imf}_upred_interval3_cond1gen16_res128_bs32_lr2e-4_wd0_ep1.yaml   all six sizes
scaling-ladder/<size>/wan22_ti2v_mix5_<size>_{imfcfg,fmctrl}_upred_..._ep1.yaml                                   tiny, small, medium
```

Each ladder file includes `../../wan22_mix5.yaml`, `../_mix5_recipe.yaml`, then its fragment, and
equals the size's `..._xpred_...` config except `transport`, the DiT's MeanFlow keys, `sampling`,
`run_name` and the W&B tags (a test checks it). The fragment sets `t_eps: null` to remove the
flow-matching `t_eps` that `_wan22_common.yaml` merges in.

| key | default | meaning |
| --- | --- | --- |
| `transport.variant` | `imf` | `mf` or `imf` |
| `transport.prediction` | `u` | the only value |
| `transport.tr_sampler` | `max_min` | or `v1` (with `r_time_dist_type` / `r_time_dist_shift`, null = t's) |
| `transport.ratio_r_neq_t` | 0.5 (MF 0.25, control 0) | fraction of rows with r != t |
| `transport.adaptive_weight` | `{p: 1.0, eps: 0.01}` | `L / sg((L + eps)^p)` |
| `transport.tangent` | `boundary` (MF `conditional`) | iMF: `boundary` or `aux_head` |
| `transport.base_model_coeff`, `aux_v_coeff` | 1.0, 1.0 | weights of the base-head and aux-v losses |
| `transport.cfg` | `{mode: none, omega: 2.0, omega_max: 8.0, beta: 1.0, interval: true, dropped_tangent: unconditional}` | see Objective |
| `transport.jvp` | `{dtype: float32, attention_backend: math, neq_subset_only: true}` | runtime only: `autocast` keeps the outer bf16; not in checkpoint metadata, so a run resumes with other values |
| `sampling.steps` | 1 | Euler steps on the grid |
| `sampling.guidance_interval` | `[0.0, 1.0]` | iMF-CFG network input |
| `sampling.time_grid` | `uniform` | `linspace(1, 0, N + 1)`, or `shifted` by `time_dist_shift` |
| `training.grad_sync_check_steps` | 2 (flow matching 0) | DDP gradient check, ignored by resume |

Model metadata records the transport's algorithm fields (`type: meanflow`, variant, distributions,
ratio, weights, tangent, cfg) and the DiT's MeanFlow keys; resuming with a different algorithm field
is refused, with different `jvp` options it is not. W&B adds `loss/train_aux_v`, `mse/train_full`,
`mse/train_base` (unweighted per-element MSE of V against the target, comparable to flow matching),
`mse/train_fm` / `mse/train_mf` (the r = t / r < t rows) and `stats/dudt_rms`. The adaptive weight
makes the gradient roughly `grad(mse) / mse`, so its scale differs from flow matching's; `clip_grad`
stays at the recipe's 1.0 and `optimizer/grad_norm` is the pre-clip norm.

### Differences from the official implementations

- The network reads t and h (official iMF reads h only): Wan's per-frame t embedding is what tells the
  clean `z0` frame (t = 0) from the noisy future frames.
- The r != t rows are a fixed number of random positions (official iMF: the first half of the batch).
- Dropped rows use the tangent `v_u` by default (official: `v_c`; `dropped_tangent: conditional`).
- Time distribution: the flow-matching shifted logit-normal (logit-normal(0, 1), shift sqrt(3)), not
  the official logit-normal(-0.4, 1).
- iMF defaults to the boundary tangent `u(z, t, 0)`, not the official 8-block auxiliary v head
  (`tangent: aux_head` with `dit.aux_v_head_depth`).
- The guidance scale and interval are added to Wan's time embedding, not in-context tokens.
- Not implemented: pixel MeanFlow (x-prediction converted to u, perceptual losses, noise scale), a Triton
  JVP kernel, `torch.compile`, FSDP.

### Commands

```bash
CONFIG=configs/training/nanovvm_video_pred/scaling-ladder/tiny/wan22_ti2v_mix5_tiny_imf_upred_interval3_cond1gen16_res128_bs32_lr2e-4_wd0_ep1.yaml
python -m vrae.training.nanovvm_video_pred.train --config $CONFIG --build-only
torchrun --standalone --nproc_per_node=2 -m vrae.training.nanovvm_video_pred.train \
  --config $CONFIG --paths configs/paths.local.yaml
# 1, 2 or more steps; iMF-CFG: sweep omega and the interval (network inputs, one forward per step).
python -m vrae.training.nanovvm_video_pred.sample --config $CONFIG --paths configs/paths.local.yaml \
  --checkpoint ckpts/nanovvm_video_pred/<run>/checkpoints/step-00010000.pt --split val --count 64 \
  --output <directory> --steps 2 [--cfg-scale 3.0 --guidance-interval 0.1 0.9] [--time-grid shifted]
torchrun --standalone --nproc_per_node=2 -m vrae.training.nanovvm_video_pred.fvd --config $CONFIG \
  --paths configs/paths.local.yaml --checkpoint <checkpoint> [--guidance-interval 0.1 0.9 --time-grid uniform]
```

The latent statistics, weights and data are the Wan path's above. With `cfg.mode: none`, a
`cfg_scale != 1` at sampling applies naive CFG to u (two forwards per step, a heuristic; warned once).

## Tests

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m unittest \
  tests.test_nanovvm_video_pred tests.test_nanovvm_scaling_ladder tests.test_remote_video_storage \
  tests.test_nanovvm_fvd tests.test_nanovvm_wan22 tests.test_nanovvm_meanflow -v
```

The tests run on CPU with synthetic videos and small weights. Hide the GPUs on a GPU node, as above,
because the end-to-end test keeps its tiny model on the CPU. They cover:

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
  - each ladder config against the size table and the recipe;
- for remote video storage (`test_remote_video_storage.py`, ported from NanoVVM's
  `tests/test_remote_video.py`; tiny tars through `file://`, no AWS access):
  - tar indexes (GNU, PAX and ustar names; links; traversal, duplicate and compressed archives
    refused), resumable builds, and stale archives;
  - the storage file (environment expansion, roots below the data root, the repository's
    `adobe_s3.yaml` covering every ladder root), and that local runs never import botocore;
  - transport errors and retries, and S3 credential isolation and range checks (with botocore);
  - the node cache: one download per video across processes, no eviction of leased files, the byte
    budget, killed producers, and failed downloads;
  - remote clips equal to local ones, prefetch that leaves the batches unchanged, the checker, and a
    training run with workers whose log has the `cache/*` metrics and whose W&B log has an in-loop FVD;
- for FVD (`test_nanovvm_fvd.py`): the ladder's `fvd` section against NanoVVM's protocol (the
  train_eval manifests mirror the val ones), rejected settings, resume identity, path resolution,
  and the engine end to end with a toy encoder: NanoVVM's pooled and per-dataset keys, the 17-frame
  mp4s, cached controls and features on a later step, the same noise per clip, and the caller's RNG;
- for the Wan2.2 path (`test_nanovvm_wan22.py`; a narrow random `WanVAE_` at 32x32, no weights): the
  loader's VAE config and official normalization, the causal `z0` (equal to frame 0 alone, unchanged
  by the future), rejection of clips other than 17 frames, causal decoding with `z0`, the VAE's own
  precision (fp32, or bf16 autocast on CUDA) under an outer autocast;
  the DiT against a literal port of Wan2.2's forward, `z0` at t = 0, the null `z0`,
  aliases, the base head, exact gradient checkpointing, and the transport; the ladder's parameter
  counts and the 5B formula; the six configs (`--build-only`, shift sqrt(3), equality with the
  EUPE res128 twins, rejected settings); statistics, train, resume and sampling end to end; and the
  controls-only FVD with frame 0 passed to the controls.
- for MeanFlow (`test_nanovvm_meanflow.py`): the default DiT's keys, parameter count and metadata and
  the flow-matching transport metadata unchanged; zero-initialized h / guidance embeddings that leave the
  output bit-identical; `torch.func.jvp` through the DiT against float64 central differences (with and
  without drop, base head and guidance) and under no_grad in training mode with checkpointing; the
  (t, r) samplers; the losses' reduction to velocity flow matching, the tangents, sub-batch vs
  full-batch JVP; one drop mask per step; gradients (dudt constant, every parameter reached, one
  forward through the wrapped model); omega / interval sampling, the guided targets against a
  closed-form mock, MF fixed CFG; the sampler (1 step = `z1 - u(z1, 1, 1)`, N steps, CFG modes); the
  engine, validation, metadata and resume rules; DDP on two gloo ranks; the 18 ladder files; and
  train, resume, 1- and 2-step sampling and FVD end to end for MF and iMF-CFG with the aux head.

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
| `src/vrae/training/nanovvm_video_pred/check.py` | data path, manifest and remote storage check |
| `src/vrae/training/nanovvm_video_pred/fvd.py` | the FVD section's paths, the V-RAE predictor, in-training and offline FVD |
| `src/vrae/evaluation/nanovvm_fvd/` | NanoVVM's FVD engine: `metrics.py` and `vendor/` vendored from NanoVVM's `eval/`; `fvd.py` is NanoVVM's `eval/fvd.py` with V-RAE generation and controls |
| `src/vrae/storage/` | remote video storage, vendored from NanoVVM: tar indexes (`tar_index.py`), S3 and `file://` reads (`remote_io.py`), the node cache (`video_cache.py`), prefetch, `backend.py`; the storage file (`config.py`), loop metrics (`metrics.py`), the CLI (`remote_video.py`) |
| `configs/storage/adobe_s3.yaml` | mix5 tars on S3 for AD nodes |
| `src/vrae/models/dit/first_frame_prediction_dit.py` | the DiT (a subclass of `VRAEVideoDiT`) |
| `src/vrae/models/dit/conditioning.py` | `FirstFrameConditionAdapter` |
| `src/vrae/models/encoders/base.py` | `encode_images` single-image entry |
| `src/vrae/models/video_vae/` | the Wan2.2 VAE source (`wan22.py`) and its backend (`backend.py`) |
| `src/vrae/models/dit/wan_ti2v_dit.py` | the Wan2.2-TI2V DiT (`wan_ti2v_dit`) |
| `src/vrae/models/dit/meanflow.py` | `MeanFlowTransport`: MF / iMF (t, r) sampling, JVP, losses, CFG, few-step sampler |

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
- Besides FVD, val sampling saves prediction/target pairs
  for external metrics.
