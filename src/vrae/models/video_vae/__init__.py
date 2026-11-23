"""Frozen video VAEs used as stage-1 latent spaces (``stage1.type``).

``wan22.py`` is the Wan2.2 VAE source (Wan-Video/Wan2.2 commit 1ea34ff, ``wan/modules/vae2_2.py``,
Apache-2.0); ``backend.py`` wraps it for nanovvm_video_pred. Model weights are not included.
"""

from vrae.models.video_vae.backend import (
    BACKEND_TYPE,
    WAN22_REVISION,
    Wan22LatentBackend,
    load_wan22_backend,
)

__all__ = ["BACKEND_TYPE", "WAN22_REVISION", "Wan22LatentBackend", "load_wan22_backend"]
