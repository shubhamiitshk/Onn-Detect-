"""The frontend contract of the frozen SigLIP/DINOv2 tower.

This lives OUTSIDE ``ml/src/data`` on purpose. The files under
``ml/src/data``, ``ml/src/models``, ``ml/src/training`` and ``ml/src/utils``
are the TRAINING-SOURCE areas: their bytes are hashed into every trained
model's provenance contract, so changing one invalidates every base
checkpoint. The VFM frontend is an EXPORT-time concern - the app and the
export path need it, the training of the base models does not - and putting
it in the training area would mean every future release recorded a
spurious provenance drift for a file no base model depends on.

The contract must equal ``Tower.frontend_id()`` in
``ml/scripts/evaluate_vfm_probes.py`` and ``VfmProbeAnalyzer.FRONTEND_ID``
in the Android app: a divergence between the three means the fusion was
trained on a different preprocessing than the app runs.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

__all__ = [
    "VFM_MEAN",
    "VFM_SIZE",
    "VFM_STD",
    "vfm_frontend_id",
    "vfm_image_tensor",
]

# SigLIP's own normalization: (x / 255 - 0.5) / 0.5.
VFM_SIZE = 224
VFM_MEAN = (0.5, 0.5, 0.5)
VFM_STD = (0.5, 0.5, 0.5)


def vfm_frontend_id(size=VFM_SIZE, repo="siglip-base-patch16-224", pooling="siglip"):
    """Versioned identifier for the SigLIP/DINOv2 frozen-tower frontend.

    This must equal ``Tower.frontend_id()`` in
    ``ml/scripts/evaluate_vfm_probes.py`` and ``VfmProbeAnalyzer.FRONTEND_ID``
    in the Android app: a divergence between the three means the fusion was
    trained on a different preprocessing than the app runs.
    """
    return f"vfm-{pooling}{size}-{repo}-bicubic-clamp-meanstd-float32"


def vfm_image_tensor(image, size=VFM_SIZE, mean=VFM_MEAN, std=VFM_STD):
    """Bicubic square resize + SigLIP mean/std normalization -> CHW float32.

    Mirrors the Android ``VfmProbeAnalyzer`` byte-for-byte. The resampling
    filter is bicubic because that is the contract the tower's embeddings
    were measured under.
    """
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1))
