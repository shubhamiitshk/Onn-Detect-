"""Tests for the Keras mirror of a frozen vision tower.

The mirror's DINOv2 branch was written against the transformers 4.x API and
crashed on the pinned 5.20.2, so it was never exercised. These tests build
the mirror for whichever towers are available locally and compare it with
the PyTorch reference for the SAME probe.

They skip when the model or the measured embedding cache is absent, because
the embedding cache lives outside the repository.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from export.keras_vfm_probe import build_keras_vfm_probe  # noqa: E402

CACHE_DIR = Path(r"D:\onn_data\runs\vfm_probe_eval")


def _available(spec) -> bool:
    """True when both the model and the measured embedding cache are present."""
    if not (CACHE_DIR / f"embeddings_{spec.key}.npz").is_file():
        return False
    try:
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def _images(spec, count=2):
    from PIL import Image

    out = []
    for seed in range(count):
        rng = np.random.default_rng(seed)
        arr = (rng.random((240, 200, 3)) * 255).astype(np.uint8)
        out.append(spec.preprocess(Image.fromarray(arr)))
    return np.stack(out).astype(np.float32)


class KerasMirrorParityTest(unittest.TestCase):
    """The mirror must reproduce the PyTorch tower to float32 noise."""

    def _check(self, repo: str) -> None:
        from evaluate_vfm_probes import TOWERS, fit_logistic_probe

        spec = TOWERS[repo]
        if not _available(spec):
            self.skipTest(f"{repo}: model or embedding cache not available locally")
        with np.load(CACHE_DIR / f"embeddings_{spec.key}.npz") as data:
            X, y = data["X_stacking"], data["y_stacking"]
        weight, bias, mean, std = fit_logistic_probe(X, y, X, 30.0)

        tower = spec.load()
        inputs_chw = _images(spec)
        features = spec.embed(tower, inputs_chw)
        z = (features - mean) / std
        reference = 1.0 / (1.0 + np.exp(-(z @ weight + bias)))

        model = build_keras_vfm_probe(
            tower,
            probe_weight=weight,
            probe_bias=float(bias),
            input_size=spec.size,
            pool_mode=spec.pooling,
            l2_normalize=True,
            feature_mean=mean,
            feature_std=std,
        )
        keras = np.array([
            float(model(x[np.newaxis], training=False).numpy()[0, 0])
            for x in np.transpose(inputs_chw, (0, 2, 3, 1))
        ])
        error = float(np.max(np.abs(keras - reference)))
        self.assertLess(error, 1e-4, f"{repo}: the mirror diverges by {error:.3e}")

    def test_siglip_base_patch16_224(self):
        self._check("google/siglip-base-patch16-224")

    def test_dinov2_with_registers_small(self):
        # The transformers 5.x attention layout, the SwiGLU MLP and the
        # `cls_mean` pooling contract. This branch did not exist before:
        # it raised an AttributeError on the pinned transformers.
        self._check("facebook/dinov2-with-registers-small")


if __name__ == "__main__":
    unittest.main()
