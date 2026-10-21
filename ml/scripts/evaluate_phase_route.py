"""Does the phase-only reconstruction carry usable signal?

Why this exists
---------------
The CVPR 2026 phase-spectrum result feeds the phase into a CROSS-MODAL
TRANSFORMER, not into a hand-crafted scalar. The measurement in
``evaluate_analytic_features.py`` shows why that matters: every scalar
summary of the phase-only reconstruction is either numerically CONSTANT
(the phase of a natural image is nearly uniform, so the reconstruction is
nearly the same image for every input) or BELOW chance.

The one way a phase-derived signal could survive a scalar pipeline is if
the EXISTING residual detector can read it: the phase-only reconstruction
is a noise-like image, which is exactly what Engine A's SRM/NPR filters
are built to detect.

This script measures:
  * Engine A's shipped score on the ORIGINAL image (the baseline);
  * Engine A's shipped score on the PHASE-ONLY RECONSTRUCTION of the same
    image, and on its laundered version;
  * the standalone AUC of each.

If the phase route is dead, this says so with a number instead of an
assertion.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.contracts import IMAGE_EXTS, discover_samples  # noqa: E402
from data.image_frontend import decode_image  # noqa: E402
from meta_classifier.phase_features import phase_only_image, _grey01  # noqa: E402
from training.metrics import compute_auc, compute_eer  # noqa: E402
from validate_export import _run_tflite  # noqa: E402

REPO = ROOT.parent
DEFAULT_SRM = REPO / "android" / "app" / "src" / "main" / "assets" / "models" / "srm_forensic_cnn_int8.tflite"


def phase_reconstruction_rgb(img: Image.Image, size: int = 128) -> np.ndarray:
    """The phase-only reconstruction, rescaled to [0, 255] as a grey RGB image."""
    grey = _grey01(img)
    reconstruction, _ = phase_only_image(grey)
    scaled = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min() + 1e-12)
    out = (scaled * 255.0).astype(np.uint8)
    rgb = np.stack([out, out, out], axis=-1)
    return rgb


def score_with_srm(srm_path: Path, rgb: np.ndarray, size: int = 128) -> float:
    resized = np.asarray(
        Image.fromarray(rgb, "RGB").resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    tensor = np.transpose(resized, (2, 0, 1))[None, ...]
    return float(_run_tflite(str(srm_path), np.transpose(tensor, (0, 2, 3, 1))))


def launder_rgb(rgb: np.ndarray, quality: int = 85) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
        tmp = handle.name
    Image.fromarray(rgb, "RGB").save(tmp, format="JPEG", quality=quality)
    out = np.asarray(Image.open(tmp).convert("RGB"))
    Path(tmp).unlink(missing_ok=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--srm", default=str(DEFAULT_SRM))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    srm_path = Path(args.srm)
    samples = discover_samples(data_root, args.split, IMAGE_EXTS)
    labels = np.asarray([label for _, label in samples], dtype=np.int64)

    original, phase, phase_laundered = [], [], []
    for index, (path, _) in enumerate(samples):
        img = decode_image(str(path)).image.convert("RGB")
        original.append(score_with_srm(srm_path, np.asarray(img)))
        reconstruction = phase_reconstruction_rgb(img)
        phase.append(score_with_srm(srm_path, reconstruction))
        phase_laundered.append(score_with_srm(srm_path, launder_rgb(reconstruction)))
        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(samples)}")

    report = {
        "split": args.split,
        "n": len(samples),
        "srm_on_original": {
            "auc": float(compute_auc(labels, original)),
            "eer": float(compute_eer(labels, original)),
            "std": float(np.std(original)),
        },
        "srm_on_phase_reconstruction": {
            "auc": float(compute_auc(labels, phase)),
            "eer": float(compute_eer(labels, phase)),
            "std": float(np.std(phase)),
        },
        "srm_on_phase_reconstruction_laundered": {
            "auc": float(compute_auc(labels, phase_laundered)),
            "eer": float(compute_eer(labels, phase_laundered)),
            "std": float(np.std(phase_laundered)),
        },
    }
    print()
    for key in ("srm_on_original", "srm_on_phase_reconstruction",
                "srm_on_phase_reconstruction_laundered"):
        entry = report[key]
        print(f"{key:<40} AUC {entry['auc']:.4f}  EER {entry['eer']:.4f}  std {entry['std']:.3e}")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
