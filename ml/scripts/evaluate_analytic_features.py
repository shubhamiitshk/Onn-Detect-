"""Measure the analytic (hand-crafted) fusion features on the sealed test split.

Why this exists
---------------
Two of the four LIVE fusion features are at or below chance:

    current_fft_score   standalone AUC 0.5013   (chance)
    current_kl_score    standalone AUC 0.4120   (BELOW chance - inverted)

yet the shipped tree splits on them in 26 and 36 nodes respectively
(``assets/fusion/XGBoostMetaClassifier.kt``). Half the model's capacity is
spent on features that carry no usable signal.

The CVPR 2026 phase-spectrum result says the AMPLITUDE spectrum is what
JPEG destroys while the PHASE survives, so a phase-only feature should
degrade far less under re-encoding than a power-spectrum slope.

This script measures every analytic feature on the sealed test split, on the
ORIGINAL files and on a LAUNDERED copy (uniform 512 px JPEG Q85, the
condition the shipped fusion already reports), so the two can be compared
directly.

Usage
-----
    python -m ml.scripts.evaluate_analytic_features \\
        --data-root D:\\onn_data\\openfake_visual \\
        --output D:\\onn_data\\runs\\vfm_probe_eval\\analytic_features.json
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
from meta_classifier.feature_extractor import (  # noqa: E402
    fft_features,
    kl_score,
    load_baseline,
    to_gray_u8,
)
from meta_classifier.phase_features import phase_candidates  # noqa: E402
from training.metrics import compute_auc, compute_eer  # noqa: E402

# The KL divergence is measured against the release's calibrated natural
# difference histogram; without it the feature is meaningless.
DEFAULT_BASELINE = Path(
    r"C:\Users\PLEXTEK\Desktop\Onn-Detect-\android\app\src\main\assets\constants\natural_diff_histogram.json"
)


def analytic_features(img: Image.Image, baseline: np.ndarray) -> dict[str, float]:
    """Every hand-crafted feature the fusion can consume.

    The fft/flatness pair comes from the SHIPPED implementation
    (`fft_features`), so what is measured here is exactly what the fusion
    consumes.
    """
    fft_score, flatness = fft_features(img)
    kl, _ = kl_score(to_gray_u8(img), baseline)
    out = {
        "fft_score": float(fft_score),
        "flatness_score": float(flatness),
        "kl_score": float(kl),
    }
    out.update({k: float(v) for k, v in phase_candidates(img).items()})
    return out


def launder(img: Image.Image, max_side: int = 512, quality: int = 85) -> Image.Image:
    """Uniform re-encode, matching the shipped 'laundered' evaluation."""
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((max(16, int(round(w * scale))), max(16, int(round(h * scale)))),
                         Image.Resampling.BILINEAR)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
        tmp = handle.name
    img.convert("RGB").save(tmp, format="JPEG", quality=quality)
    out = Image.open(tmp).convert("RGB")
    Path(tmp).unlink(missing_ok=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    samples = discover_samples(data_root, args.split, IMAGE_EXTS)
    labels = np.asarray([label for _, label in samples], dtype=np.int64)
    baseline = load_baseline(str(DEFAULT_BASELINE))
    print(f"baseline: {DEFAULT_BASELINE}")

    rows_clean: list[dict[str, float]] = []
    rows_laundered: list[dict[str, float]] = []
    for index, (path, _) in enumerate(samples):
        img = decode_image(str(path)).image
        rows_clean.append(analytic_features(img, baseline))
        rows_laundered.append(analytic_features(launder(img), baseline))
        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(samples)}")

    def summarise(rows: list[dict[str, float]]) -> dict:
        out = {}
        for key in rows[0]:
            scores = np.asarray([r[key] for r in rows], dtype=np.float64)
            entry = {"std": float(scores.std()), "mean": float(scores.mean())}
            if scores.std() < 1e-9:
                # A numerically constant feature carries no information. The
                # metrics module refuses to report an AUC for it, which is
                # the right behaviour - and it means the fusion tree is
                # splitting on a value that never varies.
                entry.update({"auc": None, "eer": None, "degenerate": "constant"})
            else:
                entry.update({
                    "auc": float(compute_auc(labels, scores)),
                    "eer": float(compute_eer(labels, scores)),
                    "degenerate": None,
                })
            out[key] = entry
        return out

    clean = summarise(rows_clean)
    laundered = summarise(rows_laundered)

    report = {
        "data_root": str(data_root),
        "split": args.split,
        "n": len(samples),
        "clean": clean,
        "laundered_512_q85": laundered,
    }
    print(f"\n{'feature':<20}{'clean AUC':>12}{'laund AUC':>12}{'delta':>10}{'std':>12}")
    print("-" * 66)
    for key in clean:
        c, l = clean[key]["auc"], laundered[key]["auc"]
        cs = "constant" if c is None else f"{c:.4f}"
        ls = "constant" if l is None else f"{l:.4f}"
        delta = "n/a" if (c is None or l is None) else f"{l - c:.4f}"
        print(f"{key:<20}{cs:>12}{ls:>12}{delta:>10}{clean[key]['std']:>12.2e}")

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
