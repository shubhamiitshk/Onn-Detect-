"""Diagnostic single-SD-VAE pixel-MSE reconstruction experiment.

This is not the official AEROBLADE multi-autoencoder/LPIPS implementation and
cannot substantiate a paper comparison. Requires separately licensed weights
and image data; it has not been executed on a detection benchmark here.
"""

import argparse
import glob
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src")))


def load_vae(device: str):
    """SD VAE in eval mode. float32: reconstruction error needs the precision;
    this runs once per image, not per frame, so the cost is acceptable."""
    import torch
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def list_images(folder: str, limit: int) -> list[str]:
    files = sorted(
        g for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp")
        for g in glob.glob(os.path.join(folder, ext))
    )
    return files[:limit]


def reconstruction_error(vae, img, device: str) -> float:
    """LPIPS if available, else MSE. Higher = worse reconstruction = real."""
    import torch
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    from PIL import Image
    x = tf(Image.open(img).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        z = posterior.mean  # deterministic; sampling adds noise to the metric
        recon = vae.decode(z).sample
    err = float(((recon - x) ** 2).mean().item())
    return err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(ROOT_DIR, "datasets", "genimage", "test"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(ROOT_DIR, "ml", "aeroblade_baseline.json"))
    args = ap.parse_args()

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    try:
        from training.metrics import (
            assert_scores_sane, compute_auc, compute_eer, polarity_report,
        )
    except ImportError:
        sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))
        from training.metrics import (
            assert_scores_sane, compute_auc, compute_eer, polarity_report,
        )

    vae = load_vae(device)

    errors, labels, files = [], [], []
    for cls, label in (("real", 0), ("fake", 1)):
        for p in list_images(os.path.join(args.images, cls), args.limit):
            try:
                errors.append(reconstruction_error(vae, p, device))
                labels.append(label)
                files.append(p)
            except Exception as e:
                print(f"skip {p}: {e}")
            if (len(errors) + 1) % 50 == 0:
                print(f"  scored {len(errors)}...")

    y = np.asarray(labels)
    # INVERSION: reconstruction error is higher for REAL. Negate so that the
    # repo convention (higher = more fake) holds for all downstream metrics.
    s = -np.asarray(errors)
    assert_scores_sane(y, s, context="aeroblade")
    auc = compute_auc(y, s, context="aeroblade")
    eer = compute_eer(y, s, context="aeroblade")
    print("=" * 50)
    print(f"AEROBLADE baseline: n={len(y)} AUC={auc:.4f} EER={eer:.4f}")
    print("(score = -reconstruction_error; higher = more fake, per convention)")
    print("=" * 50)
    print("polarity:", polarity_report(y, s))

    import json

    with open(args.out, "w") as f:
        json.dump({
            "method": "aeroblade-sd-vae-ft-mse-negmse",
            "n": int(len(y)),
            "auc": float(auc),
            "eer": float(eer),
            "score_semantics": "score = -MSE(VAE(x), x); higher = more fake",
            "images": args.images,
        }, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()