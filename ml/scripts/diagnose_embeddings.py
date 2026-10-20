"""Why does the frozen MobileCLIP2-S0 tower carry no real-vs-AI signal?

Prints embedding statistics that separate the two possible causes:
  * a DEAD tower (all embeddings near-identical, low pairwise variance), or
  * a LIVE tower whose features simply do not encode this distinction.
"""
from pathlib import Path

import numpy as np

CACHE = Path(r"D:\onn_data\runs\release_v6\checkpoints\frozen_probe_embeddings_cache.npz")


def main():
    with np.load(CACHE) as d:
        X = {n[2:]: d[n] for n in d.files if n.startswith("X_")}
        y = {n[2:]: d[n] for n in d.files if n.startswith("y_")}

    stacking = X["stacking"]
    labels = y["stacking"]
    print(f"stacking: {stacking.shape}")

    norms = np.linalg.norm(stacking, axis=1)
    print(f"row norms: mean {norms.mean():.5f}  min {norms.min():.5f}  max {norms.max():.5f}")

    per_dim_std = stacking.std(axis=0)
    print(f"per-dim std across images: mean {per_dim_std.mean():.5f} "
          f"min {per_dim_std.min():.5f} max {per_dim_std.max():.5f}")
    print(f"near-constant dims (<1e-4): {float((per_dim_std < 1e-4).mean()):.3f}")

    similarity = stacking @ stacking.T
    off = similarity[~np.eye(len(stacking), dtype=bool)]
    print(f"pairwise cosine: mean {off.mean():.4f} std {off.std():.4f} "
          f"min {off.min():.4f} max {off.max():.4f}")

    real = stacking[labels == 0]
    fake = stacking[labels == 1]
    real_centroid = real.mean(axis=0)
    fake_centroid = fake.mean(axis=0)
    cosine = float(real_centroid @ fake_centroid /
                   (np.linalg.norm(real_centroid) * np.linalg.norm(fake_centroid)))
    print(f"real vs fake centroid cosine: {cosine:.4f}  (1.0 = identical)")

    # Within-class vs between-class nearest neighbour similarity.
    for name, split in (("stacking", "stacking"), ("calibration", "calibration")):
        embeddings = X[split]
        sim = embeddings @ embeddings.T
        np.fill_diagonal(sim, -2.0)
        nearest = sim.max(axis=1)
        same = (labels if split == "stacking" else y[split])
        neighbour_same = (same == (same[np.argmax(sim, axis=1)]))
        print(f"{name}: mean NN cosine {nearest.mean():.4f}; "
              f"NN same-label rate {neighbour_same.mean():.4f} (chance ~0.5)")


if __name__ == "__main__":
    main()
