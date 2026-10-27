"""Measure frozen VFM probes under the repo's split discipline.

Why this exists
---------------
The shipped Engine C is a distilled CNN student trained on a few thousand
images. The repo already established that the tower it was distilled from
(MobileCLIP2-S0) reaches only **0.5145 held-out AUC - chance** - while the
SigLIP teacher reaches **0.8491** (``docs/DUAL_ENGINE_CASCADE.md:197-215``).

Two questions were still unanswered on the real data:

  1. How much of the shipped student's performance is lost to distillation?
  2. Does a frozen VFM probe avoid the **acquisition-source shortcut** that
     the shipped detector has (44% of real LAION images flagged synthetic)?

This script answers both, on the real dataset, with the repo's split
discipline:

  * the probe is fitted on ``stacking`` ONLY;
  * the regularisation strength is selected on ``fusion_val`` ONLY;
  * ``calibration`` and ``test`` are read once, at the end.

It additionally reports the per-source false-positive rate and the
per-generator true-positive rate, because a high AUC that comes from a
source shortcut is not a detector.

Usage
-----
    python -m ml.scripts.evaluate_vfm_probes \\
        --data-root D:\\onn_data\\openfake_visual \\
        --output-dir D:\\onn_data\\runs\\vfm_probe_eval \\
        --towers facebook/dinov2-with-registers-small google/siglip-base-patch16-224
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.contracts import IMAGE_EXTS, discover_samples  # noqa: E402
from data.image_frontend import decode_image  # noqa: E402
from training.metrics import compute_auc, compute_eer  # noqa: E402

SPLITS = ("stacking", "fusion_val", "calibration", "test")


# ---------------------------------------------------------------------------
# Tower registry
# ---------------------------------------------------------------------------

class Tower:
    """A frozen vision tower with an explicit, shippable frontend."""

    def __init__(self, repo: str, model_cls: str, size: int, mean, std, pooling: str):
        self.repo = repo
        self.model_cls = model_cls
        self.size = size
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.pooling = pooling

    @property
    def key(self) -> str:
        return self.repo.replace("/", "__")

    def frontend_id(self) -> str:
        return (
            f"vfm-{self.pooling}{self.size}-{self.repo.split('/')[-1]}"
            f"-bicubic-clamp-meanstd-float32"
        )

    def preprocess(self, img: Image.Image) -> np.ndarray:
        """Square BICUBIC resize + per-channel (x/255 - mean) / std -> CHW.

        The resampling filter is BICUBIC because the shipped Engine C
        frontend already established bicubic as the contract, and because a
        bilinear resize is not the filter these towers were trained with.
        """
        resized = img.convert("RGB").resize((self.size, self.size), Image.Resampling.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        return np.transpose(arr, (2, 0, 1))

    def load(self):
        import torch
        import transformers

        model = getattr(transformers, self.model_cls).from_pretrained(self.repo)
        model.eval()
        return model

    def embed(self, model, batch: "np.ndarray") -> "np.ndarray":
        import torch

        with torch.no_grad():
            out = model(pixel_values=torch.from_numpy(batch))
            if self.pooling == "siglip":
                # SigLIP's canonical embedding is the pooled output of its
                # attention-pooling head, not a CLS token.
                feats = out.pooler_output
            elif self.pooling == "cls":
                feats = out.last_hidden_state[:, 0]
            elif self.pooling == "cls_mean":
                # DINOv2's linear head concatenates the CLS token with the
                # mean of the patch tokens.
                states = out.last_hidden_state
                feats = torch.cat([states[:, 0], states[:, 1:].mean(dim=1)], dim=-1)
            else:  # pragma: no cover
                raise ValueError(f"Unknown pooling {self.pooling}")
            emb = feats.float().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        return emb / norms


TOWERS = {
    "facebook/dinov2-with-registers-small": Tower(
        "facebook/dinov2-with-registers-small", "Dinov2Model", 224,
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), "cls_mean",
    ),
    "facebook/dinov2-base": Tower(
        "facebook/dinov2-base", "Dinov2Model", 224,
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), "cls_mean",
    ),
    "google/siglip-base-patch16-224": Tower(
        "google/siglip-base-patch16-224", "SiglipVisionModel", 224,
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), "siglip",
    ),
    "google/siglip-so400m-patch14-384": Tower(
        "google/siglip-so400m-patch14-384", "SiglipVisionModel", 384,
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), "siglip",
    ),
    "HuggingFaceTB/SmolVLM-500M-Instruct": Tower(
        "HuggingFaceTB/SmolVLM-500M-Instruct", "AutoModelForImageTextToText", 512,
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), "siglip",
    ),
}


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(tower: Tower, data_root: Path, cache_path: Path, batch_size: int = 8):
    """Embed every split once and cache the result."""
    if cache_path.is_file():
        with np.load(cache_path) as data:
            return {k: data[k] for k in data.files}

    import torch

    model = tower.load()
    out: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for split in SPLITS:
        samples = discover_samples(data_root, split, IMAGE_EXTS)
        embeds = []
        started = time.time()
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start:start + batch_size]
            tensors = np.stack([
                tower.preprocess(decode_image(str(path)).image) for path, _ in batch_samples
            ])
            embeds.append(tower.embed(model, tensors))
            done = min(start + batch_size, len(samples))
            if done % 200 < batch_size or done == len(samples):
                rate = done / max(time.time() - started, 1e-6)
                print(f"  [{tower.repo}] {split}: {done}/{len(samples)} "
                      f"({rate:.1f} img/s, eta {(len(samples) - done) / max(rate, 1e-6):.0f}s)")
        out[f"X_{split}"] = np.concatenate(embeds, axis=0)
        labels[f"y_{split}"] = np.asarray([label for _, label in samples], dtype=np.int64)
    out.update(labels)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **out)
    return out


# ---------------------------------------------------------------------------
# Probe fitting
# ---------------------------------------------------------------------------

def fit_logistic_probe(X_fit, y_fit, X_apply, l2_reg: float):
    """L2-regularized logistic probe. Returns (weights, bias, mean, std)."""
    mean = X_fit.mean(axis=0, keepdims=True)
    std = X_fit.std(axis=0, keepdims=True) + 1e-6
    Z_fit = np.hstack([(X_fit - mean) / std, np.ones((len(X_fit), 1))])
    Z_apply = np.hstack([(X_apply - mean) / std, np.ones((len(X_apply), 1))])
    d = Z_fit.shape[1]
    w = np.zeros(d, dtype=np.float64)
    reg = np.eye(d, dtype=np.float64) * l2_reg
    reg[-1, -1] = 1e-6
    y = y_fit.astype(np.float64)
    for _ in range(80):
        logits = np.clip(Z_fit @ w, -40.0, 40.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        r = np.maximum(p * (1.0 - p), 1e-6)
        grad = Z_fit.T @ (p - y) + reg @ w
        hess = (Z_fit.T * r) @ Z_fit + reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.linalg.norm(step) < 1e-7:
            break
    return w[:-1].astype(np.float32), np.float32(w[-1]), mean.astype(np.float32), std.astype(np.float32)


def fit_source_debiased_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sources_train: list[str],
    rank_k: int = 3,
    laion_real_weight: float = 3.0,
    l2_reg: float = 30.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Project out the top-k real-source directions, then fold the projection back.

    The source nuisance subspace is estimated **strictly within the real
    class** (``y == 0``) in standardized Z-space to maintain coordinate alignment.
    Every sample there is authentic, so any variance between ``laion`` and
    ``pexels``/``docci``/``imagenet`` is 100% source nuisance and 0% synthesis
    signal. Estimating it on the mixed class would also remove the real-vs-fake
    direction, which is the signal.

    Because the probe is linear, the projection folds into the weight
    vector: ``(P e)^T w == e^T (P w)``. So the DEPLOYED head is still a
    single 768-element vector and costs nothing at inference.

    Returns ``(weight, bias, projection)``.
    """
    from sklearn.linear_model import LogisticRegression

    sources = np.asarray(sources_train)
    real_mask = y_train == 0
    if real_mask.sum() < 10:
        raise ValueError("Too few real samples to estimate the source subspace")
    if len(set(sources[real_mask].tolist())) < 2:
        raise ValueError(
            "The real class has fewer than two acquisition sources; there is "
            "no source subspace to remove"
        )

    # 1. Standardize first so Fisher source subspace is coordinate-aligned with the classifier
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    z_train = (X_train - mean) / std

    Z_real = z_train[real_mask]
    s_real = sources[real_mask]

    # 2. Fit linear source-discriminant model in standardized space with matched L2 regularization
    C_reg = 1.0 / max(1e-4, float(l2_reg))
    classifier = LogisticRegression(C=C_reg, max_iter=1000).fit(Z_real, s_real)

    # 3. SVD on the standardized source classifier's weights -> the nuisance basis V_k
    _, _, vt = np.linalg.svd(classifier.coef_, full_matrices=False)
    V_k = vt[:rank_k].T                                   # (D, k)
    projection = np.eye(X_train.shape[1], dtype=np.float64) - (V_k @ V_k.T)

    # 4. Upweight LAION reals so false positives on LAION are penalised
    sample_weight = np.ones(len(y_train), dtype=np.float64)
    sample_weight[real_mask & (sources == "laion")] = laion_real_weight

    # 5. Fit the binary probe on the STANDARDIZED, PROJECTED embeddings with matched regularization
    probe = LogisticRegression(C=C_reg, max_iter=1000)
    probe.fit(z_train @ projection, y_train, sample_weight=sample_weight)

    # Fold projection into the weight vector
    folded_weight = (projection @ probe.coef_.reshape(-1)).astype(np.float32)
    return folded_weight, float(probe.intercept_[0]), projection.astype(np.float32)


def apply_debiased_probe(
    X: np.ndarray,
    weight: np.ndarray,
    bias: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Score with a folded-projection probe.

    The folded weight is applied to the RAW features, not to the projected
    ones, because the projection is already folded into the weight.
    """
    z = (X - mean) / std
    return 1.0 / (1.0 + np.exp(-(z @ weight + bias)))


def apply_probe(X, w, b, mean, std) -> np.ndarray:
    Z = (X - mean) / std
    return 1.0 / (1.0 + np.exp(-(Z @ w + b)))


def sweep_regularisation(data: dict, l2_grid) -> tuple[float, dict, list[dict]]:
    """Fit on stacking, SELECT on fusion_val, report the rest.

    Returns ``(selected_l2, best, all_rows)``. The full sweep is kept so the
    report shows how sensitive the result is to the regularisation, which is
    exactly what the MobileCLIP2-S0 failure was about.
    """
    rows = []
    for l2 in l2_grid:
        w, b, mean, std = fit_logistic_probe(
            data["X_stacking"], data["y_stacking"], data["X_stacking"], l2
        )
        scores = {
            split: apply_probe(data[f"X_{split}"], w, b, mean, std)
            for split in SPLITS
        }
        aucs = {
            split: float(compute_auc(data[f"y_{split}"], scores[split]))
            for split in SPLITS
        }
        rows.append({"l2_reg": l2, "aucs": aucs, "scores": scores,
                     "probe": (w, b, mean, std)})
    # Selection is on fusion_val ONLY.
    best = max(rows, key=lambda r: r["aucs"]["fusion_val"])
    return best["l2_reg"], best, rows


# ---------------------------------------------------------------------------
# Group reporting
# ---------------------------------------------------------------------------

def group_report(data_root: Path, split: str, scores: np.ndarray, labels: np.ndarray) -> dict:
    """Per-source FPR and per-generator TPR for one split."""
    manifest = json.loads((data_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    # The manifest stores POSIX-style paths; the filesystem lookup yields
    # OS-native separators. Normalise both sides before matching.
    by_path = {record["path"].replace("\\", "/"): record for record in manifest["records"]}
    samples = discover_samples(data_root, split, IMAGE_EXTS)
    records = []
    for path, _ in samples:
        relative = str(Path(path).relative_to(data_root)).replace("\\", "/")
        record = by_path.get(relative)
        if record is None:
            raise KeyError(f"Manifest has no record for {relative!r}")
        records.append(record)

    out: dict[str, dict] = {"source": {}, "generator": {}}
    for axis in ("source", "generator"):
        buckets = defaultdict(list)
        for index, record in enumerate(records):
            buckets[str(record.get(axis))].append(index)
        for value, indices in sorted(buckets.items()):
            idx = np.asarray(indices)
            y = labels[idx]
            p = scores[idx]
            real, fake = int((y == 0).sum()), int((y == 1).sum())
            out[axis][value] = {
                "n_real": real, "n_fake": fake,
                "tpr": float(p[y == 1].mean()) if fake else None,
                "fpr": float(p[y == 0].mean()) if real else None,
            }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def split_sources(data_root: Path, split: str) -> list[str]:
    """The acquisition source of every sample in a split, in sample order.

    The source comes from the MANIFEST, not from the path: the shipped
    layout is ``<split>/fake/<generator>/<file>``, so the path encodes the
    generator and says nothing about where the image came from.
    """
    manifest = json.loads((data_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    by_path = {record["path"].replace("\\", "/"): record for record in manifest["records"]}
    samples = discover_samples(data_root, split, IMAGE_EXTS)
    sources = []
    for path, _ in samples:
        relative = str(Path(path).relative_to(data_root)).replace("\\", "/")
        record = by_path.get(relative)
        if record is None:
            raise KeyError(f"Manifest has no record for {relative!r}")
        sources.append(str(record.get("source")))
    return sources


def evaluate_debiased_probe(
    data: dict,
    data_root: Path,
    rank_grid=(1, 2, 3, 4),
    laion_weight: float = 3.0,
) -> dict:
    """Compare the plain probe with the source-debiased probe.

    The rank is selected on ``fusion_val`` only, exactly like the
    regularisation sweep, and the per-source FPR is reported on the sealed
    test split.
    """
    stacking_sources = split_sources(data_root, "stacking")
    fusion_sources = split_sources(data_root, "fusion_val")
    test_sources = split_sources(data_root, "test")

    plain = sweep_regularisation(data, [30.0])[1]
    plain_scores = plain["scores"]["test"]
    plain_groups = group_report(data_root, "test", plain_scores, data["y_test"])

    rows = []
    for rank in rank_grid:
        weight, bias, _ = fit_source_debiased_probe(
            data["X_stacking"], data["y_stacking"], stacking_sources,
            rank_k=rank, laion_real_weight=laion_weight,
        )
        # The SAME standardization the fit used, so the applied function is
        # the fitted one.
        mean = data["X_stacking"].mean(axis=0, keepdims=True).astype(np.float32)
        std = (data["X_stacking"].std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        scores = {
            split: apply_debiased_probe(data[f"X_{split}"], weight, bias, mean, std)
            for split in SPLITS
        }
        aucs = {
            split: float(compute_auc(data[f"y_{split}"], scores[split]))
            for split in SPLITS
        }
        groups = group_report(data_root, "test", scores["test"], data["y_test"])
        rows.append({
            "rank_k": rank,
            "aucs": aucs,
            "test_groups": groups,
            "scores": scores,
        })

    best = max(rows, key=lambda r: r["aucs"]["fusion_val"])
    return {
        "plain": {
            "aucs": {s: plain["aucs"][s] for s in SPLITS},
            "test_groups": plain_groups,
        },
        "debiased": {
            "rank_k": best["rank_k"],
            "aucs": best["aucs"],
            "test_groups": best["test_groups"],
        },
        "all_ranks": [
            {"rank_k": r["rank_k"], "aucs": r["aucs"]} for r in rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--towers", nargs="+", default=list(TOWERS.keys()))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--l2-grid", type=float, nargs="+",
                        default=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    parser.add_argument("--debias", action="store_true",
                        help="Also measure the source-debiased probe: the source nuisance "
                             "subspace is estimated within the real class only and folded "
                             "back into the weight vector.")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {"data_root": str(data_root), "towers": {}}
    for repo in args.towers:
        tower = TOWERS.get(repo)
        if tower is None:
            raise SystemExit(f"Unknown tower {repo!r}; add it to TOWERS with its frontend contract.")
        print(f"\n=== {repo} ({tower.pooling}, {tower.size}px) ===")
        cache = output_dir / f"embeddings_{tower.key}.npz"
        data = extract_embeddings(tower, data_root, cache, args.batch_size)
        print(f"  embedding dim: {data['X_stacking'].shape[1]}")

        l2_grid = list(args.l2_grid)
        best_l2, best, all_rows = sweep_regularisation(data, l2_grid)
        scores = best["scores"]
        labels = data["y_test"]
        groups = group_report(data_root, "test", scores["test"], labels)

        entry = {
            "repo": repo,
            "frontend_id": tower.frontend_id(),
            "embedding_dim": int(data["X_stacking"].shape[1]),
            "selected_l2_reg": best_l2,
            "auc": {s: best["aucs"][s] for s in SPLITS},
            "test_eer": float(compute_eer(labels, scores["test"])),
            "test_groups": groups,
            "regularisation_sweep": [
                {"l2_reg": row["l2_reg"], "aucs": row["aucs"]} for row in all_rows
            ],
        }
        report["towers"][repo] = entry

        print(f"  selected l2_reg (by fusion_val): {best_l2}")
        for split in SPLITS:
            print(f"    {split:<12} AUC {best['aucs'][split]:.4f}")
        print("  test per-source FPR:")
        for source, stats in groups["source"].items():
            if stats["fpr"] is not None:
                flag = "  <-- SHORTCUT" if stats["fpr"] > 0.05 else ""
                print(f"    {source:<12} n_real={stats['n_real']:<4} fpr={stats['fpr']:.4f}{flag}")
        zero = [g for g, s in groups["generator"].items() if s["tpr"] == 0.0]
        if zero:
            print(f"  generators with TPR=0: {zero}")

        if args.debias:
            print("\n  --- source debiasing ---")
            debiased = evaluate_debiased_probe(data, data_root)
            plain_groups = debiased["plain"]["test_groups"]["source"]
            debiased_groups = debiased["debiased"]["test_groups"]["source"]
            print(f"  plain probe     : AUC {debiased['plain']['aucs']['test']:.4f}")
            print(f"  debiased probe  : AUC {debiased['debiased']['aucs']['test']:.4f} "
                  f"(rank_k={debiased['debiased']['rank_k']})")
            print(f"  {'source':<12}{'plain FPR':>12}{'debiased FPR':>15}{'delta':>10}")
            for source in sorted(set(plain_groups) | set(debiased_groups)):
                p = (plain_groups.get(source) or {}).get("fpr")
                d = (debiased_groups.get(source) or {}).get("fpr")
                delta = "" if (p is None or d is None) else f"{d - p:+.4f}"
                print(f"  {source:<12}{'n/a' if p is None else f'{p:.4f}':>12}"
                      f"{'n/a' if d is None else f'{d:.4f}':>15}{delta:>10}")
            entry["source_debiasing"] = {
                "plain": debiased["plain"],
                "debiased": debiased["debiased"],
                "all_ranks": debiased["all_ranks"],
            }

    out = output_dir / "vfm_probe_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
