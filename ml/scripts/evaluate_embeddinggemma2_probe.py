#!/usr/bin/env python3
"""Benchmark frozen EmbeddingGemma 2 vs. SigLIP-B/16 under ONN-DETECT's split protocol.

Drop-in evaluation script for `ml/scripts/evaluate_embeddinggemma2_probe.py` in ONN-DETECT
(or run standalone in Google Colab on a free T4 GPU).

Why this script exists
----------------------
1. Enforces FP32 execution (`torch.float32`) on T4 GPUs with `audio_config=None`, avoiding
   EmbeddingGemma 2's known FP16 NaN / degradation issues on Compute Capability 7.5.
2. Replaces image-caption retrieval loss (MultipleNegativesRankingLoss) with ONN-DETECT's
   strict group-disjoint supervised forensic probe protocol:
     - Fit L2-regularized binary classifier p_AI = sigmoid(w^T e + b) on `stacking` ONLY
     - Select regularization C on `fusion_val` ONLY
     - Fit decision threshold on `calibration` ONLY
     - Evaluate once on `test` (ROC-AUC, EER, TPR@1%FPR, TPR@0.1%FPR, and per-source real FPR)
3. Exports the trained 768-d weight vector `w`, bias `b`, and calibrated threshold to JSON so
   it can be loaded directly alongside MediaPipe `UniversalEmbedder` on Android if the benchmark
   outperforms the shipped SigLIP-B/16 probe (`siglip_probe_int8.tflite`).

Usage (Colab T4 or local environment)
-------------------------------------
    pip install -U sentence-transformers transformers scikit-learn pillow numpy
    python ml/scripts/evaluate_embeddinggemma2_probe.py \
        --data-root /content/datasets/openfake_visual \
        --output-dir /content/drive/MyDrive/onn_gemma2_probe_eval

Self-Test:
    python ml/scripts/evaluate_embeddinggemma2_probe.py --self-test
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT / "src"))
if (ROOT / "scripts").exists():
    sys.path.insert(0, str(ROOT / "scripts"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS = ("stacking", "fusion_val", "calibration", "test")
C_GRID = (0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0)


def discover_split_records(data_root: Path, split: str) -> list[dict[str, Any]]:
    """Load sample records from dataset_manifest.json or fallback directory layout."""
    manifest_path = data_root / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [
            r for r in manifest.get("records", [])
            if r.get("split") == split
            and (
                (data_root / r.get("rel_path", r.get("path", ""))).exists()
                or (data_root / r.get("path", "")).exists()
            )
        ]
        if records:
            out: list[dict[str, Any]] = []
            for r in records:
                rel = r.get("rel_path", r.get("path", ""))
                p = (data_root / rel).resolve()
                if not p.exists():
                    p = (data_root / r["path"]).resolve()
                out.append(
                    {
                        "path": p,
                        "label": 1 if int(r["label"]) == 1 else 0,
                        "source": str(r.get("source_id", r.get("source", "unknown"))),
                        "generator": str(r.get("generator", "unknown")),
                    }
                )
            return out

    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    out: list[dict[str, Any]] = []
    for cls_name, label in (("real", 0), ("fake", 1)):
        cls_dir = split_dir / cls_name
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                parent_name = p.parent.name if p.parent != cls_dir else cls_name
                out.append(
                    {
                        "path": p,
                        "label": label,
                        "source": parent_name if label == 0 else "synthetic",
                        "generator": parent_name if label == 1 else "authentic",
                    }
                )
    if not out:
        raise RuntimeError(f"No images found under {split_dir}")
    return out


def compute_forensic_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    sources: list[str],
    threshold: float,
) -> dict[str, Any]:
    """Compute ROC-AUC, EER, low-FPR recall, and per-source real-image FPR."""
    auc = float(roc_auc_score(y_true, probs))
    fpr, tpr, _ = roc_curve(y_true, probs)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    tpr_at_1pct = float(np.max(tpr[fpr <= 0.01])) if np.any(fpr <= 0.01) else 0.0
    tpr_at_01pct = float(np.max(tpr[fpr <= 0.001])) if np.any(fpr <= 0.001) else 0.0

    preds = (probs >= threshold).astype(np.int32)
    acc = float(np.mean(preds == y_true))

    source_buckets: dict[str, list[int]] = defaultdict(list)
    for lbl, pred, src in zip(y_true.tolist(), preds.tolist(), sources):
        if lbl == 0:
            source_buckets[src].append(pred)

    per_source_fpr = {
        src: {
            "n_real": len(flags),
            "fpr": float(np.mean(flags)),
        }
        for src, flags in sorted(source_buckets.items())
    }

    return {
        "auc": auc,
        "eer": eer,
        "tpr_at_fpr_1pct": tpr_at_1pct,
        "tpr_at_fpr_0_1pct": tpr_at_01pct,
        "accuracy_at_threshold": acc,
        "decision_threshold": float(threshold),
        "per_source_real_fpr": per_source_fpr,
    }


def select_threshold_on_calibration(y_cal: np.ndarray, probs_cal: np.ndarray, max_fpr: float = 0.05) -> float:
    """Select operating point on calibration split maximizing Youden's J subject to FPR <= max_fpr."""
    fpr, tpr, thresholds = roc_curve(y_cal, probs_cal)
    valid = np.where(fpr <= max_fpr)[0]
    if len(valid) > 0:
        best_idx = valid[int(np.argmax(tpr[valid] - fpr[valid]))]
    else:
        best_idx = int(np.argmax(tpr - fpr))
    thr = float(thresholds[best_idx])
    return float(np.clip(thr, 1e-4, 1.0 - 1e-4))


def extract_embeddinggemma2_split(
    model: Any,
    records: list[dict[str, Any]],
    cache_path: Path,
    truncate_dim: int = 768,
) -> np.ndarray:
    """Extract L2-normalized 768-d image embeddings using EmbeddingGemma 2 in FP32."""
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (len(records), truncate_dim) and np.all(np.isfinite(cached)):
            print(f"[cache] Reusing {cache_path.name} {cached.shape}")
            return cached

    vecs = np.zeros((len(records), truncate_dim), dtype=np.float32)
    t0 = time.time()
    for idx, rec in enumerate(records):
        with Image.open(rec["path"]) as raw:
            img = raw.convert("RGB")
            emb = model.encode(
                {"text": "<|image|>", "image": [img]},
                prompt="",
                truncate_dim=truncate_dim,
                normalize_embeddings=True,
            )
        emb_np = np.asarray(emb, dtype=np.float32).reshape(-1)
        if emb_np.shape[0] != truncate_dim:
            raise RuntimeError(f"Expected {truncate_dim}-d embedding, got {emb_np.shape}")
        if not np.all(np.isfinite(emb_np)):
            raise RuntimeError(f"Non-finite embedding encountered for {rec['path']} (ensure FP32/BF16, never FP16)")
        norm = float(np.linalg.norm(emb_np))
        vecs[idx] = emb_np / (norm + 1e-12)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(records):
            elapsed = time.time() - t0
            print(f"  encoded {idx + 1}/{len(records)} ({elapsed:.1f}s)")

    np.save(cache_path, vecs)
    return vecs


class SyntheticEmbeddingGemmaModel:
    """Mock model used for self-tests and offline verification."""

    def __init__(self, truncate_dim: int = 768, seed: int = 42):
        self.truncate_dim = truncate_dim
        self.rng = np.random.default_rng(seed)

    def encode(self, inputs: dict[str, Any], prompt: str = "", truncate_dim: int = 768, normalize_embeddings: bool = True) -> np.ndarray:
        # Create deterministic synthetic feature based on image pixel sum
        img = inputs["image"][0]
        arr = np.asarray(img, dtype=np.float32)
        base_seed = int(np.sum(arr)) % 100000
        rng = np.random.default_rng(base_seed)
        v = rng.standard_normal(truncate_dim, dtype=np.float32)
        if normalize_embeddings:
            v /= (np.linalg.norm(v) + 1e-12)
        return v


def run_benchmark(
    data_root: Path,
    output_dir: Path,
    model: Any,
    model_name: str = "unsloth/embeddinggemma-2",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the strict 4-split forensic evaluation protocol."""
    output_dir.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, list[dict[str, Any]]] = {}
    split_embs: dict[str, np.ndarray] = {}
    split_labels: dict[str, np.ndarray] = {}
    split_sources: dict[str, list[str]] = {}

    for split in SPLITS:
        recs = discover_split_records(data_root, split)
        split_records[split] = recs
        split_labels[split] = np.asarray([r["label"] for r in recs], dtype=np.int32)
        split_sources[split] = [r["source"] for r in recs]
        cache_file = output_dir / f"emb_embeddinggemma2_{split}.npy"
        print(f"[{split}] Extracting {len(recs)} image embeddings...")
        split_embs[split] = extract_embeddinggemma2_split(model, recs, cache_file)

    # 1. Fit on stacking ONLY, select regularization C on fusion_val ONLY
    best_c = 1.0
    best_val_auc = -1.0
    best_clf: LogisticRegression | None = None

    for c_val in C_GRID:
        clf = LogisticRegression(C=c_val, penalty="l2", solver="lbfgs", max_iter=2000, class_weight="balanced")
        clf.fit(split_embs["stacking"], split_labels["stacking"])
        val_probs = clf.predict_proba(split_embs["fusion_val"])[:, 1]
        val_auc = float(roc_auc_score(split_labels["fusion_val"], val_probs))
        print(f"  C={c_val:<6} -> fusion_val AUC = {val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_c = c_val
            best_clf = clf

    assert best_clf is not None
    print(f"\nSelected C={best_c} (fusion_val AUC={best_val_auc:.4f})")

    # 2. Calibrate threshold on calibration split ONLY
    cal_probs = best_clf.predict_proba(split_embs["calibration"])[:, 1]
    threshold = select_threshold_on_calibration(split_labels["calibration"], cal_probs)

    # 3. Evaluate across all splits (reading sealed test once)
    report: dict[str, Any] = {
        "model_name": model_name,
        "dtype": "float32",
        "selected_C": best_c,
        "decision_threshold": threshold,
        "splits": {},
    }
    for split in SPLITS:
        probs = best_clf.predict_proba(split_embs[split])[:, 1]
        report["splits"][split] = compute_forensic_metrics(
            split_labels[split], probs, split_sources[split], threshold
        )

    # 4. Export linear head weights w (768,) and bias b for Android UniversalEmbedder scoring
    w_vec = best_clf.coef_.reshape(-1).astype(np.float32).tolist()
    b_val = float(best_clf.intercept_.reshape(-1)[0])
    head_artifact = {
        "schema_version": 1,
        "backbone": model_name,
        "embedding_dim": 768,
        "l2_normalized": True,
        "weights": w_vec,
        "bias": b_val,
        "decision_threshold": threshold,
        "test_auc": report["splits"]["test"]["auc"],
        "test_tpr_at_fpr_1pct": report["splits"]["test"]["tpr_at_fpr_1pct"],
        "diagnostic_only": True,
    }

    head_path = output_dir / "embeddinggemma2_forensic_head.json"
    report_path = output_dir / "embeddinggemma2_evaluation_report.json"
    head_path.write_text(json.dumps(head_artifact, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    test_m = report["splits"]["test"]
    print("\n=== SEALED TEST SPLIT SUMMARY ===")
    print(f"Test ROC-AUC       : {test_m['auc']:.4f}  (Compare vs. SigLIP-B/16 linear: 0.8919 | SigLIP-TAP: 0.9921)")
    print(f"Test TPR @ 1% FPR  : {test_m['tpr_at_fpr_1pct']:.4f}  (Compare vs. SigLIP-TAP: 0.7243 | Fusion v10: 0.9400)")
    print(f"Test EER           : {test_m['eer']:.4f}")
    print(f"Saved head weights : {head_path}")
    print(f"Saved full report  : {report_path}")

    return report, head_artifact


def run_self_test(target_dir: Path | None = None) -> int:
    """Run an offline end-to-end self test with synthetic images and verified splits."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_path = Path(tmp_str) if target_dir is None else target_dir
        tmp_path.mkdir(parents=True, exist_ok=True)
        data_root = tmp_path / "synthetic_dataset"

        # Create mini-splits with real and fake samples
        rng = np.random.default_rng(123)
        for split in SPLITS:
            for cls_name in ("real", "fake"):
                d = data_root / split / cls_name / "source_a"
                d.mkdir(parents=True, exist_ok=True)
                for i in range(10):
                    # For real, lower brightness; for fake, higher brightness
                    base_val = 50 if cls_name == "real" else 200
                    noise = rng.integers(-10, 10, size=(32, 32, 3), endpoint=True)
                    img_arr = np.clip(base_val + noise, 0, 255).astype(np.uint8)
                    Image.fromarray(img_arr).save(d / f"img_{i:03d}.png")

        mock_model = SyntheticEmbeddingGemmaModel(truncate_dim=768)
        out_dir = tmp_path / "out"
        report, head = run_benchmark(data_root, out_dir, mock_model, "mock-embeddinggemma-2")

        assert (out_dir / "embeddinggemma2_forensic_head.json").exists()
        assert (out_dir / "embeddinggemma2_evaluation_report.json").exists()
        assert len(head["weights"]) == 768
        assert np.isfinite(head["bias"])
        assert "test" in report["splits"]
        print("\n[SELF-TEST PASSED] Synthetic 4-split benchmark completed successfully.")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Path to openfake_visual dataset root")
    parser.add_argument("--output-dir", type=Path, help="Directory for cache, head weights, and report")
    parser.add_argument("--model-name", default="unsloth/embeddinggemma-2", help="HF model id")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda or cpu)")
    parser.add_argument("--self-test", action="store_true", help="Run offline synthetic self-test")
    args = parser.parse_args()

    if args.self_test:
        sys.exit(run_self_test(args.output_dir))

    if not args.data_root or not args.output_dir:
        parser.error("--data-root and --output-dir are required unless --self-test is specified.")

    # Gracefully bypass broken torchvision C++ extension mismatches in Colab
    try:
        import torchvision
    except Exception:
        import sys
        sys.modules["torchvision"] = None

    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # Explicit FP32 on T4 (Compute Capability 7.5) because FP16 causes NaN/degradation in EmbeddingGemma 2
    dtype = torch.float32
    print(f"Loading {args.model_name} on {device} with dtype={dtype} and audio_config=None...")
    model = SentenceTransformer(
        args.model_name,
        device=device,
        model_kwargs={"torch_dtype": dtype},
        config_kwargs={"audio_config": None},
    )

    run_benchmark(args.data_root, args.output_dir, model, args.model_name)


if __name__ == "__main__":
    main()
