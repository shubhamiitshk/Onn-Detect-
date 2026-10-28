#!/usr/bin/env python3
"""Benchmark Qwen Vision-Language Models vs. EmbeddingGemma 2 & SigLIP-B/16 under ONN-DETECT's split protocol.

Drop-in evaluation script for `ml/scripts/evaluate_qwen_probe.py` in ONN-DETECT
(or run standalone in Google Colab on a free T4 GPU).

Protocol:
1. Loads Qwen VLM (e.g., `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct`, or `Qwen/Qwen3.5-0.8B`).
2. Extracts multi-modal visual representation from the final cross-attended hidden state `h_{-1}`.
3. Applies ONN-DETECT's strict group-disjoint forensic evaluation protocol:
     - Fit L2-regularized binary classifier p_AI = sigmoid(w^T e + b) on `stacking` ONLY
     - Select regularization C on `fusion_val` ONLY
     - Fit decision threshold on `calibration` ONLY
     - Evaluate once on sealed `test` (ROC-AUC, EER, TPR@1%FPR, TPR@0.1%FPR, and per-source real FPR)
4. Compares against both the shipped SigLIP-B/16 INT8 probe and EmbeddingGemma 2.
5. Exports probe weights, bias, and evaluation report to JSON.

Usage (Google Colab T4 or local GPU):
    python ml/scripts/evaluate_qwen_probe.py \
        --model-name Qwen/Qwen3.5-2B \
        --data-root /content/datasets/openfake_visual \
        --output-dir /content/drive/MyDrive/onn_qwen_probe_eval

Self-Test:
    python ml/scripts/evaluate_qwen_probe.py --self-test
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS = ("stacking", "fusion_val", "calibration", "test")
C_GRID = (0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0)

SHIPPED_SIGLIP_BENCHMARK = {
    "auc_linear": 0.8919,
    "auc_tap": 0.9921,
    "tpr_at_1pct_fpr": 0.7243,
    "eer": 0.1250,
    "notes": "shipped on-device siglip_probe_int8.tflite (89.96 MB)",
}

GEMMA2_BENCHMARK = {
    "auc": 0.9006,
    "tpr_at_1pct_fpr": 0.2100,
    "eer": 0.1800,
    "notes": "evaluated unsloth/embeddinggemma-2 FP32 (300+ MB bloat, rejected)",
}


def ensure_benchmark_dataset(data_root: Path) -> None:
    """Ensures data_root contains the 4 strict ONN-DETECT forensic splits.
    If missing or incomplete, automatically downloads CIFAKE benchmark slice via Hugging Face Hub (8.4 MB)
    and unpacks it into stacking (50/50), fusion_val (50/50), calibration (50/50), and test (100/100).
    """
    for split in SPLITS:
        split_dir = data_root / split
        if not split_dir.is_dir():
            break
    else:
        return

    print(f"\n[Dataset] Benchmark split directory missing at {data_root}.")
    print("[Dataset] Auto-downloading CIFAKE authentic/synthetic benchmark slice from Hugging Face Hub (8.4 MB)...")
    import io
    from PIL import Image
    from huggingface_hub import hf_hub_download
    try:
        import pyarrow.parquet as pq
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyarrow"])
        import pyarrow.parquet as pq

    parquet_file = hf_hub_download(
        repo_id="dragonintelligence/CIFAKE-image-dataset",
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(parquet_file)
    df = table.to_pandas()

    reals = df[df["label"] == 0].reset_index(drop=True)
    fakes = df[df["label"] == 1].reset_index(drop=True)

    splits_spec = {
        "stacking": (reals.iloc[0:50], fakes.iloc[0:50]),
        "fusion_val": (reals.iloc[50:100], fakes.iloc[50:100]),
        "calibration": (reals.iloc[100:150], fakes.iloc[100:150]),
        "test": (reals.iloc[150:250], fakes.iloc[150:250]),
    }

    data_root.mkdir(parents=True, exist_ok=True)
    for split_name, (r_df, f_df) in splits_spec.items():
        r_dir = data_root / split_name / "real" / "authentic"
        f_dir = data_root / split_name / "fake" / "synthetic"
        r_dir.mkdir(parents=True, exist_ok=True)
        f_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in r_df.iterrows():
            img_data = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
            img = Image.open(io.BytesIO(img_data))
            img.save(r_dir / f"img_real_{idx:03d}.png")

        for idx, row in f_df.iterrows():
            img_data = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
            img = Image.open(io.BytesIO(img_data))
            img.save(f_dir / f"img_fake_{idx:03d}.png")

    print(f"[Dataset] Extracted 500 benchmark images into {data_root} successfully.\n")


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


class QwenFeatureExtractor:
    """Feature extractor wrapping Qwen VLM for representation extraction."""

    def __init__(self, model: Any, processor: Any, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = device
        self._dim: int | None = None

    def encode_image(self, img: Image.Image) -> np.ndarray:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Forensic synthetic or authentic anomaly analysis."},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][:, -1, :].to(torch.float32).squeeze(0)
            emb = last_hidden.detach().cpu().numpy()

        norm = float(np.linalg.norm(emb))
        emb_norm = emb / (norm + 1e-12)
        if self._dim is None:
            self._dim = emb_norm.shape[0]
        return emb_norm.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        if self._dim is not None:
            return self._dim
        cfg = getattr(self.model, "config", None)
        return int(getattr(cfg, "hidden_size", getattr(getattr(cfg, "text_config", None), "hidden_size", 1536)))


class SyntheticQwenModel:
    """Mock model used for self-tests and offline verification without GPU."""

    def __init__(self, embedding_dim: int = 1536, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.seed = seed

    def encode_image(self, img: Image.Image) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float32)
        base_seed = int(np.sum(arr)) % 100000 + self.seed
        rng = np.random.default_rng(base_seed)
        v = rng.standard_normal(self.embedding_dim, dtype=np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v.astype(np.float32)


def extract_qwen_split(
    extractor: Any,
    records: list[dict[str, Any]],
    cache_path: Path,
) -> np.ndarray:
    """Extract L2-normalized image embeddings using Qwen VLM."""
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(records) and np.all(np.isfinite(cached)):
            print(f"[cache] Reusing {cache_path.name} {cached.shape}")
            return cached

    vecs: list[np.ndarray] = []
    t0 = time.time()
    for idx, rec in enumerate(records):
        with Image.open(rec["path"]) as raw:
            img = raw.convert("RGB")
            # Enforce max pixel resolution of 512 for consistent latency & memory
            if max(img.size) > 512:
                scale = 512.0 / max(img.size)
                new_size = (max(1, int(round(img.size[0] * scale))), max(1, int(round(img.size[1] * scale))))
                img = img.resize(new_size, Image.Resampling.BICUBIC)
            emb = extractor.encode_image(img)
        vecs.append(emb)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(records):
            elapsed = time.time() - t0
            print(f"  encoded {idx + 1}/{len(records)} ({elapsed:.1f}s)")

    mat = np.stack(vecs, axis=0).astype(np.float32)
    np.save(cache_path, mat)
    return mat


def run_benchmark(
    data_root: Path,
    output_dir: Path,
    extractor: Any,
    model_name: str = "Qwen/Qwen3.5-2B",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the strict 4-split forensic evaluation protocol."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_benchmark_dataset(data_root)

    split_records: dict[str, list[dict[str, Any]]] = {}
    split_embs: dict[str, np.ndarray] = {}
    split_labels: dict[str, np.ndarray] = {}
    split_sources: dict[str, list[str]] = {}

    sanitized_name = model_name.replace("/", "_").replace("-", "_").lower()

    for split in SPLITS:
        recs = discover_split_records(data_root, split)
        split_records[split] = recs
        split_labels[split] = np.asarray([r["label"] for r in recs], dtype=np.int32)
        split_sources[split] = [r["source"] for r in recs]
        cache_file = output_dir / f"emb_{sanitized_name}_{split}.npy"
        print(f"[{split}] Extracting {len(recs)} image embeddings...")
        split_embs[split] = extract_qwen_split(extractor, recs, cache_file)

    emb_dim = split_embs["stacking"].shape[1]

    # 1. Fit on stacking ONLY, select regularization C on fusion_val ONLY
    best_c = 1.0
    best_val_auc = -1.0
    best_clf: LogisticRegression | None = None

    for c_val in C_GRID:
        clf = LogisticRegression(C=c_val, max_iter=2000, class_weight="balanced")
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

    # 3. Evaluate across all splits
    report: dict[str, Any] = {
        "model_name": model_name,
        "embedding_dim": emb_dim,
        "selected_C": best_c,
        "decision_threshold": threshold,
        "splits": {},
        "comparison": {
            "shipped_siglip": SHIPPED_SIGLIP_BENCHMARK,
            "embeddinggemma2": GEMMA2_BENCHMARK,
        },
    }
    for split in SPLITS:
        probs = best_clf.predict_proba(split_embs[split])[:, 1]
        report["splits"][split] = compute_forensic_metrics(
            split_labels[split], probs, split_sources[split], threshold
        )

    # 4. Export linear head weights w and bias b
    w_vec = best_clf.coef_.reshape(-1).astype(np.float32).tolist()
    b_val = float(best_clf.intercept_.reshape(-1)[0])
    head_artifact = {
        "schema_version": 1,
        "backbone": model_name,
        "embedding_dim": emb_dim,
        "l2_normalized": True,
        "weights": w_vec,
        "bias": b_val,
        "decision_threshold": threshold,
        "test_auc": report["splits"]["test"]["auc"],
        "test_tpr_at_fpr_1pct": report["splits"]["test"]["tpr_at_fpr_1pct"],
        "test_eer": report["splits"]["test"]["eer"],
        "diagnostic_only": True,
    }

    head_path = output_dir / f"{sanitized_name}_forensic_head.json"
    report_path = output_dir / f"{sanitized_name}_evaluation_report.json"
    head_path.write_text(json.dumps(head_artifact, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    test_m = report["splits"]["test"]
    print("\n" + "=" * 80)
    print(f"ONN-DETECT BENCHMARK REPORT: {model_name} vs. BASELINES")
    print("=" * 80)
    print(f"{'Metric':<28} {model_name:<24} {'EmbeddingGemma 2':<18} {'SigLIP-B/16 (Shipped)':<22}")
    print("-" * 80)
    print(f"{'ROC-AUC':<28} {test_m['auc']:<24.4f} {'0.9006':<18} {'0.8919 / 0.9921 (TAP)':<22}")
    print(f"{'TPR @ 1.0% FPR':<28} {test_m['tpr_at_fpr_1pct']:<24.4f} {'0.2100':<18} {'0.7243':<22}")
    print(f"{'Equal Error Rate (EER)':<28} {test_m['eer']:<24.4f} {'0.1800':<18} {'0.1250':<22}")
    print(f"{'Accuracy @ Calibrated Thr':<28} {test_m['accuracy_at_threshold']:<24.4f} {'0.8200':<18} {'0.8800':<22}")
    print(f"{'Calibrated Threshold':<28} {test_m['decision_threshold']:<24.4f} {'0.4851':<18} {'0.4200':<22}")
    print("=" * 80)
    print(f"Saved forensic head : {head_path}")
    print(f"Saved full report   : {report_path}")

    return report, head_artifact


def run_self_test(target_dir: Path | None = None) -> int:
    """Run an offline end-to-end self test with synthetic images and verified splits."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_path = Path(tmp_str) if target_dir is None else target_dir
        tmp_path.mkdir(parents=True, exist_ok=True)
        data_root = tmp_path / "synthetic_dataset"

        rng = np.random.default_rng(123)
        for split in SPLITS:
            for cls_name in ("real", "fake"):
                d = data_root / split / cls_name / "source_a"
                d.mkdir(parents=True, exist_ok=True)
                for i in range(10):
                    base_val = 60 if cls_name == "real" else 190
                    noise = rng.integers(-10, 10, size=(32, 32, 3), endpoint=True)
                    img_arr = np.clip(base_val + noise, 0, 255).astype(np.uint8)
                    Image.fromarray(img_arr).save(d / f"img_{i:03d}.png")

        mock_extractor = SyntheticQwenModel(embedding_dim=1536)
        out_dir = tmp_path / "out"
        report, head = run_benchmark(data_root, out_dir, mock_extractor, "mock-qwen-2-vl")

        assert (out_dir / "mock_qwen_2_vl_forensic_head.json").exists()
        assert (out_dir / "mock_qwen_2_vl_evaluation_report.json").exists()
        assert len(head["weights"]) == 1536
        assert np.isfinite(head["bias"])
        assert "test" in report["splits"]
        print("\n[SELF-TEST PASSED] Synthetic Qwen 4-split benchmark completed successfully.")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Path to openfake_visual dataset root")
    parser.add_argument("--output-dir", type=Path, help="Directory for cache, head weights, and report")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-2B", help="HF Qwen model id (default: Qwen/Qwen3.5-2B)")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda or cpu)")
    parser.add_argument("--self-test", action="store_true", help="Run offline synthetic self-test")
    args = parser.parse_args()

    if args.self_test:
        sys.exit(run_self_test(args.output_dir))

    if not args.data_root or not args.output_dir:
        parser.error("--data-root and --output-dir are required unless --self-test is specified.")

    # Gracefully bypass broken torchvision / torchaudio C++ extension mismatches in Colab
    for mod in ("torchvision", "torchaudio"):
        try:
            __import__(mod)
        except Exception:
            sys.modules[mod] = None

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    bf16_ok = torch.cuda.is_bf16_supported() if device == "cuda" else False
    dtype = torch.bfloat16 if bf16_ok else (torch.float16 if device == "cuda" else torch.float32)

    print("=" * 70)
    print(f"Loading Qwen VLM: {args.model_name}")
    print(f"Device: {device} | dtype: {dtype}")
    print("=" * 70)

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    extractor = QwenFeatureExtractor(model, processor, device=device)
    run_benchmark(args.data_root, args.output_dir, extractor, args.model_name)


if __name__ == "__main__":
    main()
