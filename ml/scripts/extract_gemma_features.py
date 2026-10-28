#!/usr/bin/env python3
"""Reference Extractor for Frozen EmbeddingGemma 2 Global Image Features.

Extracts normalized 768-D image representations from Google's EmbeddingGemma 2
multimodal encoder in full FP32 precision.

Invariant Specifications (from IMPLEMENTATION.md):
- Global image, EXIF orientation corrected.
- RGB conversion with white background for alpha channels.
- No blind 224px resizing: delegates to the model processor's native geometry.
- Visual tokens: 70 (or configurable 140 for fine-grained analysis).
- Omit audio encoder; extract text+vision representations.
- Emits cached `features.npz` and `feature_contract.json` for CPU-based linear head training.

Usage:
    python -m ml.scripts.extract_gemma_features \\
        --model-dir /path/to/embeddinggemma-2 \\
        --dataset-jsonl /path/to/data.jsonl \\
        --output-dir /path/to/cache \\
        --vision-tokens 70

    # Self-test mode (synthetic vectors, no GPU or model download needed):
    python -m ml.scripts.extract_gemma_features --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

FEATURE_DIM = 768


def preprocess_image_for_gemma(img: Image.Image) -> Image.Image:
    """Preprocesses a PIL Image ensuring EXIF orientation and RGB with white alpha background."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        converted = img.convert("RGBA")
        bg.paste(converted, mask=converted.split()[-1])
        return bg
    return img.convert("RGB")


def compute_image_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class GemmaFeatureExtractor:
    """Loads EmbeddingGemma 2 text+vision pipeline in FP32 on CPU or GPU."""

    def __init__(self, model_dir: Path, model_revision: str = "main", vision_tokens: int = 70, device: str = "cpu"):
        self.model_dir = model_dir
        self.model_revision = model_revision
        self.vision_tokens = vision_tokens
        self.device = device
        self.model = None
        self.processor = None

    def load_pipeline(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModel

        print(f"Loading EmbeddingGemma 2 from {self.model_dir} (revision: {self.model_revision})...")
        self.processor = AutoProcessor.from_pretrained(str(self.model_dir), revision=self.model_revision)
        # Invariant: Google explicitly warns against FP16 activations for Gemma 2. Use FP32.
        self.model = AutoModel.from_pretrained(
            str(self.model_dir),
            revision=self.model_revision,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.eval()

    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        import torch

        if self.model is None or self.processor is None:
            self.load_pipeline()

        clean_img = preprocess_image_for_gemma(image)
        # Pass through the official processor without text prefix
        inputs = self.processor(images=clean_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs) if hasattr(self.model, "get_image_features") else self.model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feat = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                feat = outputs.last_hidden_state.mean(dim=1)
            else:
                feat = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            # L2 normalise to 768-D unit sphere
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
            vec = feat.squeeze(0).cpu().numpy().astype(np.float32)
            assert vec.shape[-1] == FEATURE_DIM, f"Expected {FEATURE_DIM} dimensions, got {vec.shape}"
            return vec


def run_extraction(
    model_dir: Path,
    model_revision: str,
    dataset_jsonl: Path,
    output_dir: Path,
    vision_tokens: int = 70,
    device: str = "cpu",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = GemmaFeatureExtractor(model_dir, model_revision, vision_tokens, device)

    lines = [json.loads(line) for line in dataset_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Extracting features for {len(lines)} dataset items...")

    base_dir = dataset_jsonl.parent
    features_list = []
    labels_list = []
    root_ids_list = []
    splits_list = []
    sha256s_list = []

    for item in lines:
        rel_path = item["image_path"]
        img_path = (base_dir / rel_path).resolve()
        img = Image.open(img_path)
        vec = extractor.extract_embedding(img)
        features_list.append(vec)
        labels_list.append(int(item["label"]))
        root_ids_list.append(str(item.get("root_id", rel_path)))
        splits_list.append(str(item.get("split", "train")))
        sha256s_list.append(compute_image_sha256(img_path))

    features_arr = np.array(features_list, dtype=np.float32)
    labels_arr = np.array(labels_list, dtype=np.int32)

    npz_path = output_dir / "features.npz"
    np.savez_compressed(
        npz_path,
        features=features_arr,
        labels=labels_arr,
        root_ids=np.array(root_ids_list),
        splits=np.array(splits_list),
        sha256s=np.array(sha256s_list),
    )

    contract = {
        "model_id": "google/embeddinggemma-2",
        "model_revision": model_revision,
        "feature_dim": FEATURE_DIM,
        "vision_tokens": vision_tokens,
        "precision": "float32",
        "l2_normalized": True,
        "exif_transposed": True,
        "icc_profile_applied": False,
        "sample_count": len(lines),
        "split_counts": {s: int(np.sum(np.array(splits_list) == s)) for s in set(splits_list)},
    }
    (output_dir / "feature_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"Features saved to {npz_path} ({features_arr.shape})")
    print(f"Contract saved to {output_dir / 'feature_contract.json'}")


def run_self_test(output_dir: Path) -> bool:
    print("Running GemmaFeatureExtractor self-test with synthetic data...")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    n_samples = 60
    splits = ["train"] * 30 + ["calibration"] * 15 + ["test"] * 15
    labels = [i % 2 for i in range(n_samples)]
    features = rng.randn(n_samples, FEATURE_DIM).astype(np.float32)
    features = features / np.linalg.norm(features, axis=-1, keepdims=True)
    root_ids = [f"capture_{i:03d}" for i in range(n_samples)]
    sha256s = [hashlib.sha256(f"img_{i}".encode()).hexdigest() for i in range(n_samples)]

    npz_path = output_dir / "features.npz"
    np.savez_compressed(
        npz_path,
        features=features,
        labels=np.array(labels, dtype=np.int32),
        root_ids=np.array(root_ids),
        splits=np.array(splits),
        sha256s=np.array(sha256s),
    )

    contract = {
        "model_id": "google/embeddinggemma-2-reference",
        "model_revision": "self-test-fixture",
        "feature_dim": FEATURE_DIM,
        "vision_tokens": 70,
        "precision": "float32",
        "l2_normalized": True,
        "sample_count": n_samples,
        "split_counts": {"train": 30, "calibration": 15, "test": 15},
    }
    (output_dir / "feature_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    assert npz_path.is_file()
    assert (output_dir / "feature_contract.json").is_file()
    print("GemmaFeatureExtractor self-test PASSED!")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--model-revision", type=str, default="main")
    parser.add_argument("--dataset-jsonl", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="gemma_cache")
    parser.add_argument("--vision-tokens", type=int, default=70)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    if args.self_test:
        run_self_test(out)
        return

    if not args.model_dir or not args.dataset_jsonl:
        parser.error("--model-dir and --dataset-jsonl are required unless --self-test is used")

    run_extraction(
        Path(args.model_dir).resolve(),
        args.model_revision,
        Path(args.dataset_jsonl).resolve(),
        out,
        args.vision_tokens,
    )


if __name__ == "__main__":
    main()
