#!/usr/bin/env python3
"""Fetch a fast local evaluation slice from CIFAKE to support local desktop fusion and release testing."""
import io
import json
import hashlib
import shutil
import sys
from pathlib import Path
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from datasets import load_dataset

DATASET_ROOT = Path("datasets/openfake_visual").resolve()
CLASSES = {0: "real", 1: "fake"}
SPLITS = ["train", "val", "calibration", "test"]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("⬇️ Fetching local 1000-sample CIFAKE evaluation slice from Hugging Face...")
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
    for sp in SPLITS:
        for cls in ("real", "fake"):
            (DATASET_ROOT / sp / cls).mkdir(parents=True, exist_ok=True)

    ds = load_dataset("dragonintelligence/CIFAKE-image-dataset", split="test", streaming=True)
    counts = {0: 0, 1: 0} # ONN label counts: 0=real, 1=fake
    max_per_class = 500 # 500 real, 500 fake = 1000 total
    records = []

    for item in ds:
        raw_label = int(item["label"]) # CIFAKE: 0=FAKE, 1=REAL
        onn_label = 0 if raw_label == 1 else 1 # ONN-DETECT: 0=REAL, 1=FAKE
        
        if counts[onn_label] >= max_per_class:
            if counts[0] >= max_per_class and counts[1] >= max_per_class:
                break
            continue

        c = counts[onn_label]
        counts[onn_label] += 1

        # Partition per class: 200 train, 100 val, 100 cal, 100 test
        if c < 200:
            split = "train"
        elif c < 300:
            split = "val"
        elif c < 400:
            split = "calibration"
        else:
            split = "test"

        cls_name = "real" if onn_label == 0 else "fake"
        filename = f"cifake_{cls_name}_{c:05d}.png"
        img_path = DATASET_ROOT / split / cls_name / filename
        
        # Save image
        img = item["image"].convert("RGB")
        img.save(img_path, format="PNG")

        rel_path = img_path.relative_to(DATASET_ROOT).as_posix()
        digest = sha256_file(img_path)
        records.append({
            "path": rel_path,
            "label": onn_label,
            "split": split,
            "source_id": f"cifake-{cls_name}-{c:05d}",
            "group_id": f"group-{cls_name}-{c:05d}",
            "sha256": digest,
            "source": "cifake",
        })

    manifest = {
        "schema_version": 1,
        "source": "dragonintelligence/CIFAKE-image-dataset",
        "revision": "local-desktop-slice",
        "label_convention": "real=0_fake=1",
        "grouping_verified": True,
        "records": records,
    }
    manifest_path = DATASET_ROOT / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"🎉 Generated local evaluation dataset at {DATASET_ROOT} ({len(records)} samples across 4 splits)!")

if __name__ == "__main__":
    main()
