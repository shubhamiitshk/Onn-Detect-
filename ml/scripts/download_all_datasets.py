"""Non-destructive, revision-pinned download of honestly named pilot datasets.

Never partitions by counters or guesses numeric label meaning. Original source
splits are retained. Additional grouped development/calibration splits must be
prepared before these pilots satisfy the training contract.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.contracts import sha256_file, semantic_label

SOURCES = {"cifake": ("dragonintelligence/CIFAKE-image-dataset", "image", ["fake", "real"]),
           "gary-stafford-audio": ("garystafford/deepfake-audio-detection", "audio", ["real", "fake"])}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SOURCES)
    parser.add_argument("--revision", required=True, help="Immutable 40-character Hugging Face commit")
    parser.add_argument("--output-dir", required=True, help="New dataset directory; existing paths are never replaced")
    parser.add_argument("--limit", type=int, default=None, help="Optional total balanced limit per source split")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        raise ValueError("Pin an immutable source commit, not main/latest")
    if args.limit is not None and args.limit < 2:
        raise ValueError("Limit must include both classes")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".download-", dir=output.parent))
    from datasets import load_dataset, Audio, Image
    repo, media, verified_names = SOURCES[args.dataset]
    dataset = load_dataset(repo, revision=args.revision, streaming=True)
    records, split_counts = [], {}
    for source_split, stream in dataset.items():
        split = "val" if source_split in ("validation", "dev") else source_split
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported source split {source_split}")
        feature = stream.features.get("label")
        names = getattr(feature, "names", None) or verified_names
        if {semantic_label(names, index) for index in range(len(names))} != {0, 1}:
            raise ValueError("Source class schema changed")
        stream = stream.cast_column(media, Audio(decode=False) if media == "audio" else Image(decode=False))
        counts = [0, 0]
        for index, row in enumerate(stream):
            label = semantic_label(names, row["label"])
            if args.limit is not None and counts[label] >= args.limit // 2:
                if min(counts) >= args.limit // 2:
                    break
                continue
            value = row[media]
            original_path = value.get("path") or ""
            content = value.get("bytes")
            if content is None:
                path = Path(original_path)
                if not path.is_file():
                    raise ValueError(f"Source bytes unavailable for {original_path}; no substitute sample")
                content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if media == "image":
                import io
                from PIL import Image as PILImage
                with PILImage.open(io.BytesIO(content)) as image:
                    image.verify()
                    extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image.format)
                if extension is None:
                    raise ValueError("Unsupported source image encoding")
            else:
                import io
                import soundfile as sf
                info = sf.info(io.BytesIO(content))
                if info.frames <= 0:
                    raise ValueError("Empty source audio")
                extension = {"WAV": ".wav", "FLAC": ".flac", "OGG": ".ogg"}.get(info.format)
                if extension is None:
                    raise ValueError(f"Unsupported audio encoding {info.format}")
            relative = Path(split) / ("real" if label == 0 else "fake") / (digest + extension)
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                continue
            path.write_bytes(content)
            source_id = str(row.get("id", original_path or f"{source_split}:{index}"))
            records.append({"path": relative.as_posix(), "split": split, "label": label, "sha256": digest,
                            "source_id": source_id, "group_id": None, "source_split": source_split,
                            "source_label": int(row["label"]), "source_class": names[int(row["label"])]})
            counts[label] += 1
        if not all(counts):
            raise ValueError(f"Incomplete source split {source_split}: {counts}. Partial data remains in {stage}")
        split_counts[split] = counts
    if not records:
        raise ValueError(f"Download produced no records; staging retained at {stage}")
    manifest = {"schema_version": 1, "source": repo, "revision": args.revision,
                "label_convention": "real=0_fake=1", "source_label_names": verified_names,
                "grouping_verified": False, "status": "source_splits_only_requires_grouped_preparation",
                "split_counts": split_counts, "records": records}
    (stage / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Rename a complete version; no recursive removal of existing user data.
    os.rename(stage, output)
    print(f"Downloaded {len(records)} original-byte samples to {output}. Preserve test; prepare verified groups and missing splits before training.")

if __name__ == "__main__":
    main()
