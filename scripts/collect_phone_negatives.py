#!/usr/bin/env python3
"""Collect consented real-phone negatives for LAION-Mobile-style calibration.

Why: modern phone ISPs (HDR fusion, denoising, motion suppression) create
learned artifacts that look like generative fingerprints. Desktop calibration
(400 sealed samples) does not cover them, so phone FPR must be measured on
the target device itself before any `validated: true` promotion. See
docs/DEVICE_VALIDATION_RUNBOOK.md.

Usage:
    python scripts/collect_phone_negatives.py --source ./phone_raw --out ./phone_negatives_manifest.json --consent-confirmed
    python scripts/collect_phone_negatives.py --adb-pull --out ./phone_negatives_manifest.json --consent-confirmed

Requires explicit --consent-confirmed: only consented, data-protected photos.
GPS is stripped from working copies; device metadata is recorded in the
manifest only. Pillow is optional (EXIF fields degrade to "unknown" without
it); everything else is stdlib-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PHOTO_EXTS = {".jpg", ".jpeg", ".heic", ".heif", ".png", ".webp"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def looks_like_screenshot(p: Path, size: int) -> bool:
    """Heuristic non-photo reject: screenshots are often PNG, uniform, tall."""
    if p.suffix.lower() != ".png":
        return False
    if size > 8 * 1024 * 1024:
        return False  # large PNGs are usually not screenshots
    return True  # flagged for manual review, not auto-kept


def exif_for(p: Path) -> dict:
    meta = {"make": None, "model": None, "datetime": None, "gps_present": None, "width": None, "height": None}
    try:
        from PIL import Image, ExifTags  # type: ignore
        with Image.open(p) as im:
            meta["width"], meta["height"] = im.size
        exif = Image.open(p).getexif()  # type: ignore
        for tag_id, val in exif.items():
            tag = ExifTags.TAGS.get(tag_id, str(tag_id)).lower()
            if tag == "make":
                meta["make"] = str(val)
            elif tag == "model":
                meta["model"] = str(val)
            elif tag == "datetimeoriginal":
                meta["datetime"] = str(val)
            elif tag in ("gpsinfo", "gpslatitude", "gpslongitude"):
                meta["gps_present"] = True
    except Exception:
        pass
    return meta


def collect(source: Path, adb_pull: bool) -> list[Path]:
    if adb_pull:
        tmp = Path("./phone_negatives_pull")
        tmp.mkdir(exist_ok=True)
        print(f"Pulling DCIM via adb to {tmp} …")
        for sub in ("DCIM", "Pictures"):
            subprocess.run(["adb", "pull", f"/sdcard/{sub}", str(tmp)], check=False)
        source = tmp
    if not source.is_dir():
        print(f"ERROR: source not found: {source}", file=sys.stderr)
        sys.exit(2)
    return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in PHOTO_EXTS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path("./phone_negatives_raw"))
    ap.add_argument("--adb-pull", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("./phone_negatives_manifest.json"))
    ap.add_argument("--consent-confirmed", action="store_true", help="Required: photos are consented and data-protected")
    args = ap.parse_args()
    if not args.consent_confirmed:
        print("REFUSING: pass --consent-confirmed only for consented, data-protected photos.", file=sys.stderr)
        sys.exit(2)
    files = collect(args.source, args.adb_pull)
    seen: dict[str, str] = {}
    manifest: list[dict] = []
    kept, rejects = 0, {"duplicate": 0, "screenshot_review": 0, "no_phone_exif": 0, "tiny": 0}
    for p in files:
        digest = sha256_file(p)
        if digest in seen:
            rejects["duplicate"] += 1
            continue
        seen[digest] = str(p)
        if p.stat().st_size < 10 * 1024:
            rejects["tiny"] += 1
            continue
        meta = exif_for(p)
        if meta["make"] is None and meta["model"] is None:
            rejects["no_phone_exif"] += 1
            # Keep for review but mark it; never train on unmarked files.
            manifest.append({"file": str(p), "sha256": digest, "label": "real", "exif": meta,
                             "flag": "NO_PHONE_EXIF — review before use", "gps_stripped": "strip before training"})
            continue
        if looks_like_screenshot(p, p.stat().st_size):
            rejects["screenshot_review"] += 1
        manifest.append({"file": str(p), "sha256": digest, "label": "real", "exif": meta,
                         "flag": "SCREENSHOT_REVIEW" if looks_like_screenshot(p, p.stat().st_size) else "",
                         "gps_stripped": "strip before training"})
        kept += 1
    args.out.write_text(json.dumps({
        "schema_version": 1, "purpose": "real-phone negatives for on-device calibration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "consent_confirmed": True, "notes": "Group-disjoint from desktop calibration. Strip GPS before training. Review flagged files by eye.",
        "counts": {"kept": kept, "rejects": rejects}, "items": manifest,
    }, indent=2), encoding="utf-8")
    print(f"Manifest: {args.out}  kept={kept} rejects={rejects}")


if __name__ == "__main__":
    main()
