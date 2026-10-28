#!/usr/bin/env python3
"""Lightweight documentation-consistency gate.

Checks that the repository's documentation does not reference files that no
longer exist, and that the release-framing invariants from CLAUDE.md hold.
"""
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    errors = []

    # 1. No tracked file may reference the deleted root algorithm module.
    deleted = "onn_prism_v11_algorithms.py"
    canonical = "ml/src/models/prism_v11_algorithms.py"
    try:
        out = subprocess.check_output(
            ["git", "grep", "-l", deleted, "--", "."],
            cwd=root, text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        out = ""
    this_script = Path(__file__).resolve().relative_to(root).as_posix()
    stale = [p for p in out.splitlines() if p and p != deleted and Path(p).as_posix() != this_script]
    # Allow references that already point at the canonical path.
    for p in stale:
        text = (root / p).read_text(encoding="utf-8", errors="replace")
        # Flag only lines that mention the bare deleted filename without the canonical path.
        for line in text.splitlines():
            if deleted in line and canonical not in line:
                errors.append(f"{p}: stale reference to deleted '{deleted}'")
                break

    # 2. Required authoritative docs must exist.
    for required in ("CLAUDE.md", "README.md", "docs/FIX_STATUS_2026-10-07.md"):
        if not (root / required).exists():
            errors.append(f"Missing required doc: {required}")

    # 3. README must label benchmark summary as historical/aspirational.
    readme = (root / "README.md").read_text(encoding="utf-8")
    if "Historical / Aspirational" not in readme:
        errors.append("README.md missing 'Historical / Aspirational' benchmark label")

    # 4. Configs must be annotated experimental if present.
    for cfg in ("ml/configs/srm_v11_prism.yaml", "ml/configs/lcnn_v11_prism.yaml",
                "ml/configs/siglip_tap_v11_prism.yaml", "ml/configs/xgboost_v11_prism.yaml"):
        p = root / cfg
        if p.exists() and "EXPERIMENTAL / NOT SHIPPED" not in p.read_text(encoding="utf-8"):
            errors.append(f"{cfg} missing experimental annotation")

    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1
    print("Docs consistency: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())