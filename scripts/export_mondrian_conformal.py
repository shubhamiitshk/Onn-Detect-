#!/usr/bin/env python3
"""Export Mondrian (stratified) conformal calibration for the fusion.

ROADMAP v10.2: export stratified non-conformity quantiles
`q_hat_{1-alpha}(m)` into `assets/constants/conformal_calibration.json` over
the 4 primary missingness masks, expose `|C(X, m)|` in the UI, abstain to
INCONCLUSIVE when `|C| != 1` (exchangeability required for bounds).

This script computes quantiles from a scored calibration file you supply.
It never fabricates quantiles: without input it exits 2 with instructions.

Input JSON (list): [{"score": float in [0,1], "label": 0|1, "mask": str}, ...]
Masks e.g. "full", "no-probe", "no-c2pa", "no-lipsync" — must match
FusionFeatureSchema missingness masks used at runtime.

Output: assets/constants/conformal_calibration.json
  {"schema_version": 1, "alpha": 0.1, "masks": {mask: {"q_hat": q, "n_calibration": n, "split_manifest_hash": h}}}
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import math

def quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals: return float("nan")
    # Conformal finite-sample correction: ceil((n+1)*q)-th order statistic.
    n = len(sorted_vals)
    k = math.ceil((n + 1) * q)
    return sorted_vals[min(max(k - 1, 0), n - 1)]

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True, help="calibration scores JSON (score,label,mask)")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("assets/constants/conformal_calibration.json"))
    ap.add_argument("--split-manifest-hash", default="", help="SHA-256 of the split manifest these scores came from")
    args = ap.parse_args()
    if not args.scores.is_file():
        print(f"REFUSING: scores file missing: {args.scores}", file=sys.stderr); sys.exit(2)
    rows = json.loads(args.scores.read_text())
    by_mask: dict[str, list[float]] = {}
    for r in rows:
        try:
            s, y, m = float(r["score"]), int(r["label"]), str(r["mask"])
        except Exception:
            print("REFUSING: each row needs score in [0,1], label 0|1, mask str", file=sys.stderr); sys.exit(2)
        if not (0.0 <= s <= 1.0) or y not in (0, 1):
            print("REFUSING: score out of [0,1] or label not 0|1", file=sys.stderr); sys.exit(2)
        # Non-conformity for Mondrian: 1 - p(true class). Fusion raises P(AI);
        # true class score is s if y==1 else 1-s; non-conformity = 1 - that.
        nc = 1.0 - (s if y == 1 else 1.0 - s)
        by_mask.setdefault(m, []).append(nc)
    if not by_mask:
        print("REFUSING: no rows", file=sys.stderr); sys.exit(2)
    out = {"schema_version": 1, "alpha": args.alpha,
           "masks": {m: {"q_hat": quantile(sorted(v), 1 - args.alpha), "n_calibration": len(v),
                         "split_manifest_hash": args.split_manifest_hash} for m, v in sorted(by_mask.items())}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out} masks={sorted(by_mask)} alpha={args.alpha}")

if __name__ == "__main__":
    main()
