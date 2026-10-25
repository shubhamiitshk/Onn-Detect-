"""Measure the dual-engine fusion and the cascade arbitration policy.

This script produces the numbers that justify (or refute) the dual-engine
design. It never assumes a result: it re-measures every configuration on the
sealed TEST split and reports the cascade trade-off explicitly.

It reads the artifacts written by `train_xgboost_meta.py`:
  * ``meta_features_cache.npz`` - the exact feature vectors the fusion was
    trained and evaluated on (slot 6 already holds the Engine C score)
  * ``xgboost_meta.json``      - the trained booster
  * ``cascade_policy.json``     - the policy fitted on the calibration split

Reported metrics
----------------
1. Engine A alone  - the fusion with slot 6 masked (probe unavailable)
2. Engine C alone  - the semantic probe on its own
3. Dual-engine     - the fusion with slot 6 populated
4. Laundered dual-engine (512px JPEG Q85) - content-only check
5. Cascade arbitration: coverage, resolved accuracy, disagreement rate and
   the resulting INCONCLUSIVE rate

Usage:
    python -m ml.scripts.evaluate_cascade --fusion-dir D:\\onn_data\\runs\\release_v6\\fusion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ML_ROOT / "src"))
sys.path.insert(0, str(ML_ROOT / "scripts"))

from meta_classifier.dual_engine import (
    CascadePolicy,
    decide,
    fit_escalation_band,
    mask_semantic_probe,
)
from meta_classifier.feature_extractor import SEMANTIC_PROBE_INDEX
from training.metrics import compute_auc, compute_eer, eer_operating_threshold


def _auc(labels, scores) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.5


def _load_booster(fusion_dir: Path):
    import xgboost as xgb
    from meta_classifier.feature_extractor import FEATURE_NAMES

    booster = xgb.Booster()
    booster.load_model(str(fusion_dir / "xgboost_meta.json"))
    booster.feature_names = list(FEATURE_NAMES)
    return booster


BASE_MARGIN_EPS = 1e-6


def _probe_base_margin(X: np.ndarray) -> np.ndarray:
    """logit(slot 6), and 0.0 wherever the probe is unavailable (NaN)."""
    probe = np.asarray(X[:, SEMANTIC_PROBE_INDEX], dtype=np.float64)
    margin = np.zeros(len(probe), dtype=np.float64)
    available = np.isfinite(probe)
    q = np.clip(probe[available], BASE_MARGIN_EPS, 1.0 - BASE_MARGIN_EPS)
    margin[available] = np.log(q / (1.0 - q))
    return margin


def _predict(booster, X: np.ndarray, base_margin_mode: str = "probe") -> np.ndarray:
    import xgboost as xgb
    from meta_classifier.feature_extractor import FEATURE_NAMES

    if base_margin_mode == "probe":
        bm = _probe_base_margin(X)
    elif base_margin_mode == "none":
        bm = None
    else:
        raise ValueError(f"Unsupported fusion_base_margin mode: {base_margin_mode!r} (fail closed)")
    dmatrix = xgb.DMatrix(X, feature_names=list(FEATURE_NAMES), base_margin=bm)
    return booster.predict(dmatrix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-dir", default=None, help="Directory written by train_xgboost_meta.py")
    parser.add_argument("--config", default=None, help="Optional config YAML to locate fusion directory")
    parser.add_argument("--output", default=None, help="Optional path for the JSON report")
    args = parser.parse_args()

    fusion_dir = None
    if args.fusion_dir:
        fusion_dir = Path(args.fusion_dir).resolve()
    elif args.config:
        cfg_path = Path(args.config).resolve()
        candidates = [
            cfg_path.parent / "exported",
            Path("release_v9/fusion").resolve(),
            Path("fusion").resolve(),
        ]
        for c in candidates:
            if (c / "meta_features_cache.npz").is_file():
                fusion_dir = c
                break
    if fusion_dir is None:
        if Path("release_v9/fusion/meta_features_cache.npz").is_file():
            fusion_dir = Path("release_v9/fusion").resolve()
        else:
            raise SystemExit("Missing --fusion-dir or --config pointing to a directory with meta_features_cache.npz")

    cache_path = fusion_dir / "meta_features_cache.npz"
    if not cache_path.is_file():
        raise SystemExit(f"Missing {cache_path}; run train_xgboost_meta.py first")

    with np.load(cache_path) as data:
        X_cal = data["X_calibration"]
        y_cal = data["y_calibration"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        X_laund = data["X_test_laundered_512_q85"] if "X_test_laundered_512_q85" in data.files else None

    booster = _load_booster(fusion_dir)
    results_path = fusion_dir / "meta_results.json"
    meta = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else {}
    fusion_threshold = float(meta.get("calibration", {}).get("threshold", 0.5))
    base_margin_mode = (
        meta.get("fusion_base_margin")
        or (meta.get("identity") or {}).get("fusion_base_margin")
        or ("probe" if "base_margin_intercept" in meta else "none")
    )

    # A release built before the dual engine held slot 6 at a neutral
    # CONSTANT. A constant is not a measurement, so it must not be mistaken
    # for a probe. Only a varying, finite slot counts as Engine C.
    #
    # The check uses the unique-value count, not the standard deviation:
    # float32 rounding makes a constant column report std ~1.5e-08, which a
    # `std() > 0` test would wrongly accept.
    probe_column = X_cal[:, SEMANTIC_PROBE_INDEX]
    has_probe = bool(
        np.isfinite(probe_column).all() and np.unique(probe_column).size > 1
    )



    report: dict = {"release_id": meta.get("release_id"), "fusion_threshold": fusion_threshold,
                    "engine_c_present": has_probe}

    print("=" * 92)
    print("DUAL-ENGINE MEASUREMENTS")
    print("=" * 92)

    # ── Engine A alone: the probe is unavailable, slot 6 masked ────────────
    # This is only a meaningful measurement when the booster was TRAINED with
    # a probe-free path. Without masked training rows the missing branch is
    # arbitrary and the number below would be an artefact of the tree shape
    # rather than a property of Engine A.
    mask_rate = float(meta.get("dual_engine", {}).get("cascade_mask_rate", 0.0))
    X_cal_a = mask_semantic_probe(X_cal)
    X_test_a = mask_semantic_probe(X_test)
    cal_a = _predict(booster, X_cal_a, base_margin_mode)
    test_a = _predict(booster, X_test_a, base_margin_mode)
    report["engine_a"] = {
        "calibration_auc": _auc(y_cal, cal_a),
        "test_auc": _auc(y_test, test_a),
        "laundered_test_auc": _auc(y_test, _predict(booster, mask_semantic_probe(X_laund), base_margin_mode)) if X_laund is not None else None,
        "cascade_mask_rate": mask_rate,
        "probe_free_path_is_trained": mask_rate > 0.0,
    }
    print(f"Engine A (probe unavailable)  calibration AUC : {report['engine_a']['calibration_auc']:.4f}")
    print(f"Engine A (probe unavailable)  test AUC       : {report['engine_a']['test_auc']:.4f}")
    if report["engine_a"]["laundered_test_auc"] is not None:
        print(f"Engine A (probe unavailable)  laundered AUC  : {report['engine_a']['laundered_test_auc']:.4f}")
    if has_probe and mask_rate <= 0.0:
        print(
            "[WARNING] The booster was trained with slot 6 always populated, so it has "
            "no trained probe-free path. The 'Engine A alone' numbers above are an "
            "artefact of an arbitrary missing branch, NOT a property of Engine A. "
            "Retrain with --cascade-mask-rate 0.25 for an honest comparison."
        )

    if not has_probe:
        print("\n[Note] This release ships no Engine C. The dual-engine path is inactive.")
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return

    # ── Engine C alone ────────────────────────────────────────────────────
    probe_cal = X_cal[:, SEMANTIC_PROBE_INDEX]
    probe_test = X_test[:, SEMANTIC_PROBE_INDEX]
    _, probe_threshold = eer_operating_threshold(y_cal, probe_cal)
    report["engine_c"] = {
        "calibration_auc": _auc(y_cal, probe_cal),
        "test_auc": _auc(y_test, probe_test),
        "threshold": float(probe_threshold),
    }
    if X_laund is not None:
        report["engine_c"]["laundered_test_auc"] = _auc(y_test, X_laund[:, SEMANTIC_PROBE_INDEX])
    print(f"Engine C (semantic probe)     calibration AUC : {report['engine_c']['calibration_auc']:.4f}")
    print(f"Engine C (semantic probe)     test AUC       : {report['engine_c']['test_auc']:.4f}")
    if "laundered_test_auc" in report["engine_c"]:
        print(f"Engine C (semantic probe)     laundered AUC  : {report['engine_c']['laundered_test_auc']:.4f}")

    # ── Dual-engine fusion ────────────────────────────────────────────────
    cal_dual = _predict(booster, X_cal, base_margin_mode)
    test_dual = _predict(booster, X_test, base_margin_mode)
    report["dual_engine"] = {
        "calibration_auc": _auc(y_cal, cal_dual),
        "test_auc": _auc(y_test, test_dual),
    }
    if X_laund is not None:
        report["dual_engine"]["laundered_test_auc"] = _auc(y_test, _predict(booster, X_laund, base_margin_mode))
    print(f"Dual-engine fusion            calibration AUC : {report['dual_engine']['calibration_auc']:.4f}")
    print(f"Dual-engine fusion            test AUC       : {report['dual_engine']['test_auc']:.4f}")
    if "laundered_test_auc" in report["dual_engine"]:
        print(f"Dual-engine fusion            laundered AUC  : {report['dual_engine']['laundered_test_auc']:.4f}")

    lift = report["dual_engine"]["test_auc"] - report["engine_a"]["test_auc"]
    report["dual_engine"]["test_auc_lift_over_engine_a"] = float(lift)
    print(f"-> Dual-engine lift over Engine A on the sealed test split: {lift:+.4f}")

    # ── Cascade policy ────────────────────────────────────────────────────
    policy_path = fusion_dir / "cascade_policy.json"
    if policy_path.is_file():
        policy = CascadePolicy.load(policy_path)
    else:
        policy, table = fit_escalation_band(
            cal_dual, probe_cal, y_cal,
            engine_a_threshold=fusion_threshold,
            engine_c_threshold=float(probe_threshold),
        )
        print("\n[Note] No fitted policy found; refitting on the calibration split.")

    cascade = {"policy": policy.to_dict()}
    verdicts, escalated_flags, reasons = [], [], []
    for a_score, c_score in zip(test_dual, probe_test):
        decision = decide(float(a_score), float(c_score), policy)
        verdicts.append(decision.verdict)
        escalated_flags.append(decision.escalated)
        reasons.append(decision.reason)
    verdicts = np.array(verdicts)
    escalated_flags = np.array(escalated_flags)
    y_fake = y_test == 1

    classified = verdicts != "INCONCLUSIVE"
    correct = np.zeros(len(y_test), dtype=bool)
    correct[classified] = (
        (verdicts[classified] == "SYNTHETIC") == y_fake[classified]
    )
    cascade["sealed_test"] = {
        "n": int(len(y_test)),
        "classified_rate": float(classified.mean()),
        "inconclusive_rate": float((~classified).mean()),
        "escalation_rate": float(escalated_flags.mean()),
        "accuracy_over_classified": float(correct[classified].mean()) if classified.any() else 0.0,
        "accuracy_over_all": float(correct.mean()),
    }
    print("\n" + "=" * 92)
    print("CASCADE ARBITRATION (sealed test split)")
    print("=" * 92)
    print(f"Policy band              : [{policy.escalate_below:.4f}, {policy.escalate_above:.4f}]")
    print(f"Engine A threshold       : {policy.engine_a_threshold:.6f}")
    print(f"Engine C threshold       : {policy.engine_c_threshold:.6f}")
    print(f"Calibration coverage     : {policy.coverage:.4f}")
    print(f"Calibration agreement    : {policy.agreement_rate:.4f}")
    print(f"Test escalation rate     : {cascade['sealed_test']['escalation_rate']:.4f}")
    print(f"Test INCONCLUSIVE rate   : {cascade['sealed_test']['inconclusive_rate']:.4f}")
    print(f"Test accuracy (classified): {cascade['sealed_test']['accuracy_over_classified']:.4f}")
    report["cascade"] = cascade

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
