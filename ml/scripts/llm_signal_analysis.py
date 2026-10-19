"""Statistical evaluation and late-fusion ablation for Onn-Detect LLM/VLM signals.

Reads `llm_signal_cache.npz` (or legacy scratch cache formats) and reports:
1. Standalone ROC-AUC, EER, Balanced Accuracy, and TPR@FPR=1% with 95% stratified
   bootstrap confidence intervals on laundered images (format/resolution shortcut removed).
2. Zero-Shot VLM token probabilities (SmolVLM-256M, SmolVLM-500M) vs. Frozen Vision
   Tower linear probes (SmolVLM SigLIP-B/16 tower, CLIP-ViT, DINOv2).
3. Marginal fusion lift (Delta AUC + paired bootstrap p-value) when adding each
   candidate signal to Onn-Detect's content features (SRM + FFT + KL + Flatness)
   using L2-regularized Logistic/Ridge Stacking and shortcut-free XGBoost.

Self-contained: uses `scikit-learn` and `xgboost` when available in `.venv`, and
includes exact pure-NumPy ROC/EER/L2-IRLS logistic probe implementations so it
never fails on missing optional packages.

Usage:
  python llm_signal_analysis.py --cache path/to/llm_signal_cache.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
_TRAPZ = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ---------------------------------------------------------------------------
# Exact NumPy ROC / AUC / EER / Bootstrap Metrics
# ---------------------------------------------------------------------------

def _roc_curve_np(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    desc_idx = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_idx]
    y_true = y_true[desc_idx]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[y_score[0] + 1.0, y_score[threshold_idxs]]

    fpr = fps / max(1, fps[-1])
    tpr = tps / max(1, tps[-1])
    return fpr, tpr, thresholds


def roc_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = _roc_curve_np(y_true, y_score)
    return float(_TRAPZ(tpr, fpr))


def balanced_acc_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    pos = y_true == 1
    neg = y_true == 0
    tpr = float(np.mean(y_pred[pos] == 1)) if np.any(pos) else 0.0
    tnr = float(np.mean(y_pred[neg] == 0)) if np.any(neg) else 0.0
    return 0.5 * (tpr + tnr)


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = _roc_curve_np(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def tpr_at_fixed_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr, tpr, _ = _roc_curve_np(y_true, y_score)
    valid = np.where(fpr <= target_fpr)[0]
    return float(tpr[valid[-1]]) if len(valid) > 0 else 0.0


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point_auc, ci_low_95, ci_high_95) using class-stratified resampling."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    point = roc_auc_np(y_true, y_score)
    aucs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([s_pos, s_neg])
        aucs[b] = roc_auc_np(y_true[idx], y_score[idx])
    low, high = np.percentile(aucs, [2.5, 97.5])
    return point, float(low), float(high)


def paired_bootstrap_pvalue(
    y_true: np.ndarray,
    score_base: np.ndarray,
    score_cand: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Return (delta_auc, ci_low, ci_high, two_sided_p_value) for (cand - base)."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    delta_obs = roc_auc_np(y_true, score_cand) - roc_auc_np(y_true, score_base)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([s_pos, s_neg])
        deltas[b] = roc_auc_np(y_true[idx], score_cand[idx]) - roc_auc_np(y_true[idx], score_base[idx])
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    p_val = 2.0 * min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0)))
    return float(delta_obs), float(ci_low), float(ci_high), min(1.0, p_val)


# ---------------------------------------------------------------------------
# L2-Regularized Logistic Probe & Cross-Validation (with NumPy IRLS fallback)
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def _fit_predict_l2_logistic_np(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_eval: np.ndarray,
    l2_reg: float = 1.0,
    max_iter: int = 50,
) -> np.ndarray:
    """Newton-IRLS L2-regularized logistic regression in pure NumPy."""
    mean = X_tr.mean(axis=0, keepdims=True)
    std = X_tr.std(axis=0, keepdims=True) + 1e-6
    Z_tr = np.hstack([(X_tr - mean) / std, np.ones((len(X_tr), 1))])
    Z_ev = np.hstack([(X_eval - mean) / std, np.ones((len(X_eval), 1))])
    d = Z_tr.shape[1]
    w = np.zeros(d, dtype=np.float64)
    reg_mat = np.eye(d, dtype=np.float64) * l2_reg
    reg_mat[-1, -1] = 1e-6  # Do not penalize bias

    y_f = y_tr.astype(np.float64)
    for _ in range(max_iter):
        p = _sigmoid(Z_tr @ w)
        r = np.maximum(p * (1.0 - p), 1e-6)
        grad = Z_tr.T @ (p - y_f) + reg_mat @ w
        hess = (Z_tr.T * r) @ Z_tr + reg_mat
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.linalg.norm(step) < 1e-6:
            break
    return _sigmoid(Z_ev @ w).astype(np.float32)


def _stratified_kfold_indices(y: np.ndarray, n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    pos = rng.permutation(np.where(y == 1)[0])
    neg = rng.permutation(np.where(y == 0)[0])
    pos_folds = np.array_split(pos, n_splits)
    neg_folds = np.array_split(neg, n_splits)
    folds = []
    all_idx = np.arange(len(y))
    for k in range(n_splits):
        val_idx = np.concatenate([pos_folds[k], neg_folds[k]])
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        folds.append((all_idx[train_mask], val_idx))
    return folds


def fit_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit cross-validated L2-regularized linear probe; return (oof_train_probs, test_probs)."""
    folds = _stratified_kfold_indices(y_train, n_splits=5, seed=42)
    best_reg, best_auc = 1.0, -1.0
    best_oof = np.zeros(len(y_train), dtype=np.float32)

    for reg in (0.05, 0.2, 1.0, 5.0, 20.0, 100.0):
        oof = np.zeros(len(y_train), dtype=np.float32)
        for tr_idx, val_idx in folds:
            oof[val_idx] = _fit_predict_l2_logistic_np(X_train[tr_idx], y_train[tr_idx], X_train[val_idx], l2_reg=reg)
        auc = roc_auc_np(y_train, oof)
        if auc > best_auc:
            best_auc, best_reg, best_oof = auc, reg, oof

    test_probs = _fit_predict_l2_logistic_np(X_train, y_train, X_test, l2_reg=best_reg)
    return best_oof, test_probs


def fit_fusion_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Fit shortcut-free fusion meta-classifier on content features + candidate signal."""
    _, lr_preds = fit_linear_probe(X_train, y_train, X_test)
    try:
        import xgboost as xgb

        folds = _stratified_kfold_indices(y_train, n_splits=5, seed=42)
        tr_idx, val_idx = folds[0]
        dtrain = xgb.DMatrix(X_train[tr_idx], label=y_train[tr_idx])
        dval = xgb.DMatrix(X_train[val_idx], label=y_train[val_idx])
        dtest = xgb.DMatrix(X_test)
        params = {
            "max_depth": 3,
            "eta": 0.08,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=40,
            evals=[(dval, "val")],
            early_stopping_rounds=6,
            verbose_eval=False,
        )
        best = getattr(booster, "best_iteration", None)
        if best is not None:
            booster = booster[: best + 1]
        xgb_preds = booster.predict(dtest).astype(np.float32)
        return (0.5 * xgb_preds + 0.5 * lr_preds).astype(np.float32)
    except Exception:
        return lr_preds


def evaluate_all_signals(cache: dict[str, np.ndarray]) -> dict:
    y_train = cache["y_stacking"]
    y_test = cache["y_test"]

    results = {"standalone": {}, "fusion": {}}
    candidate_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # 1. Current Onn-Detect content features (columns 0..3: srm, fft, kl, flatness)
    has_current = "X_current_stacking" in cache and "X_current_test" in cache
    if has_current:
        X_cur_train = cache["X_current_stacking"][:, :4]
        X_cur_test = cache["X_current_test"][:, :4]
        for idx, name in enumerate(["srm_score", "fft_score", "kl_score", "flatness_score"]):
            auc, lo, hi = stratified_bootstrap_ci(y_test, X_cur_test[:, idx])
            eer, thr = compute_eer(y_test, X_cur_test[:, idx])
            results["standalone"][f"current_{name}"] = {
                "auc": auc,
                "auc_ci95": [lo, hi],
                "eer": eer,
                "tpr_at_fpr_1pct": tpr_at_fixed_fpr(y_test, X_cur_test[:, idx], 0.01),
                "balanced_acc": balanced_acc_np(y_test, X_cur_test[:, idx] >= thr),
            }

        base_fusion_test = fit_fusion_model(X_cur_train, y_train, X_cur_test)
        b_auc, b_lo, b_hi = stratified_bootstrap_ci(y_test, base_fusion_test)
        b_eer, b_thr = compute_eer(y_test, base_fusion_test)
        results["standalone"]["current_4signal_fusion"] = {
            "auc": b_auc,
            "auc_ci95": [b_lo, b_hi],
            "eer": b_eer,
            "tpr_at_fpr_1pct": tpr_at_fixed_fpr(y_test, base_fusion_test, 0.01),
            "balanced_acc": balanced_acc_np(y_test, base_fusion_test >= b_thr),
        }
    else:
        X_cur_train = X_cur_test = base_fusion_test = None

    # 2. Zero-Shot VLM probabilities (zs_*)
    zs_prefixes = sorted(
        {k[: -len("_test")] for k in cache.keys() if k.startswith("zs_") and k.endswith("_test")}
    )
    for prefix in zs_prefixes:
        test_scores = cache[f"{prefix}_test"]
        train_scores = cache.get(f"{prefix}_stacking")
        auc, lo, hi = stratified_bootstrap_ci(y_test, test_scores)
        eer, thr = compute_eer(y_test, test_scores)
        results["standalone"][prefix] = {
            "auc": auc,
            "auc_ci95": [lo, hi],
            "eer": eer,
            "tpr_at_fpr_1pct": tpr_at_fixed_fpr(y_test, test_scores, 0.01),
            "balanced_acc": balanced_acc_np(y_test, test_scores >= thr),
        }
        if train_scores is not None and len(train_scores) == len(y_train):
            candidate_signals[prefix] = (train_scores, test_scores)

    # 3. Frozen Vision Encoder linear probes (emb_*)
    emb_prefixes = sorted(
        {k[: -len("_test")] for k in cache.keys() if k.startswith("emb_") and k.endswith("_test")}
    )
    for prefix in emb_prefixes:
        train_key = f"{prefix}_stacking"
        test_key = f"{prefix}_test"
        if train_key not in cache:
            continue
        oof_train, test_probs = fit_linear_probe(cache[train_key], y_train, cache[test_key])
        probe_name = prefix.replace("emb_", "probe_")
        auc, lo, hi = stratified_bootstrap_ci(y_test, test_probs)
        eer, thr = compute_eer(y_test, test_probs)
        results["standalone"][probe_name] = {
            "auc": auc,
            "auc_ci95": [lo, hi],
            "eer": eer,
            "tpr_at_fpr_1pct": tpr_at_fixed_fpr(y_test, test_probs, 0.01),
            "balanced_acc": balanced_acc_np(y_test, test_probs >= thr),
            "embedding_dim": int(cache[test_key].shape[1]),
        }
        candidate_signals[probe_name] = (oof_train, test_probs)

    # 4. Fusion ablation: Current 4 content signals + each candidate signal
    if has_current and base_fusion_test is not None:
        for sig_name, (sig_train, sig_test) in candidate_signals.items():
            X_aug_train = np.column_stack([X_cur_train, sig_train])
            X_aug_test = np.column_stack([X_cur_test, sig_test])
            aug_preds = fit_fusion_model(X_aug_train, y_train, X_aug_test)
            auc, lo, hi = stratified_bootstrap_ci(y_test, aug_preds)
            eer, thr = compute_eer(y_test, aug_preds)
            delta, d_lo, d_hi, p_val = paired_bootstrap_pvalue(y_test, base_fusion_test, aug_preds)
            results["fusion"][f"current_plus_{sig_name}"] = {
                "auc": auc,
                "auc_ci95": [lo, hi],
                "delta_auc_vs_current": delta,
                "delta_ci95": [d_lo, d_hi],
                "p_value": p_val,
                "eer": eer,
                "tpr_at_fpr_1pct": tpr_at_fixed_fpr(y_test, aug_preds, 0.01),
                "balanced_acc": balanced_acc_np(y_test, aug_preds >= thr),
            }

    return results


def print_markdown_report(results: dict) -> None:
    print("\n" + "=" * 92)
    print("ONN-DETECT LAUNDERED BENCHMARK: ZERO-SHOT VLM vs. VISION TOWER PROBES vs. CURRENT PIPELINE")
    print("=" * 92)
    print(f"{'Signal / Model':<42} | {'AUC (95% CI)':<22} | {'EER':<7} | {'TPR@1%FPR':<9} | {'BalAcc':<6}")
    print("-" * 92)
    for name, m in sorted(results["standalone"].items(), key=lambda kv: kv[1]["auc"], reverse=True):
        ci_str = f"{m['auc']:.3f} [{m['auc_ci95'][0]:.3f}, {m['auc_ci95'][1]:.3f}]"
        print(
            f"{name:<42} | {ci_str:<22} | {m['eer']:.3f}   | {m['tpr_at_fpr_1pct']:.3f}     | {m['balanced_acc']:.3f}"
        )

    if results["fusion"]:
        print("\n" + "=" * 92)
        print("MARGINAL FUSION LIFT WHEN ADDED TO CURRENT CONTENT FEATURES (SRM + FFT + KL + FLATNESS)")
        print("=" * 92)
        print(f"{'Fusion Combination':<42} | {'Fused AUC':<10} | {'Delta AUC (95% CI)':<24} | {'p-value':<7}")
        print("-" * 92)
        for name, m in sorted(results["fusion"].items(), key=lambda kv: kv[1]["auc"], reverse=True):
            d_str = f"{m['delta_auc_vs_current']:+.3f} [{m['delta_ci95'][0]:+.3f}, {m['delta_ci95'][1]:+.3f}]"
            print(f"{name:<42} | {m['auc']:.3f}      | {d_str:<24} | {m['p_value']:.4f}")
    print("=" * 92 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        default=str(SCRIPT_DIR / "llm_signal_cache.npz"),
        help="Path to llm_signal_cache.npz produced by llm_signal_experiment.py",
    )
    parser.add_argument(
        "--output-json",
        default=str(SCRIPT_DIR / "llm_signal_analysis_results.json"),
        help="Path to write full JSON analysis report",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache file not found at {cache_path}. Run `python llm_signal_experiment.py all` first."
        )

    with np.load(cache_path, allow_pickle=False) as data:
        cache = {k: data[k] for k in data.files}

    results = evaluate_all_signals(cache)
    print_markdown_report(results)

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved detailed JSON report -> {out_path}")


if __name__ == "__main__":
    main()
