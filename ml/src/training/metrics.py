"""Binary detection metrics: real=0, synthetic=1, higher score=more synthetic.

No automatic polarity reversal. Below-chance OOD results remain visible and
must be diagnosed using source labels and matched frontend/model lineage.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve


class DegenerateScores(RuntimeError):
    """Raised when scores are constant/NaN - the model did not discriminate."""


def assert_scores_sane(y_true, y_pred, context: str = "") -> None:
    """Reject empty, non-finite, constant or invalid binary evaluation data.

    Below-chance AUC can reflect distribution shift as well as polarity bugs;
    report it directly and diagnose it rather than silently flipping scores.
    """
    scores = np.asarray(y_pred, dtype=np.float64)
    truth = np.asarray(y_true).reshape(-1)
    scores = scores.reshape(-1)
    if truth.size != scores.size or not np.isin(truth, [0, 1]).all():
        raise ValueError("Scores/labels must have equal length and binary labels")
    labels = np.unique(truth)
    prefix = f"[{context}] " if context else ""

    if scores.size == 0:
        raise DegenerateScores(f"{prefix}no scores produced")
    if not np.all(np.isfinite(scores)):
        bad = int((~np.isfinite(scores)).sum())
        raise DegenerateScores(
            f"{prefix}{bad}/{scores.size} non-finite scores (NaN/Inf) - "
            "the front end is producing garbage"
        )
    if float(scores.std()) < 1e-6:
        raise DegenerateScores(
            f"{prefix}scores are constant (std={scores.std():.2e}, "
            f"mean={scores.mean():.4f}) - model collapsed or front end "
            "returned zeros; AUC/EER are meaningless"
        )
    if labels.size < 2:
        raise ValueError(
            f"{prefix}need both classes to score, got {labels.tolist()}"
        )


def compute_auc(y_true, y_pred, *, sanity: bool = True, context: str = ""):
    """ROC-AUC with 1 = FAKE and higher score = more fake. 0.5 = chance."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_pred, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        raise ValueError("AUC requires both real and synthetic classes")
    if sanity:
        assert_scores_sane(y_true, y_score, context)
    return roc_auc_score(y_true, y_score)


def compute_accuracy(y_true, y_pred, threshold=0.5):
    preds = (np.asarray(y_pred, dtype=np.float64) >= threshold).astype(int)
    return accuracy_score(np.asarray(y_true), preds)


def compute_eer(y_true, y_pred, *, sanity: bool = True, context: str = ""):
    """Equal Error Rate with 1 = FAKE and higher score = more fake.

    Returns 0.0 for a perfect detector and 0.5 for chance. A return value near
    1.0 is the signature of an inverted score and should be treated as a bug,
    not as a weak model.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_pred, dtype=np.float64)
    if len(y_score) == 0 or len(np.unique(y_true)) < 2:
        raise ValueError("EER requires nonempty scores and both classes")
    if sanity:
        assert_scores_sane(y_true, y_score, context)

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_f1_score(y_true, y_pred, threshold=0.5):
    from sklearn.metrics import f1_score

    preds = (np.asarray(y_pred, dtype=np.float64) >= threshold).astype(int)
    return float(f1_score(np.asarray(y_true), preds, zero_division=0))


def eer_operating_threshold(y_true, y_pred, *, context: str = ""):
    """(eer, threshold) for ONN-DETECT polarity.

    The returned threshold lives on the SAME scale as y_pred (higher = more
    fake), so `score >= threshold` flags synthetic. Keeping both values in one
    convention is what stops the historical bug where the EER and the threshold
    were computed with opposite sign conventions and shipped together.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_pred, dtype=np.float64)
    if len(y_score) == 0 or len(np.unique(y_true)) < 2:
        raise ValueError("Threshold calibration requires both classes")
    assert_scores_sane(y_true, y_score, context)

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def polarity_report(y_true, y_pred, context: str = "") -> dict:
    """Both polarities at once, so an inversion is visible in the output."""
    y_score = np.asarray(y_pred, dtype=np.float64)
    direct = compute_auc(y_true, y_score, sanity=False)
    flipped = compute_auc(y_true, 1.0 - y_score, sanity=False)
    eer_direct = compute_eer(y_true, y_score, sanity=False)
    eer_flipped = compute_eer(y_true, 1.0 - y_score, sanity=False)
    return {
        "auc_as_model_emits": float(direct),
        "auc_inverted": float(flipped),
        "eer_as_model_emits": float(eer_direct),
        "eer_inverted": float(eer_flipped),
        "suspicious": bool(direct < 0.5),
    }


def plot_roc_curve(y_true, y_pred, save_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_pred, dtype=np.float64))
        auc_val = compute_auc(y_true, y_pred, sanity=False)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc_val:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def plot_confusion_matrix(y_true, y_pred, save_path, threshold=0.5):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        preds = (np.asarray(y_pred, dtype=np.float64) >= threshold).astype(int)
        cm = confusion_matrix(np.asarray(y_true), preds)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"]).plot(
            cmap="Blues", values_format="d"
        )
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    except Exception:
        pass