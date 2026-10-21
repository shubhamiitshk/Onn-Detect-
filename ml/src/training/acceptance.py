"""Release acceptance floors for the shipped detectors.

AUC alone is not an acceptance criterion. A detector can have a respectable
AUC and no usable low-FPR operating point: the shipped image fusion has
AUC 0.7152 and ``tpr_at_fpr_1pct`` of 0.040, which is a coin flip in the
regime anyone would actually ship.

These floors exist so that number cannot hide behind the other one.

The floors are deliberately set at what the 2026 literature reports for
comparable systems (PRL: 96.40% balanced accuracy across eight benchmarks;
REALIS / DailyBench: real-world generalisation is the hard part), and
deliberately below the academic headline numbers, because the difference
between a benchmark image and a phone photo is the whole problem.
"""

from __future__ import annotations

# module name -> (min tpr@1%fpr, min tpr@0.1%fpr, min auc)
MINIMUM_ACCEPTANCE = {
    "srm_forensic_cnn_int8.tflite": (0.15, 0.05, 0.70),
    "semantic_probe_int8.tflite": (0.25, 0.10, 0.75),
    "lcnn_audio_int8.tflite": (0.50, 0.30, 0.85),
}

# The XGBoost fusion is what the image/video route actually ships, so it is
# gated separately and more strictly than any single engine.
# -> (min tpr@1%fpr, min tpr@0.1%fpr, min auc)
MINIMUM_FUSION_ACCEPTANCE = (0.30, 0.15, 0.85)

# A 0.1% FPR operating point needs enough REAL samples to be a statistic
# rather than one image. With n_real = 200 it is defined by 0.2 expected
# false positives, i.e. the single highest-scoring real image.
MIN_REAL_FOR_LOW_FPR_GATE = 2000


def _require_metric(test: dict, key: str, context: str) -> float:
    if key not in test:
        raise ValueError(f"{context}: sealed test report is missing '{key}'")
    value = test[key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"{context}: '{key}' must be a number, got {value!r}")
    return float(value)


def check_model_acceptance(
    name: str,
    test: dict,
    floors: tuple[float, float, float] | None = None,
    context: str = "",
    require_low_fpr_statistic: bool = True,
) -> None:
    """Raise unless a sealed test report clears the acceptance floors.

    ``floors`` is ``(min tpr@1%fpr, min tpr@0.1%fpr, min auc)``. When it is
    None the per-model floor from :data:`MINIMUM_ACCEPTANCE` is used.

    The 0.1% FPR floor is only enforced when the evaluation set is large
    enough for that operating point to mean anything. When
    ``require_low_fpr_statistic`` is False the metric must still be
    PRESENT, but is not required to clear the floor - which is what lets a
    small pilot set be staged while a distribution release is held to the
    stricter rule.
    """
    prefix = f"[{context}] " if context else ""
    if floors is None:
        floors = MINIMUM_ACCEPTANCE.get(name)
        if floors is None:
            return
    min_tpr_1, min_tpr_01, min_auc = floors

    auc = _require_metric(test, "auc", f"{prefix}{name}")
    if auc < min_auc:
        raise ValueError(f"{prefix}{name}: sealed test AUC {auc:.4f} < required {min_auc}")

    tpr_1 = _require_metric(test, "tpr_at_fpr_1pct", f"{prefix}{name}")
    if tpr_1 < min_tpr_1:
        raise ValueError(
            f"{prefix}{name}: TPR@1%FPR {tpr_1:.4f} < required {min_tpr_1}. "
            "AUC alone is not an acceptance criterion: a detector can have a "
            "respectable AUC and no usable low-FPR operating point."
        )

    tpr_01 = _require_metric(test, "tpr_at_fpr_0_1pct", f"{prefix}{name}")
    n_real = int(test.get("n_real", 0) or 0)
    if n_real < MIN_REAL_FOR_LOW_FPR_GATE:
        if require_low_fpr_statistic:
            raise ValueError(
                f"{prefix}{name}: sealed test has n_real={n_real}; a 0.1% FPR operating "
                f"point is defined by {0.001 * n_real:.1f} expected false positives and "
                f"is not a stable statistic. Use at least {MIN_REAL_FOR_LOW_FPR_GATE} "
                "real samples before gating on tpr_at_fpr_0_1pct."
            )
        return
    if tpr_01 < min_tpr_01:
        raise ValueError(f"{prefix}{name}: TPR@0.1%FPR {tpr_01:.4f} < required {min_tpr_01}")
