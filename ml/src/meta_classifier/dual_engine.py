"""Dual-engine feature assembly and cascade policy.

This module is the Python source of truth for the "Engine A + Lite C"
cascade. ``android/.../domain/pipeline/CascadePolicy.kt`` is a hand-written
mirror of :class:`CascadePolicy` and of :func:`assemble_features`; the two
are checked against each other by ``ml/tests/test_dual_engine_contract.py``.

Terminology
-----------
Engine A
    Artifact engine. The SRM+NPR(+RGB) forensic CNN plus the FFT/KL spectral
    descriptors. ~141 KB, ~5 ms, 0.592-0.655 AUC on the v4 release.

Engine C ("Lite C")
    Semantic engine. A 224x224 appearance-stem CNN distilled from a frozen
    CLIP vision tower. ~3-38 MB, ~25 ms, 0.80-0.885 AUC.

Fusion
    The XGBoost meta-classifier over the canonical ten-signal schema. The
    Engine C score occupies slot 6 (``semantic_probe_score``).

Cascade arbitration
    Two independent detectors that disagree are evidence of uncertainty, not
    evidence of authenticity. The policy therefore abstains on disagreement
    rather than picking a winner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Fusion slot that carries the Engine C measurement.
SEMANTIC_PROBE_INDEX = 6

# Score convention shared by every detector: real=0, fake=1.
REAL_LABEL = 0
FAKE_LABEL = 1


def assemble_features(
    srm_score: float,
    fft_score: float,
    kl_score: float,
    flatness_score: float,
    resolution_norm: float,
    is_compressed: bool,
    semantic_probe_score: Optional[float],
    neutral_c2pa: float = 0.05,
    neutral_lip_sync: float = 0.9,
    neutral_jitter: float = 0.1,
) -> np.ndarray:
    """Build the canonical ten-signal vector.

    ``semantic_probe_score=None`` encodes Engine C as UNAVAILABLE (``NaN``).
    A neutral constant would be indistinguishable from a real low reading.
    """
    probe = float("nan") if semantic_probe_score is None else float(semantic_probe_score)
    if semantic_probe_score is not None:
        if not np.isfinite(probe):
            raise ValueError("Semantic probe score must be finite when supplied")
        if not 0.0 <= probe <= 1.0:
            raise ValueError(f"Semantic probe score out of range: {semantic_probe_score}")

    vector = np.array([
        srm_score, fft_score, kl_score, flatness_score,
        resolution_norm, 1.0 if is_compressed else 0.0,
        probe, neutral_c2pa, neutral_lip_sync, neutral_jitter,
    ], dtype=np.float32)

    finite = np.isfinite(vector)
    for index in (0, 1, 2, 3, 4, 5, 7, 8, 9):
        if not finite[index]:
            raise ValueError(f"Feature {index} must be finite")
    return vector


def mask_semantic_probe(features: np.ndarray) -> np.ndarray:
    """Return a copy with slot 6 marked unavailable (Engine C did not run)."""
    masked = np.array(features, dtype=np.float32, copy=True)
    masked[:, SEMANTIC_PROBE_INDEX] = np.nan
    return masked


def apply_semantic_probe(features: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Return a copy with slot 6 populated by the Engine C scores."""
    out = np.array(features, dtype=np.float32, copy=True)
    out[:, SEMANTIC_PROBE_INDEX] = np.asarray(scores, dtype=np.float32)
    return out


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    return -(clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped))


def normalized_margin(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Signed distance from the operating threshold, scaled to [-1, 1].

    ``+1`` is "one full confidence span above the threshold", ``-1`` is the
    same below. Used to compare how close two detectors are to their own
    operating points; the two thresholds may differ.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold out of range: {threshold}")
    return np.clip((np.asarray(scores, dtype=np.float64) - threshold) / max(threshold, 1.0 - threshold), -1.0, 1.0)


@dataclass
class CascadePolicy:
    """Measured arbitration policy for the two engines.

    Attributes:
        engine_a_threshold: Frozen operating threshold for the fusion score.
        engine_c_threshold: Frozen operating threshold for the probe score.
        escalate_below: Engine C scores below this are treated as ambiguous.
        escalate_above: Engine C scores above this are treated as ambiguous.
        coverage: Fraction of samples resolved without arbitration.
        agreement_rate: Measured agreement between the two engines.
        release_id: Release the policy was fitted against.
    """

    engine_a_threshold: float
    engine_c_threshold: float
    escalate_below: float
    escalate_above: float
    coverage: float = 1.0
    agreement_rate: float = 0.0
    release_id: str = "unvalidated"

    def validate(self) -> None:
        for name, value in (
            ("engine_a_threshold", self.engine_a_threshold),
            ("engine_c_threshold", self.engine_c_threshold),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite probability, got {value}")
        if not 0.0 <= self.escalate_below <= self.escalate_above <= 1.0:
            raise ValueError("Escalation band must satisfy 0 <= below <= above <= 1")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be a probability")
        if not 0.0 <= self.agreement_rate <= 1.0:
            raise ValueError("agreement_rate must be a probability")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CascadePolicy":
        policy = cls(
            engine_a_threshold=float(data["engine_a_threshold"]),
            engine_c_threshold=float(data["engine_c_threshold"]),
            escalate_below=float(data["escalate_below"]),
            escalate_above=float(data["escalate_above"]),
            coverage=float(data.get("coverage", 1.0)),
            agreement_rate=float(data.get("agreement_rate", 0.0)),
            release_id=str(data.get("release_id", "unvalidated")),
        )
        policy.validate()
        return policy

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CascadePolicy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class CascadeDecision:
    """Outcome of applying the policy to one sample."""

    verdict: str  # "SYNTHETIC" | "AUTHENTIC" | "INCONCLUSIVE"
    score: float
    escalated: bool
    reason: str

    @property
    def is_classified(self) -> bool:
        return self.verdict in ("SYNTHETIC", "AUTHENTIC")


def decide(
    engine_a_score: float,
    engine_c_score: Optional[float],
    policy: CascadePolicy,
) -> CascadeDecision:
    """Apply the cascade arbitration policy to one sample.

    ``engine_a_score`` is the FUSION score and is the stronger signal
    (sealed test AUC 0.7152 vs Engine C 0.6871), so it is the score that
    decides and the score that is reported.

    Engine C unavailable -> the fusion decides (no arbitration possible).
    Engine C available and unambiguous -> the fusion decides.
    Engine C ambiguous -> arbitrate: agreement resolves, disagreement abstains.

    The reported score and the threshold used to compare it must come from
    the SAME engine. Reporting Engine C's score while deciding with the
    fusion would produce a score that contradicts its own verdict, and
    reporting Engine C's score when the fusion decided would apply the
    wrong operating point to the stronger signal.

    Mirrors ``CascadePolicy.decide`` in Kotlin exactly, including the input
    guards, so the desktop measurements describe the shipped behaviour.
    """
    policy.validate()
    if not np.isfinite(engine_a_score) or not 0.0 <= engine_a_score <= 1.0:
        raise ValueError(f"Engine A score must be a finite probability, got {engine_a_score}")
    if engine_c_score is not None and (
        not np.isfinite(engine_c_score) or not 0.0 <= engine_c_score <= 1.0
    ):
        raise ValueError(f"Engine C score must be a finite probability, got {engine_c_score}")

    if engine_c_score is None or not np.isfinite(engine_c_score):
        verdict = "SYNTHETIC" if engine_a_score >= policy.engine_a_threshold else "AUTHENTIC"
        return CascadeDecision(verdict, float(engine_a_score), False, "Engine C unavailable; fusion alone")

    engine_c_score = float(engine_c_score)
    a_says_fake = engine_a_score >= policy.engine_a_threshold
    c_says_fake = engine_c_score >= policy.engine_c_threshold

    if policy.escalate_below <= engine_c_score <= policy.escalate_above:
        if a_says_fake == c_says_fake:
            verdict = "SYNTHETIC" if a_says_fake else "AUTHENTIC"
            return CascadeDecision(verdict, float(engine_a_score), True, "Fusion and Engine C agree")
        return CascadeDecision(
            "INCONCLUSIVE", float(engine_a_score), True,
            "Fusion/Engine C disagreement; abstaining rather than selecting a winner",
        )

    verdict = "SYNTHETIC" if a_says_fake else "AUTHENTIC"
    return CascadeDecision(verdict, float(engine_a_score), False, "Fusion confident")


def fit_escalation_band(
    engine_a_scores: np.ndarray,
    engine_c_scores: np.ndarray,
    labels: np.ndarray,
    engine_a_threshold: float,
    engine_c_threshold: float,
    coverage_grid: Optional[np.ndarray] = None,
    min_classified_rate: float = 0.85,
    min_escalation_rate: float = 0.05,
) -> tuple[CascadePolicy, list[dict[str, float]]]:
    """Fit the escalation band on the CALIBRATION split.

    The band is the interval around ``engine_c_threshold`` in which Engine C
    is treated as ambiguous and arbitration is required.

    The half-width is chosen by INVERSE CDF, not by a fixed fraction of the
    threshold: for a target coverage ``c`` the half-width is the
    ``(1 - c)`` quantile of ``|engine_c_score - engine_c_threshold|`` on the
    calibration split. That makes the requested coverage an ACHIEVED
    coverage. A fixed half-width would cover a wildly different fraction of
    samples depending on how the score distribution is shaped, and the
    shipped policy would silently disagree with the measured table.

    Which engine decides
    --------------------
    ``engine_a_scores`` is the FUSION score, and it is the stronger signal
    (sealed test AUC 0.7152 vs Engine C 0.6871). The fusion therefore
    decides OUTSIDE the band. Inside the band the two are compared, and a
    disagreement abstains.

    Objective
    ---------
    The band is chosen to maximise SELECTIVE ACCURACY - accuracy over the
    samples that actually receive a verdict - subject to two constraints:

    * ``min_classified_rate``: at least this fraction of samples must get a
      definite verdict. Without it the fitter can always widen the band
      until only the samples it agrees with remain, which is the same
      failure as ranking on resolved accuracy alone.
    * ``min_escalation_rate``: at least this fraction must go through
      arbitration. Without it the fitter collapses to a zero-width band,
      which disables the cascade entirely.

    Returns the fitted policy and the full Pareto table (every candidate band
    with its coverage, accuracy and disagreement rate).
    """
    if coverage_grid is None:
        coverage_grid = np.arange(0.30, 1.0001, 0.02)
    coverage_grid = np.clip(np.asarray(coverage_grid, dtype=np.float64), 0.0, 1.0)
    if not 0.0 < min_classified_rate <= 1.0:
        raise ValueError("min_classified_rate must be in (0, 1]")
    if not 0.0 <= min_escalation_rate < 1.0:
        raise ValueError("min_escalation_rate must be in [0, 1)")

    a_scores = np.asarray(engine_a_scores, dtype=np.float64)
    c_scores = np.asarray(engine_c_scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if not (len(a_scores) == len(c_scores) == len(y)):
        raise ValueError("Engine A/C scores and labels must be the same length")
    if len(y) == 0:
        raise ValueError("Cannot fit an escalation band on an empty split")
    if not 0.0 <= engine_c_threshold <= 1.0:
        raise ValueError("Engine C threshold must be a probability")

    a_says_fake = a_scores >= engine_a_threshold
    c_says_fake = c_scores >= engine_c_threshold
    distances = np.abs(c_scores - engine_c_threshold)

    table: list[dict[str, float]] = []
    # (selective_accuracy, classified_rate, below, above, disagreement_rate)
    best: Optional[tuple[float, float, float, float, float]] = None

    for coverage in coverage_grid:
        coverage = float(coverage)
        escalate_fraction = 1.0 - coverage
        if escalate_fraction <= 0.0:
            half_width = 0.0
        else:
            half_width = float(np.quantile(distances, escalate_fraction))
        below = max(0.0, engine_c_threshold - half_width)
        above = min(1.0, engine_c_threshold + half_width)
        if below > above:  # pragma: no cover - defensive
            below, above = above, below

        escalated = (c_scores >= below) & (c_scores <= above)
        resolved = ~escalated
        disagreed = escalated & (a_says_fake != c_says_fake)
        arbitrated = escalated & ~disagreed
        # Samples that receive a definite verdict: outside the band the
        # fusion decides; inside it, agreement decides. A disagreement
        # abstains and is NOT classified.
        classified = resolved | arbitrated

        resolved_accuracy = (
            float((a_says_fake[resolved] == (y[resolved] == FAKE_LABEL)).mean())
            if resolved.any() else 0.0
        )
        arbitration_accuracy = (
            float((a_says_fake[arbitrated] == (y[arbitrated] == FAKE_LABEL)).mean())
            if arbitrated.any() else 0.0
        )
        disagreement_rate = float(disagreed.mean())
        achieved_coverage = float(resolved.mean())
        classified_rate = float(classified.mean())
        escalation_rate = float(escalated.mean())
        selective_accuracy = (
            float((a_says_fake[classified] == (y[classified] == FAKE_LABEL)).mean())
            if classified.any() else 0.0
        )

        table.append({
            "target_coverage": coverage,
            "coverage": achieved_coverage,
            "escalate_below": below,
            "escalate_above": above,
            "half_width": half_width,
            "resolved_accuracy": resolved_accuracy,
            "arbitration_accuracy": arbitration_accuracy,
            "selective_accuracy": selective_accuracy,
            "classified_rate": classified_rate,
            "escalation_rate": escalation_rate,
            "disagreement_rate": disagreement_rate,
            "feasible": bool(classified_rate >= min_classified_rate
                             and escalation_rate >= min_escalation_rate),
        })

        # Constraint check. A candidate that abstains on too many samples,
        # or that never arbitrates at all, is not deployable.
        if classified_rate < min_classified_rate or escalation_rate < min_escalation_rate:
            continue
        if not classified.any():
            continue

        # Rank by selective accuracy, then by how many samples still get a
        # verdict, then by agreement. The previous key ranked on
        # ``resolved_accuracy`` alone, which improved monotonically as the
        # band widened and therefore always drove to the coverage floor.
        key = (selective_accuracy, classified_rate, -disagreement_rate)
        if best is None or key > (best[0], best[1], -best[4]):
            # best[4] is the disagreement rate. The previous code used
            # best[3], which is ``escalate_above`` - an unrelated quantity.
            best = (selective_accuracy, classified_rate, below, above, disagreement_rate)

    if best is None:
        raise ValueError(
            "No escalation band satisfied the constraints "
            f"(min_classified_rate={min_classified_rate}, "
            f"min_escalation_rate={min_escalation_rate}). "
            "Widen the band search or relax the constraints."
        )

    policy = CascadePolicy(
        engine_a_threshold=float(engine_a_threshold),
        engine_c_threshold=float(engine_c_threshold),
        escalate_below=float(best[2]),
        escalate_above=float(best[3]),
        coverage=float(best[1]),
        agreement_rate=1.0 - float(best[4]),
        release_id="calibration-fit",
    )
    return policy, table
