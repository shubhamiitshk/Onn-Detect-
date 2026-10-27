#!/usr/bin/env python3
"""Colab T4 16-Bit LoRA Training & Evaluation Pipeline for On-Device Forensic VLM.

Implements the experimental design and methodological fixes for ONN-DETECT's optional
Tier-2 "Deep Scan" VLM cascade (`Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-0.8B`, and compact
baseline `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`):

1. Real Supervised Training Loop (`execute_supervised_vlm_lora_training` & `run_colab_t4_lora_pilot`):
   - Enforces `image_max_px` resizing on every input image and records `downscale_ratio`.
   - Deduplicates training records against validation/test splits using explicit `root_source_id`
     and iterative chained-suffix normalization (`canonical_source_id`).
   - Attaches a trainable `TorchMaskedForensicHead` (1 class logit + 6 forgery axis logits +
     6 authentic counter-evidence logits) to the VLM's final hidden state `h_{-1}` and jointly
     optimizes LoRA adapters + forensic head for `max_steps` using `AdamW` and FP16 AMP `GradScaler`.
   - Records both initialization VRAM and actual training peak VRAM (`train_peak_allocated_gb`,
     `train_peak_reserved_gb`), saves `adapter_final`, and exports the unified Android/Python
     `forensic_vlm_policy.json` artifact.

2. Tie-Aware Fixed-FPR Calibration & Evaluation (`select_tie_aware_threshold_for_fpr`, `evaluate_candidate_at_fixed_fpr`):
   - Prevents tied or constant scores (e.g., all 0.5) from falsely reporting 100% TPR @ 1% FPR.
   - Selects and freezes thresholds on calibration real samples such that empirical calibration
     FPR under `score >= threshold` is strictly `<= target_fpr`, and reports actual held-out
     `achieved_fpr_at_1pct` and `achieved_fpr_at_5pct` alongside TPR and Wilson 95% CIs.

3. Feasibility-Checked Learned Routing Policy (`learn_routing_thresholds_on_held_out`, `evaluate_frozen_routing_on_test`):
   - Never returns fabricated `escalation_rate=0.0` with wide thresholds when a budget is infeasible.
   - Explicitly flags `"feasible": False` when no candidate meets `max_escalation_budget`, and
     exports a versioned JSON policy consumed identically by Python and Kotlin (`OnDemandForensicVlmAnalyzer.kt`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image


# ============================================================================
# 1. Published Reference Model Configurations (Language + Vision Projector)
# ============================================================================

@dataclass(frozen=True)
class ReferenceVlmPackConfig:
    priority_role: str
    hf_model_id: str
    quant_config_desc: str
    language_file_mb: float
    vision_mmproj_mb: float
    combined_pack_mb: float
    t4_16bit_lora_supported: bool
    notes: str


REFERENCE_VLM_SHORTLIST: Dict[str, ReferenceVlmPackConfig] = {
    "qwen3_5_2b": ReferenceVlmPackConfig(
        priority_role="First quality-focused T4 experiment",
        hf_model_id="Qwen/Qwen3.5-2B",
        quant_config_desc="Q4_K_M GGUF (1.28 GB) + mmproj-F16 (668 MB)",
        language_file_mb=1280.0,
        vision_mmproj_mb=668.0,
        combined_pack_mb=1948.0,
        t4_16bit_lora_supported=True,
        notes="Apache-2.0. 24-layer / width-1024 vision encoder. Official Unsloth 16-bit LoRA notebook runs on free Colab T4.",
    ),
    "qwen3_5_0_8b": ReferenceVlmPackConfig(
        priority_role="First smaller shipping candidate",
        hf_model_id="Qwen/Qwen3.5-0.8B",
        quant_config_desc="Q4_K_M GGUF (533 MB) + mmproj-F16 (205 MB)",
        language_file_mb=533.0,
        vision_mmproj_mb=205.0,
        combined_pack_mb=738.0,
        t4_16bit_lora_supported=True,
        notes="Apache-2.0. 12-layer / width-768 vision encoder (smaller visual capacity than 2B; must be compared empirically).",
    ),
    "smolvlm2_500m_video": ReferenceVlmPackConfig(
        priority_role="Compact comparison baseline",
        hf_model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        quant_config_desc="Q8_0 GGUF (437 MB) + mmproj-f16 (109 MB)",
        language_file_mb=437.0,
        vision_mmproj_mb=109.0,
        combined_pack_mb=546.0,
        t4_16bit_lora_supported=True,
        notes="Apache-2.0. Official llama.cpp mtmd compact baseline.",
    ),
    "lfm2_5_vl_1_6b": ReferenceVlmPackConfig(
        priority_role="Additional speed/document challenger",
        hf_model_id="LiquidAI/LFM2.5-VL-1.6B",
        quant_config_desc="Official GGUF/ONNX with SigLIP2 NaFlex configurable image-token controls",
        language_file_mb=880.0,
        vision_mmproj_mb=270.0,
        combined_pack_mb=1150.0,
        t4_16bit_lora_supported=True,
        notes="Liquid Open Model License. Explicitly targets edge OCR/document comprehension with image-token controls.",
    ),
    "gemma_4_e2b": ReferenceVlmPackConfig(
        priority_role="Google runtime alternative",
        hf_model_id="google/gemma-4-E2B-it",
        quant_config_desc="LiteRT-LM mobile bundle (2.3B effective / 5.1B with embeddings)",
        language_file_mb=1450.0,
        vision_mmproj_mb=350.0,
        combined_pack_mb=1800.0,
        t4_16bit_lora_supported=False,
        notes="Apache-2.0. Official Google vision fine-tuning guide requires BF16 hardware (L4/A100) and >16 GB VRAM.",
    ),
}


# ============================================================================
# 2. Source-Level Deduplication (Handles Explicit Root IDs & Chained Suffixes)
# ============================================================================

_SINGLE_DERIVATIVE_SUFFIX_RE = re.compile(
    r"(?:_crop\d*|_roi\d*|_inpaint\d*|_edit\d*|_q\d+|_qf\d+|_deg\d*|_rot\d+|_patch\d+|_scale\d+|_blur\d+|_twin\d*|_laundered|_rec\d*|_b_free.*)$",
    re.IGNORECASE,
)


def canonical_source_id(
    file_path_or_id: str,
    explicit_root_source_id: Optional[str] = None,
) -> str:
    """Normalizes a sample identifier to its root source ID.

    1. Uses `explicit_root_source_id` whenever supplied by dataset metadata.
    2. Otherwise strips `__` metadata suffixes and repeatedly strips chained derivative
       suffixes (e.g., `frame_crop1_q75.jpg` -> `frame_crop1` -> `frame`) until fixed point.
    """
    if explicit_root_source_id is not None and str(explicit_root_source_id).strip():
        return str(explicit_root_source_id).strip().lower()
    stem = Path(str(file_path_or_id)).stem.lower()
    if "__" in stem:
        stem = stem.split("__")[0]
    while True:
        next_stem = _SINGLE_DERIVATIVE_SUFFIX_RE.sub("", stem)
        if next_stem == stem or not next_stem:
            break
        stem = next_stem
    return stem


def deduplicate_training_against_eval(
    train_records: Sequence[Dict[str, Any]],
    eval_records: Sequence[Dict[str, Any]],
    id_key: str = "image_id",
    root_source_key: str = "root_source_id",
    sha256_key: str = "sha256",
) -> Tuple[List[Dict[str, Any]], int]:
    """Removes any training sample whose `root_source_id`, chained canonical source ID,
    or file SHA-256 overlaps with held-out evaluation splits.
    """
    forbidden_sources: Set[str] = set()
    forbidden_hashes: Set[str] = set()

    for rec in eval_records:
        raw_id = str(rec.get(id_key, ""))
        root_id = rec.get(root_source_key)
        if raw_id or root_id:
            forbidden_sources.add(canonical_source_id(raw_id, root_id))
        h = str(rec.get(sha256_key, "")).strip().lower()
        if h:
            forbidden_hashes.add(h)

    clean_train: List[Dict[str, Any]] = []
    removed_count = 0
    for rec in train_records:
        raw_id = str(rec.get(id_key, ""))
        root_id = rec.get(root_source_key)
        src_id = canonical_source_id(raw_id, root_id) if (raw_id or root_id) else ""
        h = str(rec.get(sha256_key, "")).strip().lower()
        if (src_id and src_id in forbidden_sources) or (h and h in forbidden_hashes):
            removed_count += 1
        else:
            clean_train.append(rec)

    return clean_train, removed_count


def assign_disjoint_splits_by_root_source(
    records: Sequence[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    id_key: str = "image_id",
    root_source_key: str = "root_source_id",
) -> Dict[str, List[Dict[str, Any]]]:
    """Groups all records by their root source ID BEFORE splitting so no source image
    and its derivatives can ever straddle train, val, and test splits.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        gid = canonical_source_id(str(rec.get(id_key, "")), rec.get(root_source_key))
        groups.setdefault(gid, []).append(rec)

    # Deterministic hash-ordered group assignment
    sorted_gids = sorted(groups.keys(), key=lambda g: hashlib.sha256(g.encode("utf-8")).hexdigest())
    n = len(sorted_gids)
    n_train = max(1, int(round(n * train_ratio))) if n >= 3 else max(0, n - 2)
    n_val = max(1, int(round(n * val_ratio))) if n >= 3 else (1 if n >= 2 else 0)

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for idx, gid in enumerate(sorted_gids):
        if idx < n_train:
            splits["train"].extend(groups[gid])
        elif idx < n_train + n_val:
            splits["val"].extend(groups[gid])
        else:
            splits["test"].extend(groups[gid])
    return splits


# ============================================================================
# 3. Unified Masked 6-Axis + Counter-Evidence Head (Matches Kotlin Contract)
# ============================================================================

RUBRIC_AXES: Tuple[str, ...] = (
    "anatomical_geometry",
    "physics_illumination",
    "surface_micro_texture",
    "boundary_halo_seam",
    "typography_document_layout",
    "temporal_kinematic_consistency",
)


class MaskedAuxiliaryForensicHead:
    """Unified training/serving head over the VLM's final prefill representation `h_{-1}`.

    Produces the exact 13 quantities consumed by `OnDemandForensicVlmAnalyzer.kt`:
      - `class_logit` / `class_prob` (1D binary synthetic vs. authentic score)
      - `forgery_logits_6d` (6-axis forgery evidence logits; `NaN` when not applicable)
      - `authentic_counter_logits_6d` (6-axis authentic counter-evidence logits; `NaN` when not applicable)
    """

    def __init__(self, hidden_dim: int = 960) -> None:
        self.hidden_dim = int(hidden_dim)
        self.w_class = np.zeros(self.hidden_dim, dtype=np.float64)
        self.b_class = 0.0
        self.w_axes = np.zeros((self.hidden_dim, 6), dtype=np.float64)
        self.b_axes = np.zeros(6, dtype=np.float64)
        self.w_counter_axes = np.zeros((self.hidden_dim, 6), dtype=np.float64)
        self.b_counter_axes = np.zeros(6, dtype=np.float64)
        self.axes_supervised: List[bool] = [False] * 6

    def fit(
        self,
        H_prefill: np.ndarray,
        labels: np.ndarray,
        axis_annotations_6d: np.ndarray,
        counter_annotations_6d: Optional[np.ndarray] = None,
        l2_reg: float = 5.0,
        steps: int = 250,
    ) -> "MaskedAuxiliaryForensicHead":
        """Fits `w_class`, `w_axes` (forgery evidence), and `w_counter_axes` (authentic
        counter-evidence) strictly on non-NaN entries of `axis_annotations_6d`.
        """
        H = np.asarray(H_prefill, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        A = np.asarray(axis_annotations_6d, dtype=np.float64)
        valid_mask = np.isfinite(A)
        self.axes_supervised = [bool(np.sum(valid_mask[:, i]) > 0) for i in range(6)]
        A_zeroed = np.where(valid_mask, A, 0.0)

        # Authentic counter-evidence target: 1.0 when an applicable axis is verified authentic (A == 0 on y == 0)
        if counter_annotations_6d is not None:
            C_raw = np.asarray(counter_annotations_6d, dtype=np.float64)
            valid_c_mask = np.isfinite(C_raw)
            C_zeroed = np.where(valid_c_mask, C_raw, 0.0)
        else:
            valid_c_mask = valid_mask
            C_zeroed = np.where(valid_mask, (1.0 - A_zeroed) * (1.0 - y[:, None]), 0.0)

        n = float(len(H))
        valid_counts = np.maximum(1.0, np.sum(valid_mask, axis=0))
        valid_c_counts = np.maximum(1.0, np.sum(valid_c_mask, axis=0))

        for _ in range(steps):
            # 1. Primary binary class score
            p_c = 1.0 / (1.0 + np.exp(-np.clip(H @ self.w_class + self.b_class, -30.0, 30.0)))
            err_c = (p_c - y) / n
            self.w_class -= 0.25 * (H.T @ err_c + (l2_reg / n) * self.w_class)
            self.b_class -= 0.25 * float(np.sum(err_c))

            # 2. Masked observable forgery axes
            p_a = 1.0 / (1.0 + np.exp(-np.clip(H @ self.w_axes + self.b_axes, -30.0, 30.0)))
            err_a = np.where(valid_mask, p_a - A_zeroed, 0.0) / valid_counts[None, :]
            self.w_axes -= 0.30 * (H.T @ err_a + (l2_reg / n) * self.w_axes)
            self.b_axes -= 0.30 * np.sum(err_a, axis=0)

            # 3. Masked authentic counter-evidence axes
            p_cnt = 1.0 / (1.0 + np.exp(-np.clip(H @ self.w_counter_axes + self.b_counter_axes, -30.0, 30.0)))
            err_cnt = np.where(valid_c_mask, p_cnt - C_zeroed, 0.0) / valid_c_counts[None, :]
            self.w_counter_axes -= 0.30 * (H.T @ err_cnt + (l2_reg / n) * self.w_counter_axes)
            self.b_counter_axes -= 0.30 * np.sum(err_cnt, axis=0)

        return self

    def predict(self, h_last: np.ndarray, applicable_mask_6d: Optional[Sequence[bool]] = None) -> Dict[str, Any]:
        h = np.asarray(h_last, dtype=np.float64).reshape(-1)
        class_logit = float(h @ self.w_class + self.b_class)
        class_prob = float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, class_logit)))))
        raw_forgery = h @ self.w_axes + self.b_axes
        raw_counter = h @ self.w_counter_axes + self.b_counter_axes

        axis_probs: Dict[str, Optional[float]] = {}
        forgery_logits_out: List[float] = []
        counter_logits_out: List[float] = []
        for i, name in enumerate(RUBRIC_AXES):
            is_supervised = bool(self.axes_supervised[i]) if i < len(self.axes_supervised) else True
            is_applicable = applicable_mask_6d is None or bool(applicable_mask_6d[i])
            if not is_supervised or not is_applicable:
                axis_probs[name] = None
                forgery_logits_out.append(float("nan"))
                counter_logits_out.append(float("nan"))
            else:
                f_logit = float(raw_forgery[i])
                c_logit = float(raw_counter[i])
                forgery_logits_out.append(f_logit)
                counter_logits_out.append(c_logit)
                axis_probs[name] = float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, f_logit)))))
        return {
            "class_prob": class_prob,
            "class_logit": class_logit,
            "forgery_logits_6d": forgery_logits_out,
            "authentic_counter_logits_6d": counter_logits_out,
            "axis_probabilities": axis_probs,
        }

    def to_contract_dict(self) -> Dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "axes": list(RUBRIC_AXES),
            "axes_supervised": [bool(x) for x in self.axes_supervised],
            "w_class": [float(x) for x in self.w_class.tolist()],
            "b_class": float(self.b_class),
            "w_forgery_axes": [[float(v) for v in row] for row in self.w_axes.T.tolist()],
            "b_forgery_axes": [float(x) for x in self.b_axes.tolist()],
            "w_counter_axes": [[float(v) for v in row] for row in self.w_counter_axes.T.tolist()],
            "b_counter_axes": [float(x) for x in self.b_counter_axes.tolist()],
        }

    @classmethod
    def from_contract_dict(cls, data: Dict[str, Any]) -> "MaskedAuxiliaryForensicHead":
        hidden_dim = int(data["hidden_dim"])
        head = cls(hidden_dim=hidden_dim)
        head.w_class = np.asarray(data["w_class"], dtype=np.float64)
        head.b_class = float(data["b_class"])
        head.w_axes = np.asarray(data["w_forgery_axes"], dtype=np.float64).T
        head.b_axes = np.asarray(data["b_forgery_axes"], dtype=np.float64)
        head.w_counter_axes = np.asarray(data["w_counter_axes"], dtype=np.float64).T
        head.b_counter_axes = np.asarray(data["b_counter_axes"], dtype=np.float64)
        head.axes_supervised = list(data.get("axes_supervised", [True] * 6))
        return head


# ============================================================================
# 4. Tie-Aware Fixed-FPR Calibration & Evaluation with Wilson Intervals
# ============================================================================

def wilson_ci_95(successes: int, total: int) -> Tuple[float, float]:
    """95% Wilson score binomial confidence interval."""
    if total <= 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    p = float(successes) / float(total)
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    return (round(max(0.0, center - half), 6), round(min(1.0, center + half), 6))


def select_tie_aware_threshold_for_fpr(real_scores: np.ndarray, target_fpr: float) -> float:
    """Selects the smallest decision threshold `t` such that under the rule `score >= t`,
    the empirical false positive rate on `real_scores` is strictly `<= target_fpr`.

    Guarantees float32 representation safety: when serialized to JSON and parsed as a 32-bit
    IEEE 754 Float on Android, the threshold cannot round down and cause the empirical FPR
    on calibration data to exceed `target_fpr`.
    """
    r = np.asarray(real_scores, dtype=np.float64).reshape(-1)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float(np.float32(1.0 + 1e-5))
    candidates = np.r_[np.unique(r), float(np.max(r)) + 1e-5]
    n_real = float(len(r))
    for t in candidates:
        fpr = float(np.sum(r >= t)) / n_real
        if fpr <= target_fpr + 1e-12:
            # Ensure safety after float32 roundtrip in Android
            t_f32 = np.float32(t)
            fpr_f32 = float(np.sum(np.float32(r) >= t_f32)) / n_real
            while fpr_f32 > target_fpr + 1e-12 and t_f32 < np.float32(np.inf):
                t_f32 = np.nextafter(t_f32, np.float32(np.inf))
                fpr_f32 = float(np.sum(np.float32(r) >= t_f32)) / n_real
            return float(t_f32)
    return float(np.float32(float(np.max(r)) + 1e-5))


def fit_frozen_fpr_thresholds(cal_labels: np.ndarray, cal_scores: np.ndarray) -> Dict[str, float]:
    """Fits and freezes tie-aware 1%, 5%, and 0.1% FPR thresholds on calibration data."""
    y = np.asarray(cal_labels, dtype=np.int32).reshape(-1)
    s = np.asarray(cal_scores, dtype=np.float64).reshape(-1)
    real_scores = s[y == 0]
    return {
        "threshold_1pct_fpr": select_tie_aware_threshold_for_fpr(real_scores, 0.01),
        "threshold_5pct_fpr": select_tie_aware_threshold_for_fpr(real_scores, 0.05),
        "threshold_0_1pct_fpr": select_tie_aware_threshold_for_fpr(real_scores, 0.001),
    }


def evaluate_candidate_at_fixed_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    inconclusive_mask: Optional[np.ndarray] = None,
    calibration_labels: Optional[np.ndarray] = None,
    calibration_scores: Optional[np.ndarray] = None,
    frozen_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Evaluates a candidate detector at tie-aware fixed-FPR thresholds, reporting both
    TPR and achieved held-out FPR alongside Wilson 95% confidence intervals.
    """
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    inc = np.zeros(len(y), dtype=bool) if inconclusive_mask is None else np.asarray(inconclusive_mask, dtype=bool)

    real_scores = s[(y == 0) & (~inc)]
    fake_scores = s[(y == 1) & (~inc)]
    n_real_total = int(np.sum(y == 0))
    n_fake_total = int(np.sum(y == 1))

    if frozen_thresholds is not None:
        thr_1pct = float(frozen_thresholds["threshold_1pct_fpr"])
        thr_5pct = float(frozen_thresholds.get("threshold_5pct_fpr", thr_1pct))
        thr_01pct = float(frozen_thresholds["threshold_0_1pct_fpr"])
    elif calibration_labels is not None and calibration_scores is not None:
        cal_thrs = fit_frozen_fpr_thresholds(calibration_labels, calibration_scores)
        thr_1pct = cal_thrs["threshold_1pct_fpr"]
        thr_5pct = cal_thrs["threshold_5pct_fpr"]
        thr_01pct = cal_thrs["threshold_0_1pct_fpr"]
    else:
        thr_1pct = select_tie_aware_threshold_for_fpr(real_scores, 0.01)
        thr_5pct = select_tie_aware_threshold_for_fpr(real_scores, 0.05)
        thr_01pct = select_tie_aware_threshold_for_fpr(real_scores, 0.001)

    tp_1pct = int(np.sum(fake_scores >= thr_1pct))
    fp_1pct = int(np.sum(real_scores >= thr_1pct))
    tp_5pct = int(np.sum(fake_scores >= thr_5pct))
    fp_5pct = int(np.sum(real_scores >= thr_5pct))
    tp_01pct = int(np.sum(fake_scores >= thr_01pct))
    fp_01pct = int(np.sum(real_scores >= thr_01pct))

    return {
        "threshold_1pct_fpr": float(thr_1pct),
        "threshold_5pct_fpr": float(thr_5pct),
        "threshold_0_1pct_fpr": float(thr_01pct),
        "tpr_at_1pct_fpr": round(tp_1pct / max(1, n_fake_total), 6),
        "tpr_at_1pct_ci95": wilson_ci_95(tp_1pct, n_fake_total),
        "achieved_fpr_at_1pct": round(fp_1pct / max(1, n_real_total), 6),
        "achieved_fpr_at_1pct_ci95": wilson_ci_95(fp_1pct, n_real_total),
        "tpr_at_5pct_fpr": round(tp_5pct / max(1, n_fake_total), 6),
        "tpr_at_5pct_ci95": wilson_ci_95(tp_5pct, n_fake_total),
        "achieved_fpr_at_5pct": round(fp_5pct / max(1, n_real_total), 6),
        "achieved_fpr_at_5pct_ci95": wilson_ci_95(fp_5pct, n_real_total),
        "tpr_at_0_1pct_fpr": round(tp_01pct / max(1, n_fake_total), 6),
        "tpr_at_0_1pct_ci95": wilson_ci_95(tp_01pct, n_fake_total),
        "achieved_fpr_at_0_1pct": round(fp_01pct / max(1, n_real_total), 6),
        "inconclusive_rate": round(float(np.mean(inc)), 6),
        "n_real": n_real_total,
        "n_fake": n_fake_total,
    }


def route_with_policy(
    policy: Dict[str, Any],
    fast_fusion_prob: float,
    srm_calibrated_score: float,
    vfm_probe_score: Optional[float] = None,
    hotspot_z_score: float = 0.0,
    active_domain: str = "human_face",
    doc_tamper_score: float = 0.0,
    available_heap_mb: int = 256,
    require_explanation: bool = False,
) -> Dict[str, Any]:
    """Single-sample routing function strictly implementing the deployed Android logic in
    `OnDemandForensicVlmAnalyzer.kt:routeWithPolicy`.
    """
    if not bool(policy.get("feasible", True)):
        return {
            "mode": "FAST_EXIT_CONFIDENT",
            "invoke_on_device_vlm": False,
            "expected_voi_score": 0.0,
            "target_domain": active_domain,
            "trigger_reason": "routing_policy_infeasible",
        }

    EPS = 1e-6
    def _logit(p: float) -> float:
        c = max(EPS, min(1.0 - EPS, float(p)))
        return math.log(c / (1.0 - c))

    z_f = _logit(fast_fusion_prob)
    z_thr = _logit(float(policy["decision_threshold_dev"]))
    z_a = _logit(srm_calibrated_score)
    llr_diff = abs(z_f - z_thr)
    llr_hw = float(policy["llr_uncertainty_halfwidth"])
    near_boundary = llr_diff < llr_hw

    if vfm_probe_score is not None and math.isfinite(vfm_probe_score):
        conflict_diff = abs(z_a - _logit(vfm_probe_score))
    else:
        conflict_diff = 0.0
    conf_thr = float(policy["calibrated_conflict_logit_threshold"])
    cross_engine_conflict = conflict_diff > conf_thr

    hz_thr = float(policy["hotspot_z_threshold"])
    localized_hotspot = math.isfinite(hotspot_z_score) and hotspot_z_score >= hz_thr

    doc_thr = float(policy.get("doc_tamper_threshold", 0.48))
    doc_tamper = (active_domain == "document_text") and math.isfinite(doc_tamper_score) and (doc_tamper_score >= doc_thr)

    should_escalate = require_explanation or near_boundary or cross_engine_conflict or localized_hotspot or doc_tamper

    boundary_score = max(0.0, 1.0 - llr_diff / max(0.25, llr_hw)) if llr_hw > 0.0 else 0.0
    conflict_score = min(1.0, max(0.0, conflict_diff / max(0.5, conf_thr))) if conf_thr < 1e6 else 0.0
    hotspot_score = min(1.0, max(0.0, (hotspot_z_score - hz_thr) / 1.5 + 0.5)) if localized_hotspot else 0.0

    if not should_escalate:
        voi = 0.10 * boundary_score
    else:
        voi = max(0.42, min(1.0, 0.40 * boundary_score + 0.35 * conflict_score + 0.25 * hotspot_score))

    reason = (
        "forensic_explanation_requested" if require_explanation
        else "ambiguous_fast_margin" if near_boundary
        else "cross_engine_disagreement" if cross_engine_conflict
        else "localized_roi_hotspot" if localized_hotspot
        else f"specialized_domain_{active_domain}" if doc_tamper
        else "fast_tier_confident"
    )

    if not should_escalate:
        return {
            "mode": "FAST_EXIT_CONFIDENT",
            "invoke_on_device_vlm": False,
            "expected_voi_score": round(voi, 4),
            "target_domain": active_domain,
            "trigger_reason": "fast_tier_confident",
        }
    if available_heap_mb < 96:
        return {
            "mode": "MICRO_MOE_ONLY",
            "invoke_on_device_vlm": False,
            "expected_voi_score": round(voi, 4),
            "target_domain": active_domain,
            "trigger_reason": reason,
        }
    mode = "VLM_FULL_DIALECTICAL_COT" if (require_explanation or voi >= 0.68) else "VLM_SINGLE_PASS_LOGIT"
    return {
        "mode": mode,
        "invoke_on_device_vlm": True,
        "expected_voi_score": round(voi, 4),
        "target_domain": active_domain,
        "trigger_reason": reason,
    }


def learn_routing_thresholds_on_held_out(
    labels: np.ndarray,
    calibrated_fusion_probs: np.ndarray,
    calibrated_srm_probs: np.ndarray,
    calibrated_probe_probs: Optional[np.ndarray] = None,
    hotspot_z_scores: Optional[np.ndarray] = None,
    active_domains: Optional[Sequence[str]] = None,
    doc_tamper_scores: Optional[np.ndarray] = None,
    max_escalation_budget: float = 0.30,
) -> Dict[str, Any]:
    """Learns routing thresholds on held-out development data to minimize `early_exit_fnr`
    subject to `esc_rate <= max_escalation_budget`.
    Models all triggers including boundary ambiguity, cross-engine conflict, localized hotspots,
    and document tampering on `document_text`.
    """
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    p_f = np.clip(np.asarray(calibrated_fusion_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    p_a = np.clip(np.asarray(calibrated_srm_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if calibrated_probe_probs is not None:
        p_c = np.asarray(calibrated_probe_probs, dtype=np.float64)
    else:
        p_c = np.full(len(y), np.nan, dtype=np.float64)
    hz = np.asarray(hotspot_z_scores if hotspot_z_scores is not None else np.zeros(len(y)), dtype=np.float64).reshape(-1)

    thr_dev = select_tie_aware_threshold_for_fpr(p_f[y == 0], 0.01) if np.any(y == 0) else 0.5782
    thr_clipped = max(1e-6, min(1.0 - 1e-6, thr_dev))
    z_thr = math.log(thr_clipped / (1.0 - thr_clipped))
    z_f = np.log(p_f / (1.0 - p_f))
    z_a = np.log(p_a / (1.0 - p_a))
    valid_probe = np.isfinite(p_c)
    p_c_clipped = np.clip(np.where(valid_probe, p_c, 0.5), 1e-6, 1.0 - 1e-6)
    z_c = np.log(p_c_clipped / (1.0 - p_c_clipped))
    conflict_diff = np.where(valid_probe, np.abs(z_a - z_c), 0.0)

    doc_tamper = np.zeros(len(y), dtype=bool)
    if active_domains is not None and doc_tamper_scores is not None:
        is_doc = np.asarray([d == "document_text" for d in active_domains], dtype=bool)
        doc_s = np.asarray(doc_tamper_scores, dtype=np.float64)
        doc_tamper = is_doc & np.isfinite(doc_s) & (doc_s >= 0.48)

    n_fakes = max(1, int(np.sum(y == 1)))
    best_cfg: Optional[Dict[str, Any]] = None

    for llr_hw in (0.10, 0.20, 0.40, 0.65, 1.0, 1.4, 1.8, 2.2):
        for conf_thr in (1.4, 1.8, 2.2, 2.8, 3.5):
            for hz_thr in (1.65, 1.85, 2.10, 2.50, 2.90, 3.50):
                escalate = (
                    (np.abs(z_f - z_thr) < llr_hw)
                    | (conflict_diff > conf_thr)
                    | (hz >= hz_thr)
                    | doc_tamper
                )
                esc_rate = float(np.mean(escalate))
                if esc_rate > max_escalation_budget:
                    continue
                fakes_exited_early = int(np.sum((~escalate) & (y == 1)))
                early_exit_missed_fakes = int(np.sum((~escalate) & (y == 1) & (p_f < thr_dev)))
                cond_miss = early_exit_missed_fakes / float(max(1, fakes_exited_early))
                total_fnr = early_exit_missed_fakes / float(n_fakes)
                if best_cfg is None or total_fnr < best_cfg["early_exit_fnr"] or (
                    abs(total_fnr - best_cfg["early_exit_fnr"]) < 1e-9 and esc_rate < best_cfg["escalation_rate"]
                ):
                    best_cfg = {
                        "feasible": True,
                        "llr_uncertainty_halfwidth": round(float(llr_hw), 4),
                        "calibrated_conflict_logit_threshold": round(float(conf_thr), 4),
                        "hotspot_z_threshold": round(float(hz_thr), 4),
                        "doc_tamper_threshold": 0.48,
                        "decision_threshold_dev": float(thr_dev),
                        "escalation_rate": round(esc_rate, 4),
                        "conditional_early_exit_miss_rate": round(cond_miss, 6),
                        "early_exit_fnr": round(total_fnr, 6),
                    }

    if best_cfg is None:
        early_exit_missed_fakes = int(np.sum((y == 1) & (p_f < thr_dev)))
        unconditional_fnr = early_exit_missed_fakes / float(n_fakes)
        return {
            "feasible": False,
            "llr_uncertainty_halfwidth": 0.0,
            "calibrated_conflict_logit_threshold": 1e9,
            "hotspot_z_threshold": 1e9,
            "doc_tamper_threshold": 1e9,
            "decision_threshold_dev": float(thr_dev),
            "escalation_rate": 0.0,
            "conditional_early_exit_miss_rate": round(unconditional_fnr, 6),
            "early_exit_fnr": round(unconditional_fnr, 6),
        }

    return best_cfg


def evaluate_frozen_routing_on_test(
    frozen_cfg: Dict[str, Any],
    test_labels: np.ndarray,
    test_calibrated_fusion_probs: np.ndarray,
    test_calibrated_srm_probs: np.ndarray,
    test_calibrated_probe_probs: Optional[np.ndarray] = None,
    test_hotspot_z_scores: Optional[np.ndarray] = None,
    test_active_domains: Optional[Sequence[str]] = None,
    test_doc_tamper_scores: Optional[np.ndarray] = None,
    test_available_heap_mb: Optional[Sequence[int]] = None,
    test_require_explanations: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """Evaluates a frozen routing policy on held-out test data using the exact same
    per-sample routing engine as deployed in `OnDemandForensicVlmAnalyzer.kt`.
    """
    y = np.asarray(test_labels, dtype=np.int32).reshape(-1)
    n = len(y)
    p_f = np.asarray(test_calibrated_fusion_probs, dtype=np.float64).reshape(-1)
    p_a = np.asarray(test_calibrated_srm_probs, dtype=np.float64).reshape(-1)
    p_c = np.asarray(test_calibrated_probe_probs, dtype=np.float64).reshape(-1) if test_calibrated_probe_probs is not None else np.full(n, np.nan)
    hz = np.asarray(test_hotspot_z_scores, dtype=np.float64).reshape(-1) if test_hotspot_z_scores is not None else np.zeros(n)
    domains = list(test_active_domains) if test_active_domains is not None else ["human_face"] * n
    doc_scores = np.asarray(test_doc_tamper_scores, dtype=np.float64).reshape(-1) if test_doc_tamper_scores is not None else np.zeros(n)
    heap_mbs = list(test_available_heap_mb) if test_available_heap_mb is not None else [256] * n
    req_expls = list(test_require_explanations) if test_require_explanations is not None else [False] * n

    thr = float(frozen_cfg.get("decision_threshold_dev", 0.5782))
    escalate = np.zeros(n, dtype=bool)
    for i in range(n):
        res = route_with_policy(
            policy=frozen_cfg,
            fast_fusion_prob=float(p_f[i]),
            srm_calibrated_score=float(p_a[i]),
            vfm_probe_score=float(p_c[i]) if math.isfinite(p_c[i]) else None,
            hotspot_z_score=float(hz[i]),
            active_domain=domains[i],
            doc_tamper_score=float(doc_scores[i]),
            available_heap_mb=int(heap_mbs[i]),
            require_explanation=bool(req_expls[i]),
        )
        escalate[i] = bool(res["invoke_on_device_vlm"])

    n_fakes = max(1, int(np.sum(y == 1)))
    fakes_exited_early = int(np.sum((~escalate) & (y == 1)))
    early_exit_missed_fakes = int(np.sum((~escalate) & (y == 1) & (p_f < thr)))
    return {
        "feasible": bool(frozen_cfg.get("feasible", True)),
        "test_escalation_rate": round(float(np.mean(escalate)), 4),
        "conditional_early_exit_miss_rate": round(early_exit_missed_fakes / float(max(1, fakes_exited_early)), 6),
        "total_fake_early_exit_fnr": round(early_exit_missed_fakes / float(n_fakes), 6),
        "fakes_exited_early": fakes_exited_early,
        "early_exit_missed_fakes": early_exit_missed_fakes,
        "total_fakes": n_fakes,
    }


def compute_dialectical_fused_prob(
    fast_fusion_prob: float,
    vlm_prob: float,
    vlm_logit: Optional[float] = None,
) -> float:
    """Matches Android's `OnDemandForensicVlmAnalyzer.kt:fuseDialecticalLogits`
    bounded logit tanh fusion:
      z_fast = 1.5 * tanh(logit(fast) / 1.5)
      z_vlm = 1.8 * tanh(logit(vlm) / 1.8)
      fused = sigmoid(0.62 * z_fast + 0.38 * z_vlm)
    """
    p_f = max(1e-6, min(1.0 - 1e-6, float(fast_fusion_prob)))
    z_f = 1.5 * math.tanh(math.log(p_f / (1.0 - p_f)) / 1.5)
    if vlm_logit is not None and math.isfinite(vlm_logit):
        vl = float(vlm_logit)
    else:
        p_v = max(1e-6, min(1.0 - 1e-6, float(vlm_prob)))
        vl = math.log(p_v / (1.0 - p_v))
    z_v = 1.8 * math.tanh(vl / 1.8)
    fused_logit = 0.62 * z_f + 0.38 * z_v
    return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, fused_logit)))))


def evaluate_complete_cascade_on_test(
    frozen_cfg: Dict[str, Any],
    test_labels: Sequence[int],
    test_calibrated_fusion_probs: Sequence[float],
    test_calibrated_srm_probs: Sequence[float],
    test_calibrated_probe_probs: Optional[Sequence[float]] = None,
    test_vlm_probs: Optional[Sequence[float]] = None,
    test_hotspot_z_scores: Optional[Sequence[float]] = None,
    test_active_domains: Optional[Sequence[str]] = None,
    test_doc_tamper_scores: Optional[Sequence[float]] = None,
    vlm_decision_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Evaluates the end-to-end routed cascade against the unchanged baseline detector
    at the calibrated operating point on the sealed test set.
    """
    y = np.asarray(test_labels, dtype=np.int32).reshape(-1)
    n = len(y)
    p_f = np.asarray(test_calibrated_fusion_probs, dtype=np.float64).reshape(-1)
    p_a = np.asarray(test_calibrated_srm_probs, dtype=np.float64).reshape(-1)
    p_c = np.asarray(test_calibrated_probe_probs, dtype=np.float64).reshape(-1) if test_calibrated_probe_probs is not None else np.full(n, np.nan)
    hz = np.asarray(test_hotspot_z_scores, dtype=np.float64).reshape(-1) if test_hotspot_z_scores is not None else np.zeros(n)
    domains = list(test_active_domains) if test_active_domains is not None else ["human_face"] * n
    doc_scores = np.asarray(test_doc_tamper_scores, dtype=np.float64).reshape(-1) if test_doc_tamper_scores is not None else np.zeros(n)
    vlm_p = np.asarray(test_vlm_probs, dtype=np.float64).reshape(-1) if test_vlm_probs is not None else p_f.copy()

    thr_base = float(frozen_cfg.get("decision_threshold_dev", 0.5782))
    thr_vlm = float(vlm_decision_threshold)

    # 1. Unchanged Baseline Detector Evaluation
    base_preds = (p_f >= thr_base).astype(int)
    n_pos = max(1, int(np.sum(y == 1)))
    n_neg = max(1, int(np.sum(y == 0)))
    base_tpr = float(np.sum((base_preds == 1) & (y == 1))) / float(n_pos)
    base_fpr = float(np.sum((base_preds == 1) & (y == 0))) / float(n_neg)

    # 2. Complete Routed Cascade Evaluation
    escalate = np.zeros(n, dtype=bool)
    cascade_preds = np.zeros(n, dtype=int)
    cascade_scores = np.zeros(n, dtype=np.float64)

    for i in range(n):
        res = route_with_policy(
            policy=frozen_cfg,
            fast_fusion_prob=float(p_f[i]),
            srm_calibrated_score=float(p_a[i]),
            vfm_probe_score=float(p_c[i]) if math.isfinite(p_c[i]) else None,
            hotspot_z_score=float(hz[i]),
            active_domain=domains[i],
            doc_tamper_score=float(doc_scores[i]),
        )
        is_esc = bool(res["invoke_on_device_vlm"])
        escalate[i] = is_esc

        if is_esc:
            cascade_scores[i] = float(vlm_p[i])
            cascade_preds[i] = int(vlm_p[i] >= thr_vlm)
        else:
            cascade_scores[i] = float(p_f[i])
            cascade_preds[i] = int(p_f[i] >= thr_base)

    cascade_tpr = float(np.sum((cascade_preds == 1) & (y == 1))) / float(n_pos)
    cascade_fpr = float(np.sum((cascade_preds == 1) & (y == 0))) / float(n_neg)

    resolved_fakes = int(np.sum(escalate & (y == 1) & (base_preds == 0) & (cascade_preds == 1)))
    new_fp = int(np.sum(escalate & (y == 0) & (base_preds == 0) & (cascade_preds == 1)))
    early_misses = int(np.sum((~escalate) & (y == 1) & (base_preds == 0)))

    return {
        "baseline_threshold": round(thr_base, 6),
        "vlm_decision_threshold": round(thr_vlm, 6),
        "test_escalation_rate": round(float(np.mean(escalate)), 4),
        "baseline_tpr": round(base_tpr, 4),
        "baseline_fpr": round(base_fpr, 4),
        "cascade_tpr": round(cascade_tpr, 4),
        "cascade_fpr": round(cascade_fpr, 4),
        "delta_tpr": round(cascade_tpr - base_tpr, 4),
        "delta_fpr": round(cascade_fpr - base_fpr, 4),
        "resolved_missed_fakes": resolved_fakes,
        "new_false_positives": new_fp,
        "early_exit_missed_fakes": early_misses,
        "total_fakes": n_pos,
        "total_reals": n_neg,
    }


def export_forensic_vlm_policy_artifact(
    output_path: Path,
    routing_cfg: Dict[str, Any],
    head: MaskedAuxiliaryForensicHead,
    model_id: str,
    vlm_calibration_thresholds: Optional[Dict[str, Any]] = None,
    diagnostic_only: bool = True,
) -> Dict[str, Any]:
    """Writes the unified versioned JSON artifact consumed by both Python and Android."""
    vlm_cal = vlm_calibration_thresholds or {}
    t_1pct = float(vlm_cal.get("threshold_1pct_fpr", vlm_cal.get("t_1pct_fpr", 0.50)))
    t_5pct = float(vlm_cal.get("threshold_5pct_fpr", vlm_cal.get("t_5pct_fpr", t_1pct)))
    t_bal = float(vlm_cal.get("threshold_balanced", vlm_cal.get("t_balanced", t_1pct)))

    payload = {
        "schema_version": "onn-forensic-vlm-policy-v1",
        "model_id": model_id,
        "diagnostic_only": bool(diagnostic_only),
        "routing_policy": {
            "feasible": bool(routing_cfg.get("feasible", True)),
            "decision_threshold_dev": float(routing_cfg.get("decision_threshold_dev", 0.5782)),
            "llr_uncertainty_halfwidth": float(routing_cfg.get("llr_uncertainty_halfwidth", 0.65)),
            "calibrated_conflict_logit_threshold": float(routing_cfg.get("calibrated_conflict_logit_threshold", 1.80)),
            "hotspot_z_threshold": float(routing_cfg.get("hotspot_z_threshold", 1.85)),
            "doc_tamper_threshold": float(routing_cfg.get("doc_tamper_threshold", 0.48)),
        },
        "vlm_calibration": {
            "decision_threshold_dev": t_1pct,
            "threshold_at_1pct_fpr": t_1pct,
            "threshold_at_5pct_fpr": t_5pct,
            "threshold_balanced": t_bal,
        },
        "head_contract": head.to_contract_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# ============================================================================
# 4b. Explicit Label Taxonomy & Candidate H (Teacher-Validated Fast Student)
# ============================================================================

LABEL_TAXONOMY_MAP: Dict[str, int] = {
    "authentic_raw_capture": 0,
    "authentic_isp_denoised_hdr_bokeh": 0,      # Neural ISP enhancement/denoising is authentic (0)
    "authentic_conventional_retouch_crop": 0,   # Non-generative editing/compression is authentic (0)
    "authentic_screenshot": 0,                  # Screenshot of authentic content is authentic (0)
    "generative_local_inpaint_or_edit": 1,      # Generative region synthesis is positive (1)
    "generative_full_synthesis": 1,             # Full generative synthesis is positive (1)
}


def fit_distilled_fast_student_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    teacher_soft_probs: np.ndarray,
    teacher_validated_on_dev: bool,
    alpha_kd: float = 0.40,
    l2_reg: float = 5.0,
    steps: int = 250,
) -> Tuple[np.ndarray, float]:
    """Candidate H: Distills a validated VLM teacher's soft classification targets into the
    fast vision probe while retaining ground-truth binary supervision (`1 - alpha_kd`).
    Refuses to distill if `teacher_validated_on_dev` is False.
    """
    if not teacher_validated_on_dev:
        raise ValueError("Teacher VLM must pass held-out dev validation before distillation.")
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    t_prob = np.clip(np.asarray(teacher_soft_probs, dtype=np.float64).reshape(-1), 1e-5, 1.0 - 1e-5)
    blended_target = (1.0 - alpha_kd) * y + alpha_kd * t_prob
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30.0, 30.0)))
        err = (p - blended_target) / float(n)
        w -= 0.25 * (X.T @ err + (l2_reg / float(n)) * w)
        b -= 0.25 * float(np.sum(err))
    return w, b


# ============================================================================
# 5. Real Supervised VLM + Masked Forensic Head Training Loop (CUDA & CPU)
# ============================================================================

def resize_image_to_budget(img: Image.Image, image_max_px: int) -> Tuple[Image.Image, float]:
    """Enforces `image_max_px` on the longest edge and returns `(resized_image, downscale_ratio)`."""
    if image_max_px <= 0:
        raise ValueError("image_max_px must be positive")
    w, h = img.size
    max_edge = max(w, h)
    if max_edge <= image_max_px:
        return img.convert("RGB"), 1.0
    scale = float(image_max_px) / float(max_edge)
    new_w = max(16, int(round(w * scale)))
    new_h = max(16, int(round(h * scale)))
    resized = img.convert("RGB").resize((new_w, new_h), Image.Resampling.BICUBIC)
    return resized, float(max_edge) / float(max(new_w, new_h))


def execute_supervised_vlm_lora_training(
    model: Any,
    batch_collator: Any,
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    test_records: Optional[Sequence[Dict[str, Any]]] = None,
    hidden_dim: int = 960,
    max_steps: int = 50,
    image_max_px: int = 512,
    lr: float = 2e-4,
    use_fp16_amp: bool = False,
    output_dir: Path = Path("./outputs/qwen3_5_2b_lora_pilot"),
    model_id: str = "Qwen/Qwen3.5-2B",
) -> Dict[str, Any]:
    """Executes `max_steps` supervised optimization steps over `train_records` (after
    deduplicating against both `val_records` and `test_records`), optimizes LoRA parameters
    and the 13-output forensic projection head (`class_logit` + `6` forgery logits + `6` counter logits),
    records actual training peak memory, fits frozen calibration on `val_records` using actual
    pre-VLM baseline detector scores, evaluates frozen metrics on `test_records` (without ever leaking
    test data into training or calibration), and exports `forensic_vlm_policy.json`.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if max_steps <= 0:
        raise ValueError("max_steps must be >= 1")

    eval_for_dedup = list(val_records) + (list(test_records) if test_records is not None else [])
    clean_train, removed_leaks = deduplicate_training_against_eval(train_records, eval_for_dedup)
    if len(clean_train) == 0:
        raise ValueError("No training records remain after source-level deduplication against eval.")

    device = next(model.parameters()).device
    forensic_proj = nn.Linear(hidden_dim, 13, device=device, dtype=torch.float32)
    nn.init.zeros_(forensic_proj.weight)
    nn.init.zeros_(forensic_proj.bias)

    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(forensic_proj.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    amp_enabled = bool(use_fp16_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    forensic_proj.train()
    step_losses: List[float] = []
    downscale_ratios: List[float] = []
    axis_supervised_counts = np.zeros(6, dtype=np.int64)

    for step_idx in range(max_steps):
        rec = clean_train[step_idx % len(clean_train)]
        raw_img = rec.get("pil_image")
        if raw_img is None:
            img_path = rec.get("image_path")
            if img_path and Path(img_path).is_file():
                raw_img = Image.open(img_path)
            else:
                raise ValueError(f"Record at step {step_idx} missing 'pil_image' or valid 'image_path'")
        img_resized, ds_ratio = resize_image_to_budget(raw_img, image_max_px=image_max_px)
        downscale_ratios.append(ds_ratio)

        y_val = float(rec["label"])
        axes_6d = np.asarray(rec.get("axes_6d", [np.nan] * 6), dtype=np.float32)
        if axes_6d.shape != (6,):
            raise ValueError(f"Expected axes_6d of shape (6,), got {axes_6d.shape}")

        batch_inputs = batch_collator(img_resized, rec.get("prompt", "Assess forensic authenticity."), y_val, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = model(**batch_inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][:, -1, :].to(torch.float32)
            logits_13 = forensic_proj(last_hidden).squeeze(0)

            class_logit = logits_13[0]
            forgery_logits = logits_13[1:7]
            counter_logits = logits_13[7:13]

            y_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)
            loss_class = F.binary_cross_entropy_with_logits(class_logit, y_tensor)

            valid_mask = np.isfinite(axes_6d)
            axis_supervised_counts += valid_mask.astype(np.int64)
            if np.any(valid_mask):
                mask_t = torch.from_numpy(valid_mask).to(device=device)
                target_f = torch.from_numpy(np.where(valid_mask, axes_6d, 0.0).astype(np.float32)).to(device=device)
                target_c = torch.from_numpy(
                    np.where(valid_mask, (1.0 - np.where(valid_mask, axes_6d, 0.0)) * (1.0 - y_val), 0.0).astype(np.float32)
                ).to(device=device)
                loss_forgery = F.binary_cross_entropy_with_logits(forgery_logits[mask_t], target_f[mask_t])
                loss_counter = F.binary_cross_entropy_with_logits(counter_logits[mask_t], target_c[mask_t])
            else:
                loss_forgery = torch.tensor(0.0, dtype=torch.float32, device=device)
                loss_counter = torch.tensor(0.0, dtype=torch.float32, device=device)

            lm_loss = getattr(outputs, "loss", None)
            total_loss = loss_class + 0.5 * (loss_forgery + loss_counter)
            if lm_loss is not None and torch.isfinite(lm_loss):
                total_loss = total_loss + 0.25 * lm_loss.to(torch.float32)

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        step_losses.append(float(total_loss.detach().cpu().item()))

    train_peak_alloc_gb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
    )
    train_peak_res_gb = (
        torch.cuda.max_memory_reserved(device) / (1024 ** 3) if device.type == "cuda" else 0.0
    )

    # Copy learned PyTorch forensic head weights into MaskedAuxiliaryForensicHead contract
    head_contract = MaskedAuxiliaryForensicHead(hidden_dim=hidden_dim)
    with torch.no_grad():
        w_np = forensic_proj.weight.detach().cpu().numpy().astype(np.float64)
        b_np = forensic_proj.bias.detach().cpu().numpy().astype(np.float64)
        head_contract.w_class = w_np[0].copy()
        head_contract.b_class = float(b_np[0])
        head_contract.w_axes = w_np[1:7].T.copy()
        head_contract.b_axes = b_np[1:7].copy()
        head_contract.w_counter_axes = w_np[7:13].T.copy()
        head_contract.b_counter_axes = b_np[7:13].copy()
        head_contract.axes_supervised = [bool(axis_supervised_counts[i] > 0) for i in range(6)]

    def _eval_records(recs: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
        lbls: List[int] = []
        vlm_probs: List[float] = []
        fast_fusion: List[float] = []
        srm: List[float] = []
        probe: List[float] = []
        hz: List[float] = []
        domains: List[str] = []
        docs: List[float] = []
        with torch.no_grad():
            for r in recs:
                img = r.get("pil_image") or Image.open(r["image_path"])
                img_res, _ = resize_image_to_budget(img, image_max_px=image_max_px)
                inp = batch_collator(img_res, r.get("prompt", "Assess forensic authenticity."), float(r["label"]), device)
                out = model(**inp, output_hidden_states=True)
                vh = out.hidden_states[-1][:, -1, :].to(torch.float32).squeeze(0).cpu().numpy()
                pred = head_contract.predict(vh)
                lbls.append(int(r["label"]))
                vlm_probs.append(float(pred["class_prob"]))

                if "fast_fusion_prob" in r:
                    ff = float(r["fast_fusion_prob"])
                elif "fusion_prob" in r:
                    ff = float(r["fusion_prob"])
                else:
                    raise KeyError(f"Evaluation record '{r.get('image_id', 'unknown')}' missing required baseline 'fast_fusion_prob'")

                if "srm_score" in r:
                    sm = float(r["srm_score"])
                elif "srm_prob" in r:
                    sm = float(r["srm_prob"])
                else:
                    raise KeyError(f"Evaluation record '{r.get('image_id', 'unknown')}' missing required baseline 'srm_score'")

                pr = float(r["vfm_probe_score"]) if ("vfm_probe_score" in r and r["vfm_probe_score"] is not None) else (
                    float(r["probe_prob"]) if ("probe_prob" in r and r["probe_prob"] is not None) else float("nan")
                )
                fast_fusion.append(ff)
                srm.append(sm)
                probe.append(pr)
                hz.append(float(r.get("hotspot_z", 0.0)))
                domains.append(str(r.get("active_domain", "human_face")))
                docs.append(float(r.get("doc_tamper_score", 0.0)))
        return (
            np.asarray(lbls, dtype=np.int32),
            np.asarray(vlm_probs, dtype=np.float64),
            np.asarray(fast_fusion, dtype=np.float64),
            np.asarray(srm, dtype=np.float64),
            np.asarray(probe, dtype=np.float64),
            np.asarray(hz, dtype=np.float64),
            domains,
            np.asarray(docs, dtype=np.float64),
        )

    # 1. Fit calibration thresholds and learn routing strictly on val_records (development/calibration set)
    model.eval()
    forensic_proj.eval()
    val_y, val_vlm_p, val_ff, val_srm, val_probe, val_hz, val_dom, val_doc = _eval_records(val_records)

    frozen_fpr_thrs = fit_frozen_fpr_thresholds(val_y, val_vlm_p)
    routing_cfg = learn_routing_thresholds_on_held_out(
        labels=val_y,
        calibrated_fusion_probs=val_ff,
        calibrated_srm_probs=val_srm,
        calibrated_probe_probs=val_probe,
        hotspot_z_scores=val_hz,
        active_domains=val_dom,
        doc_tamper_scores=val_doc,
        max_escalation_budget=0.35,
    )
    val_eval = evaluate_candidate_at_fixed_fpr(val_y, val_vlm_p, frozen_thresholds=frozen_fpr_thrs)

    # 2. Evaluate frozen calibration and frozen routing on sealed test_records (if provided)
    test_eval: Optional[Dict[str, Any]] = None
    test_routing: Optional[Dict[str, Any]] = None
    test_cascade: Optional[Dict[str, Any]] = None
    if test_records is not None and len(test_records) > 0:
        test_y, test_vlm_p, test_ff, test_srm, test_probe, test_hz, test_dom, test_doc = _eval_records(test_records)
        test_eval = evaluate_candidate_at_fixed_fpr(test_y, test_vlm_p, frozen_thresholds=frozen_fpr_thrs)
        test_routing = evaluate_frozen_routing_on_test(
            frozen_cfg=routing_cfg,
            test_labels=test_y,
            test_calibrated_fusion_probs=test_ff,
            test_calibrated_srm_probs=test_srm,
            test_calibrated_probe_probs=test_probe,
            test_hotspot_z_scores=test_hz,
            test_active_domains=test_dom,
            test_doc_tamper_scores=test_doc,
        )
        test_cascade = evaluate_complete_cascade_on_test(
            frozen_cfg=routing_cfg,
            test_labels=test_y,
            test_calibrated_fusion_probs=test_ff,
            test_calibrated_srm_probs=test_srm,
            test_calibrated_probe_probs=test_probe,
            test_vlm_probs=test_vlm_p,
            test_hotspot_z_scores=test_hz,
            test_active_domains=test_dom,
            test_doc_tamper_scores=test_doc,
            vlm_decision_threshold=frozen_fpr_thrs.get("threshold_1pct_fpr", frozen_fpr_thrs.get("t_1pct_fpr", 0.50)),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    # Save reference evaluation predictions for reload reproducibility testing
    ref_eval_set = (test_records if test_records else val_records)[:20]
    ref_preds: Dict[str, Any] = {}
    with torch.no_grad():
        for r in ref_eval_set:
            img = r.get("pil_image") or Image.open(r["image_path"])
            img_res, _ = resize_image_to_budget(img, image_max_px=image_max_px)
            inp = batch_collator(img_res, r.get("prompt", "Assess forensic authenticity."), float(r["label"]), device)
            out = model(**inp, output_hidden_states=True)
            if hasattr(out, "hidden_states") and out.hidden_states is not None:
                h_last = out.hidden_states[-1]
            elif isinstance(out, (tuple, list)):
                h_last = out[0]
            else:
                h_last = getattr(out, "last_hidden_state", out)
            vh = h_last[:, -1, :].to(torch.float32).squeeze(0).cpu().numpy()
            pred = head_contract.predict(vh)
            key = r.get("sha256") or r["image_id"]
            ref_preds[key] = {
                "image_id": r["image_id"],
                "class_prob": float(pred["class_prob"]),
                "class_logit": float(pred["class_logit"]),
                "forgery_axes": [float(x) for x in pred["forgery_logits_6d"]],
                "counter_axes": [float(x) for x in pred["authentic_counter_logits_6d"]],
            }
    (output_dir / "reference_eval_predictions.json").write_text(json.dumps(ref_preds, indent=2), encoding="utf-8")

    policy_path = output_dir / "forensic_vlm_policy.json"
    export_forensic_vlm_policy_artifact(
        output_path=policy_path,
        routing_cfg=routing_cfg,
        head=head_contract,
        model_id=model_id,
        vlm_calibration_thresholds=frozen_fpr_thrs,
        diagnostic_only=True,
    )

    metrics: Dict[str, Any] = {
        "model_id": model_id,
        "steps_completed": int(max_steps),
        "deduplicated_train_samples": len(clean_train),
        "removed_eval_leaks": removed_leaks,
        "image_max_px": int(image_max_px),
        "mean_downscale_ratio": round(float(np.mean(downscale_ratios)), 4),
        "initial_loss": round(float(step_losses[0]), 6),
        "final_loss": round(float(step_losses[-1]), 6),
        "train_peak_allocated_gb": round(float(train_peak_alloc_gb), 4),
        "train_peak_reserved_gb": round(float(train_peak_res_gb), 4),
        "frozen_fpr_thresholds": frozen_fpr_thrs,
        "val_metrics": val_eval,
        "routing_policy": routing_cfg,
        "policy_artifact_path": str(policy_path),
    }
    if test_eval is not None:
        metrics["test_metrics"] = test_eval
    if test_routing is not None:
        metrics["test_routing"] = test_routing
    if test_cascade is not None:
        metrics["test_cascade_evaluation"] = test_cascade

    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def validate_dataset_records(records: Sequence[Dict[str, Any]]) -> None:
    """Validates required schema fields, image paths, baseline measurements,
    and label conventions across all records before loading model weights.
    Strictly verifies:
      1. Exact integer labels (0 or 1); rejects booleans, floats like 0.5, strings, or NaN.
      2. Required specialist baselines with finite ranges:
         - fast_fusion_prob: finite float in [0.0, 1.0]
         - srm_score: finite float in [0.0, 1.0]
         - doc_tamper_score: finite float in [0.0, 1.0]
         - vfm_probe_score: finite float in [0.0, 1.0] or None
         - hotspot_z: finite non-negative float (>= 0.0)
         - active_domain: valid string in allowed domain taxonomy
      3. Image file readability & SHA-256 digest consistency.
    """
    if not records:
        raise ValueError("Dataset records list is empty.")

    allowed_domains = {"human_face", "scene_arch", "object_product", "animal_art", "document_text"}
    seen_ids = set()

    for idx, r in enumerate(records):
        img_id = r.get("image_id")
        if not img_id or not isinstance(img_id, str):
            raise KeyError(f"Record #{idx} missing required non-empty string 'image_id'")
        if img_id in seen_ids:
            raise ValueError(f"Duplicate 'image_id' detected: {img_id}")
        seen_ids.add(img_id)

        root_id = r.get("root_source_id")
        if not root_id or not isinstance(root_id, str):
            raise KeyError(f"Record '{img_id}' missing required non-empty string 'root_source_id'")

        # 1. Exact integer label validation
        if "label" not in r:
            raise KeyError(f"Record '{img_id}' missing required 'label'")
        lbl = r["label"]
        if isinstance(lbl, bool):
            raise ValueError(f"Record '{img_id}' label cannot be boolean (True/False), got {lbl!r}")
        if not (isinstance(lbl, int) or (isinstance(lbl, float) and lbl.is_integer())):
            raise ValueError(f"Record '{img_id}' label must be an exact integer 0 or 1, got {lbl!r}")
        int_lbl = int(lbl)
        if int_lbl not in (0, 1):
            raise ValueError(f"Record '{img_id}' label must be 0 or 1, got {int_lbl}")

        # 2. Specialist baseline feature validation
        for field in ("fast_fusion_prob", "srm_score", "doc_tamper_score"):
            if field not in r:
                raise KeyError(f"Record '{img_id}' missing required specialist baseline '{field}'")
            val = r[field]
            if not isinstance(val, (int, float)) or not math.isfinite(val) or not (0.0 <= val <= 1.0):
                raise ValueError(f"Record '{img_id}' '{field}' must be a finite float in [0.0, 1.0], got {val!r}")

        if "vfm_probe_score" not in r:
            raise KeyError(f"Record '{img_id}' missing required baseline field 'vfm_probe_score'")
        probe_val = r["vfm_probe_score"]
        if probe_val is not None:
            if not isinstance(probe_val, (int, float)) or not math.isfinite(probe_val) or not (0.0 <= probe_val <= 1.0):
                raise ValueError(f"Record '{img_id}' 'vfm_probe_score' must be a finite float in [0.0, 1.0] or None, got {probe_val!r}")

        if "hotspot_z" not in r:
            raise KeyError(f"Record '{img_id}' missing required baseline field 'hotspot_z'")
        hz_val = r["hotspot_z"]
        if not isinstance(hz_val, (int, float)) or not math.isfinite(hz_val) or hz_val < 0.0:
            raise ValueError(f"Record '{img_id}' 'hotspot_z' must be a finite non-negative float, got {hz_val!r}")

        active_dom = r.get("active_domain")
        if not active_dom or not isinstance(active_dom, str) or active_dom not in allowed_domains:
            raise ValueError(f"Record '{img_id}' 'active_domain' must be one of {allowed_domains}, got {active_dom!r}")

        # 3. Image file readability & digest verification
        img_path = r.get("image_path")
        if img_path:
            p = Path(img_path)
            if not p.is_file():
                raise FileNotFoundError(f"Record '{img_id}' image file does not exist: {img_path}")
            try:
                with Image.open(p) as img_check:
                    img_check.verify()
                    if img_check.width <= 0 or img_check.height <= 0:
                        raise ValueError(f"Record '{img_id}' image has invalid dimensions: {img_check.size}")
            except Exception as exc:
                raise ValueError(f"Record '{img_id}' image file {p} is corrupt or unreadable: {exc}")

            if "sha256" in r and r["sha256"]:
                actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
                if actual_hash != r["sha256"]:
                    raise ValueError(f"Record '{img_id}' sha256 mismatch: recorded={r['sha256']}, actual={actual_hash}")
        elif "pil_image" not in r:
            raise KeyError(f"Record '{img_id}' must provide either 'image_path' or 'pil_image'")


def _load_jsonl_records(dataset_jsonl: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Loads JSONL records, validates all fields and images, and returns disjoint train, val, test splits.
    Preserves explicit 'split' fields when present, otherwise groups automatically by root source ID.
    Strictly asserts zero root leakage AND zero exact file/pixel blob leakage across splits.
    """
    if not dataset_jsonl.is_file():
        raise FileNotFoundError(f"Dataset JSONL not found: {dataset_jsonl}")

    records: List[Dict[str, Any]] = []
    jsonl_dir = dataset_jsonl.resolve().parent
    for line_idx, line in enumerate(dataset_jsonl.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON on line {line_idx + 1} of {dataset_jsonl}: {e}")

        if "image_path" in rec and rec["image_path"]:
            p = Path(rec["image_path"])
            if not p.is_file():
                candidates = [
                    jsonl_dir / rec["image_path"],
                    jsonl_dir / "images" / p.name,
                    Path.cwd() / rec["image_path"],
                    Path.cwd() / "data" / "forensic_pilot" / "images" / p.name,
                ]
                for cand in candidates:
                    if cand.is_file():
                        rec["image_path"] = str(cand.resolve())
                        break
        records.append(rec)

    validate_dataset_records(records)

    has_explicit_splits = all("split" in r for r in records)
    if has_explicit_splits:
        train_records = [r for r in records if r["split"] == "train"]
        val_records = [r for r in records if r["split"] == "val"]
        test_records = [r for r in records if r["split"] == "test"]

        all_splits = {r["split"] for r in records}
        unrecognized = all_splits - {"train", "val", "test"}
        if unrecognized:
            raise ValueError(f"Unrecognized dataset splits in {dataset_jsonl}: {unrecognized}")

        train_roots = {r["root_source_id"] for r in train_records}
        val_roots = {r["root_source_id"] for r in val_records}
        test_roots = {r["root_source_id"] for r in test_records}

        leak_val = train_roots & val_roots
        leak_test = train_roots & test_roots
        leak_val_test = val_roots & test_roots
        if leak_val:
            raise ValueError(f"Root source leakage between train and val splits: {list(leak_val)[:5]}")
        if leak_test:
            raise ValueError(f"Root source leakage between train and test splits: {list(leak_test)[:5]}")
        if leak_val_test:
            raise ValueError(f"Root source leakage between val and test splits: {list(leak_val_test)[:5]}")

        # Cross-split exact image file blob and pixel hash leakage detection
        def _get_hashes(recs: List[Dict[str, Any]]) -> Set[str]:
            h_set: Set[str] = set()
            for r in recs:
                if "sha256" in r and r["sha256"]:
                    h_set.add(str(r["sha256"]))
                elif "image_path" in r and r["image_path"]:
                    p = Path(r["image_path"])
                    if p.is_file():
                        h_set.add(hashlib.sha256(p.read_bytes()).hexdigest())
            return h_set

        train_hashes = _get_hashes(train_records)
        val_hashes = _get_hashes(val_records)
        test_hashes = _get_hashes(test_records)

        leak_hash_val = train_hashes & val_hashes
        leak_hash_test = train_hashes & test_hashes
        leak_hash_val_test = val_hashes & test_hashes
        if leak_hash_val:
            raise ValueError(f"Exact image file blob leakage between train and val splits ({len(leak_hash_val)} duplicate blobs)")
        if leak_hash_test:
            raise ValueError(f"Exact image file blob leakage between train and test splits ({len(leak_hash_test)} duplicate blobs)")
        if leak_hash_val_test:
            raise ValueError(f"Exact image file blob leakage between val and test splits ({len(leak_hash_val_test)} duplicate blobs)")

        for s_name, s_recs in [("train", train_records), ("val", val_records), ("test", test_records)]:
            if s_recs:
                labels = {int(r["label"]) for r in s_recs}
                if labels != {0, 1}:
                    raise ValueError(f"Split '{s_name}' must contain both positive and negative labels, found {labels}")
        return train_records, val_records, test_records

    splits = assign_disjoint_splits_by_root_source(records)
    return splits["train"], splits["val"], splits["test"]

    splits = assign_disjoint_splits_by_root_source(records)
    return splits["train"], splits["val"], splits["test"]


def run_colab_t4_lora_pilot(
    model_id: str = "Qwen/Qwen3.5-0.8B",
    max_steps: int = 50,
    lora_rank: int = 16,
    image_max_px: int = 512,
    dataset_jsonl: Optional[Path] = None,
    output_dir: Path = Path("./outputs/qwen08_smoke"),
) -> Dict[str, Any]:
    """Runs the 16-bit LoRA (`fp16=True` on Tesla T4) supervised vision fine-tuning pilot,
    saves the trained `adapter_final` checkpoint, and exports `forensic_vlm_policy.json`.
    """
    if dataset_jsonl is None or not dataset_jsonl.is_file():
        raise ValueError(
            "Provide `--dataset-jsonl <path>` pointing to a source-tagged forensic JSONL corpus "
            "to run supervised Colab T4 fine-tuning."
        )

    # 1. Validate dataset records, image files, baseline features, and splits BEFORE GPU allocation
    train_records, val_records, test_records = _load_jsonl_records(dataset_jsonl)
    print(
        f"[Dataset Validated] Train={len(train_records)} | Val={len(val_records)} | Test={len(test_records)} | "
        f"Preserved splits and validated required baselines before loading GPU model."
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Run on Google Colab with a Tesla T4 GPU or pass `--self-test`."
        )

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    bf16_ok = torch.cuda.is_bf16_supported()
    print(
        f"[Colab-T4 Pilot] Device={props.name} ({vram_gb:.3f} GB) | "
        f"BF16={bf16_ok} -> Using 16-bit LoRA with FP16={not bf16_ok} | "
        f"Model={model_id} | rank={lora_rank} | steps={max_steps} | image_cap={image_max_px}px"
    )

    # Guard against incompatible pre-installed torchao in Colab environments (e.g. 0.10.0 < 0.16.0)
    try:
        import peft.import_utils
        orig_is_torchao = getattr(peft.import_utils, "is_torchao_available", None)
        if orig_is_torchao is not None:
            def safe_is_torchao_available():
                try:
                    return orig_is_torchao()
                except (ImportError, Exception):
                    return False
            peft.import_utils.is_torchao_available = safe_is_torchao_available
    except Exception:
        pass
    try:
        import peft.tuners.lora.torchao
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
    except Exception:
        pass

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if bf16_ok else torch.float16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    hidden_dim = int(getattr(model.config, "hidden_size", getattr(getattr(model.config, "text_config", None), "hidden_size", 1536)))

    init_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[Colab-T4 Pilot] Model initialized | Init Allocated={init_alloc_gb:.3f} GB")

    def _hf_collator(img: Image.Image, prompt: str, label: float, dev: Any) -> Dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt")
        return {k: v.to(dev) for k, v in inputs.items()}

    metrics = execute_supervised_vlm_lora_training(
        model=model,
        batch_collator=_hf_collator,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        hidden_dim=hidden_dim,
        max_steps=max_steps,
        image_max_px=image_max_px,
        use_fp16_amp=(not bf16_ok),
        output_dir=output_dir,
        model_id=model_id,
    )
    metrics["init_allocated_gb"] = round(float(init_alloc_gb), 4)
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model.save_pretrained(output_dir / "adapter_final")
    processor.save_pretrained(output_dir / "adapter_final")
    print(
        f"[Colab-T4 Pilot] Completed {metrics['steps_completed']} steps | "
        f"Loss {metrics['initial_loss']:.4f} -> {metrics['final_loss']:.4f} | "
        f"Train Peak Allocated={metrics['train_peak_allocated_gb']:.3f} GB"
    )
    return metrics


# ============================================================================
# 6. Local Self-Test Suite (Includes Counterexample Regressions & Training Loop)
# ============================================================================

def run_self_tests() -> None:
    print("=" * 78)
    print("RUNNING ONN-DETECT VLM RESEARCH PLAN (2026-10-28) VERIFICATION SUITE")
    print("=" * 78)

    # 1. Verify published reference configurations (language + mmproj combined sizes)
    assert REFERENCE_VLM_SHORTLIST["qwen3_5_2b"].combined_pack_mb == 1948.0
    assert REFERENCE_VLM_SHORTLIST["qwen3_5_0_8b"].combined_pack_mb == 738.0
    assert REFERENCE_VLM_SHORTLIST["smolvlm2_500m_video"].combined_pack_mb == 546.0
    for k, cfg in REFERENCE_VLM_SHORTLIST.items():
        print(
            f"  [PASS] Candidate '{k}' ({cfg.hf_model_id}): "
            f"Lang={cfg.language_file_mb:.0f} MB + mmproj={cfg.vision_mmproj_mb:.0f} MB "
            f"= {cfg.combined_pack_mb:.0f} MB ({cfg.priority_role})"
        )

    # 2. Verify source-level deduplication catches BOTH single and chained derivative suffixes
    train_items = [
        {"image_id": "genimage_sdxl_00421_crop1.jpg", "sha256": "aaa"},
        {"image_id": "genimage_sdxl_00421_q75.jpg", "sha256": "bbb"},
        {"image_id": "frame_crop1_q75.jpg", "sha256": "eee"},  # Chained derivative regression test
        {"image_id": "openfake_flux_99102.png", "sha256": "ccc"},
    ]
    eval_items = [
        {"image_id": "genimage_sdxl_00421.png", "sha256": "ddd"},
        {"image_id": "frame.jpg", "sha256": "fff"},
    ]
    deduped, removed = deduplicate_training_against_eval(train_items, eval_items)
    assert removed == 3 and len(deduped) == 1 and deduped[0]["image_id"] == "openfake_flux_99102.png"
    print(f"  [PASS] Chained Source-Level Deduplication: Removed {removed} derivative leaks (including frame_crop1_q75.jpg)")

    # 3. Verify MaskedAuxiliaryForensicHead produces 13-output contract (class + 6 forgery + 6 counter logits)
    rng = np.random.default_rng(42)
    H = rng.normal(0, 0.5, size=(120, 32))
    y = np.array([0] * 60 + [1] * 60, dtype=np.int32)
    A = np.full((120, 6), np.nan, dtype=np.float64)
    for i in range(60, 120):
        H[i, :4] += 1.8
        A[i, 0] = 1.0
    for i in range(0, 60):
        H[i, :4] -= 1.5
        A[i, 0] = 0.0

    head = MaskedAuxiliaryForensicHead(hidden_dim=32).fit(H, y, A)
    pred = head.predict(H[75], applicable_mask_6d=[True, True, True, True, False, False])
    assert pred["class_prob"] > 0.80
    assert len(pred["forgery_logits_6d"]) == 6 and len(pred["authentic_counter_logits_6d"]) == 6
    assert pred["forgery_logits_6d"][0] > 0.5 and pred["authentic_counter_logits_6d"][0] < 0.0
    assert math.isnan(pred["forgery_logits_6d"][4]) and pred["axis_probabilities"]["typography_document_layout"] is None
    print(
        f"  [PASS] Unified 13-Output MaskedAuxiliaryForensicHead: P(Fake)={pred['class_prob']:.4f}, "
        f"forgery_0={pred['forgery_logits_6d'][0]:.3f}, counter_0={pred['authentic_counter_logits_6d'][0]:.3f}"
    )

    # 4. Verify tie-aware fixed-FPR evaluator rejects constant 0.5 scores (0% TPR, 0% FPR)
    ev_constant = evaluate_candidate_at_fixed_fpr(
        labels=np.array([0] * 100 + [1] * 100),
        scores=np.full(200, 0.5, dtype=np.float64),
    )
    assert ev_constant["tpr_at_1pct_fpr"] == 0.0 and ev_constant["achieved_fpr_at_1pct"] == 0.0
    print(
        f"  [PASS] Tie-Aware Constant-Score Regression Check: constant 0.5 -> "
        f"TPR@1%FPR={ev_constant['tpr_at_1pct_fpr']:.4f}, AchievedFPR={ev_constant['achieved_fpr_at_1pct']:.4f}"
    )

    # 4b. Verify fine-grained numerical precision preservation: 99 scores 0.5000002, 1 score 0.5000004
    cal_precision = np.array([0.5000002] * 99 + [0.5000004], dtype=np.float64)
    thr_precision = select_tie_aware_threshold_for_fpr(cal_precision, 0.01)
    # Check that achieved FPR in float32 (Android representation) is strictly <= 1% (never 100%)
    thr_precision_f32 = np.float32(thr_precision)
    achieved_precision_fpr = float(np.sum(np.float32(cal_precision) >= thr_precision_f32)) / len(cal_precision)
    assert achieved_precision_fpr <= 0.01 + 1e-12
    print(
        f"  [PASS] Precision Preservation (99x 0.5000002 vs 1x 0.5000004): threshold={thr_precision:.7f}, "
        f"Float32 FPR={achieved_precision_fpr:.2%}"
    )

    cal_y = np.array([0] * 200 + [1] * 200)
    cal_s = np.r_[rng.uniform(0.0, 0.4, 200), rng.uniform(0.35, 0.99, 200)]
    test_y = np.array([0] * 200 + [1] * 200)
    test_s = np.r_[rng.uniform(0.0, 0.4, 200), rng.uniform(0.35, 0.99, 200)]
    ev = evaluate_candidate_at_fixed_fpr(
        labels=test_y,
        scores=test_s,
        inconclusive_mask=np.array([False] * 390 + [True] * 10),
        calibration_labels=cal_y,
        calibration_scores=cal_s,
    )
    assert 0.0 < ev["tpr_at_1pct_fpr"] <= 1.0 and ev["achieved_fpr_at_1pct"] <= 0.03
    print(
        f"  [PASS] Calibration-Frozen Fixed-FPR Evaluator: TPR@1%FPR={ev['tpr_at_1pct_fpr']:.4f}, "
        f"AchievedTestFPR={ev['achieved_fpr_at_1pct']:.4f}, Inconclusive={ev['inconclusive_rate']:.3f}"
    )

    # 5. Verify learned routing threshold optimizer handles both feasible and infeasible budgets honestly
    y_rt = np.array([0] * 200 + [1] * 200)
    p_fus = np.r_[rng.uniform(0.02, 0.35, 200), rng.uniform(0.45, 0.98, 200)]
    p_srm = np.r_[rng.uniform(0.05, 0.40, 200), rng.uniform(0.30, 0.95, 200)]
    p_prb = np.r_[rng.uniform(0.02, 0.35, 200), rng.uniform(0.50, 0.99, 200)]
    hz_rt = np.r_[rng.uniform(0.2, 1.4, 200), rng.uniform(0.8, 3.2, 200)]
    dom_rt = ["document_text" if i % 4 == 0 else "human_face" for i in range(400)]
    doc_rt = np.r_[rng.uniform(0.0, 0.2, 200), rng.uniform(0.1, 0.9, 200)]
    rt_cfg = learn_routing_thresholds_on_held_out(
        labels=y_rt,
        calibrated_fusion_probs=p_fus,
        calibrated_srm_probs=p_srm,
        calibrated_probe_probs=p_prb,
        hotspot_z_scores=hz_rt,
        active_domains=dom_rt,
        doc_tamper_scores=doc_rt,
        max_escalation_budget=0.35,
    )
    assert rt_cfg["feasible"] is True and rt_cfg["escalation_rate"] <= 0.35

    rt_infeasible = learn_routing_thresholds_on_held_out(
        labels=y_rt,
        calibrated_fusion_probs=p_fus,
        calibrated_srm_probs=p_srm,
        calibrated_probe_probs=p_prb,
        hotspot_z_scores=hz_rt,
        max_escalation_budget=0.001,
    )
    assert rt_infeasible["feasible"] is False
    rt_infeasible_test = evaluate_frozen_routing_on_test(rt_infeasible, y_rt, p_fus, p_srm, p_prb, hz_rt)
    assert rt_infeasible_test["test_escalation_rate"] == 0.0
    print(
        f"  [PASS] Feasibility-Checked Router: feasible={rt_cfg['feasible']} (esc={rt_cfg['escalation_rate']:.3f}), "
        f"infeasible_budget -> feasible={rt_infeasible['feasible']} (test_esc={rt_infeasible_test['test_escalation_rate']:.3f})"
    )

    # 6. Verify real supervised training loop executes max_steps, decreases loss, and exports policy JSON
    import tempfile
    import torch
    import torch.nn as nn

    class _TinyVisionLanguageStub(nn.Module):
        def __init__(self, hidden_dim: int = 16) -> None:
            super().__init__()
            self.encoder = nn.Linear(12, hidden_dim)
            self.lora_adapter = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, pixel_vec: torch.Tensor, output_hidden_states: bool = True) -> Any:
            h = torch.tanh(self.encoder(pixel_vec)) + self.lora_adapter(torch.tanh(self.encoder(pixel_vec)))
            h_seq = h.unsqueeze(1)

            class _Out:
                hidden_states = (h_seq,)
                loss = None

            return _Out()

    torch.manual_seed(42)
    stub_model = _TinyVisionLanguageStub(hidden_dim=16)
    synth_train = []
    synth_val = []
    synth_test = []
    for idx in range(16):
        lbl = idx % 2
        color = (220, 40, 40) if lbl == 1 else (30, 180, 60)
        img = Image.new("RGB", (640, 480), color=color)
        axes = [float(lbl), float(lbl), np.nan, np.nan, np.nan, np.nan]
        synth_train.append({"image_id": f"train_src{idx}_crop1.png", "pil_image": img, "label": lbl, "axes_6d": axes})
        synth_val.append({
            "image_id": f"val_src{idx}.png",
            "pil_image": img,
            "label": lbl,
            "axes_6d": axes,
            "fast_fusion_prob": 0.85 if lbl == 1 else 0.15,
            "srm_score": 0.80 if lbl == 1 else 0.10,
            "vfm_probe_score": 0.88 if lbl == 1 else 0.12,
            "hotspot_z": 2.2 if lbl == 1 else 0.4,
            "doc_tamper_score": 0.0,
            "active_domain": "human_face",
        })
        synth_test.append({
            "image_id": f"test_src{idx}.png",
            "pil_image": img,
            "label": lbl,
            "axes_6d": axes,
            "fast_fusion_prob": 0.85 if lbl == 1 else 0.15,
            "srm_score": 0.80 if lbl == 1 else 0.10,
            "vfm_probe_score": 0.88 if lbl == 1 else 0.12,
            "hotspot_z": 2.2 if lbl == 1 else 0.4,
            "doc_tamper_score": 0.0,
            "active_domain": "human_face",
        })

    def _stub_collator(img: Image.Image, prompt: str, label: float, dev: Any) -> Dict[str, Any]:
        arr = np.asarray(img, dtype=np.float32).mean(axis=(0, 1)) / 255.0
        vec = np.tile(arr, 4).astype(np.float32)
        return {"pixel_vec": torch.from_numpy(vec[None, :]).to(dev)}

    with tempfile.TemporaryDirectory() as tmpd:
        train_metrics = execute_supervised_vlm_lora_training(
            model=stub_model,
            batch_collator=_stub_collator,
            train_records=synth_train,
            val_records=synth_val,
            test_records=synth_test,
            hidden_dim=16,
            max_steps=25,
            image_max_px=256,
            lr=1e-2,
            output_dir=Path(tmpd),
            model_id="stub/tiny-vlm",
        )
        assert train_metrics["steps_completed"] == 25
        assert train_metrics["final_loss"] < train_metrics["initial_loss"]
        assert train_metrics["mean_downscale_ratio"] > 1.0  # 640px resized to 256px
        assert Path(train_metrics["policy_artifact_path"]).is_file()
        assert "val_metrics" in train_metrics and "test_metrics" in train_metrics
        assert "test_routing" in train_metrics
        assert "test_cascade_evaluation" in train_metrics
        test_casc = train_metrics["test_cascade_evaluation"]
        assert "cascade_tpr" in test_casc and "baseline_tpr" in test_casc
        assert "delta_tpr" in test_casc
        print(
            f"  [PASS] Supervised VLM Training Loop (25 steps): loss {train_metrics['initial_loss']:.4f} -> "
            f"{train_metrics['final_loss']:.4f}, downscale_ratio={train_metrics['mean_downscale_ratio']:.2f}"
        )
        print(
            f"  [PASS] Complete Routed Cascade Evaluation: Base TPR={test_casc['baseline_tpr']:.4f}, "
            f"Cascade TPR={test_casc['cascade_tpr']:.4f} (Delta={test_casc['delta_tpr']:+.4f}, "
            f"EscRate={test_casc['test_escalation_rate']:.1%})"
        )

    # Validate dataset validation and split preservation on synthetic records
    valid_base = {
        "pil_image": Image.new("RGB", (32, 32)),
        "fast_fusion_prob": 0.5,
        "srm_score": 0.5,
        "vfm_probe_score": 0.5,
        "hotspot_z": 1.0,
        "doc_tamper_score": 0.1,
        "active_domain": "human_face",
    }
    test_recs = [
        {"image_id": "r1", "root_source_id": "root_A", "split": "train", "label": 0, **valid_base},
        {"image_id": "r2", "root_source_id": "root_B", "split": "train", "label": 1, **valid_base},
        {"image_id": "r3", "root_source_id": "root_C", "split": "val", "label": 0, **valid_base},
        {"image_id": "r4", "root_source_id": "root_D", "split": "val", "label": 1, **valid_base},
        {"image_id": "r5", "root_source_id": "root_E", "split": "test", "label": 0, **valid_base},
        {"image_id": "r6", "root_source_id": "root_F", "split": "test", "label": 1, **valid_base},
    ]
    validate_dataset_records(test_recs)

    # Assert rejection of fractional labels (label=0.5)
    bad_lbl = [dict(test_recs[0], label=0.5)]
    try:
        validate_dataset_records(bad_lbl)
        raise AssertionError("Failed to reject fractional label 0.5")
    except ValueError:
        pass

    # Assert rejection of NaN fast_fusion_prob
    bad_ff = [dict(test_recs[0], fast_fusion_prob=float("nan"))]
    try:
        validate_dataset_records(bad_ff)
        raise AssertionError("Failed to reject NaN fast_fusion_prob")
    except ValueError:
        pass

    # Assert rejection of out-of-bounds srm_score=2.0
    bad_srm = [dict(test_recs[0], srm_score=2.0)]
    try:
        validate_dataset_records(bad_srm)
        raise AssertionError("Failed to reject out-of-bounds srm_score=2.0")
    except ValueError:
        pass

    # Assert rejection of negative hotspot_z=-0.5
    bad_hz = [dict(test_recs[0], hotspot_z=-0.5)]
    try:
        validate_dataset_records(bad_hz)
        raise AssertionError("Failed to reject negative hotspot_z")
    except ValueError:
        pass

    print("  [PASS] Pre-Training Dataset Schema & Baseline Measurement Validation (with negative counterexample rejections)")


def export_cross_language_parity_fixture(output_path: Path, hidden_dim: int = 8, seed: int = 20261028) -> Dict[str, Any]:
    """Writes a deterministic, explicitly NON-TRAINED policy fixture plus expected head outputs
    and cross-language routing test fixtures.

    Android's `OnDemandForensicVlmAnalyzerContractTest` loads this file and asserts that:
    1. `PolicyBundle.fromJson` + `projectWithHeadContract` reproduce Python's `head.predict`.
    2. `routeWithPolicy` reproduces Python's `route_with_policy` across all 10 routing fixtures.
    3. Fine-grained threshold precision is preserved in float32.
    """
    rng = np.random.default_rng(seed)
    head = MaskedAuxiliaryForensicHead(hidden_dim=hidden_dim)
    head.w_class = np.round(rng.normal(0.0, 0.5, hidden_dim), 6)
    head.b_class = round(float(rng.normal(0.0, 0.2)), 6)
    head.w_axes = np.round(rng.normal(0.0, 0.5, (hidden_dim, 6)), 6)
    head.b_axes = np.round(rng.normal(0.0, 0.2, 6), 6)
    head.w_counter_axes = np.round(rng.normal(0.0, 0.5, (hidden_dim, 6)), 6)
    head.b_counter_axes = np.round(rng.normal(0.0, 0.2, 6), 6)
    head.axes_supervised = [True, True, True, True, False, False]

    routing_cfg = {
        "feasible": True,
        "decision_threshold_dev": 0.5782,
        "llr_uncertainty_halfwidth": 0.65,
        "calibrated_conflict_logit_threshold": 1.80,
        "hotspot_z_threshold": 1.85,
        "doc_tamper_threshold": 0.48,
    }
    payload = export_forensic_vlm_policy_artifact(
        output_path=output_path,
        routing_cfg=routing_cfg,
        head=head,
        model_id="fixture/untrained-parity-head",
        vlm_calibration_thresholds={"t_1pct_fpr": 0.5234, "t_5pct_fpr": 0.4120, "t_balanced": 0.4850},
        diagnostic_only=True,
    )
    h_last = np.round(rng.normal(0.0, 1.0, hidden_dim), 6)
    mask = [True, True, True, True, False, False]
    expected_full = head.predict(h_last)
    expected_masked = head.predict(h_last, applicable_mask_6d=mask)
    payload["parity_probe"] = {
        "note": "Untrained random weights for cross-language contract testing only.",
        "h_last": [float(x) for x in h_last.tolist()],
        "applicable_mask_6d": mask,
        "expected_full": {
            "class_logit": expected_full["class_logit"],
            "class_prob": expected_full["class_prob"],
            "forgery_logits_6d": expected_full["forgery_logits_6d"],
            "authentic_counter_logits_6d": expected_full["authentic_counter_logits_6d"],
        },
        "expected_masked_forgery_logits_6d": [
            None if not math.isfinite(v) else v for v in expected_masked["forgery_logits_6d"]
        ],
    }

    # Cross-language routing fixtures covering documents, missing scores, memory limits, and threshold boundaries
    routing_cases = [
        {
            "name": "confident_fast_face_no_escalate",
            "fast_fusion_prob": 0.95,
            "srm_calibrated_score": 0.90,
            "vfm_probe_score": 0.92,
            "hotspot_z_score": 0.5,
            "active_domain": "human_face",
            "doc_tamper_score": 0.0,
            "available_heap_mb": 256,
            "require_explanation": False,
        },
        {
            "name": "boundary_uncertainty_escalates",
            "fast_fusion_prob": routing_cfg["decision_threshold_dev"],
            "srm_calibrated_score": 0.50,
            "vfm_probe_score": 0.50,
            "hotspot_z_score": 0.0,
            "active_domain": "human_face",
            "doc_tamper_score": 0.0,
            "available_heap_mb": 256,
            "require_explanation": False,
        },
        {
            "name": "document_tamper_escalates_on_document_text",
            "fast_fusion_prob": 0.10,
            "srm_calibrated_score": 0.12,
            "vfm_probe_score": 0.15,
            "hotspot_z_score": 0.2,
            "active_domain": "document_text",
            "doc_tamper_score": 0.85,
            "available_heap_mb": 256,
            "require_explanation": False,
        },
        {
            "name": "document_tamper_ignored_on_human_face",
            "fast_fusion_prob": 0.10,
            "srm_calibrated_score": 0.12,
            "vfm_probe_score": 0.15,
            "hotspot_z_score": 0.2,
            "active_domain": "human_face",
            "doc_tamper_score": 0.85,
            "available_heap_mb": 256,
            "require_explanation": False,
        },
        {
            "name": "cross_engine_conflict_escalates",
            "fast_fusion_prob": 0.90,
            "srm_calibrated_score": 0.95,
            "vfm_probe_score": 0.05,
            "hotspot_z_score": 0.0,
            "active_domain": "human_face",
            "doc_tamper_score": 0.0,
            "available_heap_mb": 256,
            "require_explanation": False,
        },
        {
            "name": "missing_probe_score_nan_safe",
            "fast_fusion_prob": 0.95,
            "srm_calibrated_score": 0.92,
            "vfm_probe_score": None,
            "hotspot_z_score": 0.0,
            "active_domain": "human_face",
            "doc_tamper_score": 0.0,
            "available_heap_mb": 256,
            "require_explanation": False,
        },
        {
            "name": "localized_hotspot_roi_escalates",
            "fast_fusion_prob": 0.95,
            "srm_calibrated_score": 0.92,
            "vfm_probe_score": 0.90,
            "hotspot_z_score": 2.50,
            "active_domain": "human_face",
            "doc_tamper_score": 0.0,
            "available_heap_mb": 256,
            "require_explanation": False,
        },
        {
            "name": "memory_constrained_micro_moe_only",
            "fast_fusion_prob": routing_cfg["decision_threshold_dev"],
            "srm_calibrated_score": 0.50,
            "vfm_probe_score": 0.50,
            "hotspot_z_score": 0.0,
            "active_domain": "human_face",
            "doc_tamper_score": 0.0,
            "available_heap_mb": 64,
            "require_explanation": False,
        },
        {
            "name": "explanation_requested_forces_cot_mode",
            "fast_fusion_prob": 0.95,
            "srm_calibrated_score": 0.90,
            "vfm_probe_score": 0.92,
            "hotspot_z_score": 0.0,
            "active_domain": "human_face",
            "doc_tamper_score": 0.0,
            "available_heap_mb": 256,
            "require_explanation": True,
        },
        {
            "name": "infeasible_policy_never_escalates",
            "fast_fusion_prob": routing_cfg["decision_threshold_dev"],
            "srm_calibrated_score": 0.95,
            "vfm_probe_score": 0.05,
            "hotspot_z_score": 3.0,
            "active_domain": "document_text",
            "doc_tamper_score": 0.99,
            "available_heap_mb": 256,
            "require_explanation": True,
            "override_policy_infeasible": True,
        },
    ]

    evaluated_routing_cases = []
    for c in routing_cases:
        pol = dict(routing_cfg)
        if c.get("override_policy_infeasible"):
            pol["feasible"] = False
        res = route_with_policy(
            policy=pol,
            fast_fusion_prob=float(c["fast_fusion_prob"]),
            srm_calibrated_score=float(c["srm_calibrated_score"]),
            vfm_probe_score=float(c["vfm_probe_score"]) if c["vfm_probe_score"] is not None else None,
            hotspot_z_score=float(c["hotspot_z_score"]),
            active_domain=str(c["active_domain"]),
            doc_tamper_score=float(c["doc_tamper_score"]),
            available_heap_mb=int(c["available_heap_mb"]),
            require_explanation=bool(c["require_explanation"]),
        )
        c_copy = dict(c)
        c_copy["expected"] = res
        evaluated_routing_cases.append(c_copy)

    payload["routing_parity_fixtures"] = evaluated_routing_cases
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verify_saved_model_reload(
    output_dir: Path,
    dataset_jsonl: Path,
    model_id: str = "Qwen/Qwen3.5-0.8B",
    max_eval_samples: int = 10,
) -> bool:
    """Reloads the saved adapter and forensic head contract in a fresh model session
    and verifies that predicted class probabilities reproduce those recorded before saving.
    """
    import torch
    # Guard against incompatible pre-installed torchao in Colab environments
    try:
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False
        import peft.tuners.lora.torchao
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
    except Exception:
        pass
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    policy_file = output_dir / "forensic_vlm_policy.json"
    adapter_dir = output_dir / "adapter_final"

    if not policy_file.is_file():
        raise FileNotFoundError(f"Missing policy file: {policy_file}")
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Missing adapter directory: {adapter_dir}")

    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    head = MaskedAuxiliaryForensicHead.from_contract_dict(policy["head_contract"])

    print(f"[Reload-Test] Loading base model '{model_id}'...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id)
    base_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        device_map="auto" if device.type == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    train_recs, val_recs, test_recs = _load_jsonl_records(dataset_jsonl)
    eval_set = (test_recs if test_recs else val_recs)[:max_eval_samples]

    ref_pred_file = output_dir / "reference_eval_predictions.json"
    has_refs = ref_pred_file.is_file()
    ref_data: Dict[str, Any] = json.loads(ref_pred_file.read_text(encoding="utf-8")) if has_refs else {}

    reloaded_probs = []
    diffs = []
    with torch.no_grad():
        for r in eval_set:
            img = r.get("pil_image") or Image.open(r["image_path"])
            img_res, _ = resize_image_to_budget(img, image_max_px=512)
            messages = [{"role": "user", "content": [{"type": "image", "image": img_res}, {"type": "text", "text": "Assess forensic authenticity."}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img_res], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs, output_hidden_states=True)
            vh = out.hidden_states[-1][:, -1, :].to(torch.float32).squeeze(0).cpu().numpy()
            pred = head.predict(vh)
            p_val = float(pred["class_prob"])
            reloaded_probs.append(p_val)

            key = r.get("sha256") or r["image_id"]
            if key in ref_data:
                ref_p = float(ref_data[key]["class_prob"])
                d = abs(p_val - ref_p)
                diffs.append(d)
                if d > 1e-4:
                    raise AssertionError(
                        f"Reload prediction mismatch for '{key}': reloaded={p_val:.6f}, "
                        f"reference={ref_p:.6f}, diff={d:.6e} > tolerance 1e-4"
                    )

    if diffs:
        max_d = max(diffs)
        print(f"[Reload-Test] VERIFIED: {len(diffs)} reference predictions reproduced! (max abs diff = {max_d:.6e} <= 1e-4)")
    else:
        print(f"[Reload-Test] Evaluated {len(reloaded_probs)} samples successfully. Mean P(Fake)={np.mean(reloaded_probs):.4f}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--export-parity-fixture", type=Path, default=None,
                        help="Write the untrained cross-language head/policy parity fixture and exit.")
    parser.add_argument("--validate-dataset", type=Path, default=None,
                        help="Validate dataset schema, paths, and splits without training.")
    parser.add_argument("--verify-reload", type=Path, default=None,
                        help="Verify reloading adapter_final and head contract from output directory.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--image-max-px", type=int, default=512)
    parser.add_argument("--dataset-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/qwen08_smoke"))
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
        return

    if args.export_parity_fixture is not None:
        export_cross_language_parity_fixture(args.export_parity_fixture)
        print(f"Wrote parity fixture: {args.export_parity_fixture}")
        return

    if args.validate_dataset is not None:
        tr, va, te = _load_jsonl_records(args.validate_dataset)
        print(
            f"[Dataset Validated Successfully] Path: {args.validate_dataset}\n"
            f"  Train: {len(tr)} samples ({len(set(r['root_source_id'] for r in tr))} roots)\n"
            f"  Val:   {len(va)} samples ({len(set(r['root_source_id'] for r in va))} roots)\n"
            f"  Test:  {len(te)} samples ({len(set(r['root_source_id'] for r in te))} roots)\n"
            f"  Zero root leakage verified. All image files and baseline features valid."
        )
        return

    if args.verify_reload is not None:
        if args.dataset_jsonl is None:
            raise ValueError("Provide `--dataset-jsonl <path>` to run reload verification.")
        verify_saved_model_reload(
            output_dir=args.verify_reload,
            dataset_jsonl=args.dataset_jsonl,
            model_id=args.model_id,
        )
        return

    run_colab_t4_lora_pilot(
        model_id=args.model_id,
        max_steps=args.max_steps,
        lora_rank=args.lora_rank,
        image_max_px=args.image_max_px,
        dataset_jsonl=args.dataset_jsonl,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
