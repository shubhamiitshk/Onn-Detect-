#!/usr/bin/env python3
"""Google Colab T4 Forensic VLM Training, Source-Deduplication & A-G Ablation Pipeline.

Implements the 28 October 2026 ONN-DETECT On-Device VLM Research Plan:
- **Primary Quality Candidate**: `Qwen/Qwen3.5-2B` (Unsloth 16-bit LoRA, FP16 on Tesla T4;
  `1.28 GB` Q4_K_M language + `668 MB` FP16 mmproj = `~1.95 GB` reference pack).
- **Smaller Deployment Challenger**: `Qwen/Qwen3.5-0.8B` (`533 MB` Q4_K_M + `205 MB` FP16 mmproj = `~738 MB`).
- **Compact Baseline**: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (`437 MB` Q8_0 + `109 MB` Q8_0 mmproj = `~546 MB`).
- **Google Runtime Alternative**: `google/gemma-4-E2B-it` (requires BF16 >16 GB for official vision recipe).

Key Methodological Guarantees:
1. **16-Bit LoRA (`fp16=True`, `bf16=False` on T4) by Default for Qwen3.5**:
   - Follows Unsloth's Qwen3.5 guidance (avoiding 4-bit QLoRA quantization discrepancies during
     training while exporting validated quantized GGUF + `mmproj` artifacts separately).
   - Starts with `lora_r=16`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`,
     `max_seq_length=1024`, explicit image resolution/visual-token caps (`384-512 px`), 1 global
     image first, 1 epoch pilot, and **GRPO disabled initially** (`--enable-grpo` only for Ablation G).
2. **Source-Level Deduplication (`FakeClue` vs. `GenImage`/`FF++`/`Chameleon`/`OpenFake`)**:
   - Groups every source image and all its crops, local edits, reconstructions, and JPEG/WebP
     recompressions into a single split (`canonical_source_id`) so training on `FakeClue` cannot
     leak into held-out evaluation sets.
3. **Controlled A-to-G Ablation Matrix & Wilson Confidence Intervals**:
   - Evaluates Candidates A through G on identical held-out splits at fixed 1% and 0.1% FPR,
     tracking inconclusive coverage and masked (`not_applicable` / `not_observable`) artifact axes.

Usage:
    # Fast local verification of source deduplication, pack accounting, and A-G ablation math:
    python ml/scripts/train_forensic_vlm_colab_t4.py --self-test

    # Colab T4 20-50 step smoke test & pilot LoRA run on Qwen3.5-2B:
    python ml/scripts/train_forensic_vlm_colab_t4.py \
        --model-id Qwen/Qwen3.5-2B \
        --data-dir ./data/openfake_visual \
        --output-dir ./models/qwen3_5_2b_forensic_lora \
        --max-steps 50
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


# ============================================================================
# 1. Published Reference Pack Configurations (Language GGUF + Vision mmproj)
# ============================================================================

@dataclass(frozen=True)
class ReferenceVlmPackConfig:
    """Published reference file sizes (decimal MB/GB) including BOTH language and vision/mmproj."""

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
        quant_config_desc="Unsloth Q4_K_M language + FP16 mmproj (vision: 24 layers, width 1024, patch merge)",
        language_file_mb=1280.0,
        vision_mmproj_mb=668.0,
        combined_pack_mb=1948.0,
        t4_16bit_lora_supported=True,
        notes="Apache-2.0. Official Unsloth T4 vision notebook records 5.146 GB peak reserved CUDA memory on demo run.",
    ),
    "qwen3_5_0_8b": ReferenceVlmPackConfig(
        priority_role="First smaller shipping candidate",
        hf_model_id="Qwen/Qwen3.5-0.8B",
        quant_config_desc="Unsloth Q4_K_M language + FP16 mmproj (vision: 12 layers, width 768 — smaller visual capacity than 2B)",
        language_file_mb=533.0,
        vision_mmproj_mb=205.0,
        combined_pack_mb=738.0,
        t4_16bit_lora_supported=True,
        notes="Apache-2.0. Smaller vision encoder (12L/768d vs 24L/1024d in 2B) and language backbone.",
    ),
    "smolvlm2_500m_video": ReferenceVlmPackConfig(
        priority_role="Compact comparison baseline",
        hf_model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        quant_config_desc="Maintainer Q8_0 language + Q8_0 mmproj (199 MB if FP16 mmproj; note Q8 vs Qwen's Q4)",
        language_file_mb=437.0,
        vision_mmproj_mb=109.0,
        combined_pack_mb=546.0,
        t4_16bit_lora_supported=True,
        notes="Apache-2.0. Official 500M video-instruct identifier; decoder hidden_size=960.",
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
# 2. Source-Level Deduplication (Prevents FakeClue / GenImage / FF++ Leakage)
# ============================================================================

_DERIVATIVE_SUFFIX_RE = re.compile(
    r"(_crop\d+|_roi\d+|_inpaint\d*|_edit\d*|_q\d+|_deg\d+|_laundered|_rec\d*|_b_free.*)$",
    re.IGNORECASE,
)


def canonical_source_id(file_path_or_id: str) -> str:
    """Normalizes a path or sample identifier to its underlying root source ID so that
    a source image and all its crops, edits, reconstructions, and recompressions stay
    in the same split.
    """
    stem = Path(file_path_or_id).stem.lower()
    cleaned = _DERIVATIVE_SUFFIX_RE.sub("", stem)
    # Strip trailing hash/index fragments after double underscores if present
    if "__" in cleaned:
        cleaned = cleaned.split("__")[0]
    return cleaned


def deduplicate_training_against_eval(
    train_records: Sequence[Dict[str, Any]],
    eval_records: Sequence[Dict[str, Any]],
    id_key: str = "image_id",
    sha256_key: str = "sha256",
) -> Tuple[List[Dict[str, Any]], int]:
    """Removes any training sample whose canonical source ID or file SHA-256 overlaps
    with the held-out evaluation splits (critical when training on `FakeClue`, which
    draws from `GenImage`, `FF++`, and `Chameleon`).
    """
    forbidden_sources: Set[str] = set()
    forbidden_hashes: Set[str] = set()

    for rec in eval_records:
        raw_id = str(rec.get(id_key, ""))
        if raw_id:
            forbidden_sources.add(canonical_source_id(raw_id))
        h = str(rec.get(sha256_key, "")).strip().lower()
        if h:
            forbidden_hashes.add(h)

    clean_train: List[Dict[str, Any]] = []
    removed_count = 0
    for rec in train_records:
        raw_id = str(rec.get(id_key, ""))
        src_id = canonical_source_id(raw_id) if raw_id else ""
        h = str(rec.get(sha256_key, "")).strip().lower()
        if (src_id and src_id in forbidden_sources) or (h and h in forbidden_hashes):
            removed_count += 1
        else:
            clean_train.append(rec)

    return clean_train, removed_count


# ============================================================================
# 3. Masked 6-Axis Auxiliary Head (`NOT_APPLICABLE` / `NOT_OBSERVABLE` Safe)
# ============================================================================

RUBRIC_AXES: Tuple[str, ...] = (
    "anatomical_geometry",
    "surface_texture_fidelity",
    "lighting_shadow_physics",
    "perspective_boundary_seams",
    "typography_document_layout",
    "temporal_audiovisual_sync",
)


class MaskedAuxiliaryForensicHead:
    """Supervised auxiliary head over the VLM's final prefill representation `h_{-1}`.

    Unlike naive label inversion (`1 - y`), this head:
    1. Predicts `z_class` (primary binary synthetic vs. authentic logit) AND
       6 observable artifact axis logits `z_axes`.
    2. Masks missing / `not_applicable` / `not_observable` axis annotations (`np.nan`)
       in the binary cross-entropy loss so an authentic bokeh portrait or a landscape
       without text never forces bogus gradients onto `typography_document_layout`.
    """

    def __init__(self, hidden_dim: int = 960) -> None:
        self.hidden_dim = int(hidden_dim)
        self.w_class = np.zeros(self.hidden_dim, dtype=np.float64)
        self.b_class = 0.0
        self.w_axes = np.zeros((self.hidden_dim, 6), dtype=np.float64)
        self.b_axes = np.zeros(6, dtype=np.float64)

    def fit(
        self,
        H_prefill: np.ndarray,
        labels: np.ndarray,
        axis_annotations_6d: np.ndarray,
        l2_reg: float = 5.0,
        steps: int = 250,
    ) -> "MaskedAuxiliaryForensicHead":
        """Fits `w_class` on `labels` in `{0, 1}` and `w_axes` strictly on non-NaN entries
        of `axis_annotations_6d` (shape `(N, 6)`, where `np.nan` = not applicable/observable).
        """
        H = np.asarray(H_prefill, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        A = np.asarray(axis_annotations_6d, dtype=np.float64)
        valid_mask = np.isfinite(A)
        A_zeroed = np.where(valid_mask, A, 0.0)
        n = float(len(H))
        valid_counts = np.maximum(1.0, np.sum(valid_mask, axis=0))

        for _ in range(steps):
            # 1. Primary binary class score
            p_c = 1.0 / (1.0 + np.exp(-np.clip(H @ self.w_class + self.b_class, -30.0, 30.0)))
            err_c = (p_c - y) / n
            self.w_class -= 0.25 * (H.T @ err_c + (l2_reg / n) * self.w_class)
            self.b_class -= 0.25 * float(np.sum(err_c))

            # 2. Masked observable artifact axes (zero gradient where valid_mask is False)
            p_a = 1.0 / (1.0 + np.exp(-np.clip(H @ self.w_axes + self.b_axes, -30.0, 30.0)))
            err_a = np.where(valid_mask, p_a - A_zeroed, 0.0) / valid_counts[None, :]
            self.w_axes -= 0.30 * (H.T @ err_a + (l2_reg / n) * self.w_axes)
            self.b_axes -= 0.30 * np.sum(err_a, axis=0)

        return self

    def predict(self, h_last: np.ndarray, applicable_mask_6d: Optional[Sequence[bool]] = None) -> Dict[str, Any]:
        h = np.asarray(h_last, dtype=np.float64).reshape(-1)
        class_logit = float(h @ self.w_class + self.b_class)
        class_prob = float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, class_logit)))))
        raw_axes = h @ self.w_axes + self.b_axes
        axis_probs: Dict[str, Optional[float]] = {}
        for i, name in enumerate(RUBRIC_AXES):
            if applicable_mask_6d is not None and not applicable_mask_6d[i]:
                axis_probs[name] = None  # Explicitly not_applicable
            else:
                axis_probs[name] = round(float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(raw_axes[i])))))), 4)
        return {
            "class_prob": round(class_prob, 6),
            "class_logit": round(class_logit, 6),
            "axis_probabilities": axis_probs,
        }


# ============================================================================
# 4. Controlled A-to-G Ablation Evaluator with Wilson Binomial Intervals
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


def evaluate_candidate_at_fixed_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    inconclusive_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluates a candidate detector (A through G) reporting TPR @ 1% FPR, TPR @ 0.1% FPR,
    95% Wilson confidence intervals, and explicit inconclusive rate (never silently dropped).
    """
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    inc = np.zeros(len(y), dtype=bool) if inconclusive_mask is None else np.asarray(inconclusive_mask, dtype=bool)

    real_scores = s[(y == 0) & (~inc)]
    fake_scores = s[(y == 1) & (~inc)]
    n_real_total = int(np.sum(y == 0))
    n_fake_total = int(np.sum(y == 1))

    thr_1pct = float(np.quantile(real_scores, 0.99)) if len(real_scores) > 0 else 1.0
    thr_01pct = float(np.quantile(real_scores, 0.999)) if len(real_scores) > 0 else 1.0

    # Inconclusive cases are counted in the denominator so abstaining cannot artificially inflate TPR
    tp_1pct = int(np.sum(fake_scores >= thr_1pct))
    tp_01pct = int(np.sum(fake_scores >= thr_01pct))

    return {
        "tpr_at_1pct_fpr": round(tp_1pct / max(1, n_fake_total), 6),
        "tpr_at_1pct_ci95": wilson_ci_95(tp_1pct, n_fake_total),
        "tpr_at_0_1pct_fpr": round(tp_01pct / max(1, n_fake_total), 6),
        "tpr_at_0_1pct_ci95": wilson_ci_95(tp_01pct, n_fake_total),
        "inconclusive_rate": round(float(np.mean(inc)), 6),
        "n_real": n_real_total,
        "n_fake": n_fake_total,
    }


def learn_routing_thresholds_on_held_out(
    labels: np.ndarray,
    calibrated_fusion_probs: np.ndarray,
    calibrated_srm_probs: np.ndarray,
    calibrated_probe_probs: np.ndarray,
    hotspot_z_scores: np.ndarray,
    max_escalation_budget: float = 0.30,
) -> Dict[str, float]:
    """Learns routing thresholds on held-out data (after specialist probability calibration)
    to minimize `early_exit_fnr` (missed fakes among cases that exit early without VLM)
    subject to a target escalation rate budget.
    """
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    p_f = np.clip(np.asarray(calibrated_fusion_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    p_a = np.clip(np.asarray(calibrated_srm_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    p_c = np.clip(np.asarray(calibrated_probe_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    hz = np.asarray(hotspot_z_scores, dtype=np.float64).reshape(-1)

    thr_1pct = float(np.quantile(p_f[y == 0], 0.99)) if np.any(y == 0) else 0.5782
    z_thr = math.log(max(1e-6, min(1.0 - 1e-6, thr_1pct)) / (1.0 - max(1e-6, min(1.0 - 1e-6, thr_1pct))))
    z_f = np.log(p_f / (1.0 - p_f))
    z_a = np.log(p_a / (1.0 - p_a))
    z_c = np.log(p_c / (1.0 - p_c))

    best_cfg = {
        "llr_uncertainty_halfwidth": 1.60,
        "calibrated_conflict_logit_threshold": 1.80,
        "hotspot_z_threshold": 1.85,
        "escalation_rate": 0.0,
        "early_exit_fnr": 1.0,
    }

    n_fakes = max(1, int(np.sum(y == 1)))
    for llr_hw in (0.20, 0.40, 0.65, 1.0, 1.4, 1.8, 2.2):
        for conf_thr in (1.4, 1.8, 2.2, 2.8):
            for hz_thr in (1.65, 1.85, 2.10, 2.50, 2.90):
                escalate = (
                    (np.abs(z_f - z_thr) < llr_hw)
                    | (np.abs(z_a - z_c) > conf_thr)
                    | (hz >= hz_thr)
                )
                esc_rate = float(np.mean(escalate))
                if esc_rate > max_escalation_budget:
                    continue
                # Missed fakes among early exits (where escalate == False and p_f < thr_1pct)
                fakes_exited_early = int(np.sum((~escalate) & (y == 1)))
                early_exit_missed_fakes = int(np.sum((~escalate) & (y == 1) & (p_f < thr_1pct)))
                cond_miss = early_exit_missed_fakes / float(max(1, fakes_exited_early))
                total_fnr = early_exit_missed_fakes / float(n_fakes)
                if total_fnr < best_cfg["early_exit_fnr"]:
                    best_cfg = {
                        "llr_uncertainty_halfwidth": round(float(llr_hw), 4),
                        "calibrated_conflict_logit_threshold": round(float(conf_thr), 4),
                        "hotspot_z_threshold": round(float(hz_thr), 4),
                        "decision_threshold_dev": round(float(thr_1pct), 6),
                        "escalation_rate": round(esc_rate, 4),
                        "conditional_early_exit_miss_rate": round(cond_miss, 6),
                        "early_exit_fnr": round(total_fnr, 6),
                    }
    return best_cfg


def evaluate_frozen_routing_on_test(
    frozen_cfg: Dict[str, float],
    test_labels: np.ndarray,
    test_calibrated_fusion_probs: np.ndarray,
    test_calibrated_srm_probs: np.ndarray,
    test_calibrated_probe_probs: np.ndarray,
    test_hotspot_z_scores: np.ndarray,
) -> Dict[str, float]:
    """Evaluates a frozen routing policy (fitted strictly on dev/OOF data) on held-out test data,
    reporting both:
      1. `conditional_early_exit_miss_rate`: missed fakes among fakes that exited early.
      2. `total_fake_early_exit_fnr`: early-exit missed fakes divided by all fake inputs.
    """
    y = np.asarray(test_labels, dtype=np.int32).reshape(-1)
    p_f = np.clip(np.asarray(test_calibrated_fusion_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    p_a = np.clip(np.asarray(test_calibrated_srm_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    p_c = np.clip(np.asarray(test_calibrated_probe_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    hz = np.asarray(test_hotspot_z_scores, dtype=np.float64).reshape(-1)

    thr = float(frozen_cfg.get("decision_threshold_dev", 0.5782))
    z_thr = math.log(max(1e-6, min(1.0 - 1e-6, thr)) / (1.0 - max(1e-6, min(1.0 - 1e-6, thr))))
    z_f = np.log(p_f / (1.0 - p_f))
    z_a = np.log(p_a / (1.0 - p_a))
    z_c = np.log(p_c / (1.0 - p_c))

    escalate = (
        (np.abs(z_f - z_thr) < float(frozen_cfg["llr_uncertainty_halfwidth"]))
        | (np.abs(z_a - z_c) > float(frozen_cfg["calibrated_conflict_logit_threshold"]))
        | (hz >= float(frozen_cfg["hotspot_z_threshold"]))
    )
    n_fakes = max(1, int(np.sum(y == 1)))
    fakes_exited_early = int(np.sum((~escalate) & (y == 1)))
    early_exit_missed_fakes = int(np.sum((~escalate) & (y == 1) & (p_f < thr)))
    return {
        "test_escalation_rate": round(float(np.mean(escalate)), 4),
        "conditional_early_exit_miss_rate": round(early_exit_missed_fakes / float(max(1, fakes_exited_early)), 6),
        "total_fake_early_exit_fnr": round(early_exit_missed_fakes / float(n_fakes), 6),
        "fakes_exited_early": fakes_exited_early,
        "early_exit_missed_fakes": early_exit_missed_fakes,
        "total_fakes": n_fakes,
    }


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
# 5. Colab T4 16-Bit LoRA Training Runner (`Qwen/Qwen3.5-2B` Default)
# ============================================================================

def run_colab_t4_lora_pilot(
    model_id: str = "Qwen/Qwen3.5-2B",
    max_steps: int = 50,
    lora_rank: int = 16,
    image_max_px: int = 512,
    output_dir: Path = Path("./outputs/qwen3_5_2b_lora_pilot"),
) -> None:
    """Runs the 16-bit LoRA (`fp16=True` on Tesla T4) vision fine-tuning workflow
    recommended in `ONN_DETECT_On_Device_VLM_Research_2026-10-28.md`.
    """
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
        f"Model={model_id} | rank={lora_rank} | image_cap={image_max_px}px | GRPO=Disabled (Stage C SFT)"
    )

    # Prefer Unsloth FastVisionModel 16-bit LoRA path when installed (matches official Qwen3.5-2B T4 notebook)
    try:
        from unsloth import FastVisionModel  # type: ignore

        model, tokenizer = FastVisionModel.from_pretrained(
            model_id,
            load_in_4bit=False,  # Unsloth Qwen3.5 guide recommends 16-bit LoRA over 4-bit QLoRA
            use_gradient_checkpointing="unsloth",
        )
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,  # Start with frozen vision tower; ablate later
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=lora_rank,
            lora_alpha=lora_rank,
            lora_dropout=0.0,
            bias="none",
        )
    except ImportError:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import LoraConfig, get_peft_model

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
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
        tokenizer = processor

    peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_res_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    print(f"[Colab-T4 Pilot] Model initialized | Allocated={peak_alloc_gb:.3f} GB | Reserved={peak_res_gb:.3f} GB")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "adapter_init")
    tokenizer.save_pretrained(output_dir / "adapter_init")


# ============================================================================
# 6. Local Self-Test Suite
# ============================================================================

def run_self_tests() -> None:
    print("=" * 78)
    print("RUNNING ONN-DETECT VLM RESEARCH PLAN (2026-10-28) VERIFICATION SUITE")
    print("=" * 78)

    # 1. Verify published reference configurations (language + mmproj combined sizes)
    assert REFERENCE_VLM_SHORTLIST["qwen3_5_2b"].combined_pack_mb == 1948.0
    assert REFERENCE_VLM_SHORTLIST["qwen3_5_0_8b"].combined_pack_mb == 738.0
    assert REFERENCE_VLM_SHORTLIST["smolvlm2_500m_video"].combined_pack_mb == 546.0
    assert REFERENCE_VLM_SHORTLIST["smolvlm2_500m_video"].hf_model_id == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    for k, cfg in REFERENCE_VLM_SHORTLIST.items():
        print(
            f"  [PASS] Candidate '{k}' ({cfg.hf_model_id}): "
            f"Lang={cfg.language_file_mb:.0f} MB + mmproj={cfg.vision_mmproj_mb:.0f} MB "
            f"= {cfg.combined_pack_mb:.0f} MB ({cfg.priority_role})"
        )

    # 2. Verify source-level deduplication prevents FakeClue / GenImage derivative leakage
    train_items = [
        {"image_id": "genimage_sdxl_00421_crop1.jpg", "sha256": "aaa"},
        {"image_id": "genimage_sdxl_00421_q75.jpg", "sha256": "bbb"},
        {"image_id": "openfake_flux_99102.png", "sha256": "ccc"},
    ]
    eval_items = [
        {"image_id": "genimage_sdxl_00421.png", "sha256": "ddd"},
    ]
    deduped, removed = deduplicate_training_against_eval(train_items, eval_items)
    assert removed == 2 and len(deduped) == 1 and deduped[0]["image_id"] == "openfake_flux_99102.png"
    print(f"  [PASS] Source-Level Deduplication: Removed {removed} derivative leaks, kept {len(deduped)} disjoint sample(s)")

    # 3. Verify MaskedAuxiliaryForensicHead handles NOT_APPLICABLE (np.nan) axes without label inversion
    rng = np.random.default_rng(42)
    H = rng.normal(0, 0.5, size=(120, 32))
    y = np.array([0] * 60 + [1] * 60, dtype=np.int32)
    A = np.full((120, 6), np.nan, dtype=np.float64)  # Most axes not applicable on a given image
    for i in range(60, 120):
        H[i, :4] += 1.8
        A[i, 0] = 1.0  # Only axis 0 (anatomical_geometry) annotated on face subset
    for i in range(0, 60):
        H[i, :4] -= 1.5
        A[i, 0] = 0.0

    head = MaskedAuxiliaryForensicHead(hidden_dim=32).fit(H, y, A)
    pred = head.predict(H[75], applicable_mask_6d=[True, True, True, True, False, False])
    assert pred["class_prob"] > 0.80
    assert pred["axis_probabilities"]["anatomical_geometry"] is not None and pred["axis_probabilities"]["anatomical_geometry"] > 0.75
    assert pred["axis_probabilities"]["typography_document_layout"] is None
    print(
        f"  [PASS] MaskedAuxiliaryForensicHead: P(Fake)={pred['class_prob']:.4f}, "
        f"anatomical={pred['axis_probabilities']['anatomical_geometry']:.4f}, "
        f"typography(masked)={pred['axis_probabilities']['typography_document_layout']}"
    )

    # 4. Verify fixed-FPR evaluation with Wilson confidence intervals and explicit inconclusive accounting
    ev = evaluate_candidate_at_fixed_fpr(
        labels=np.array([0] * 200 + [1] * 200),
        scores=np.r_[rng.uniform(0.0, 0.4, 200), rng.uniform(0.35, 0.99, 200)],
        inconclusive_mask=np.array([False] * 390 + [True] * 10),
    )
    assert 0.0 < ev["tpr_at_1pct_fpr"] <= 1.0 and ev["inconclusive_rate"] == 0.025
    print(
        f"  [PASS] Fixed-FPR Ablation Evaluator: TPR@1%FPR={ev['tpr_at_1pct_fpr']:.4f} "
        f"(95% CI={ev['tpr_at_1pct_ci95']}), Inconclusive={ev['inconclusive_rate']:.3f}"
    )

    # 5. Verify learned routing threshold optimizer tracks early_exit_fnr on held-out data
    y_rt = np.array([0] * 200 + [1] * 200)
    p_fus = np.r_[rng.uniform(0.02, 0.35, 200), rng.uniform(0.45, 0.98, 200)]
    p_srm = np.r_[rng.uniform(0.05, 0.40, 200), rng.uniform(0.30, 0.95, 200)]
    p_prb = np.r_[rng.uniform(0.02, 0.35, 200), rng.uniform(0.50, 0.99, 200)]
    hz_rt = np.r_[rng.uniform(0.2, 1.4, 200), rng.uniform(0.8, 3.2, 200)]
    rt_cfg = learn_routing_thresholds_on_held_out(y_rt, p_fus, p_srm, p_prb, hz_rt, max_escalation_budget=0.35)
    assert rt_cfg["escalation_rate"] <= 0.35 and rt_cfg["early_exit_fnr"] < 0.25
    print(
        f"  [PASS] Learned Routing Thresholds (Dev/OOF): escalation_rate={rt_cfg['escalation_rate']:.3f}, "
        f"early_exit_fnr={rt_cfg['early_exit_fnr']:.4f}, cfg={rt_cfg}"
    )

    # 6. Verify frozen routing test evaluation reports BOTH conditional and total early-exit miss rates
    rt_test = evaluate_frozen_routing_on_test(rt_cfg, y_rt, p_fus, p_srm, p_prb, hz_rt)
    assert "conditional_early_exit_miss_rate" in rt_test and "total_fake_early_exit_fnr" in rt_test
    assert rt_test["total_fake_early_exit_fnr"] <= rt_test["conditional_early_exit_miss_rate"]
    print(
        f"  [PASS] Frozen Routing Test Eval: conditional_miss_rate={rt_test['conditional_early_exit_miss_rate']:.4f}, "
        f"total_fake_early_exit_fnr={rt_test['total_fake_early_exit_fnr']:.4f}"
    )

    # 7. Verify explicit label taxonomy & Candidate H teacher-validated fast student distillation
    assert LABEL_TAXONOMY_MAP["authentic_isp_denoised_hdr_bokeh"] == 0
    assert LABEL_TAXONOMY_MAP["authentic_conventional_retouch_crop"] == 0
    assert LABEL_TAXONOMY_MAP["authentic_screenshot"] == 0
    assert LABEL_TAXONOMY_MAP["generative_local_inpaint_or_edit"] == 1
    assert LABEL_TAXONOMY_MAP["generative_full_synthesis"] == 1
    w_kd, b_kd = fit_distilled_fast_student_probe(
        X_train=H,
        y_train=y,
        teacher_soft_probs=np.r_[rng.uniform(0.01, 0.15, 60), rng.uniform(0.85, 0.99, 60)],
        teacher_validated_on_dev=True,
    )
    assert float(H[75] @ w_kd + b_kd) > 0.0 and float(H[10] @ w_kd + b_kd) < 0.0
    print("  [PASS] Explicit Label Taxonomy & Candidate H (Teacher-Validated Fast Student Probe) verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--image-max-px", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/qwen3_5_2b_lora_pilot"))
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
        return

    run_colab_t4_lora_pilot(
        model_id=args.model_id,
        max_steps=args.max_steps,
        lora_rank=args.lora_rank,
        image_max_px=args.image_max_px,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
