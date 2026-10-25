# Benchmark Results — ONN-DETECT (Release v9)

**Release:** `onn-prod-v9`  
**Evaluation Date:** October 2026  
**Primary Evaluation Record:** [`docs/RESULTS_2026-10-20.md`](file:///docs/RESULTS_2026-10-20.md)  
**Corpora:**
- Visual: `openfake_visual` (3,600 records, 6 disjoint splits, 68 generative models including Midjourney v6, Flux, SDXL, Sora, Ideogram).
- Audio: `asvspoof_inthewild_audio` (2,400 records, 4 disjoint splits, ASVspoof 2019 LA + in-the-wild recordings).

---

## 1. Headline Empirical Metrics (Desktop Test Split)

| Component / Task | Model Architecture | Metric | Clean Test | Laundered (512px JPEG Q85) | Latency (ARM64 XNNPACK) |
|---|---|---|---|---|---|
| **Vision Foundation Probe** | INT8 SigLIP-B/16 (224px, 89.96 MB) + TAP | ROC-AUC | 0.8910 | 0.8972 | ~140–180 ms |
| **Residual Spatial Engine** | INT8 SRM+NPR Forensic CNN (128px) | ROC-AUC | 0.6871 | 0.6620 | ~15 ms |
| **Shipped Fusion v9** | **SigLIP-B/16 + SRM/NPR + 16-Tree XGBoost Base-Margin** | **ROC-AUC** | **0.8941** | **0.8999** | **<180 ms total** |
| **Low-FPR Recall (Images)** | Fusion v9 Base-Margin Cascade | **TPR @ 1% FPR** | **0.2800** (28.0%) | **0.2950** (29.5%) | — |
| **Speech Anti-Spoofing** | INT8 LCNN-MFM (508 KiB, thr=0.35547) | ROC-AUC | **0.8749** | 0.8620 (Opus VoIP) | **<25 ms (4s window)** |
| **Speech Low-FPR Recall** | INT8 LCNN-MFM (508 KiB) | **TPR @ 1% FPR** | **0.5950** (59.5%) | 0.5400 | — |
| **Provenance Verification** | C2PA Content Credentials | Recall | **1.0000 (Exact)** | — | <5 ms |
| **Neural Watermark (AudioSeal)** | Export pipeline (TFLite weights unreleased) | Recall | — | — | — |

---

## 2. Evidence Classification & Validation Status

Per repository governance and release integrity audit standards, results are strictly partitioned into measured desktop evidence vs provisional research indicators:

| Claim / Metric | Evidence Status | Measurement Source | Operational Decision |
|---|---|---|---|
| **Visual Fusion (16 trees, base-margin)** | **Desktop Verified** | `meta_results.json`, `release_manifest.json` | Active in production runtime (`XGBoostMetaClassifier.kt`) |
| **Audio Anti-Spoofing (LCNN-MFM)** | **Desktop Verified** | `lcnn_audio_threshold.json` (thr=0.35546875) | Active in production runtime (`AudioClassifier.kt`) |
| **Offline C2PA Verification** | **Desktop & Unit Verified** | `c2pa_trust_anchors.pem`, C2PA replay suite | Fail-closed cryptographic evaluation |
| **Video Temporal Blend (0.65/0.20/0.15)** | **Provisional Indicator** | Multi-frame heuristic blend | Verdict marked provisional until independent video corpus certification |
| **Ultra-Low 0.1% FPR Image Recall** | **Research Aspiration** | Finite held-out sample quantiles | Unresolved promise on limited real-sample sizes |
| **AudioSeal Watermark Detector** | **Unreleased / Awaiting Conversion** | PyTorch/ONNX pipeline | Watermark absence treated as unknown, not authentic |

---

## 3. Why Base-Margin Cascade Matters

In `v6`, stacking a tree ensemble directly over the visual backbone created plateaus at the top of the score distribution, collapsing `TPR@1%FPR` to **0.040**.

In `v9`, formulating the continuous SigLIP-B/16 logit as XGBoost's `base_margin` ($\text{margin}(x) = \text{logit}(p_{\text{probe}}) + f_{\text{tree}}(x)$) over 16 shallow trees enables the booster to act as a residual corrector while preserving continuous tail ordering:
- **TPR @ 1% FPR:** 0.040 $\rightarrow$ **0.280** (**7× gain**)
- **Tail Discretization:** Zero ties; monotonically ranked continuous margins.

---

## 4. Verification Test Suite

Every commit is gated by automated suites:
- **Python ML tests:** Model contracts, preprocessing parity, Keras channels-last mirror, DCPT patchcraft, B-Free paired twins, SigLIP-TAP pooling, and Dual-SSL distillation.
- **Script parity tests:** Resampling precision, threshold calibration, and schema validation.
- **Kotlin Android unit tests:** Zero-network enforcement, Room migrations, XGBoost base_margin inference, StrongBox Keystore attestation, streaming audio buffers, Voiced Duration speech gate, and BioLip/AVFF lip-sync synchrony.
- **Lineage & APK artifact gates:** `onn_release_lineage_gate.py` (strict model SHA256 binding, header checks, parity bounds), merged permissions audit (0 internet permissions), and 16 KiB ELF alignment for 64-bit Android 15/16 compatibility.
