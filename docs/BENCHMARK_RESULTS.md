# Benchmark Results — ONN-DETECT (Release `onn-prod-v10` / `v10.1`)

**Release:** `onn-prod-v10`  
**Evaluation Date:** October 2026  
**Primary Evaluation Record:** [`android/app/src/main/assets/evaluation_report.json`](file:///android/app/src/main/assets/evaluation_report.json)  
**Corpora:**
- Visual: Full-resolution multi-generator dataset (20,000 samples across `train`, `val`, `stacking`, `fusion_val`, `calibration`, `test` spanning Midjourney, Stable Diffusion v1.4/v1.5, ADM, GLIDE, Wukong, VQDM, BigGAN + authentic camera photographs, strictly filtering out $32 \times 32$ CIFAKE thumbnails).
- Audio: `asvspoof_inthewild_audio` (2,400 records, 4 disjoint splits, ASVspoof 2019 LA + in-the-wild recordings).

---

## 1. Headline Empirical Metrics (Sealed Desktop Test Split)

| Component / Task | Model Architecture | Metric | Clean Test | Laundered (Social-Media Recompressed) | Latency (ARM64 XNNPACK) |
|---|---|---|---|---|---|
| **Vision Foundation Probe** | INT8 SigLIP-B/16 (224px, 89.96 MB) + TAP | ROC-AUC | **0.9921** | — | ~130–160 ms |
| **Vision Probe Low-FPR Recall** | INT8 SigLIP-B/16 + TAP | **TPR @ 1% FPR** | **0.7243** (72.43%) | — | — |
| **Residual Spatial Engine** | INT8 SRM+NPR+RGB + $4\times 4$ PatchCraft Mosaic + DCPT (128px) | ROC-AUC | **0.9655** | **0.9391** (WhatsApp Q85) / **0.8949** (Mean) | ~15 ms |
| **Residual Spatial Low-FPR Recall** | INT8 SRM+NPR+RGB + PatchCraft Mosaic | **TPR @ 1% FPR** | **0.5314** (53.14%) | — | — |
| **Shipped Fusion v10** | **SigLIP-TAP + PatchCraft SRM + 7-Tree XGBoost Base-Margin** | **ROC-AUC** | **0.9974** | **0.9972** | **<180 ms total** |
| **Low-FPR Recall (Images, v10)** | **Fusion v10 Base-Margin Cascade (`thr=0.5782`)** | **TPR @ 1% FPR** | **0.9400** (94.0%) | **0.9000** (90.0%) | — |
| **Ultra-Low-FPR Recall (Images, v10)** | **Fusion v10 Base-Margin Cascade** | **TPR @ 0.1% FPR** | **0.9150** (91.5%) | **0.8250** (82.5%) | — |
| **Speech Anti-Spoofing** | INT8 LCNN-MFM (508 KiB, `thr=0.35547`) | ROC-AUC | **0.8747** | 0.8620 (Opus VoIP) | **<25 ms (4s window)** |
| **Speech Low-FPR Recall** | INT8 LCNN-MFM (508 KiB) | **TPR @ 1% FPR** | **0.6500** (65.0%) | 0.5400 | — |
| **Provenance Verification** | C2PA Content Credentials (`C2PAVerifier.kt`) | Recall | **1.0000 (Exact)** | — | <5 ms |
| **Neural Watermark (AudioSeal)** | Export pipeline (TFLite weights unreleased) | Recall | — | — | — |

---

## 2. Evidence Classification & Validation Status

Per repository governance and release integrity audit standards, results are strictly partitioned into measured desktop evidence vs provisional research indicators:

| Claim / Metric | Evidence Status | Measurement Source | Operational Decision |
|---|---|---|---|
| **Visual Fusion (`onn-prod-v10`, 7 trees, base-margin)** | **Desktop Verified** | `evaluation_report.json`, `release_manifest.json` | Active in production runtime (`XGBoostMetaClassifier.kt`) |
| **Audio Anti-Spoofing (LCNN-MFM)** | **Desktop Verified** | `lcnn_audio_threshold.json` (`thr=0.35546875`) | Active in production runtime (`AudioClassifier.kt`) |
| **Offline C2PA Verification** | **Desktop & Unit Verified** | `c2pa_trust_anchors.pem`, C2PA replay suite | Fail-closed cryptographic evaluation |
| **Video Asymmetric Multimodal Escalation** | **Provisional Indicator** | `VideoScorePolicy.combineMultimodalScore` | Verdict marked provisional until independent video corpus certification |
| **Localized ROI & Rich-vs-Poor Contrast** | **XAI Diagnostic Telemetry** | `ResidualHeatmapExtractor.kt`, `SRMForensicAnalyzer.kt` | Exposed in `signals.perturbationScore` while preserving unmutated slot-0 calibration parity |
| **AudioSeal Watermark Detector** | **Unreleased / Awaiting Conversion** | PyTorch/ONNX pipeline | Watermark absence treated as unknown, not authentic |

---

## 3. Why `onn-prod-v10` Overcame the `v9` Ceiling

1. **Eliminating the $32 \times 32$ CIFAKE Domain Shift:** In `v9`, Track 1 (`SRMForensicCNN`) was trained on $32 \times 32$ thumbnails upscaled to $128 \times 128$, which destroyed high-frequency camera PRNU and diffusion upsampler residuals (`0.6871` AUC). Retraining on full-resolution multi-generator images with native $4 \times 4$ PatchCraft high-frequency texture mosaics raised standalone Track 1 Test AUC from `0.6871` $\rightarrow$ **`0.9655`**.
2. **Tunable Attention Pooling (TAP) on SigLIP-B/16:** Training the TAP head across 196 spatial patch tokens raised standalone Track 2 Test AUC from `0.8910` $\rightarrow$ **`0.9921`**.
3. **7-Tree Base-Margin XGBoost Fusion:** Formulating the continuous SigLIP-TAP logit as XGBoost's `base_margin` ($\text{margin}(x) = \text{logit}(p_{\text{probe}}) + f_{\text{tree}}(x)$) alongside the newly active `srm_score` splits (`0.168..0.824`) raised Clean Test AUC to **`0.9974`** (`TPR @ 1% FPR = 94.0%`) and Laundered Test AUC to **`0.9972`** (`TPR @ 1% FPR = 90.0%`).

---

## 4. Verification Test Suite

Every commit is gated by automated suites:
- **Python ML tests (162 tests):** Model contracts, preprocessing parity, Keras channels-last mirror, DCPT patchcraft, B-Free paired twins, SigLIP-TAP pooling, and Dual-SSL distillation.
- **Script parity tests:** Resampling precision, threshold calibration, and schema validation.
- **Kotlin Android unit tests (235 tests):** Zero-network enforcement, Room migrations, XGBoost base_margin inference, StrongBox Keystore attestation, streaming audio buffers, Voiced Duration speech gate, BioLip/AVFF lip-sync synchrony, `video/*` Share-to-Verify, and `VideoScorePolicy` multimodal escalation.
- **Lineage & APK artifact gates:** `onn_release_lineage_gate.py` (strict model SHA256 binding, header checks, parity bounds), merged permissions audit (0 internet permissions), and 16 KiB ELF alignment for 64-bit Android 15/16 compatibility.
