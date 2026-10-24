# Benchmark Results — ONN-DETECT (Release v7 / Research Upgrade)

**Release:** `onn-20261021-v7`  
**Evaluation Date:** October 2026  
**Primary Evaluation Record:** [`docs/RESULTS_2026-10-20.md`](file:///docs/RESULTS_2026-10-20.md)  
**Corpora:**
- Visual: `openfake_visual` (3,600 records, 6 disjoint splits, 68 generative models including Midjourney v6, Flux, SDXL, Sora, Ideogram).
- Audio: `asvspoof_inthewild_audio` (2,400 records, 4 disjoint splits, ASVspoof 2019 LA + in-the-wild recordings).

---

## 1. Headline Empirical Metrics

| Component / Task | Model Architecture | Metric | Clean Test | Laundered (512px JPEG Q85) | Latency (ARM64 XNNPACK) |
|---|---|---|---|---|---|
| **Vision Foundation Probe** | INT8 SigLIP-B/16 (224px, 89.96 MB) + TAP | ROC-AUC | 0.8910 | 0.8972 | ~140–180 ms |
| **Residual Spatial Engine** | INT8 SRM+NPR Forensic CNN (128px) | ROC-AUC | 0.6871 | 0.6620 | ~15 ms |
| **Shipped Fusion v7** | **SigLIP-B/16 + SRM/NPR + XGBoost Base-Margin** | **ROC-AUC** | **0.8941** | **0.8999** | **<180 ms total** |
| **Low-FPR Recall (Images)** | Fusion v7 Base-Margin Cascade | **TPR @ 1% FPR** | **0.2800** (28.0%) | **0.2950** (29.5%) | — |
| **Ultra-Low FPR (Images)** | Fusion v7 Base-Margin Cascade | **TPR @ 0.1% FPR** | **0.2500** (25.0%) | **0.2600** (26.0%) | — |
| **Speech Anti-Spoofing** | INT8 LCNN-MFM (508 KiB) + VAD Gate | ROC-AUC | **0.8749** | 0.8620 (Opus VoIP) | **<25 ms (4s window)** |
| **Speech Low-FPR Recall** | INT8 LCNN-MFM (508 KiB) | **TPR @ 1% FPR** | **0.5950** (59.5%) | 0.5400 | — |
| **Provenance Verification** | C2PA Content Credentials + AudioSeal | Recall | **1.0000 (Exact)** | 0.9850 (AudioSeal) | <5 ms |

---

## 2. Why Base-Margin Cascade Matters

In `v6`, stacking a tree ensemble directly over the visual backbone created plateaus at the top of the score distribution, collapsing `TPR@1%FPR` to **0.040**.

In `v7`, formulating the continuous SigLIP-B/16 logit as XGBoost's `base_margin` ($\text{margin}(x) = \text{logit}(p_{\text{probe}}) + f_{\text{tree}}(x)$) enables the trees to act as residual correctors while preserving continuous tail ordering:
- **TPR @ 1% FPR:** 0.040 $\rightarrow$ **0.280** (**7× gain**)
- **TPR @ 0.1% FPR:** 0.005 $\rightarrow$ **0.250** (**50× gain**)

---

## 3. Social Media Laundering Hardening

Under harsh 512px JPEG Q85 compression laundering (simulating WhatsApp, Telegram, and social feed re-encoding), the `v7` cascade's ROC-AUC actually **improves from 0.8941 to 0.8999**.

This occurs because uniform re-compression strips away container and quantization discrepancies between web JPEGs (`laion`) and camera originals (`pexels`/`docci`), allowing the underlying semantic and high-pass feature representations to drive discrimination without dataset container artifacts.

---

## 4. Verification Test Suite

Every commit is gated by **417 automated tests**:
- **162 Python ML tests:** Model contracts, preprocessing parity, Keras channels-last mirror, DCPT patchcraft, B-Free paired twins, SigLIP-TAP pooling, Dual-SSL distillation, and MIL patch extraction.
- **31 Script parity tests:** Resampling precision, threshold calibration, and schema validation.
- **224 Kotlin Android unit tests:** Zero-network enforcement, Room migrations, XGBoost base_margin inference, StrongBox Keystore attestation, streaming audio buffers, Voiced Duration speech gate, and BioLip/AVFF lip-sync synchrony.
- **3 Lineage & APK artifact gates:** `onn_release_lineage_gate.py` (strict model SHA256 binding), merged permissions audit (0 internet permissions), and 16 KiB ELF alignment for 64-bit Android 15/16 compatibility.

For the exhaustive 952-line evaluation report, including per-generator breakdowns across 68 models and detailed latency benchmarks, refer to [`docs/RESULTS_2026-10-20.md`](file:///docs/RESULTS_2026-10-20.md).
