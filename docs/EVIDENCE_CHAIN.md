# ONN-DETECT: Verifiable Research Evidence Chain & Traceability Matrix

**Document Version:** 1.0-AUDIT-2026  
**Target Repository:** `shubhamiitshk/Onn-Detect-`  
**Evaluation Standard:** Independent Reproducibility & Research Source Traceability  
**Release Manifest Invariant:** `validated: false` until physical ARM64 device execution  

---

## 1. Executive Traceability Matrix

Every forensic capability claimed in ONN-DETECT is mapped below to its primary literature citation, exact mathematical adaptation status, corresponding training/export pipeline, artifact hash, and physical verification status.

| Research Paper | Component / Modality | Mathematical Nature of Integration | Training / Export Script | Shipped Model / Asset SHA-256 | Verification Status |
|---|---|---|---|---|---|
| **SigLIP** (Zhai et al., Google 2023) | Vision Foundation Probe (Track 2) | **Frozen ViT Backbone + Folded Linear Probe**: Uses `google/siglip-base-patch16-224` with frozen MAP attention query folding in Keras mirror. | `ml/src/export/keras_vfm_probe.py` | `siglip_probe_int8.tflite`<br>`1b8c4f3de7822196...` (89.96 MB) | **Desktop Verified**; On-Device Physical Evaluation Pending |
| **TAP** (CVPRW 2026) | Vision Probe Attention | **Experimental Staging**: Tunable Attention Pooling over 196 patch tokens. Implemented in training scripts; linear probe remains shipped baseline. | `ml/scripts/train_siglip_tap.py` | Staged in `ml/runs/` (Unbundled from prod manifest) | Experimental Prototype |
| **PatchCraft** (Zhong et al., 2023) | Texture Preprocessing | **Adaptation**: Native $4 \times 4$ texture mosaic patchification (`-patchcraft-mosaic-tile32`) preserving 128x128 high-frequency grids. | `ml/src/data/patch_extractor.py`<br>`train_srm_dcpt_patchcraft.py` | Integrated in `srm_forensic_cnn_int8.tflite` input contract | **Desktop Verified** |
| **SRM** (Fridrich & Kodovsky, 2012) | High-Pass Residual Stem | **Direct Tensor Implementation**: 12 fixed 5x5 steganalysis filters embedded in input Conv2D stem. Verified via `verify_srm_stem_weights.py`. | `ml/src/models/srm_forensic_cnn.py` | `srm_forensic_cnn_int8.tflite`<br>`08397e56b53e506e...` (142 KiB) | **Verified** against frozen `.tflite` buffer ($e < 0.002$) |
| **NPR** (Tan et al., CVPR 2024) | Upsampling Residual Stem | **Direct Tensor Implementation**: 4 fixed 5x5 neighboring pixel relationship filters in stem channels 12–15. | `ml/src/models/srm_forensic_cnn.py` | `srm_forensic_cnn_int8.tflite` (Channels 12–15) | **Exact Match** ($e = 0.000000$) |
| **DCPT** (Yang et al., 2026) | Degradation Consistency | **Training Loss Recipe**: Clean/degraded paired consistency loss across JPEG/resizing. Evaluated in training suite. | `ml/tests/test_dcpt_losses.py`<br>`train_srm_dcpt_patchcraft.py` | Incorporated in SRM model weights | **Desktop Verified** |
| **Effort** (Yan et al., ICML 2025) | Domain Subspace Adaptation | **Custom Lightweight Adaptation**: Source-centroid nuisance projection across real-class subsets (LAION vs. Pexels/DocCI). Not the full SVD orthogonal subspace adaptation of the original paper. | `ml/src/models/prism_v11_algorithms.py` (experimental) | Code adaptation in `ml/src/models/prism_v11_algorithms.py` (experimental) | Custom Adaptation |
| **LCNN-MFM** (Lavrentyeva et al., 2019) | Speech Anti-Spoofing | **Quantized INT8 Architecture**: 508 KiB Light CNN with Max-Feature-Map activation over 40-D LFCC spectrograms. | `ml/scripts/train_lcnn_audio.py` | `lcnn_audio_int8.tflite`<br>`6963363f0e51f633...` (508 KiB) | **Desktop Verified**; On-Device Physical Evaluation Pending |
| **AudioSeal** (San Roman et al., Meta 2024) | Proactive Watermarking | **Sample-Level Neural Watermarking**: 1D SEANet detector emitting temporal presence score in $[0, 1]$ (`audioseal-presence-mean-v2`) with Bonferroni multi-window adjustment. | `ml/scripts/export_audioseal_detector.py`<br>`AudioSealDetector.kt` | Unbundled from core APK manifest (Reference / Optional) | Standalone Parity Verified; Decoupled from synthetic verdict |
| **WASP** (Müller & Debus, 2026) | Watermark Shortcut Mitigation | **Runtime Invariant Policy**: Watermark absence is strictly neutral (does not lower authenticity score). Balanced training decorrelation remains future work. | `docs/STATE_OF_THE_ART_REVIEW_2026.md`<br>`AudioClassifier.kt` | Architectural Policy Invariant | Policy Invariant |
| **MN-P** (Xu et al., 2026) | Non-Vocal / Silence Residuals | **Custom Lightweight Adaptation**: 16-D spectral variance vector extracted from unvoiced frames. Distinct from XLS-R token-level cross-attention gating in the paper. | `ml/src/models/prism_v11_algorithms.py` (experimental) | Experimental code adaptation | Custom Adaptation |
| **LADDER** (Chen & Xu, 2026) | Lip-Sync Dynamics | **Telemetry Heuristic**: 2nd-order discrete Euclidean difference on MediaPipe 64 perioral landmarks. Decoupled from production verdict (telemetry only). | `PerioralLandmarkAnalyzer.kt`<br>`VideoForensicAnalyzer.kt` | Diagnostic Telemetry (`landmarkJitter`) | Diagnostic Only (Decoupled) |
| **CHROM rPPG** (de Haan & Jeanne, 2013) | Facial Physiological Blood Pulse | **Telemetry Heuristic**: Bilateral cheek green chromaticity difference across contiguous frame runs. Decoupled from production verdict (telemetry only). | `BioLipLandmarkAnalyzer.kt`<br>`VideoForensicAnalyzer.kt` | Diagnostic Telemetry (`rppgPhaseAnomaly`) | Diagnostic Only (Decoupled) |
| **C2PA Content Credentials** | Cryptographic Provenance | **Deterministic Verification**: JUMBF box parser, APP11 reassembly, COSE_Sign1 ES256 signature and exclusion-range check against production trust list. | `android/app/src/main/java/com/onndetect/detection/provenance/` | `assets/trust/c2pa_trust_anchors.pem` | **Verified Offline** (< 5 ms deterministic) |
| **Mondrian Conformal Prediction** (Angelopoulos & Bates, 2021) | Uncertainty Quantification | **Statistical Calibration**: Calibration sets bound false alarms under exchangeability assumption. Does not guarantee 0.1% FPR under unseen generator distribution shifts. | `ml/src/training/evaluation.py` | Calibration records in `release_manifest.json` | Statistically Conditional |
| **EmbeddingGemma 2** (Google / Unsloth, 2026) | Frozen Multimodal Embedder | **Empirically Measured (Sealed Test 0.9006 AUC vs. SigLIP Linear 0.8919 / SigLIP-TAP 0.9921)**: Linear probe reaches 0.9006 AUC (+0.0087 delta), but TPR@1%FPR collapses to 0.2100 (vs. SigLIP-TAP 0.7243) due to global pooling bottleneck. Android APK inclusion rejected per Decision Gate. | `ml/scripts/evaluate_embeddinggemma2_probe.py`<br>`ml/notebooks/evaluate_embeddinggemma2_probe_colab.ipynb` | `embeddinggemma2_forensic_head.json` (diagnostic only) | Empirically Measured (Rejected for APK) |
| **Qwen 3.5 2B** (Alibaba, 2026) | Dense Vision-Language Backbone | **Empirically Measured (Sealed Test 0.9295 AUC vs. Gemma 2 0.9006 & SigLIP-TAP 0.9921)**: Linear probe reaches 0.9295 AUC (+0.0289 over Gemma 2) and 0.4100 TPR@1%FPR (+95% relative increase over Gemma 2 0.2100). Exceeds Gemma 2 in visual capacity, but remains below specialized SigLIP probe (0.7243). Decision Gate: Rejected for Slot 6 APK inclusion (1.95 GB Q4 footprint violates <150MB APK limit); approved as primary Tier-2 Deep Scan candidate (`qwen3.5-2b.zip` via `VlmPackManager.kt`). | `ml/scripts/evaluate_qwen_probe.py`<br>`docs/test_artifacts/qwen35_2b_eval/` | `qwen_qwen3.5_2b_forensic_head.json` | Empirically Measured (Tier-2 Deep Scan Candidate) |

---

## 2. Model Hash Verification & Architecture Specifications

### 2.1 Vision Foundation Probe (`siglip_probe_int8.tflite`)
* **SHA-256:** `1b8c4f3de7822196c3c0ebf59f683d86f24d5fa2476fb1d9312a1e14b1519c7b`
* **Size:** 89.96 MB (94,334,144 bytes)
* **Input Tensor Contract:** `[1, 224, 224, 3]` float32 in $[0, 1]$
* **Output Tensor Contract:** `[1, 1]` float32 in $[0, 1]$
* **True Architecture Status:** Frozen SigLIP-B/16 tower with linear probe folded directly into the pre-trained `_SiglipMAPAttention` head (`vm.head`). Experimental TAP and PatchHead heads exist in training scripts (`train_siglip_tap.py`, `ml/src/models/prism_v11_algorithms.py` (experimental)) and are tracked separately.

### 2.2 Spatial-Frequency Residual CNN (`srm_forensic_cnn_int8.tflite`)
* **SHA-256:** `08397e56b53e506ef89b290f114f167babb2ca530a52320ec770b519bf307ca1`
* **Size:** 142 KiB (145,568 bytes)
* **Input Tensor Contract:** `[1, 128, 128, 3]` float32
* **Output Tensor Contract:** `[1, 1]` float32
* **Stem Verification:** Fixed 20-channel $5 \times 5$ convolutional stem verified with max absolute quantization error $0.001969 < 0.005$ against canonical SRM and NPR mathematical definitions (`scripts/verify_srm_stem_weights.py`).

### 2.3 Audio Anti-Spoofing (`lcnn_audio_int8.tflite`)
* **SHA-256:** `6963363f0e51f6338ceac34575e4c9fdeeb8ecbebe33605695721232377af508`
* **Size:** 508 KiB (520,384 bytes)
* **Input Tensor Contract:** `[1, 401, 40, 1]` float32 (LFCC v3 frontend)
* **Output Tensor Contract:** `[1, 1]` float32
* **Operating Point:** Calibrated threshold $0.35547$ with voiced duration gating $\ge 300\text{ ms}$.

---

## 3. Physical Device Validation Gate Requirements

To promote `"validated": false` to `"validated": true` in `release_manifest.json`, the following criteria must be satisfied on physical hardware:
1. **Target Hardware:** Snapdragon 8 Gen 2 / 8 Gen 3 (e.g. Galaxy S23/S24, OnePlus 11/12) and Google Tensor G3 / G4 (Pixel 8/9).
2. **Cold Start & Warm Latency:** Automated 50-scan profile measuring p50, p90, and p99 inference latency under OS battery/thermal management.
3. **Memory Profile:** PSS and RSS instrumentation confirming ART heap bloat remains $< 32\text{ MB}$ and mmap page-cache eviction operates cleanly under memory pressure.
4. **Thermal Stability:** Verification that sustained repeated inference does not cause CPU throttling exceeding 25% throughput degradation.
5. **Signed Log Invariant:** Device benchmark reports must be committed to `device_evaluation.json` with hardware attestation signatures.
