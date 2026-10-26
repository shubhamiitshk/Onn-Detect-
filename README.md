# ONN-DETECT: Zero-Cloud On-Device AI Media Forensics

**ONN-DETECT** is an on-device, zero-cloud AI media forensics engine and Android SDK designed to detect AI-generated images, synthetic voice clones, and deepfake videos directly on smartphones in **under 180 ms** (sub-25 ms for streaming audio).

All media decoding, neural inference, cryptographic verification, and tamper analysis execute strictly within an OS-level **zero-network sandbox** (`android.permission.INTERNET` is completely excluded from the application manifest).

---

## Key Capabilities & Features

### 1. Multi-Modal On-Device Cascade (Release `onn-prod-v10` / `v10.1`)
- **Vision Foundation Probe (`siglip_probe_int8.tflite`):** INT8 SigLIP-B/16 + Tunable Attention Pooling (TAP) over 196 tokens (224px, 89.96 MB) memory-mapped directly into the OS page cache with **0 MB Java heap bloat**, achieving **0.9921 ROC-AUC** (`72.43%` TPR @ 1% FPR) on full-resolution multi-generator test splits.
- **Spatial-Frequency Residual Engine (`srm_forensic_cnn_int8.tflite`):** 20-channel spatial-frequency CNN (12 fixed 5×5 SRM high-pass filters + 4 NPR upsampling kernels + 4 centered RGB low-pass stem, verified from the shipped tensor `(20,5,5,3)`) over $4 \times 4$ PatchCraft high-frequency texture mosaics (`-patchcraft-mosaic-tile32`) + DCPT frequency pooling, achieving **0.9655 ROC-AUC** standalone (`0.8949` mean AUC under social-media WhatsApp/Instagram recompression). Includes Rich-vs-Poor texture contrast (`analyzeWithContrast`) and $8 \times 8$ sliding-window hotspot localization (`extractHotspotBoundingBox`) as XAI diagnostics while preserving exact slot-0 calibration parity.
- **Base-Margin XGBoost Meta-Fusion (`XGBoostMetaClassifier.kt`):** 7-tree multimodal booster combining 10 forensic signals with SigLIP continuous logit base-margin prior (`threshold = 0.57819986`), achieving **0.9974 Clean Test ROC-AUC** (**94.0% TPR @ 1% FPR**) and **0.9972 Laundered Test ROC-AUC** (**90.0% TPR @ 1% FPR**).
- **Speech Anti-Spoofing (`lcnn_audio_int8.tflite`):** Quantized INT8 LCNN-MFM (508 KiB) scoring 4-second LFCC spectrograms in **<25 ms** with calibrated threshold **0.35547** (`lcnn_audio_threshold.json`), **0.8747 ROC-AUC**, and **65.0% TPR @ 1% FPR**.
- **Cryptographic & Neural Provenance (`C2PAVerifier.kt`):** Offline C2PA Content Credentials verification against the official production trust list (Adobe, Google, DigiCert, ... anchors in `assets/trust/c2pa_trust_anchors.pem`): full JUMBF walk, multi-segment APP11 reassembly (ISO 19566-5), COSE_Sign1 ES256 and c2pa.hash.data exclusion-range binding, with a fail-closed `signed = hashVerified && coseSignatureValid && certChainValid` conjunction. The Meta AudioSeal watermark detector ships its verified ONNX export pipeline (`ml/scripts/export_audioseal_detector.py`); the app-side TFLite weights are pending a numerically faithful conversion — watermark absence represents unknown provenance rather than authenticity.

### 2. Consumer Security Suite
- **Family Safe-Voice Vault (dual-gate):** Offline enrollment requires **≥ 8 s of voiced speech** (20 ms frame voicing gate) AND an LCNN-MFM anti-spoof verdict below the 0.35547 operating point — a synthetic sample can never become a trusted reference. Embeddings are pooled over 3 s windows at a 1.5 s hop and sealed with **AES-256-GCM in the Android Keystore** (alias `onn_voice_vault`), with the GCM AAD bound to each row's identity; a manifest swap between rows fails authentication loudly.
- **WhatsApp / Telegram / Instagram 1-Tap Forward Guard (`image/*`, `audio/*`, `video/*`):** Share Sheet integration (`ACTION_SEND`) to inspect forwarded photos, voice notes, viral videos/Reels, and payment receipts in 1 tap without leaving chat apps. Interactive **Tampering X-Ray Reticle**, Thermal Map, and Spike Isolate overlays (`HeatmapViewer.kt`) pinpoint localized high-pass residual spikes and document forgery bounding boxes.
- **BioLip Perioral Forensics & Asymmetric Multimodal Video Analysis:** MediaPipe 478-point face mesh (64-point perioral set, inter-ocular pose normalization around the nose tip) measuring second-order acceleration jitter (0.6) and MAR flicker (0.4) in talking-head video, plus experimental bilateral malar rPPG chromaticity correlation (`rppgPhaseAnomaly`). In video mode, `VideoScorePolicy.combineMultimodalScore` applies threshold-aligned audio escalation and one-sided temporal/BioLip anomaly boosting so smooth motion or authentic background music never penalizes a synthetic video score.

*(The earlier planned UPI live-call shield and AI-reel accessibility automation were removed for Google Play 2026 policy compliance — the shipped suite above is 100% Play-compliant and network-free: the manifest actively removes `INTERNET` and `ACCESS_NETWORK_STATE`.)*

### 3. Enterprise B2B SDK
- **Video-KYC Injection Shield (`VirtualCameraDetector.kt`):** Defeats virtual camera injection (OBS/ManyCam/VCam), dynamic hooking frameworks (Xposed/LSPosed/Frida/Pine), and face-swap deepfakes for RBI V-CIP and SEBI digital onboarding compliance.
- **Hardware-Signed Forensic Receipts & Shareable Visual Certificates (`StrongBoxReceiptSigner.kt` & `ForensicCertificateExporter.kt`):** Cryptographically binds scan verdicts, media SHA-256, and model release hashes inside the physical **Android StrongBox / TEE Keystore**, exporting both canonical JSON proofs (verifiable server-side via `scripts/verify_forensic_receipt.py`) and $1080 \times 1440$ high-resolution PNG Forensic Verification Certificates.
- **Payment & Document Forgery Detector (`DocumentForgeryAnalyzer.kt`):** High-pass residual analysis detects spliced numbers on UPI payment confirmation screenshots and forged identity cards.

---

## Empirical Benchmark Summary

Evaluated on sealed high-resolution multi-generator benchmark splits (Midjourney, Stable Diffusion v1.4/v1.5, ADM, GLIDE, Wukong, VQDM, BigGAN + authentic camera photographs) and in-the-wild audio datasets (release `onn-prod-v10`):

| Benchmark / Task | Metric | Clean Test | Laundered (Social-Media Recompressed) | Latency (ARM64 XNNPACK) |
|---|---|---|---|---|
| **Image Forensics (`onn-prod-v10` Fusion)** | ROC-AUC | **0.9974** | **0.9972** | **~140–180 ms** |
| **Low-FPR Recall (Images, Fusion v10)** | TPR @ 1% FPR | **0.9400 (94.0%)** | **0.9000 (90.0%)** | — |
| **SigLIP-B/16 + TAP Probe Standalone** | ROC-AUC | **0.9921** | — | ~130 ms |
| **SRM + PatchCraft Mosaic CNN Standalone** | ROC-AUC | **0.9655** | **0.9391** (WhatsApp Q85) / **0.8949** (Mean) | ~15 ms |
| **Speech Anti-Spoofing (LCNN-MFM)** | ROC-AUC | **0.8747** | 0.8620 (Opus VoIP) | **<25 ms** |
| **Speech Low-FPR Recall** | TPR @ 1% FPR | **0.6500 (65.0%)** | 0.5400 | — |
| **Provenance Verification (C2PA)** | fail-closed conjunction | chain + COSE_Sign1 + data-hash all required for `signed=true` | official production trust list (32 anchors) | <5 ms |
| **Neural Watermark (AudioSeal)** | export pipeline | ONNX parity **0.000e+00** vs PyTorch | app-side TFLite weights pending faithful conversion | — |

*See [`docs/BENCHMARK_RESULTS.md`](docs/BENCHMARK_RESULTS.md) and [`docs/RESULTS_2026-10-20.md`](docs/RESULTS_2026-10-20.md) for full empirical breakdowns.*

---

## Build, Test & Verification

### Prerequisites
- JDK 17
- Android SDK 36 (Build Tools 35.0.0)
- Python 3.11 with virtual environment (`.venv`)

### 1. Android APK & Unit Tests
```powershell
cd android
.\gradlew.bat testDebugUnitTest assembleRelease
cd ..
```
*All 235 Kotlin unit tests pass; the release APK is generated at `android/app/build/outputs/apk/release/app-release.apk` (signed, zero internet permissions).*

### 2. Consistency & Integrity Gating
```powershell
python onn_release_lineage_gate.py --repo .
python scripts/check_android_consistency.py
python -m unittest discover -s scripts/tests -v
python scripts/verify_android_artifact.py --apk android/app/build/outputs/apk/release/app-release.apk
```
*Verifies 16 KiB ELF alignment for Android 15, asset hashes, and manifest zero-network isolation.*

### 3. ML Pipeline & Parity Tests
```powershell
.\.venv\Scripts\python.exe -m pytest ml/tests
```
*All 162 Python ML tests pass (PyTorch vs. LiteRT parity, Keras channels-last mirror, Multi-Instance Learning, B-Free Twins, SigLIP-TAP, and Dual-SSL distillation).*

---

## Pitch Deck & Program Documentation

- **Pitch Deck (PowerPoint):** [`ONN_DETECT_PITCH_DECK_Z21_KGP_FORGE.pptx`](ONN_DETECT_PITCH_DECK_Z21_KGP_FORGE.pptx)
- **Pitch Deck & Q&A Guide (Markdown):** [`docs/PITCH_DECK_Z21_KGP_FORGE.md`](docs/PITCH_DECK_Z21_KGP_FORGE.md)
- **Technical Architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Dual-Engine Cascade Specifications:** [`docs/DUAL_ENGINE_CASCADE.md`](docs/DUAL_ENGINE_CASCADE.md)

---

## License

Apache 2.0 for repository software. Neural network checkpoints and evaluation corpora are subject to their respective upstream licenses.
