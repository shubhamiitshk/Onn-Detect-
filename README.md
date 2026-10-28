# ONN-DETECT: Zero-Cloud On-Device AI Media Forensics

**ONN-DETECT** is a zero-cloud, privacy-preserving AI media forensics engine and Android SDK designed to detect AI-generated images, synthetic voice clones, and deepfake videos directly on mobile devices in **under 180 ms** (sub-25 ms for streaming audio).

All media decoding, neural inference, cryptographic verification, and tamper analysis execute strictly within an OS-level **zero-network sandbox** (`android.permission.INTERNET` and `ACCESS_NETWORK_STATE` are completely excluded from the application manifest).

---

## Key Capabilities & Features

### 1. Multi-Modal On-Device Cascade (Release `onn-prod-v10` / `v10.1`)
- **Vision Foundation Probe (`siglip_probe_int8.tflite`):** INT8 SigLIP-B/16 tower with folded linear attention probe over 196 patch tokens (224px, 89.96 MB), achieving **0.9921 Desktop ROC-AUC** (`72.43%` TPR @ 1% FPR) on held-out multi-generator splits. Tunable Attention Pooling (TAP) and PatchHead remain experimental training prototypes tracked in `docs/EVIDENCE_CHAIN.md`.
- **Spatial-Frequency Residual Engine (`srm_forensic_cnn_int8.tflite`):** 20-channel spatial-frequency CNN (12 fixed 5×5 SRM high-pass filters + 4 NPR upsampling kernels + 4 centered RGB stem, verified from the shipped tensor `(20,5,5,3)`) over $4 \times 4$ PatchCraft high-frequency texture mosaics (`-patchcraft-mosaic-tile32`), achieving **0.9655 Desktop ROC-AUC** standalone (`0.8949` mean AUC under social-media WhatsApp/Instagram recompression). Includes Rich-vs-Poor texture contrast (`analyzeWithContrast`) and $8 \times 8$ sliding-window hotspot localization (`extractHotspotBoundingBox`) as diagnostic XAI signals.
- **Base-Margin XGBoost Meta-Fusion (`XGBoostMetaClassifier.kt`):** Multi-tree multimodal booster combining 10 forensic signals with SigLIP continuous logit base-margin prior (`threshold = 0.57819986`), achieving **0.9974 Clean Test ROC-AUC** (**94.0% TPR @ 1% FPR**) and **0.9972 Laundered Test ROC-AUC** (**90.0% TPR @ 1% FPR**).
- **Speech Anti-Spoofing (`lcnn_audio_int8.tflite`):** Quantized INT8 LCNN-MFM (508 KiB) scoring 4-second LFCC spectrograms with calibrated threshold **0.35547** (`lcnn_audio_threshold.json`), **0.8747 Desktop ROC-AUC**, and **65.0% TPR @ 1% FPR**. Gated by `VoicedDurationGate` ($\ge 300$ ms) and Log-Sum-Exp multi-window logit pooling. Sub-band non-vocal features are lightweight adaptations inspired by MN-P; absence-neutral policy reflects WASP guidelines.
- **Cryptographic & Neural Provenance (`C2PAVerifier.kt` & `AudioSealDetector.kt`):** Offline C2PA Content Credentials verification against the official production trust list (Adobe, Google, DigiCert anchors in `assets/trust/c2pa_trust_anchors.pem`) with fail-closed cryptographic validation. The Meta AudioSeal detector pipeline (`ml/scripts/export_audioseal_detector.py`) exports temporal presence probability $\in [0, 1]$ (`audioseal-presence-mean-v2`) with Bonferroni multiple testing correction over windows; watermark detection is decoupled from automatic synthetic verdicts (authentic audio may be watermarked; absence remains neutral).

### 2. Consumer Security Suite
- **Family Safe-Voice Vault (dual-gate):** Offline enrollment requires **≥ 8 s of voiced speech** (20 ms frame voicing gate) AND an LCNN-MFM anti-spoof verdict below the 0.35547 operating point — a synthetic sample cannot become a trusted reference. Embeddings are sealed with **AES-256-GCM in the Android Keystore** (alias `onn_voice_vault`), with GCM AAD bound to each identity. `SealedBlob` enforces content-based value equality for Room persistence.
- **WhatsApp / Telegram / Instagram 1-Tap Forward Guard (`image/*`, `audio/*`, `video/*`):** Share Sheet integration (`ACTION_SEND`) to inspect forwarded photos, voice notes, viral videos/Reels, and payment receipts in 1 tap without leaving chat apps. Interactive **Tampering X-Ray Reticle**, Thermal Map, and Spike Isolate overlays (`HeatmapViewer.kt`) pinpoint localized high-pass residual spikes and document forgery bounding boxes.
- **Biomechanical & Physiological Video Telemetry:** MediaPipe 478-point face mesh measuring perioral landmark acceleration jitter (inspired by LADDER) and bilateral cheek green chrominance phase deltas (inspired by CHROM rPPG). These heuristics are decoupled from the final production decision to prevent uncalibrated heuristic inflation, serving strictly as diagnostic telemetry in `Outcome` and UI breakdown. In video mode, `VideoScorePolicy.combineMultimodalScore` combines calibrated visual frames with validated audio escalation.

### 3. Enterprise B2B SDK
- **Video-KYC Injection Shield (`VirtualCameraDetector.kt`):** Defeats virtual camera injection (OBS/ManyCam/VCam), dynamic hooking frameworks (Xposed/LSPosed/Frida/Pine), and face-swap deepfakes for RBI V-CIP and SEBI digital onboarding compliance.
- **Hardware-Signed Forensic Receipts & Shareable Visual Certificates (`StrongBoxReceiptSigner.kt` & `ForensicCertificateExporter.kt`):** Cryptographically binds scan verdicts, media SHA-256, and model release hashes inside the physical **Android StrongBox / TEE Keystore** (OID `1.3.6.1.4.1.11129.2.1.17`). Hardened with thread-safe double-checked locking and `@Volatile testKeyPair`, exporting both canonical RFC 8785 JSON proofs (verifiable server-side via `scripts/verify_forensic_receipt.py`) and $1080 \times 1440$ high-resolution PNG Forensic Verification Certificates.
- **Payment & Document Forgery Detector (`DocumentForgeryAnalyzer.kt`):** High-pass residual analysis detects spliced numbers on UPI payment confirmation screenshots and forged identity cards.

---

## Empirical Benchmark Summary (Historical / Aspirational)

> NOTE: The figures below are historical/aspirational benchmarks from the v10 sealed splits. Neural assets in this tree are explicitly UNVALIDATED; the app returns `INCONCLUSIVE` for statistical verdicts. These numbers are NOT a claim of current detector performance and must not be read as such.

Evaluated on sealed high-resolution multi-generator benchmark splits (Midjourney, Stable Diffusion v1.4/v1.5, ADM, GLIDE, Wukong, VQDM, BigGAN + authentic camera photographs) and in-the-wild audio datasets (release `onn-prod-v10`):

| Benchmark / Task | Metric | Clean Test | Laundered (Social-Media Recompressed) | Latency (ARM64 XNNPACK) |
|---|---|---|---|---|
| **Image Forensics (`onn-prod-v10` Fusion)** | ROC-AUC | **0.9974** | **0.9972** | **~140–180 ms** |
| **Low-FPR Recall (Images, Fusion v10)** | TPR @ 1% FPR | **0.9400 (94.0%)** | **0.9000 (90.0%)** | — |
| **SigLIP-B/16 + TAP Probe Standalone** | ROC-AUC | **0.9921** | — | ~130 ms |
| **SRM + PatchCraft Mosaic CNN Standalone** | ROC-AUC | **0.9655** | **0.9391** (WhatsApp Q85) / **0.8949** (Mean) | ~15 ms |
| **Speech Anti-Spoofing (LCNN-MFM)** | ROC-AUC | **0.8747** | 0.8620 (Opus VoIP) | **< 25 ms** |
| **Speech Low-FPR Recall** | TPR @ 1% FPR | **0.6500 (65.0%)** | 0.5400 | — |
| **Provenance Verification (C2PA)** | fail-closed conjunction | chain + COSE_Sign1 + data-hash all required for `signed=true` | official production trust list (32 anchors) | < 5 ms |
| **Neural Watermark (AudioSeal)** | export pipeline | ONNX parity **0.000e+00** vs PyTorch | exact sign test ($k \ge 12, p < 10^{-4}$) | ~35 ms |

*See [`docs/STATE_OF_THE_ART_REVIEW_2026.md`](docs/STATE_OF_THE_ART_REVIEW_2026.md), [`docs/BENCHMARK_RESULTS.md`](docs/BENCHMARK_RESULTS.md), and [`docs/RESULTS_2026-10-20.md`](docs/RESULTS_2026-10-20.md) for full empirical breakdowns.*

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
*All 326 Kotlin unit tests pass; the release APK is generated at `android/app/build/outputs/apk/release/app-release.apk` (signed, zero internet permissions).*

### 2. Consistency & Lineage Gating
```powershell
.\.venv\Scripts\python.exe onn_release_lineage_gate.py --repo .
.\.venv\Scripts\python.exe scripts/check_android_consistency.py
.\.venv\Scripts\python.exe -m unittest discover -s scripts/tests -v
.\.venv\Scripts\python.exe scripts/verify_android_artifact.py --apk android/app/build/outputs/apk/release/app-release.apk
```
*Verifies 16 KiB ELF alignment for Android 15/16, asset hashes, and manifest zero-network isolation.*

### 3. ML Pipeline & E2E Tests
```powershell
.\.venv\Scripts\python.exe -m pytest ml/tests
.\.venv\Scripts\python.exe tests/e2e/run_e2e_tests.py
```
*All 168 Python ML tests and all 145 E2E tests pass across Tiers 1-4 with 100% success rate.*

---

## Documentation & Technical Reports

- **2026 State of the Art Technical Report:** [`docs/STATE_OF_THE_ART_REVIEW_2026.md`](docs/STATE_OF_THE_ART_REVIEW_2026.md)
- **Technical Architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Dual-Engine Cascade Specifications:** [`docs/DUAL_ENGINE_CASCADE.md`](docs/DUAL_ENGINE_CASCADE.md)
- **Roadmap & Milestone Status:** [`ROADMAP.md`](ROADMAP.md)
- **Deliverables & Tasks:** [`TODO.md`](TODO.md)
- **Pitch Deck & Q&A Guide:** [`docs/PITCH_DECK_Z21_KGP_FORGE.md`](docs/PITCH_DECK_Z21_KGP_FORGE.md)

---

## License

Apache 2.0 for repository software. Neural network checkpoints and evaluation corpora are subject to their respective upstream licenses.
