# ONN-DETECT: Zero-Cloud On-Device AI Media Forensics

**ONN-DETECT** is an on-device, zero-cloud AI media forensics engine and Android SDK designed to detect AI-generated images, synthetic voice clones, and deepfake videos directly on smartphones in **under 180 ms** (sub-25 ms for streaming audio).

All media decoding, neural inference, cryptographic verification, and tamper analysis execute strictly within an OS-level **zero-network sandbox** (`android.permission.INTERNET` is completely excluded from the application manifest).

---

## Key Capabilities & Features

### 1. Multi-Modal On-Device Cascade (Release v7)
- **Vision Foundation Probe:** Frozen INT8 SigLIP-B/16 (224px, 89.96 MB) memory-mapped directly into the OS page cache with **0 MB Java heap bloat**.
- **Spatial-Frequency Residual Engine:** 20-channel spatial-frequency CNN (12 fixed 5×5 SRM high-pass filters + 4 NPR upsampling kernels + 4 centered RGB low-pass stem, verified from the shipped tensor `(20,5,5,3)`) detecting generative grid artifacts and resampled boundaries.
- **Base-Margin XGBoost Meta-Fusion:** Injects SigLIP continuous logits as the booster `base_margin` ($\text{margin} = \text{logit}(p) + f_{\text{tree}}$), eliminating tree quantization ties and boosting low-FPR recall by **7× (TPR@1%FPR: 0.040 $\rightarrow$ 0.280)**.
- **Speech Anti-Spoofing:** Quantized INT8 LCNN-MFM (508 KiB) scoring 4-second audio windows in **<25 ms** with **0.875 AUC** and **59.5% TPR @ 1% FPR**.
- **Cryptographic & Neural Provenance:** Offline C2PA Content Credentials verification against the official production trust list (Adobe, Google, DigiCert, ... anchors in `assets/trust/c2pa_trust_anchors.pem`): full JUMBF walk, multi-segment APP11 reassembly (ISO 19566-5), COSE_Sign1 ES256 and c2pa.hash.data exclusion-range binding, with a fail-closed `signed = hashVerified && coseSignatureValid && certChainValid` conjunction. The Meta AudioSeal watermark detector ships its verified ONNX export pipeline (`ml/scripts/export_audioseal_detector.py`); the app-side TFLite weights are pending a numerically faithful conversion — the exporter refuses to emit divergent graphs.

### 2. Consumer Security Suite
- **Family Safe-Voice Vault (dual-gate):** Offline enrollment requires **≥ 8 s of voiced speech** (20 ms frame voicing gate) AND an LCNN-MFM anti-spoof verdict below the 0.375 operating point — a synthetic sample can never become a trusted reference. Embeddings are pooled over 3 s windows at a 1.5 s hop and sealed with **AES-256-GCM in the Android Keystore** (alias `onn_voice_vault`), with the GCM AAD bound to each row's identity; a manifest swap between rows fails authentication loudly.
- **WhatsApp / Telegram 1-Tap Forward Guard:** Share Sheet integration (`ACTION_SEND`) to inspect forwarded voice notes, viral videos, and payment receipts without leaving chat apps. 32×32 residual tamper heatmaps and in-memory ELA pinpoint the edited regions.
- **BioLip Perioral Forensics:** MediaPipe 478-point face mesh (64-point perioral set, inter-ocular pose normalization around the nose tip) measuring second-order acceleration jitter (0.6) and MAR flicker (0.4) in talking-head video, with a CPU fallback for threads without an EGL context.

*(The earlier planned UPI live-call shield and AI-reel accessibility automation were removed for Google Play 2026 policy compliance — the shipped suite above is 100% Play-compliant and network-free: the manifest actively removes `INTERNET` and `ACCESS_NETWORK_STATE`.)*

### 3. Enterprise B2B SDK
- **Video-KYC Injection Shield:** Defeats virtual camera injection (OBS/ManyCam) and face-swap deepfakes for RBI V-CIP and SEBI digital onboarding compliance.
- **Hardware-Signed Forensic Receipts:** Cryptographically binds scan verdicts, media SHA-256, and model release hashes inside the physical **Android StrongBox / TEE Keystore**, creating tamper-proof audit trails for insurance claim investigations.
- **Payment & Document Forgery Detector:** High-pass residual analysis detects spliced numbers on UPI payment confirmation screenshots and forged identity cards.

---

## Empirical Benchmark Summary

Evaluated on sealed benchmark corpora spanning 68 generative models (Midjourney v6, Flux, SDXL, Sora, Ideogram) and in-the-wild audio datasets:

| Benchmark / Task | Metric | Clean Test | Laundered (512px JPEG Q85) | Latency (ARM64 XNNPACK) |
|---|---|---|---|---|
| **Image Forensics (v7 Cascade)** | ROC-AUC | **0.8941** | **0.8999** | **~140–180 ms** |
| **Low-FPR Recall (Images)** | TPR @ 1% FPR | **0.2800 (28.0%)** | **0.2950 (29.5%)** | — |
| **Speech Anti-Spoofing (LCNN-MFM)** | ROC-AUC | **0.8749** | 0.8620 (Opus VoIP) | **<25 ms** |
| **Speech Low-FPR Recall** | TPR @ 1% FPR | **0.5950 (59.5%)** | 0.5400 | — |
| **Provenance Verification (C2PA)** | fail-closed conjunction | chain + COSE_Sign1 + data-hash all required for `signed=true` | official production trust list (32 anchors) | <5 ms |
| **Neural Watermark (AudioSeal)** | export pipeline | ONNX parity **0.000e+00** vs PyTorch | app-side TFLite weights pending faithful conversion | — |

*See [`docs/RESULTS_2026-10-20.md`](docs/RESULTS_2026-10-20.md) and [`docs/BENCHMARK_RESULTS.md`](docs/BENCHMARK_RESULTS.md) for full empirical breakdowns.*

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
*All 222 Kotlin unit tests pass; the release APK is generated at `android/app/build/outputs/apk/release/app-release.apk` (110.93 MB, signed, zero internet permissions).*

### 2. Consistency & Integrity Gating
```powershell
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
