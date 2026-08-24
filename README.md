# ONN-DETECT: Zero-Trust Offline Multimodal Deepfake Forensics

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform: Android](https://img.shields.io/badge/Platform-Android%2014%2B-green.svg)](https://developer.android.com)
[![PyTorch: 2.x](https://img.shields.io/badge/PyTorch-2.x%20%7C%20CUDA-orange.svg)](https://pytorch.org)
[![TFLite: INT8](https://img.shields.io/badge/Inference-TFLite%20INT8%20(On--Device)-red.svg)](https://tensorflow.org/lite)
[![Zero-Trust: 100% Offline](https://img.shields.io/badge/Privacy-100%25%20Airgapped%20Offline-black.svg)]()

**ONN-DETECT** is an offline multimodal deepfake detection engine for
Android. It fuses high-frequency residual CNNs, spectral physics, pixel
co-occurrence statistics and cryptographic metadata provenance in a
transpiled **XGBoost meta-classifier** to expose synthetic media without
ever transmitting data off the device (no INTERNET permission; empty TLS
trust anchors).

---

## Core Forensic Modalities

```
┌──────────────────────────────────────────────────────────────────────┐
│                            ONN-DETECT PIPELINE                       │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
 📸 Visual Forensics         🎙️ Audio Forensics         🎬 Temporal & Provenance
 ├─ SRM-CNN (noise residuals)├─ LCNN-MFM (LFCC)         ├─ FrameCosineDelta consistency
 ├─ FFT 1/f^2 slope analysis ├─ EER-calibrated verdict  ├─ EXIF anomaly detection
 ├─ KL co-occurrence         └─ Encrypted voiceprint    ├─ C2PA JUMBF manifest scan
 └─ PRNU sensor-noise check     enrollment (Keystore)   └─ SynthID/C2PA fusion rules
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   ▼
                    🧠 XGBoost decision trees (transpiled Kotlin, <100 µs)
                                   ▼
                    📊 Calibrated risk breakdown (Compose dark UI)
```

### 1. Visual Forensics
* **SRM-CNN**: fixed spatial high-pass filters expose noise residuals left by
  GAN / diffusion generators. Runs at **native resolution** (the shipped model
  is trained on 32x32 inputs; never upscale before inference).
* **FFT**: 2D radix-2 power spectrum -> azimuthal average -> log-log slope
  deviation from the natural 1/f^2 power law.
* **KL co-occurrence**: adjacent-pixel-difference histograms (21 bins,
  clamped to ±10) scored against a calibrated natural-image baseline.
* **PRNU**: Laplacian-variance sensor-noise heuristic, reported as a signal.

### 2. Audio Forensics
* **LCNN-MFM**: Light CNN with Max-Feature-Map over pure-Kotlin LFCC features
  (verified against torchaudio golden vectors), thresholded at the model's
  EER operating point from `lcnn_audio_threshold.json`.
* **Voice enrollment**: encrypted voiceprint storage via
  `EncryptedSharedPreferences` (Android Keystore).

### 3. Temporal & Provenance
* **FrameCosineDelta** temporal consistency for video scans.
* **EXIF analyzer** flags manipulation-software tags and scrubbed timestamps.
* **C2PA scanner** detects JUMBF manifests (byte-scan; no signature
  verification yet). `SignalFusion` forces a definitive AI verdict on
  SynthID presence and dampens scores for trusted C2PA.

> Experimental stubs (documented in code): MobileCLIP semantic scoring,
> MediaPipe lip-sync correlation, facial-landmark jitter, optical-flow
> morphing, ECAPA-TDNN speaker verification. They are wired as neutral
> signals and do not affect verdicts yet.

---

## Benchmark Results

From `ml/evaluation_results.json` (real held-out evaluations produced by
`ml/scripts/run_full_pipeline.py`; see `docs/BENCHMARK_RESULTS.md`):

| Model | Eval split | n | Metric | Result |
| :--- | :--- | :--- | :--- | :--- |
| SRMForensicCNN | genimage/val (native 32x32) | 4000 | AUC / Acc / EER | 0.956 / 0.889 / 0.110 |
| LCNN-MFM-Audio | asvspoof2019/test | 280 | EER / AUC | 0.050 / 0.976 |
| XGBoost meta fusion | held-out feature split | 1000 | AUC / Acc | 0.962 / 0.893 |

The transpiled Kotlin classifier matches `booster.predict()` to <1e-4 at
export time.

## The 10-Feature Meta-Fusion Vector

Order is a cross-platform contract between
`ml/src/meta_classifier/feature_extractor.py` and
[`XGBoostMetaClassifier.kt`](android/app/src/main/java/com/onndetect/detection/fusion/XGBoostMetaClassifier.kt):

```kotlin
val features = floatArrayOf(
    srmScore,          // [0] SRM residual CNN anomaly score
    spectral.score,    // [1] FFT 2D spectral power-law deviation
    klScore,           // [2] Pixel co-occurrence KL divergence
    spectral.flatness, // [3] Spectral flatness measure
    quality.resNorm,   // [4] Normalized image resolution (min(w,h)/1024)
    quality.isComp,    // [5] Compression flag (1.0 = compressed)
    exifAnomaly,       // [6] EXIF tampering anomaly (measured from file)
    c2paPresent,       // [7] C2PA presence score (measured from file)
    lipSyncCorr,       // [8] Temporal signal - neutral constant for stills
    landmarkJitter     // [9] Temporal signal - neutral constant for stills
)
val rawScore = metaClassifier.predict(features)
```

Features 6-9 carry neutral constants until retrained with real labels.

## Reproducing Training & Export

Python backend (Windows PowerShell from repo root):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision torchaudio
.venv\Scripts\python.exe -m pip install -r ml\requirements.txt
.venv\Scripts\python.exe -m pip install -e ml

cd ml
make train-all      # SRM-CNN + LCNN-MFM + XGBoost meta (+ Kotlin transpile)
make export         # PyTorch -> Keras -> TFLite INT8 (parity-checked)
python scripts/generate_android_constants.py
cd ..
python scripts/check_android_consistency.py
```

See `docs/TRAINING_GUIDE.md` for dataset layout and options.

## Android App Architecture

Jetpack Compose, Coroutines/Flow, Hilt, Room, LiteRT:

```
android/app/src/main/java/com/onndetect/
├── data/local/           # Room scan history
├── detection/
│   ├── audio/            # AudioClassifier, AudioDecoder, LFCCExtractor, SpeakerVerifier
│   ├── crossmodal/       # LipSyncCorrelation (stub)
│   ├── fusion/           # XGBoostMetaClassifier.kt (transpiled, auto-generated)
│   ├── provenance/       # EXIFAnalyzer, C2PAManifestChecker
│   ├── temporal/         # VideoDecoder, FrameCosineDelta, LandmarkJitterDetector (stub)
│   ├── video/            # VideoForensicAnalyzer, OpticalFlowAnalyzer (stub)
│   └── visual/           # SRMForensicAnalyzer, FFTSpectralAnalyzer, FFTUtil,
│                         # KLCooccurrenceAnalyzer, PRNUAnalyzer
├── domain/pipeline/      # DetectionPipeline orchestrator, SignalFusion
├── inference/            # LiteRT engine, memory-mapped loader, delegate
│                         # selector, SHA-256 ModelIntegrityChecker
└── ui/                   # Compose screens (home/scanner/result/history/
                          # enrollment) + cyber-forensic dark theme
```

## Building the APK

```powershell
cd android
.\gradlew.bat assembleDebug
```

CI builds the APK, runs unit tests, and executes the cross-platform
consistency checks on every push (`.github/workflows/`).

## License

This project is licensed under the Apache 2.0 License.
