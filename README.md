# ONN-DETECT: Zero-Trust Offline Multimodal Deepfake Forensics

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform: Android](https://img.shields.io/badge/Platform-Android%2014%2B-green.svg)](https://developer.android.com)
[![PyTorch: 2.x](https://img.shields.io/badge/PyTorch-2.x%20%7C%20CUDA-orange.svg)](https://pytorch.org)
[![TFLite: INT8](https://img.shields.io/badge/Inference-TFLite%20INT8%20(On--Device)-red.svg)](https://tensorflow.org/lite)
[![Zero-Trust: 100% Offline](https://img.shields.io/badge/Privacy-100%25%20Airgapped%20Offline-black.svg)]()

**ONN-DETECT** is an enterprise-grade, multimodal deepfake detection engine designed for **100% on-device, air-gapped forensic analysis** on Android. It fuses high-frequency residual CNNs, neural vocoder spectrogram analyzers, spatial spectral physics, hardware sensor noise fingerprinting (PRNU), optical flow velocity morphing heuristics, and cryptographic provenance into a transpiled **XGBoost Meta-Classifier** to expose synthetic media without ever transmitting sensitive user data to the cloud.

---

## ⚡ Core Forensic Modalities (100% On-Device)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 ONN-DETECT PIPELINE                                     │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        ▼                                   ▼                                   ▼
 📸 Visual Forensics                🎙️ Audio Forensics                 🎬 Temporal & Cross-Modal
 ├─ SRM-CNN (Noise Residuals)        ├─ LCNN-MFM (LFCC Cepstrals)        ├─ Lip-Sync Audio Correlation
 ├─ FFT (2D Power Spectrum)          ├─ Neural Vocoder Phase Forensics   ├─ Facial Landmark Jitter
 ├─ KL Co-occurrence Divergence      └─ Hardware-Backed Voice Liveness   └─ Optical Flow Velocity Morphing
 └─ PRNU Camera Sensor Pattern          (EncryptedSharedPreferences)
        │                                   │                                   │
        └───────────────────────────────────┼───────────────────────────────────┘
                                            │
                                            ▼
                             🔒 Cryptographic Provenance
                             ├─ C2PA JUMBF Manifest Parser
                             └─ EXIF Manipulation Detector
                                            │
                                            ▼
                             🧠 XGBoost Decision Trees
                             (Pure Kotlin Transpilation: < 100 µs)
                                            │
                                            ▼
                             📊 Calibrated Risk Breakdown
                             (Cyber-Forensic Compose Dark UI)
```

### 1. 📸 Visual Forensics
* **Spatial Rich Model (SRM-CNN)**: Fixed spatial high-pass filters extract high-frequency noise residuals left behind by Latent Diffusion (Stable Diffusion, Midjourney, DALL-E 3, Flux) and GAN models.
* **Fast Fourier Transform (FFT)**: Computes 2D radial power spectra and azimuthal integration to flag unnatural high-frequency grid harmonics and deviations from the natural photographic $1/f^2$ power law.
* **Pixel Co-occurrence (KL Divergence)**: Evaluates adjacent horizontal/vertical pixel difference histograms against calibrated natural image distributions.
* **Camera Sensor Fingerprinting (PRNU)**: Isolates sensor Photo Response Non-Uniformity patterns using discrete Laplacian matrices to verify if an image passed through physical smartphone camera optics.

### 2. 🎙️ Audio Forensics
* **Light-CNN with Max-Feature-Map (LCNN-MFM)**: Analyzes 40-channel Linear Frequency Cepstral Coefficients (LFCC) across linear frequency bands to catch phase distortions and spectral discontinuities characteristic of neural vocoders (HiFi-GAN, WaveGlow, Tacotron, FastSpeech).
* **Voice Liveness & Speaker Verification**: Compares incoming speech against an enrolled trusted biometric voiceprint using 192-dimensional embeddings and cosine similarity. All voiceprints are encrypted at rest using AES-256-GCM via the Android Keystore (`EncryptedSharedPreferences`).

### 3. 🎬 Video & Cross-Modal Forensics
* **Cross-Modal Lip-Sync Correlation**: Correlates MediaPipe facial landmark motion (mouth/lips) with the audio amplitude envelope to catch audio-driven talking-head deepfakes (Wav2Lip, Audio2Head).
* **Temporal Landmark Jitter**: Measures inter-frame standard deviation of 3D facial anchors to flag temporal flickering.
* **Optical Flow Velocity Morphing**: Dense Kotlin block-matching evaluates pixel motion vectors to detect unnatural "melting" physics common in generative video models (Sora, Runway Gen-2).

### 4. 🔒 Cryptographic Provenance & Model Security
* **C2PA Manifest Scanner**: Inspects JPEG/PNG binary headers for `c2pa.org/seam/cg` JUMBF metadata containers.
* **EXIF Anomaly Detection**: Flags synthetic software tags (e.g. Photoshop, ComfyUI, Automatic1111) and stripped timestamp arrays.
* **Runtime Model Verification**: SHA-256 integrity checks execute on app boot to guarantee `.tflite` model files have not been tampered with.

---

## 📊 Trained Model Benchmark Results

All models were trained on an **NVIDIA RTX A4000 GPU (16GB VRAM)** across high-resolution multi-generator datasets:

| Model Architecture | Input Resolution / Shape | Target Dataset | Primary Metric | Result |
| :--- | :--- | :--- | :--- | :--- |
| **SRM-Forensic-CNN** | $256 \times 256 \times 3$ (RGB) | GenImage / Diffusion Artifacts | **Validation AUC** | **98.64%** |
| **LCNN-MFM-Audio** | $[1, 401, 40, 1]$ (LFCC) | ASVspoof 2019 / In-The-Wild | **Validation AUC**<br/>**Test AUC**<br/>**Validation EER** | **99.92%**<br/>**98.53%**<br/>**2.15%** |
| **XGBoost Meta-Fusion** | 10-Feature Vector | Multi-Signal Real Calibration | **Test AUC**<br/>**Test Accuracy** | **96.18%**<br/>**89.40%** |
| **TFLite INT8 Visual** | $(1, 32, 32, 3)$ | Real Image Calibration | **Decision Agreement** | **98.00%** |
| **TFLite INT8 Audio** | $(1, 401, 40, 1)$ | Real Audio Calibration | **Decision Agreement** | **100.00%** (Mean diff: 0.006) |

---

## 🧠 The 10-Feature Meta-Fusion Vector

The transpiled Kotlin meta-classifier ([`XGBoostMetaClassifier.kt`](file:///c:/Users/gaura/.gemini/antigravity/scratch/Onn-Detect--main/android/app/src/main/java/com/onndetect/detection/fusion/XGBoostMetaClassifier.kt)) evaluates the following feature vector in $< 100\,\mu\text{s}$:

```kotlin
val features = floatArrayOf(
    srmScore,          // [0] SRM residual CNN anomaly score
    spectral.score,    // [1] FFT 2D spectral power law deviation
    klScore,           // [2] Pixel co-occurrence KL divergence
    spectral.flatness, // [3] Spectral flatness measure
    quality.resNorm,   // [4] Normalized image resolution
    quality.isComp,    // [5] Compression flag (1.0 = compressed)
    exifAnomaly,       // [6] EXIF metadata tampering anomaly score
    c2paPresent,       // [7] C2PA synthetic metadata presence (1.0 / 0.0)
    lipSyncCorr,       // [8] Cross-modal audio-to-mouth motion correlation
    landmarkJitter     // [9] Inter-frame facial landmark jitter variance
)
val rawScore = metaClassifier.predict(features)
```

---

## 🛠️ Reproducing Training & Exporting Models

The ML pipeline uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python dependency management.

### 1. Environment Setup
```powershell
cd ml
uv pip install -r requirements.txt
```

### 2. Dataset Download
Download high-resolution visual and in-the-wild audio datasets:
```powershell
# Stream multi-generator visual (50k images) and deepfake audio (10k samples)
uv run python scripts/download_advanced_datasets.py --mode all --visual-samples 50000 --audio-samples 10000
```

### 3. Model Training
```powershell
# 1. Train Visual SRM-CNN (50 Epochs, AMP FP16)
$env:DATASET_DIR = "../datasets/highres_visual"
$env:IMG_SIZE = "256"
uv run python scripts/train_srm_cnn.py --device cuda --amp --epochs 50 --batch-size 64 --num-workers 4

# 2. Train Audio LCNN-MFM (35 Epochs, AMP FP16)
uv run python scripts/train_lcnn_audio.py --config configs/lcnn_inthewild.yaml --device cuda --amp --epochs 35 --num-workers 4
```

### 4. Quantization, Parity Verification & Android Export
```powershell
# 1. Quantize PyTorch checkpoints to INT8 TFLite with numerical parity checks
uv run python scripts/export_all_models.py

# 2. Regenerate Android calibration constants & SHA-256 hashes
uv run python scripts/generate_android_constants.py

# 3. Train XGBoost meta-classifier and transpile decision trees into Kotlin
uv run python scripts/train_xgboost_meta.py
```

---

## 📱 Android App Architecture

Built natively with **Jetpack Compose**, **Kotlin Coroutines / Flow**, and **Dagger Hilt**:

```
android/app/src/main/java/com/onndetect/
├── data/
│   ├── integrity/          # TFLite SHA-256 checksum validator
│   └── preferences/        # EncryptedSharedPreferences store
├── detection/
│   ├── audio/              # AudioClassifier, SpeakerVerifier
│   ├── crossmodal/         # LipSyncCorrelation
│   ├── fusion/             # XGBoostMetaClassifier.kt (Transpiled)
│   ├── provenance/         # EXIFAnalyzer, C2PAManifestChecker
│   ├── temporal/           # LandmarkJitterDetector
│   ├── video/              # OpticalFlowAnalyzer
│   └── visual/             # SRMForensicAnalyzer, FFTSpectralAnalyzer, PRNUAnalyzer
├── domain/pipeline/        # DetectionPipeline orchestrator
└── ui/
    ├── screens/            # HomeScreen, ScanScreen, VoiceEnrollmentScreen
    └── theme/              # Cyber-Forensic Dark Mode Theme
```

### Building the APK
```powershell
cd android
./gradlew assembleDebug
```

---

## ⚖️ License
This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
