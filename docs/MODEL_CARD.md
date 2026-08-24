# Model Card - ONN-DETECT v1.0

## Models

| Model | Architecture | Input | Output | Size (assets) |
|---|---|---|---|---|
| SRMForensicCNN | frozen 5x5 SRM filters -> depthwise-separable CNN | `(1,32,32,3)` RGB float [0,1] | P(AI-generated) | ~98 KB INT8 |
| LCNN-MFM-Audio | LightCNN with Max-Feature-Map | `(1,T,40,1)` LFCC (T=401 @ 4s/16kHz) | P(synthetic) | ~492 KB INT8 |
| XGBoost meta fusion | 19 boosted trees, transpiled to Kotlin if/else | 10-feature vector | P(AI-generated) | source only |

## Training data

- Visual: GenImage thumbnails (real vs AI-generated), native 32x32.
- Audio: ASVspoof 2019 (bonafide vs spoof), 16 kHz mono, 4 s windows.
- Meta: features extracted from genimage/val with the trained SRM-CNN;
  provenance/temporal slots (features 6-9) are neutral constants pending
  real per-file labels.

## Measured performance (held-out; see docs/BENCHMARK_RESULTS.md)

- SRMForensicCNN: AUC 0.956 / accuracy 0.889 on genimage/val (n=4000).
- LCNN-MFM: EER 0.050 / AUC 0.976 on asvspoof2019/test (n=280).
- XGBoost meta: AUC 0.962 on held-out feature split (n=1000).

## Intended use & limits

- Purpose: triage support for suspected AI-generated media, fully offline.
- NOT a court-grade proof of authenticity or forgery.
- Trained on specific generators/datasets; expect degraded accuracy on
  unseen generators, heavy recompression, screenshots, or scanned documents.
- The visual model requires native-resolution input; upscaling inputs
  invalidates it.
- Features 6-9 carry no signal until the meta-model is retrained with real
  provenance/temporal labels.

## Security

- `.tflite` files verified against SHA-256 (`model_hashes.json`) at app start.
- Voiceprints (if enrolled) stored via EncryptedSharedPreferences
  (AES-256-GCM/SIV, Android Keystore).
- App has no INTERNET permission; network security config removes all trust
  anchors.
