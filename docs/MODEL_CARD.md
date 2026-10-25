# Model Card: ONN-DETECT Production & Research Stack

## Intended Use

Offline forensic media analysis in an Android application. Designed to detect synthetic and manipulated media across visual, speech, and video modalities without cloud dependencies, preserving user privacy (`android.permission.INTERNET` is strictly absent).

## Assets and Lineage

Lineage is cryptographically bound in `android/app/src/main/assets/constants/release_manifest.json` and enforced by `onn_release_lineage_gate.py`:
- **SRM Forensic CNN (`srm_forensic_cnn_int8.tflite`)**: Input is float32 NHWC `[1, 128, 128, 3]`. Evaluates residual high-pass spatial noise. Supports both PatchCraft mosaic extraction (`-patchcraft-mosaic-tile32`, assembling 16 high-frequency $32\times 32$ crops) and bilinear scaling (`-bilinear-square-v2-size128-float32`). Calibrated operating threshold: `0.53125`.
- **Vision Foundation Probe (`siglip_probe_int8.tflite`)**: Input is float32 NHWC `[1, 224, 224, 3]`. SigLIP-B/16 tower with Tunable Attention Pooling (TAP) head fine-tuned with DCPT loss and $14\times 14$ patch token supervision. Calibrated probe threshold: `0.48364`.
- **Speech Countermeasure (`lcnn_audio_int8.tflite`)**: Input is float32 NHWC `[1, 401, 40, 1]` (LFCC power spectrogram). 0.72 MB LCNN-MFM student architecture with multi-window audio aggregation (`WINDOWS_PER_FILE = 3` support). Calibrated operating threshold: `0.35546875` (`lcnn_audio_threshold.json`).
- **Face Landmarker (`face_landmarker.task`)**: MediaPipe FaceMesh task evaluating 478 perioral and facial landmarks for BioLip micro-jitter and AVFF lip-audio cross-correlation.
- **Fusion Booster (`XGBoostMetaClassifier.kt` / `xgboost_meta.json`)**: 16 shallow trees combining 10 forensic signals with SigLIP logit base-margin prior. Decision threshold: `0.58559` (`meta_classifier.json`).

## Output and Gating Policy

- **Verdict States**: `AUTHENTIC`, `SYNTHETIC`, or `INCONCLUSIVE`.
- **Speech Content Gating**: Audio evaluation is gated by `VoicedDurationGate.voicedMs >= 300L` with zero-crossing rate and spectral flux checks; non-speech audio fails safely to `INCONCLUSIVE`. Evaluated against the modal audio threshold (`0.35546875`).
- **Video Sampling & AVFF Synchrony**: Timeline frames are sampled and evaluated with keyframe reuse (up to 4 probe calls). Lip-sync correlation is aligned via `computeAvffLipSyncCorrelation`. Video temporal blend (`0.65` visual, `0.20` audio, `0.15` temporal) provides multi-frame scoring, with video verdicts marked provisional until independent video-corpus calibration is certified.
- **Provenance Verification**: Standalone offline C2PA Content Credentials verification (`C2PAManifestChecker`, `C2PATrustStore`, `C2PAVerifier`) against bundled X.509 root anchors (`c2pa_trust_anchors.pem`).
- **Voice Vault**: Hardware-backed Android KeyStore AES-256-GCM encrypted local speaker verification. Gated on anti-spoof threshold (`0.35546875`).

## Limitations

- **Compression Laundering**: Aggressive re-compression (e.g. repeated messaging platform transcode below JPEG Q50 or Opus <12 kbps) attenuates high-frequency noise residuals. The cascade mitigates this via semantic base-margin priors.
- **Novel Generative Architectures**: Unseen generator architectures with distinct patch token semantics require retraining probe weights via `bfree_paired_corpus.py` and `train_bfree_tap_and_ssl_distill.py`.
- **Physical Device Execution**: Production deployment requires physical device latency validation (P95 <250 ms) and 16 KiB ELF alignment verification on target Android 15/16 hardware.
