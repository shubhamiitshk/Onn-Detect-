# ONN-DETECT

Offline, privacy-first AI-media forensics for Android. PyTorch backend trains
compact detectors; the Android app runs them fully on-device via TFLite
(LiteRT). **No INTERNET permission** - scans never leave the phone.

## Status (working, verified)

| Stage | Metric | Value |
|---|---|---|
| SRMForensicCNN (visual) | AUC / Acc on genimage/val | 0.956 / 0.889 |
| LCNN-MFM (audio)        | EER / AUC on asvspoof test | 0.050 / 0.976 |
| XGBoost meta fusion     | AUC on held-out features   | 0.962 |

All numbers come from `ml/evaluation_results.json`, produced by real
evaluations (`ml/scripts/run_full_pipeline.py`). The transpiled Kotlin
meta-classifier matches xgboost predictions to <1e-4 (export-time parity
check).

## Layout

- `ml/` - training + export backend (see `ml/Makefile`)
  - `src/models/` SRMForensicCNN, LCNN-MFM; `src/meta_classifier/` feature
    extraction mirroring the on-device analyzers
  - `scripts/train_*.py` -> `scripts/export_all_models.py` ->
    `scripts/generate_android_constants.py` ships models+constants into the app
- `android/` - Compose + Hilt app. Image scan path is end-to-end:
  gallery pick -> SRM TFLite + FFT slope + KL co-occurrence -> transpiled
  XGBoost fusion -> Room history.
- `datasets/genimage` 32x32 thumbnails; `datasets/asvspoof2019` FLACs.

## Critical implementation notes

1. **Never resize the 32x32 dataset images up** before SRM-CNN inference or
   training. Native resolution is what makes it work (AUC 0.96 vs 0.50).
2. Feature contract shared by Python and Kotlin (order matters), 10 slots:
   `[srm_score, fft_score, kl_score, flatness_score, resolution_norm,
   is_compressed, exif_anomaly, c2pa_present, lip_sync_corr,
   landmark_jitter]` - defined in `ml/src/meta_classifier/feature_extractor.py`,
   consumed by `android/.../detection/fusion/XGBoostMetaClassifier.kt`.
   Slots 6-7 are measured from real file metadata on-device; slots 8-9 are
   temporal signals that use the neutral constants in
   `DetectionPipeline.Companion.NEUTRAL_*` (= feature_extractor NEUTRAL_*)
   for still images. NEVER synthesize these features from the SRM score
   during training - that leaks the label and inflates meta AUC.
3. KL histogram semantics must stay identical on both platforms: diffs are
   CLAMPED to [-10,10] into 21 bins, epsilon 1e-6, renormalized.
4. Model hashes in `assets/constants/model_hashes.json` are generated at
   export time; `ModelIntegrityChecker` verifies them at startup.

## Commands

```powershell
# ML backend (Windows, from repo root)
.venv\Scripts\python.exe ml\scripts\train_srm_cnn.py --device cuda
cd ml; make train-meta export eval

# Android
# open android/ in Android Studio, or from android/: gradle wrapper && .\gradlew assembleDebug
```

Verify cross-platform consistency anytime:
`.venv\Scripts\python.exe scripts\check_android_consistency.py`
