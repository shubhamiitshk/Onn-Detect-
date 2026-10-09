# ONN-DETECT

An Android app for offline image, speech and sampled video analysis. Media decoding, feature extraction, inference and history storage run on the device. The APK gate checks the merged permissions, asset hashes and 16 KiB compatibility of 64-bit native libraries.

**Current bundled models are unvalidated.** Their hashes are preserved, but their original data, score orientation, preprocessing and calibration cannot be established from this checkout. Scans therefore report **Inconclusive**, with available diagnostic signals and explicit failures or coverage limitations. A diagnostic score is not a calibrated probability or proof of authenticity. No current accuracy claim covers all generators, ChatGPT images, Nano Banana, edited images, music or unseen voices.

## Implemented behavior

- Typed scan status and verdict, with unavailable evidence kept distinct from zero.
- Hash-bound model, tensor, frontend, threshold and fusion contracts; serialized CPU LiteRT inference.
- Bounded media decoding, cancellation, PCM-format handling, shared anti-alias resampling, YUV crop/stride/color/rotation handling and sampled coverage.
- Status-aware result/history UI, lifecycle-safe scanning and explicit Room migrations that retain legacy history as unvalidated.
- Unsupported biometric enrollment is labeled unavailable. EXIF, byte-pattern provenance and temporal heuristics do not override verdicts.
- Standalone CPU LiteRT is bundled; no runtime model download is needed.

See [implementation status](docs/FIX_STATUS_2026-10-07.md), [architecture](docs/ARCHITECTURE.md) and the [complete review and dataset research](docs/COMPREHENSIVE_REVIEW_2026-10-06.md). Review files describe the pre-fix snapshot; implementation status describes the current work.

## Build and verify

Use JDK 17 and Android SDK 36. The repository pins AGP, Gradle, Kotlin, Compose compiler, KSP and the standalone LiteRT runtime. Android 8.0/API 26 is the minimum supported OS.

```powershell
cd android
.\gradlew.bat testDebugUnitTest lintDebug assembleDebug assembleRelease assembleDebugAndroidTest
cd ..
python scripts/check_android_consistency.py
python -m unittest discover -s scripts/tests -v
python scripts/verify_android_artifact.py --apk android/app/build/outputs/apk/debug/app-debug.apk
python scripts/verify_android_artifact.py --apk android/app/build/outputs/apk/release/app-release-unsigned.apk
```

The release APK is unsigned. The ordinary APK gate permits explicitly unvalidated diagnostic builds. Distribution as a validated detector additionally requires `--require-validated-models`, a matched trained release and independent device evidence. JVM tests do not replace codec, bitmap, performance or migration tests on devices.

## Training

Use a separate Python 3.11 environment. Training may run on a GPU workstation or dedicated runner; detection remains offline on Android.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ml/requirements.txt -r ml/requirements-export.txt -r ml/requirements-data.txt -r ml/requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e ml
.\.venv\Scripts\python.exe -m unittest discover -s ml/tests -p "test_*.py" -v
```

Prepared corpora need verified source/group identities, file hashes, both labels and disjoint train/validation/calibration/test splits. Image fusion additionally needs independent stacking and fusion-validation splits. Missing splits, corrupt files and unknown label schemas fail instead of silently becoming training examples.

```powershell
cd ml
..\.venv\Scripts\python.exe scripts/run_training_pipeline.py --visual-config configs/srm_cnn_openfake.yaml --audio-config configs/lcnn_asvspoof.yaml --data-root D:/verified-datasets --device cuda --output-dir runs/experiment-001
```

This stages a bundle and evaluation evidence. It does not overwrite Android assets or promote desktop evidence to device validation. Read [training guide](docs/TRAINING_GUIDE.md), [model card](docs/MODEL_CARD.md) and [deployment](docs/DEPLOYMENT.md) before running a large experiment.

## Research priorities

The existing 32px image model is a pilot baseline. Test a higher-resolution residual model and a compact semantic model against unseen generator families, including independently acquired modern platform samples. Speech training should combine licensed ASVspoof, multilingual TTS and realistic codec/channel data, with speaker/source/generator-disjoint holdouts. Speech, music and environmental audio need distinct routing and evaluation.

The full review contains verified primary-source links, dataset sizes, license/access restrictions and a staged teacher/student strategy. Large datasets or new models do not by themselves guarantee generalization. Held-out families, calibrated low false-positive operating points and actual phone parity determine what can be released.

License: Apache 2.0 for repository code. Dataset and model licenses are separate.
