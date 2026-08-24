# ONN-DETECT Android

Compose + Hilt app. Everything runs on-device; the manifest has **no INTERNET
permission** and the network security config blocks all cleartext/HTTPS.

## Build

The Gradle wrapper is committed - no local Gradle install needed:

```bash
.\gradlew.bat assembleDebug     # build
.\gradlew.bat testDebugUnitTest # unit tests (FFT, KL fusion math)
```

## Architecture

```
MainActivity (permissions -> Compose)
  └─ ONNNavGraph { Home, Scanner, Result, History }
       └─ DetectionPipeline.analyzeMedia(bitmap):
            SRMForensicAnalyzer   (srm_forensic_cnn_int8.tflite, 32x32 RGB [0,1])
            FFTSpectralAnalyzer   (radix-2 FFT -> log-log slope deviation)
            KLCooccurrenceAnalyzer(21-bin diff histogram vs calibrated baseline)
                 ↓ feature vector [srm, fft, kl, flatness, res_norm, compressed,
                    exif, c2pa, lip_sync*, jitter*]  (* neutral for stills)
            XGBoostMetaClassifier (AUTO-GENERATED transpiled trees)
                 ↓
            SignalFusion.fuse()   (SynthID/C2PA provenance override)
                 ↓
            ScanResult -> Room scan history
```

## Generated files - do not edit by hand

- `app/src/main/java/com/onndetect/detection/fusion/XGBoostMetaClassifier.kt`
- `app/src/main/assets/models/*.tflite`
- `app/src/main/assets/constants/*.json`

Regenerate from the ML backend:
`cd ../ml && make export && python scripts/generate_android_constants.py`

The integrity checker (`ModelIntegrityChecker`) verifies every `.tflite`
against `constants/model_hashes.json` at startup.
