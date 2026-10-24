# Deployment

The source builds an offline forensic diagnostic Android application. Scans execute purely on-device via LiteRT CPU without network communication.

## Android Toolchain and Build

Use JDK 17, Android SDK platform 36, and the checked-in Gradle wrapper. The app specifies `minSdk=26`, `targetSdk=36`, AGP 8.13.2, Gradle 8.13, Kotlin/Compose plugin 2.2.20, and standalone LiteRT CPU 1.4.2.

From the repository root on Windows PowerShell:

```powershell
Push-Location android
try {
    .\gradlew.bat testDebugUnitTest assembleDebug lintDebug assembleRelease assembleDebugAndroidTest
    if ($LASTEXITCODE -ne 0) { throw 'Android validation failed' }
} finally { Pop-Location }
```

The debug APK is output to `android/app/build/outputs/apk/debug/app-debug.apk`. The release APK is output to `android/app/build/outputs/apk/release/app-release.apk`. The build configuration specifies `androidResources.noCompress += "tflite"`, enabling memory-mapping directly from uncompressed asset records.

## Automated Lineage & APK Verification Gates

From the repository root:

```powershell
# 1. Verify model lineage and fusion alignment (P0-1 gate)
python onn_release_lineage_gate.py --repo .

# 2. Verify Android APK artifact (permissions and 16 KiB ELF alignment)
python scripts/verify_android_artifact.py --apk android/app/build/outputs/apk/release/app-release.apk

# 3. Check Android consistency and workflow syntax
python scripts/check_android_consistency.py
python scripts/verify_workflow_syntax.py
```

The APK verifier inspects:
- **Merged Permissions Audit**: Verifies that `android.permission.INTERNET` and `ACCESS_NETWORK_STATE` are strictly absent.
- **16 KiB ELF Alignment**: Verifies that all 64-bit native shared libraries (`arm64-v8a`) comply with 16 KiB page size boundaries for Android 15/16.
- **Model Contract & Hash Integrity**: Verifies SHA-256 hashes against `constants/release_manifest.json`.

## Device Tests

With a supported emulator or test phone connected via ADB:

```powershell
Push-Location android
try {
    .\gradlew.bat connectedDebugAndroidTest
    if ($LASTEXITCODE -ne 0) { throw 'Device tests failed' }
} finally { Pop-Location }
```

`PackagedInferenceTest` exercises bundled model interfaces, verifies image analysis, runs audio sampling with synthetic signals, and evaluates C2PA verification with offline test fixtures.

## Bundled Asset Layout

```text
android/app/src/main/assets/
  models/
    srm_forensic_cnn_int8.tflite     (145 KiB residual high-pass CNN)
    siglip_probe_int8.tflite         (89.96 MB INT8 SigLIP-B/16 + TAP)
    lcnn_audio_int8.tflite           (508 KiB LCNN-MFM speech countermeasure)
    face_landmarker.task             (3.6 MB MediaPipe FaceMesh model)
  trust/
    c2pa_trust_anchors.pem           (Offline PEM X.509 root trust anchors)
  fusion/
    XGBoostMetaClassifier.kt         (Transpiled Kotlin booster, 16 trees)
    xgboost_meta.json                (Compiled XGBoost booster definition)
    meta_results.json                (Fusion evaluation and threshold stats)
  constants/
    model_hashes.json                (Model filenames to expected SHA-256)
    release_manifest.json            (Authoritative release metadata & splits)
    meta_classifier.json             (Fusion feature schema & model bindings)
    cascade_policy.json              (Two-stage cascade evaluation policy)
    semantic_probe_threshold.json    (SigLIP probe calibration metrics)
    lcnn_audio_threshold.json        (Speech countermeasure thresholds)
    natural_diff_histogram.json      (Horizontal pixel-difference baseline)
  evaluation_report.json             (Exhaustive benchmark evaluation record)
  source_snapshot.json               (Git commit provenance snapshot)
```

## Security and Privacy Boundary

- **Zero Network Egress**: The manifest removes all networking permissions.
- **Biometric Security (Voice Vault)**: Biometric speaker embeddings are sealed using hardware-backed Android KeyStore AES-256-GCM (`AndroidKeyStoreVoiceVaultCrypto`) with optional StrongBox attestation.
- **Offline C2PA Verification**: Manifest verification parses CBOR and COSE Sign1 signatures purely offline against local trust anchors, preventing tracking or surveillance via online OCSP / AIA network lookups.
