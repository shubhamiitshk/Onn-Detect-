# Deployment

## Build the APK

```powershell
cd android
.\gradlew.bat assembleDebug          # debug
.\gradlew.bat assembleRelease        # release (minified + shrunk)
```

Requirements: JDK 17, Android SDK 34. The app targets `minSdk 26`,
`targetSdk 34`.

## Shipped assets (all offline)

```
android/app/src/main/assets/
  models/  srm_forensic_cnn_int8.tflite   (~98 KB)
           lcnn_audio_int8.tflite         (~492 KB)
  constants/ model_hashes.json            SHA-256 of every .tflite
             natural_diff_histogram.json  KL baseline calibrated on real photos
             lcnn_audio_threshold.json    EER operating point of the LCNN
             meta_classifier.json         feature contract + decision threshold
```

`noCompress "tflite"` is set in `app/build.gradle.kts`; keep it, or
memory-mapped loading will fail.

## Integrity verification

At startup `HomeViewModel` runs `ModelIntegrityChecker.verifyAllModelsDetailed()`,
which recomputes SHA-256 over each `.tflite` and compares against
`model_hashes.json`. A mismatch surfaces as "INTEGRITY COMPROMISED" on the
home screen. Regenerate hashes after every model update:

```bash
python ml/scripts/generate_android_constants.py
```

## Privacy posture

- Manifest declares **no INTERNET permission**.
- `network_security_config.xml` ships an empty trust-anchor set, so even a
  bundled library cannot open HTTPS.
- `allowBackup=false` + data-extraction rules exclude scan history and
  voiceprints from cloud backups.

## Release checklist

1. `cd ml && make train-all export eval` (or reuse committed checkpoints)
2. `python scripts/generate_android_constants.py`
3. `python scripts/check_android_consistency.py` -> exit 0
4. `cd android && .\gradlew.bat testDebugUnitTest assembleRelease`
5. Verify "SYSTEMS ONLINE" badge after installing the release build.
