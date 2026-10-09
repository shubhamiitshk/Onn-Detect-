# Deployment

The source builds an offline diagnostic app. The existing trained weights and metric assets are explicitly **unvalidated**, so image/audio/video scans return an inconclusive assessment rather than a fabricated authentic/synthetic verdict. Building a release APK is not evidence that a detector release passed calibration, domain or phone validation.

## Android toolchain and build

Use JDK 17, Android SDK platform 36, and the checked-in Gradle wrapper. The app has `minSdk=26`, `targetSdk=36`, AGP 8.13.2, Gradle 8.13, Kotlin/Compose plugin 2.2.20, and standalone LiteRT CPU 1.4.2. Consult the checked-in Gradle files if versions change. The project caps Gradle workers for the supported 8 GiB development machine.

From the repository root on Windows PowerShell:

```powershell
$env:JAVA_HOME = 'C:/Program Files/Microsoft/jdk-17.0.20.101-hotspot'
Push-Location android
try {
    .\gradlew.bat testDebugUnitTest assembleDebug lintDebug assembleRelease assembleDebugAndroidTest
    if ($LASTEXITCODE -ne 0) { throw 'Android validation failed' }
} finally { Pop-Location }
```

On Linux/macOS use the appropriate local JDK 17 and `bash gradlew` from `android/`. Release builds are minified and resource-shrunk; configure signing credentials separately before distribution. No signing key is committed.

The debug APK is `android/app/build/outputs/apk/debug/app-debug.apk`. The locally unsigned release APK is `android/app/build/outputs/apk/release/app-release-unsigned.apk`. Keep `androidResources.noCompress += "tflite"`; mapping requires an uncompressed model entry.

## Check the actual APK

From the repository root:

```powershell
python scripts/verify_android_artifact.py --apk android/app/build/outputs/apk/debug/app-debug.apk
python scripts/verify_android_artifact.py --apk android/app/build/outputs/apk/release/app-release-unsigned.apk
```

The verifier locates SDK `aapt`/`aapt2`, or accepts an explicit `--aapt` path. It parses the APK's merged permissions, validates bundled hashes/contracts, and checks 64-bit ELF load segments plus ZIP alignment for 16 KiB pages. Reading only the source manifest is insufficient: dependencies can add permissions/native libraries.

For a distributable detector add `--require-validated-models`. **That strict check is expected to fail for the current legacy bundle.** Do not edit `validated` to bypass it. Native alignment success is necessary for 16 KiB loading but is not a substitute for installation and smoke tests on a 16 KiB device/emulator.

## Device tests

With a supported emulator or test phone connected:

```powershell
Push-Location android
try {
    .\gradlew.bat connectedDebugAndroidTest
    if ($LASTEXITCODE -ne 0) { throw 'Device tests failed' }
} finally { Pop-Location }
```

`PackagedInferenceTest` runs the actual bundled model interfaces, verifies that legacy image analysis abstains, and checks audio sampling/silence handling with a deterministic local WAV. History migration tests exercise the schema upgrade. These smoke tests contain no accuracy claims. No emulator or phone result should be called a modern-generator detection benchmark.

Before promotion, test real PNG/JPEG/WebP/HEIF orientation/color/alpha, native-resolution image preprocessing, real WAV/FLAC/AAC/MP3/Opus files and PCM encodings/rates/stereo, and video chroma stride/crop/rotation/color/vendor codecs. Verify unsupported HDR, empty/truncated files, deadlines, cancellation, repeated/concurrent scans, airplane-mode cold launch, and Java/native peak memory. Measure latency, energy and false-positive/coverage metrics on actual target phones. Match Python and Android tensors and final operating decisions using the same file bytes.

## Bundled asset layout

```text
android/app/src/main/assets/
  models/
    srm_forensic_cnn_int8.tflite
    lcnn_audio_int8.tflite
  constants/
    model_hashes.json
    release_manifest.json
    natural_diff_histogram.json
    lcnn_audio_threshold.json
    meta_classifier.json
```

The legacy assets contain about 98 KiB image and 492 KiB audio weights with float32 interfaces around quantized internals. Old thresholds/metrics are retained as historical unvalidated data; they are not a supported accuracy statement. No neural binaries were retrained or silently inverted by the correctness fixes.

A future complete bundle additionally contains its evaluation report, fusion booster/source/results, and independent device report. Every file must be declared and hash-bound. A raw model file plus an unrelated constants JSON is not an installable release.

## Train and stage a replacement

Use a separate prepared, licensed dataset and a suitable Python environment. Training cannot proceed from the repo alone: the reviewed dataset/checkpoint artifacts were not present. Large external training is compatible with offline Android inference.

Prepare dataset manifests with verified real=0/fake=1 labels, source revisions, byte hashes and original-photo/recording group IDs. Explicit `train`, `val`, `calibration`, `test` splits are required; visual fusion also needs independent `stacking` and `fusion_val`. See the training documentation for dataset preparation and pinned environments.

From `ml/`, after those prerequisites exist:

```powershell
python scripts/run_training_pipeline.py --visual-config configs/srm_cnn_genimage.yaml --audio-config configs/lcnn_asvspoof.yaml --device auto --output-dir runs/my-release --release-id my-release
```

This trains verified checkpoints, exports into a new staging directory, evaluates export parity and held-out scores, trains fusion from quantized base-model scores, attaches measured constants, and validates the staged bundle. It does **not** replace Android assets. Output remains `validated:false` until independent device evidence is attached.

Inspect a complete staged bundle from the repository root:

```powershell
python ml/scripts/release_bundle.py --bundle ml/runs/my-release/bundle --validate-only
```

The bundle's device report must identify the same release and model/frontend hashes, bind source_snapshot_sha256 and fusion_kotlin_sha256 to the staged source snapshot and Kotlin artifact, pass tensor/model/fusion parity, and record offline verification. A repeated release ID cannot substitute for matching source bytes. Supply independently measured evidence rather than constructing a report to force a flag:

```powershell
python ml/scripts/release_bundle.py --bundle ml/runs/my-release/bundle --device-report /absolute/path/to/device_evaluation.json --require-validated
```

No such replacement accuracy/device report was produced by the code fixes. General-audio speech/music routing and video-level calibrated aggregation remain separate requirements; an image/speech model bundle cannot unlock those unsupported promises.

## Install a validated bundle into the project

The installer requires the full bundle and the matching compiled fusion destination. The root staging manifest moves to `assets/constants/release_manifest.json`; hashes still refer to the same artifact bytes. Previous assets are retained in a recoverable sibling directory.

From the repository root:

```powershell
python ml/scripts/release_bundle.py --bundle ml/runs/my-release/bundle --install android/app/src/main/assets --kotlin-destination android/app/src/main/java/com/onndetect/detection/fusion/XGBoostMetaClassifier.kt
```

On a Bash host the wrapper supplies those destinations:

```bash
./scripts/package_for_android.sh /absolute/path/to/validated/bundle
```

The wrapper never copies a subset of whichever models happen to exist. Installation validates first, stages a full copy, pairs the compiled Kotlin source with the release, and retains prior assets for rollback. Use a new release ID rather than overwriting a prior backup.

Rebuild, rerun unit/instrumented checks, then inspect the actual final release APK with `--require-validated-models`. Sign and install that exact release artifact, repeat offline/device checks, and record its APK/model/release hashes with the evaluation results. Publish operating-point metrics and supported generator/language/media coverage; do not promise detection of every AI-generated file.

## Offline and local-storage checks

The manifest explicitly removes `INTERNET` and `ACCESS_NETWORK_STATE` added by transitive libraries. Unused MediaPipe, transport and GPU components are removed; the standalone runtime and model files are bundled, with no first-run network model download. The APK verifier enforces the permission invariant.

`allowBackup=false` and extraction rules exclude local scan data. Network security configuration alone does not block every possible socket or replace permission checks. Test final APK permissions and cold first-run operation in airplane mode. The current implementation does not perform cryptographic C2PA/SynthID attestation or biometric voice identity enrollment.
