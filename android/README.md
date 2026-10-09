# ONN-DETECT Android

The app analyzes selected media with a standalone bundled CPU LiteRT runtime. Source manifest merger rules remove Internet and network-state permissions. No camera/microphone or broad library access is requested: the system picker supplies a grant for the selected file. A cloud file provider can use its own connectivity to download an input; select locally available media for fully disconnected use.

## Current detector status

The existing shipped model binaries lack reproducible training/calibration lineage. They are explicitly unvalidated. Results therefore show **Inconclusive**, while available individual detector scores are diagnostic measurements rather than calibrated AI probabilities. A completed scan or verified model hash does not prove detection accuracy or camera origin. Metadata markers are not cryptographic verification.

Images are bounded to 64 MiB/100 megapixels, oriented from EXIF, converted to sRGB with white alpha compositing, and decoded to a bounded working bitmap. Speech analysis evaluates sampled windows; video evaluates selected frames. Coverage is shown with the result. Music, environmental sounds, lip-sync and voice enrollment are not validated features.

Selected content is copied into private temporary storage before any decoder or metadata reader runs, with limits of 64 MiB for images, 128 MiB for audio and 512 MiB for video. The import has a 20-second deadline. Actual bytes enforce the limit even when a provider omits or misreports size. Cancellation signals the provider and closes owned streams/descriptors; partial imports are cleaned up. The complete scan has a 120-second timeout, and decoded bitmaps are recycled if cancellation prevents their delivery.

## Build and verification

Use JDK 17 and Android SDK 36. Versions are pinned: AGP 8.13.2, Gradle 8.13 with its official distribution SHA-256, Kotlin/Compose compiler 2.2.20, KSP 2.2.20-2.0.4, Hilt 2.57.2, standalone LiteRT 1.4.2. The app supports API 26+ and targets API 36. CPU inference is serialized by the engine. Unused CameraX, MediaPipe, GPU, Coil, DataStore and simulated biometric-storage dependencies were removed.

From this directory:

```powershell
.\gradlew.bat testDebugUnitTest assembleDebug lintDebug assembleRelease assembleDebugAndroidTest
python ../scripts/verify_android_artifact.py --apk app/build/outputs/apk/debug/app-debug.apk
```

On Linux use `bash gradlew` with the same tasks. Ordinary build dependency downloads require network; app inference does not. Release APKs are unsigned unless you configure your own signing credentials. A successful minified build does not substitute for release inference/UI smoke tests on a phone.

Room v3 exports schemas and explicitly migrates v1 to v2 to v3 without deleting saved history. Legacy assessments are retained and marked inconclusive. Latest 1,000 rows are shown, older rows remain until deletion. History stores assessment metadata, model/preprocessing/calibration identifiers and evaluated coverage; it does not store source media, URI, EXIF GPS or voiceprints. Deletion requires confirmation. Backup/device-transfer exclusions apply to the local database.

The current unit suite has 78 passing tests across 12 suites. Lint reports zero errors and 25 warnings, primarily dependency-update suggestions and KTX style recommendations. `HistoryMigrationTest` seeds actual old SQL and verifies preservation through Room on a connected emulator/device (`connectedDebugAndroidTest`).

A 12-test instrumentation run passed on an API 37 x86_64 emulator with 16 KiB pages: large-font navigation, missing/unreadable input, actual Room migrations, packaged-model/decoder checks, delayed-provider query timeout, blocked-pipe cancellation/cleanup, unknown-size byte limits and simulated Error propagation. A subsequent targeted saved-audio coverage UI regression passed after restricting sampled-frame wording to video. The final minified APK also cold-launched and completed a local 4-second audio scan through the system picker in airplane mode, retaining an inconclusive verdict and correct seconds coverage. These are runtime smoke checks, not accuracy or performance benchmarks. Logs and UI evidence are in `app/build/validation/`.

Remaining device verification includes representative arm64 phones, broader media/codec/provider formats, rotation/background/process death, landscape/TalkBack and sustained resource use. Real image/frontend parity and held-out detection accuracy still require an evaluated model release.

## Model release contract

Model assets, calibration JSON and generated fusion trees must be exported together from the ML pipeline, with hashes, tensor shapes/dtypes, frontend version and independent evaluation lineage. Use the root training/release documentation for the current commands. Never set validation flags merely to enable binary verdicts. A corrected high-resolution student requires training/export/runtime preprocessing parity and measured phone performance.

Tooling references: [AGP compatibility](https://developer.android.com/build/releases/agp-8-13-0-release-notes), [LiteRT standalone runtime versions](https://developers.google.com/edge/litert/android), [Compose compiler migration](https://kotlinlang.org/docs/compose-compiler-migration-guide.html), [16 KiB-page verification](https://developer.android.com/guide/practices/page-sizes).
