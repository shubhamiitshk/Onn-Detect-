# Android application review — 6 October 2026

This review covers the application, screens, persistence, configuration, resources, permissions, and deployment behavior in the current working tree. Existing source changes were preserved. Detection mathematics, training, and datasets are examined in the companion reviews; the detection files referenced here were read to trace their user-facing effects.

## Scope and verification

- Read all **45 assigned Android source/configuration text files**, listed individually below, including the empty ProGuard file and both complete Gradle launcher scripts.
- Read **8 additional Android detection/domain/inference files** completely to follow the result contract: `ScanResult.kt`, `SignalFusion.kt`, `DetectionPipeline.kt`, `AudioClassifier.kt`, `VideoForensicAnalyzer.kt`, `LiteRTInferenceEngine.kt`, `ModelIntegrityChecker.kt`, and `ModelIntegrityGate.kt`.
- Read **13 ignored IDE configuration files** and **2 ignored local configurations** (`android/local.properties`, `android/.gradle/config.properties`). That is **68 Android text files read in full**, of which 45 are this review's nonoverlapping source/configuration scope. `CLAUDE.md` was also read for repository context.
- Inspected the wrapper JAR's size/hash, freshly generated merged debug/release manifests, manifest merger provenance, resolved dependency provenance, JUnit results, and the full lint text report. Parsed ELF program headers for all **15 packaged native libraries**. Generated build/cache files and the IDE device-streaming cache are not maintained source and were not represented as individually reviewed source files. Model binaries/constants and unit-test source are assigned to the companion detection review.
- Android SDK platforms 34, 35, 36, and 36.1 were available. The initial command inherited Android Studio's JBR and failed `JdkImageTransform`; explicitly selecting the installed Microsoft **JDK 17** resolved that environment problem. An offline lint attempt needed an uncached lint dependency; the final normal Gradle invocation downloaded ordinary build dependencies and completed.
- Final fresh command: `gradlew.bat -Dorg.gradle.java.home=C:/Program Files/Microsoft/jdk-17.0.20.101-hotspot --no-daemon testDebugUnitTest assembleDebug lintDebug processReleaseMainManifest --console=plain --rerun-tasks`.
- **BUILD SUCCESSFUL**: 56 tasks executed in 2 minutes 57 seconds. **42 JUnit tests**, across six suites, with **0 failures, 0 errors, 0 skipped**. Debug APK built: **88,959,750 bytes**, approximately **84.84 MiB**. **Lint: 0 errors, 35 warnings**.
- These checks establish compilation and the existing test outcomes. They do **not** establish detection accuracy, release-device runtime correctness, calibrated confidence, UI accessibility, or performance on real phones. No release APK/device instrumentation was run.

References below use repository-relative paths with one-based line numbers. The checkout root is `C:/Users/sumit/OneDrive/Desktop/Onn-Detect-/`.

## Highest-priority changes

### 1. Enforce the offline promise in the merged artifact, not only in the source manifest

**Priority: P1 — privacy/release blocker.**

The source manifest says “NO INTERNET PERMISSION,” but both freshly generated **debug and release** manifests include `android.permission.INTERNET` and `ACCESS_NETWORK_STATE`. The dependency chain is verified, rather than inferred:

`app -> com.google.mediapipe:tasks-vision:0.20230731 -> tasks-core -> com.google.android.datatransport:transport-backend-cct:3.1.0`.

Evidence:

- `android/app/src/main/AndroidManifest.xml:5` — the claimed no-Internet policy.
- `android/app/build.gradle.kts:105` — direct MediaPipe dependency.
- `android/app/build/intermediates/merged_manifest/release/AndroidManifest.xml:28` and `:29` — network permissions in the release manifest.
- `android/app/build/outputs/logs/manifest-merger-debug-report.txt:513` — Internet permission added by transport-backend-cct.
- Successful `:app:dependencyInsight --dependency transport-backend-cct --configuration debugRuntimeClasspath` — confirms the MediaPipe chain.

No current production source imports or uses MediaPipe. Its dependency remains after the landmark/lip-sync implementation was removed. The network-security resource blocks cleartext and configures no trusted CAs, but that is not an operating-system prohibition on sockets and is not a substitute for removing Internet permission. This finding proves the permission promise is currently false; it is **not evidence that media has actually been uploaded**.

Remove unused MediaPipe. Add manifest merger removal rules for Internet/network-state permissions using `tools:node="remove"` so future dependencies cannot silently reintroduce them. Check the packaged release APK/AAB permissions in CI. Verify real scans in airplane mode with external connectivity unavailable. Keep the no-backup rules. Correct `android/README.md:3` and the equivalent marketing claims until the artifact enforces them. Android documents network security configuration separately from permissions: [Network security configuration](https://developer.android.com/privacy-and-security/security-config).

### 2. Introduce an explicit inconclusive/error result instead of manufacturing authenticity

**Priority: P1 — correctness and trust.**

`DetectionPipeline.kt:197` creates an aborted scan as a normal `ScanResult` with score `0.5`, `isAiGenerated=false`, and confidence zero. `ScannerViewModel.kt:77` records **any** emitted result and navigates to it. `ResultScreen.kt:55` and `HistoryScreen.kt:72` render every false Boolean as “Likely Authentic.” A phone with insufficient memory therefore receives a recorded authentic verdict for an analysis that never happened.

The same binary contract cannot express unsupported files, unknown generators, too-short speech, no usable frames, model failure, or contradictory modalities. It also makes the UI unable to show the video domain-shift caveat documented inside `VideoForensicAnalyzer.kt:28`.

Replace the Boolean-only decision with a verdict/status such as `LIKELY_SYNTHETIC`, `NO_STRONG_SYNTHETIC_EVIDENCE`, `INCONCLUSIVE`, `UNSUPPORTED`, `FAILED`, and `CANCELLED`. Persist the reason and evaluated modalities. The pipeline should return an abstention for the memory gate, and the scanner should save/navigate it as an inconclusive result only if history policy includes failed attempts. Do not map “no strong evidence of synthesis” to proof of camera origin.

The audio model failure path in `AudioClassifier.kt:41` returns `0.5` as a normal score; the shipped decision threshold is below 0.5, so a failed inference can become “AI generated.” The companion detection review covers this in depth. The UI must carry failure/availability separately from numeric scores.

### 3. Stop presenting raw scores as calibrated AI probabilities

**Priority: P1 — accuracy interpretation.**

- `ConfidenceMeter.kt:40` labels the raw model/fusion score “AI-generation probability.”
- `HistoryScreen.kt:79` repeats “AI probability.”
- `ScannerViewModel.kt:212` defines audio confidence as `abs(score - 0.5) * 2`.
- `DetectionPipeline.kt:217` and `VideoForensicAnalyzer.kt:138` use the same distance heuristic.
- `ResultScreen.kt:72` displays `score` and does not use the separately persisted `confidence` value.

Distance from 0.5 is not measured predictive confidence. It is especially misleading when a calibrated operating threshold differs from 0.5, or when the video result is the maximum of two scores. A raw score of 0.25 might pass a synthetic threshold while the heuristic calls it moderately confident in the opposite direction. Changing the label alone does not fix reliability.

Until independent calibration is measured, call this a **detector score**, show an inconclusive band, and explain that the model has limited validated generator coverage. Fit calibration on a separate, representative calibration partition, assess reliability/Brier score/ECE and false-positive rates, and calibrate modality-specific outputs before any cross-modal fusion. Persist calibration/model version, operating threshold, decision policy, and coverage warnings with every result. For video, show separate visual and audio decisions, the sampled duration/frames, and the aggregate policy.

### 4. Modernize Android deployment and replace incompatible native binaries

**Priority: P1 for distribution/device support.**

`android/app/build.gradle.kts:10` and `:15` set compile/target SDK 34. As of this review, Google's official requirement says new mobile apps and updates must target **API 36 or higher from 31 August 2026**. Existing apps targeting API 34 have availability restrictions. [Google Play target requirements](https://developer.android.com/google/play/requirements/target-sdk).

The project uses AGP 8.1.4, Gradle 8.4, Kotlin 1.9.0, TFLite 2.14.0, and old CameraX/MediaPipe prebuilts. Direct inspection of **every one of the 15 `.so` files** in the freshly merged debug native library directory found all `PT_LOAD` segments aligned at **4,096 bytes**, including all arm64 TFLite, GPU, MediaPipe, and CameraX JNI libraries. These artifacts have not been rebuilt/aligned for native 16 KiB-page support. Do not assume merely changing `targetSdk` fixes native compatibility. Android may provide compatibility modes on some versions, but these are not a sound deployment target.

Upgrade to a compatible, supported AGP/Gradle/JDK/Kotlin/Compose combination, compile and target API 36+, and replace/rebuild all remaining native prebuilts with versions verified to support 16 KiB pages. Remove unused MediaPipe and CameraX first to reduce the compatibility burden. Android recommends AGP **8.5.1+** together with compatible prebuilts; its current page lists the release restriction starting **1 February 2027**. Recheck the current policy rather than copying older November 2025 deadlines. [Android 16 KiB-page guide](https://developer.android.com/guide/practices/page-sizes).

Validate ELF alignment **and** APK/AAB ZIP alignment, run on an arm64 16 KiB emulator/device, and test both CPU and delegate inference with the shipped model graph. Check Android 15/16 edge-to-edge/insets, photo picking, back navigation, and lifecycle behavior after the SDK upgrade. Keep minSdk 26 if that remains the intended supported audience; it is independent of the target SDK.

## Further concrete findings and improvements

### Signals need typed availability; the current UI prints missing values as measurements

**Priority: P2.** `SignalBreakdown.kt:19` through `:21` display SRM/FFT/KL unconditionally. Audio scans initialize those values to `-1` in `ScannerViewModel.kt:215`, so the UI displays `-1.000` as a measured score. `SignalBreakdown.kt:24` through `:26` treat any nonzero audio/temporal value as available, which means audio DSP `-1` is printed, video without audio can print LCNN `-1`, and a legitimate score of exactly zero disappears. The JSON decoder defaults missing fields to `-1` (`SignalBreakdownJson.kt:31`), making legacy/corrupt schema handling inconsistent across rows.

Use nullable scores or a sealed signal result containing `AVAILABLE`, `NOT_APPLICABLE`, `UNSUPPORTED`, `FAILED`, and a measured value. Validate finite values and semantic range before rendering; allow legitimate zero. Hide irrelevant rows by scan mode or show “not evaluated.” Include provenance as verified/unverified/absent rather than setting `provenanceScore` to the overall score (`ScannerViewModel.kt:157`, `:222`). Rename the PRNU row: the current analyzer estimates a residual-noise proxy and its own caveat says it is not camera-fingerprint PRNU. Avoid leaving absent MobileCLIP/frequency-perturbation features looking like supported detectors.

### Remove the simulated enrollment and unimplemented lip-sync claims from the user flow

**Priority: P2.** `VoiceEnrollmentScreen.kt:21` only runs a timer; it never records audio or calls `SpeakerVerifier`. A visible implementation warning at `:55` correctly acknowledges this, but the screen still offers “HOLD TO RECORD” (`:90`) and ultimately states “IDENTITY SECURED” (`:72`). The button starts on click; it is not a hold gesture. No microphone permission is declared because no recording occurs.

Remove/disable the enrollment entry in `HomeScreen.kt:137` for the usable app. If later implementing enrollment, separate **speaker identity** from **liveness/anti-spoofing**, provide a random prompted challenge, record with just-in-time microphone permission, reject silence/poor-quality audio, store biometric templates through a real Keystore-backed design, and provide deletion. A speaker embedding alone does not prove a live speaker.

`HomeScreen.kt:120` calls video “Cross-Modal Forensics” and `:121` promises “Lip-Sync.” No actual lip-sync detector remains in the current app; frame statistics plus embedded-audio classification do not establish lip synchrony. Rename this to video/audio screening, and expose the measured coverage limitation. Keep code TODOs and model names out of end-user text.

### Simplify permissions to the selected-file access actually needed

**Priority: P2.** `MainActivity.kt:32` requests camera and all broad media permissions at launch; the result callback (`:25`) ignores denial. The current app uses `PickVisualMedia`/`OpenDocument` in `HomeScreen.kt:29`, `:37`, and `:45`. Those selected-file flows can provide access to the chosen item without asking to read the entire library. No production camera capture path exists. The pre-Android-13 branch requests `READ_EXTERNAL_STORAGE` (`MainActivity.kt:65`), but the manifest does not declare it. That prompt is ineffective and unnecessary for the selected-document route.

Remove startup permission requests and unnecessary manifest permissions. Request camera/microphone only at a future capture/recording action, with useful denial handling. If re-scanning/resuming selected documents across device restarts is a real product feature, take/release persistable URI grants and associate them with a user-selected item; do not retain every grant by default. Cloud document providers can serve a selected file through their own network process even when this app has no Internet permission: clearly distinguish local-only selection from on-device analysis and handle unavailable remote content. [Storage Access Framework permissions](https://developer.android.com/training/data-storage/shared/documents-files).

### Add cancellation, lifecycle-aware state, and complete scanner/error states

**Priority: P2.** `ScannerViewModel.kt:49`, `:118`, and `:183` start `viewModelScope` launches without keeping an active Job or exposing a cancel method. `ScannerScreen.kt:72` shows progress but provides no cancel action. Blocking audio/video decode, feature loops, and inference can outlive navigation until they cooperate with cancellation. A second scanner instance can then touch shared singleton interpreters while the first worker is still active; the companion detection review covers interpreter synchronization.

Keep a scan Job/session ID, provide cancel, rethrow `CancellationException` rather than catching all `Throwable` (`ScannerViewModel.kt:108`, `:174`, `:233`), and add cancellation checks in decode/analysis loops. Release bitmaps/resources in `finally` when the scan owns them. Serialize inference access or use properly isolated interpreters. Save request state with `SavedStateHandle` when resumability is desired and avoid silently restarting a failed scan when the screen re-enters.

Reset all progress fields when starting a scan, and make loading/decoding/analysis/saving/error/completion explicit rather than combining stage strings with Boolean flags. Use `collectAsStateWithLifecycle` across screens. Show supported file kinds, cancellable import progress, actionable error reasons, and retry/reselect controls. `ModelIntegrityGate.isOk` runs inside the audio/video launch on the main dispatcher (`ScannerViewModel.kt:124`, `:189`); move its first hash read to IO and share a readiness result with Home, which currently performs its own separate verification (`HomeViewModel.kt:27`). Model failure should show a recoverable state rather than a raw exception or mislabeled verdict.

### Preserve source geometry and validate real inputs before running a detector

**Priority: P2.** The image decoder bounds memory by downsampling the long edge to approximately 1024 pixels (`ScannerViewModel.kt:245`–`:270`), a useful protection. It correctly preserves original minimum-edge information for one metadata feature. However, it does not apply EXIF rotation/mirroring, explicitly canonicalize color/HDR/alpha behavior, or preserve native-resolution crops for forensic traces. The compression feature relies on a nullable provider MIME string (`:67`) despite the bounds decoder having its own MIME observation. No preview confirms the selected input, and no file/geometry/decode-time budget is surfaced.

Build one shared, documented preprocessing contract with the training pipeline: orientation, RGB/color conversion, alpha handling, resize interpolation, and patch sampling. Use original-resolution crops alongside a global view if the next trained model requires them; do not simply raise decode size and memory. Validate actual content and dimensions, cap import bytes/time, support or clearly reject HEIC/AVIF/WebP/GIF and malformed media, and handle provider I/O/security failures. Test native-resolution camera photos and messaging-compressed media. Log content URI identifiers only in debug builds (`ScannerViewModel.kt:253` currently logs them unconditionally), and show a user-friendly message rather than echoing a raw URI (`:61`).

### Persist the evidence needed to interpret and reproduce a scan

**Priority: P2.** `ScanHistoryRepository.kt:19` and `ScanHistoryEntity.kt:12` keep score, Boolean decision, heuristic confidence, duration, mode, and signal JSON. They drop quality information, provenance details, thresholds, evaluated frame/audio windows, model hashes/version, preprocessing version, decision status, and warnings. `ResultScreen.kt:80` always passes a null heatmap, and `HeatmapViewer.kt:25` only prints a cell count even if a heatmap exists. The user never sees the quality findings already measured by the pipeline. A previous score cannot be interpreted reliably after a model/calibration update.

Persist a versioned result schema: model/calibration/preprocessing identifiers; decision status and threshold; modality availability; quality gates; evaluated coverage; and provenance verification status. Show resolution/compression, speech duration/window coverage, frame coverage, and warnings that actually affect interpretation. Keep raw media excluded by default. If a local report/export is added, make it user-initiated and explain that a detector assessment is not verified provenance. Preserve a computed result if Room saving fails and offer retry instead of discarding it inside the scanner's broad catch.

### Replace destructive migrations and make history useful at scale

**Priority: P2.** `ScanHistoryDatabase.kt:14` disables schema export and `:28` uses `fallbackToDestructiveMigration`. Existing logs are erased on a schema version change even though the UI presents them as a forensic log. Adding the nullable `signals_json` column is a straightforward migration. The “ephemeral” rationale in the comment is a product assumption, not a visible retention policy.

Export Room schemas, implement explicit migrations, and test upgrades using seeded history. Add a timestamp index and Paging or bounded retention: `ScanHistoryDao.kt:15` loads the entire ordered table every update. `HistoryScreen.kt:60` passes rows to a nonclickable Card (`:69`), and the `NavController` parameter is unused; users cannot open a saved result. Wire row selection to `result/{id}`, show scan mode and status, and support per-item delete using the repository method already present. “Clear” (`HistoryScreen.kt:43`) immediately deletes everything; use a reversible undo/confirmation appropriate to a forensic log and an error state if deletion fails.

`ResultViewModel.kt:23` uses null for both loading and not-found and does not catch repository errors. `ResultScreen.kt:47` therefore says “Loading result…” forever for a missing/deleted/invalid ID. Model the states `Loading`, `Ready`, `NotFound`, and `Error` explicitly and provide a useful destination from each. The `scanId` screen parameter is unused because the VM retrieves it from `SavedStateHandle`; remove duplicate contracts or use typed navigation consistently.

### Make screens scroll, respect system insets, and fix low-contrast text

**Priority: P2.** Home, Result, Scanner, and Enrollment use fixed full-screen Columns with padding and no scrolling or safe-drawing insets. Home's stacked cards/buttons and Result's many signal rows will overflow small/landscape screens and enlarged text. The lower Result “Done” action may be unreachable. Use scrolling/adaptive layouts, `Scaffold`, safe-drawing/system-bar insets, and suitable widths on tablets/foldables. Keep actions accessible at 200% text size and in landscape. Verify TalkBack reading order and accessible progress/state announcements.

`Color.kt:17` defines outline text as `#38435E`, used for timestamps, duration, empty history, unavailable scores, and heatmap explanations. Calculated WCAG contrast is only **1.96:1** on background `#0A0E17` and **1.76:1** on surface `#151A28`. Use a text token with adequate contrast rather than a border-outline token. The normal secondary text `#8B9BB4` has approximately 6.84:1/6.15:1 on those surfaces. Add semantic labels tying detector score to verdict/availability; text plus color already helps, but progress should explain whether it is a measured fraction or a stage estimate.

### Localize the existing text and clean the theme contract

**Priority: P3.** `strings.xml` contains eight useful strings, but the screens hardcode nearly every user-facing label and error, and the manifest hardcodes the app label (`AndroidManifest.xml:20`). Lint reports all string resources unused. Move strings and formatted quantities into resources, add plurals, and test Hindi/other intended languages and RTL. Numeric `.format`/DateFormat naturally reflect device locale, but sentence templates remain English.

`Theme.kt:49` and `:50` accept dark/dynamic flags but ignore both. The dark palette itself is coherent; either expose a real theme preference/use the system setting or remove the unused API and be explicit about the design. The XML launch theme is light while Compose is forced dark, which can create an inconsistent launch appearance. Use appropriate launch/system-bar behavior. Avoid casting arbitrary `LocalView.context` directly to Activity without a safe context-unwrapping strategy if reuse outside this Activity is intended. Add monochrome adaptive icon layers for themed launchers; both icon XMLs are flagged by lint.

### Trim dependencies and strengthen the release verification path

**Priority: P2 for footprint/runtime; P3 for hygiene.** No current production imports use CameraX, MediaPipe, Coil, or DataStore. Removing unused dependency families reduces package size, initialization code, native compatibility risk, and the MediaPipe network permission issue. The measured debug universal APK is 84.84 MiB; this is **not** the per-device release download size. arm64 MediaPipe JNI alone is **13,351,848 bytes** uncompressed, and the x86 copy is **20,570,284 bytes**. Use release AAB/per-ABI delivery when appropriate, and measure actual release installation size, cold/warm latency, peak RSS, battery use, and thermal behavior on intended phones before choosing model size.

Do not add blanket ProGuard keep rules merely because `proguard-rules.pro` is empty. Dependency consumer rules may already be sufficient. Run minified-release inference/UI smoke tests to discover any real shrinker issue. Verify offline asset availability and delegate fallback on low/mid/high device tiers. Cache models according to measured process budgets; the DI device-tier heuristic in `AppModule.kt:26` uses total system RAM and one heap budget check, so it should not replace direct latency/memory measurements. Provide supported-device diagnostics without exposing technical implementation details in normal user flows.

The lint run found an **obsolete Compose runtime lint registry**, causing **14 Compose checks to be skipped**. A successful lint exit is therefore incomplete coverage; align AGP/lint/Compose versions. Lint's `StaticFieldLeak` on the VM's `@ApplicationContext` is likely a false positive: this field is application context, not Activity context. The two `UnusedAttribute` findings are normal compatibility behavior, not correctness failures. Do not blindly fix every warning by raising minSdk.

Pin compatible dependencies in one version catalog, use Gradle dependency verification/locking where useful, and add the distribution SHA-256 to wrapper properties. The wrapper JAR inspected here is 63,721 bytes with SHA-256 `0336f591bc0ec9aa0c9988929b93ecc916b3c1d52aed202c7381db144aa0ef15`; this records the local artifact, not a claim of official checksum verification. The POSIX wrapper has Git mode `100644`; the root orchestration review covers the Linux CI execution failure and executable-bit fix. Keep SDK/JDK machine paths and IDE/cache files ignored. The current Security Crypto dependency is a 2023 alpha; its APIs were later deprecated in favor of platform APIs/direct Android Keystore use, so a real future biometric storage feature should be designed accordingly. [AndroidX Security release notes](https://developer.android.com/jetpack/androidx/releases/security).

## Recommended user-facing result design

Show one assessment with a clearly visible status: likely synthetic, no strong synthetic evidence, or inconclusive. Under it, show the detector score only if meaningful, the evaluated modalities, source quality, and limited coverage. For video, separate image-frame and audio decisions and identify when either modality was unavailable. For audio, show how much speech was actually assessed. If quality is too low or inference fails, explain the specific reason and offer a better source/retry. Provenance should be a separate verified/unverified/absent field; a detector should not imply cryptographic authenticity.

Make the app's usable flow “select local media -> preview -> analyze on device with cancel -> assessment + limitations -> optional local history.” Remove the enrollment simulation and unavailable lip-sync/XAI promises. Keep detailed model names and debugging stage terminology behind an optional technical details view.

## Meaningful next tests

The existing 42 tests validate signal/feature math; no maintained `src/androidTest` files were present despite instrumentation dependencies. The most valuable additions are:

1. End-to-end device tests using the **packaged TFLite assets** with known inputs and expected decision/status, compared to export golden outputs on CPU and supported delegates.
2. Scanner-state tests for decode failure, inference failure, missing hash manifest, memory abstention, cancellation, Room failure, and repeated/rotated screen entry. These must prove failures never become authentic/synthetic verdicts.
3. Room migration and JSON round-trip/schema evolution tests, including legitimate zero, missing values, and rejected NaN/infinity.
4. Compose tests for history-row navigation, clear/undo, missing result, inconclusive assessment, screen-reader semantics, large font, and small/landscape layouts.
5. Artifact checks for release merged permissions, required assets/hashes, 16 KiB ELF/ZIP alignment, and minified-release inference smoke tests.
6. Performance/airplane-mode checks on actual target phones: startup, cold/warm inference, media-provider import, peak memory, cancellation latency, thermal throttling, and sustained scan batches. Measure these before selecting larger on-device models.

## Exhaustive assigned per-file coverage

Every file in the following 45-row table was read completely. Entries with no independent defect still record the relevant validation/improvement.

| File | Coverage and conclusion |
|---|---|
| `android/settings.gradle.kts` | Read repository/plugin resolution and module setup; standard Google/Maven Central setup. Add verification/locking as needed. |
| `android/build.gradle.kts` | Read all plugin versions; AGP/Kotlin/Hilt/KSP need a coordinated compatibility upgrade. |
| `android/gradle.properties` | Read JVM, cache, parallelism, AndroidX flags; comment for Gradle 9.4 tooling does not describe wrapper 8.4. Keep properties appropriate to actual version. |
| `android/gradle/wrapper/gradle-wrapper.properties` | Read distribution URL and download policy; Gradle 8.4; add official distribution checksum when upgrading. |
| `android/gradlew` | Read complete POSIX launcher; standard generated script. Git executable mode missing; root review covers CI blocker. |
| `android/gradlew.bat` | Read complete Windows launcher; worked with explicit JDK17; inherited JAVA_HOME can select incompatible JBR. |
| `android/app/build.gradle.kts` | Read every build option and dependency; target API, native compatibility, unused dependency, release validation, and lint alignment findings above. |
| `android/app/proguard-rules.pro` | File is empty; avoid unsupported blanket rules; validate the minified release. |
| `android/README.md` | Read all claims/build instructions/architecture; no-Internet promise contradicted by merged artifact, outdated provenance/lip-sync descriptions. |
| `android/app/src/main/AndroidManifest.xml` | Read complete component/permission/backup policy; selected-file app needs fewer permissions; explicitly remove merged Internet; app label should use resources. |
| `android/app/src/main/java/com/onndetect/MainActivity.kt` | Read full startup/permissions/Compose flow; remove unused startup requests, implement feature-specific denial handling. |
| `android/app/src/main/java/com/onndetect/OnnDetectApplication.kt` | Read debug-only StrictMode policies; useful leak/main-thread diagnostics; no independent blocker. |
| `android/app/src/main/java/com/onndetect/di/AppModule.kt` | Read singleton/context/database/device-tier providers; app context qualification is appropriate; use measured performance/heap budgets for tiers. |
| `android/app/src/main/java/com/onndetect/di/PipelineModule.kt` | Read all analyzer/loader/engine providers; singleton ownership increases importance of inference serialization and cancellation. |
| `android/app/src/main/java/com/onndetect/data/local/ScanHistoryEntity.kt` | Read full schema; lacks status/model/calibration/quality/provenance/coverage metadata. |
| `android/app/src/main/java/com/onndetect/data/local/ScanHistoryDao.kt` | Read all SQL operations; unbounded list/no timestamp index; replace-all insertion semantics acceptable for UUID IDs but history paging/retention needed. |
| `android/app/src/main/java/com/onndetect/data/local/ScanHistoryDatabase.kt` | Read initialization/version policy; destructive migrations and disabled schema export should be replaced. |
| `android/app/src/main/java/com/onndetect/data/local/SignalBreakdownJson.kt` | Read complete encoder/decoder; fallback -1 sentinels inconsistent with UI; version and finite-value validation needed. |
| `android/app/src/main/java/com/onndetect/data/repository/ScanHistoryRepository.kt` | Read recording/retrieval/deletion; drops evidence fields; useful per-item deletion method currently unused in UI. |
| `android/app/src/main/java/com/onndetect/ui/navigation/NavGraph.kt` | Read all routes/defaults/arguments; URI encoding is supplied by callers; move to typed routes where practical and validate mode/scan ID. |
| `android/app/src/main/java/com/onndetect/ui/screens/home/HomeViewModel.kt` | Read integrity check/state; separate from gate cache; readiness/error detail and shared IO verification preferable. |
| `android/app/src/main/java/com/onndetect/ui/screens/home/HomeScreen.kt` | Read every picker/card/entry; permission-minimal selected-file flow, unsupported lip-sync/enrollment claims, no scroll/insets, hardcoded text. |
| `android/app/src/main/java/com/onndetect/ui/screens/scanner/ScannerViewModel.kt` | Read all image/audio/video/result/decoder logic; false abort persistence, raw confidence, missing availability, main-thread hash check, cancellation/resource and preprocessing issues. |
| `android/app/src/main/java/com/onndetect/ui/screens/scanner/ScannerScreen.kt` | Read scan start/completion effects and all branches; no cancel, retry/resume state weakness, only image wording, no scroll/insets. |
| `android/app/src/main/java/com/onndetect/ui/screens/result/ResultViewModel.kt` | Read saved scan ID/load flow; null cannot distinguish Loading/NotFound/Error. |
| `android/app/src/main/java/com/onndetect/ui/screens/result/ResultScreen.kt` | Read verdict/meter/breakdown/heatmap/fallback/actions; false authenticity, missing quality/provenance, hardcoded null heatmap, no scroll. |
| `android/app/src/main/java/com/onndetect/ui/screens/history/HistoryViewModel.kt` | Read state flow and deletion; add loading/error/delete feedback and bounded query. |
| `android/app/src/main/java/com/onndetect/ui/screens/history/HistoryScreen.kt` | Read history/empty/clear/row presentation; rows cannot open result, raw probability label, irreversible clear UI, low-contrast detail text. |
| `android/app/src/main/java/com/onndetect/ui/screens/enrollment/VoiceEnrollmentScreen.kt` | Read entire timed simulation; disclosure exists but IDENTITY SECURED/recording claims remain misleading; remove usable menu entry. |
| `android/app/src/main/java/com/onndetect/ui/components/ConfidenceMeter.kt` | Read numeric/bar/color logic; raw score is not probability; clamp doesn't reject nonfinite values; update deprecated progress overload. |
| `android/app/src/main/java/com/onndetect/ui/components/SignalBreakdown.kt` | Read all labels/availability filters/rendering; -1/zero sentinel bugs; PRNU labeling; typed modality-aware results needed. |
| `android/app/src/main/java/com/onndetect/ui/components/HeatmapViewer.kt` | Read placeholder; not a heatmap renderer even with data; disable feature promise until real export/display exists. |
| `android/app/src/main/java/com/onndetect/ui/components/ScanProgressIndicator.kt` | Read determinate/indeterminate display; useful basic stage progress, but no cancellation/semantic description and deprecated overload. |
| `android/app/src/main/java/com/onndetect/ui/theme/Color.kt` | Read all color tokens; measured outline-text contrast failure; normal text colors otherwise plausible. |
| `android/app/src/main/java/com/onndetect/ui/theme/Theme.kt` | Read all palette/window/typography code; ignored theme flags, unsafe context cast for reuse, launch/insets modernization. |
| `android/app/src/main/java/com/onndetect/ui/theme/Typography.kt` | Read typography overrides; normal body/title choices; verify 11sp labels and font scaling with contrast/accessibility. |
| `android/app/src/main/res/xml/network_security_config.xml` | Read empty trust anchors/cleartext policy; policy is defense in depth, not no-Internet permission enforcement. |
| `android/app/src/main/res/xml/backup_rules.xml` | Read all legacy exclusions; good exclusion of database/preferences/external data; verify packaged policy. |
| `android/app/src/main/res/xml/data_extraction_rules.xml` | Read cloud/device-transfer exclusions; useful privacy policy; verify if new storage domains/features are added. |
| `android/app/src/main/res/values/themes.xml` | Read launch/status-bar XML; light launch theme differs from forced-dark Compose; modernize with SDK upgrade. |
| `android/app/src/main/res/values/strings.xml` | Read all strings; currently unused; replace screen/manifest hardcoded text and localize. |
| `android/app/src/main/res/values/colors.xml` | Read launcher/status colors; scan colors unused/duplicate code tokens; optional cleanup. |
| `android/app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml` | Read adaptive icon; add monochrome layer; v26 qualifier redundant at minSdk26. |
| `android/app/src/main/res/mipmap-anydpi-v26/ic_launcher_round.xml` | Read adaptive round icon; same monochrome improvement and redundant version qualifier. |
| `android/app/src/main/res/drawable/ic_launcher_foreground.xml` | Read complete vector/path geometry; no detection/security defect; reusable monochrome-compatible glyph possible. |

## Additional configuration/artifact coverage

The 13 ignored IDE files read were `.idea/.gitignore`, `.name`, `AndroidProjectSystem.xml`, `compiler.xml`, `deploymentTargetSelector.xml`, `deviceManager.xml`, `gradle.xml`, `markdown.xml`, `migrations.xml`, `misc.xml`, `runConfigurations.xml`, `vcs.xml`, and `workspace.xml`. They contain IDE settings, local JDK/emulator choices, and run state; no application behavior should depend on them. `local.properties` contains the local Android SDK path; `.gradle/config.properties` contains an IDE-local Java path. Preserve them as local artifacts, not portable build requirements.

Binary inspection covered the Gradle wrapper JAR and the following native artifact families: `libtensorflowlite_jni.so` (four ABIs), `libtensorflowlite_gpu_jni.so` (four), `libimage_processing_util_jni.so` (four), and `libmediapipe_tasks_vision_jni.so` (arm64-v8a, armeabi-v7a, x86). All fifteen showed 4 KiB PT_LOAD alignment. No x86_64 MediaPipe library was present in this merged set.

JUnit suite counts: FFT 4, FrameCosineDelta 6, KLCooccurrence 6, LFCCExtractor 10, PcmUtilResample 11, SignalFusion 5. Lint warning counts: GradleDependency 17, UnusedResources 10, UnusedAttribute 2, MonochromeLauncherIcon 2, OldTargetApi 1, ObsoleteSdkInt 1, ObsoleteLintCustomCheck 1, StaticFieldLeak 1. Generated reports remain under `android/app/build/reports/`; the debug APK remains under `android/app/build/outputs/apk/debug/`.

## Practical sequencing

1. Enforce merged-artifact offline permissions, remove misleading unavailable features, add explicit failed/inconclusive statuses, and fix signal availability.
2. Make the validated existing pipeline trustworthy end to end: packaged-model tests, typed result metadata, preprocessing parity, cancellation, calibration labels, quality/coverage display, database migrations.
3. Upgrade Android tooling/target/native dependencies and validate minified release, airplane mode, 16 KiB support, accessibility, and target-device performance.
4. Introduce stronger retrained models only after the evaluation/export/on-device contract is reliable. Use measured device budgets to select/distill/quantize them; larger datasets or models will not fix mislabeled failed scans, deployment mismatch, or uncalibrated probabilities.
