# Implementation and validation status — 7–8 October 2026

The code corrections below are implemented. The existing image and audio model binaries are preserved and explicitly **unvalidated**. The app returns an **Inconclusive** assessment for these models; a diagnostic score does not establish synthetic or authentic origin. No replacement model was trained, no benchmark accuracy was invented, and no claim covers every generator or all kinds of audio.

The [complete review](COMPREHENSIVE_REVIEW_2026-10-06.md) and its four companion reviews cover the original 171-file snapshot, including dataset and model research with primary-source links. Those historical reports describe the code before these fixes. This document describes the implementation and actual verification afterward.

## Changes against the 16 principal audit findings

| Finding | Implemented change | Remaining acceptance work |
|---|---|---|
| 1. Incorrect labels and misleading corpus names | Downloaders use explicit semantic label maps and actual source names, require pinned revisions and preserve original source splits and bytes. No counter-based random split or destructive replacement. | Acquire licensed larger corpora and independently verify original/group identities and labels. The supported pilot sources are not full GenImage or ASVspoof. |
| 2. Float64 audio enters a float32 neural network | Shared LFCC output and model inputs are float32; DSP reference precision remains explicit. A real PCM16 WAV exercises decoding, resampling, LFCC, LCNN, loss and backward. | Retrain weights against this frontend; test additional native codec paths and rates on phones. |
| 3. Split leakage and silent dataset fallback | Required splits/classes, hashes, authoritative group metadata and disjointness are checked. Images have independent stacking/fusion-validation/calibration/test pools. Corrupt media fails rather than becoming black images or silence. | Authoritative groups and near-duplicate detection need real source metadata and a separate audit; supplied group IDs cannot prove their own correctness. |
| 4. Conflicting historical metrics and unknown lineage | Legacy weights/metrics are labeled unvalidated. Model, frontend, calibration, source snapshot and evidence are bound to release hashes. Below-chance output is not silently inverted. | Evaluate new exact binaries on verified data before making an accuracy claim. |
| 5. Fusion does not match deployed features | Fusion training consumes actual INT8 base scores, independent stacking/fusion-validation data and the ordered mobile feature contract. Generated Kotlin handles missing branches, margins and numeric precision; its bytes/release ID are checked. Unsupported provenance/temporal fields use the same fixed placeholders in Python and Android, rather than pretending they are measured evidence. | Train and evaluate a real fusion release; measure Kotlin outputs on matched device fixtures. |
| 6. Failures become normal verdicts | Typed success, inconclusive, unsupported input, model failure and cancellation outcomes; missing scores remain distinct from zero. Decoder rejects and model failures retain different statuses. Persistence and UI tests exercise the verdict policy. | Continue fault injection across additional vendor decoders. |
| 7. Higher-resolution training cannot reach Android | Versioned, dynamic image tensor contracts and a shared v2 policy connect training, export, calibration and runtime: EXIF orientation, sRGB, white transparency, bounded working resolution and resize. Original dimensions and actual encoding remain feature metadata. | Native Pillow/BitmapFactory pixel parity must be measured. A production native-patch/global ensemble and stronger compact student still require implementation experiments and retraining. |
| 8. Quantization and parity use the wrong samples/thresholds | Genuine INT8 is the default, with both labels in training representatives; conversion cannot silently fall back. Labeled parity checks numeric error, agreement and task degradation at actual operating thresholds. Full sealed-test evaluation is separate from small parity pools. | Evaluate a trained release, including low-FPR sensitivity and hard transformation slices. Synthetic conversion tests are implementation checks. |
| 9. Merged permissions break the offline contract | Removed unused MediaPipe and other dependencies; explicitly remove network permissions. The APK verifier checks actual merged permissions, assets and native layout. Standalone CPU LiteRT ships in the APK. | Measure disconnected first-run analysis on representative target phones. |
| 10. PCM/YUV decoding and resampling disagree | PCM encodings, channels, bounded source windows and a named anti-alias resampler are handled explicitly. YUV conversion handles stride, crop, color and rotation; unsupported HDR is rejected. | Expand real AAC/MP3/Opus/FLAC/HEIF/vendor-video fixtures and compare decoded values on phones. |
| 11. Central-only audio and sparse video hide coverage | Audio uses bounded timestamped windows and quality checks; video reports sampled frame coverage, seeks the first requested frame and checks deadlines. Silence/empty input cannot become a normal verdict. | Partial edits outside sampled regions can be missed. Recording-level audio and video aggregation need separate calibration. |
| 12. Scores are presented as confidence/probability | Results/history show detector scores, limitations and explicit assessments. Failure/unvalidated/unsupported results cannot be called authentic. Video/audio file decisions abstain until their routing/aggregation is validated. | Train calibrated uncertainty policies. Speech, music, environmental and mixed audio routing/models remain research work. |
| 13. Shared resources, cancellation and memory are unsafe | Serialized interpreter lifecycle/invocation and owned buffers; finite tensor checks. Provider imports use cancellation signals, bounded private copies and deadlines; cancellation closes owned streams/descriptors and cleans temporary files. Decoders work from the bounded local copy. Worker errors reach the caller, and scanner allocation failures have explicit recovery. | Measure latency, memory, cancellation, thermal behavior and sustained scans on target hardware. |
| 14. Partial or mismatched releases can pass | Strict typed manifest and exact asset hash set; tensor/frontend/calibration/model checks. Desktop/device evidence binds source/model/fusion hashes. Packaging installs matching assets and Kotlin source transactionally, retaining prior/failed candidates and rolling back failures. | Attach actual release-specific device reports. Hashes bind artifacts; they do not independently authenticate a report's truth. Signed distribution remains separate. |
| 15. Placeholder capabilities overstate detection | Removed dead simulated landmark, optical-flow, lip-sync and semantic-model paths; enrollment is unavailable. Byte-pattern C2PA/EXIF cues cannot override verdicts; texture proxy and unsupported heatmap/watermark capabilities are labeled honestly. | Add cryptographic provenance, watermark verification or localization only with a real offline verifier/model and task-specific evaluation. |
| 16. Toolchain, CI and orchestration fail | SDK 36/toolchain/runtime updates, explicit Room migrations, smaller dependency graph, minified builds and a 16 KiB APK gate. CI actions are pinned; CPU contract/export checks and explicit provisioned-runner training are wired. Full-checkout source snapshots, resume state and early output-preservation guards are implemented. | Docker and GitHub jobs were statically checked but not executed remotely. Sign/package a validated replacement only after its data and device gates pass. |

## Verification performed

These are correctness/build checks, not detector accuracy benchmarks.

| Check | Observed result |
|---|---|
| Final Android build | `testDebugUnitTest lintDebug assembleDebug assembleRelease assembleDebugAndroidTest` passed, including the provider fixture and low-memory recovery changes. |
| JVM regression suite | 78 tests across 12 suites; zero failures, errors or skips. Counts were read directly from final JUnit XML. |
| Android lint | Zero errors; 25 warnings: 12 dependency/tool update notices, 12 KTX-style suggestions and one API-specific manifest attribute notice. Dependencies remain pinned to the tested toolchain. |
| Full Android instrumentation | 12 tests passed in 59.595 s; zero failed or ignored on an API 37 x86_64 emulator with 16,384-byte pages. Includes three app flows, two real Room migrations, three actual packaged-model/image/audio checks and four provider/error regressions. This full run preceded the final coverage-wording-only change. |
| Final-build saved-audio UI regression | One additional test passed in 21.017 s on the final debug APK: saved audio shows audio time coverage and cannot be labeled as sampled video frames. The full 12-test run and this targeted run are separate evidence. |
| ML CPU suite with required real SoundFile I/O | 37 tests run in 4.741 s: 36 passed; one Windows symlink test skipped because the account lacks symlink privilege. Real 48 kHz PCM16 WAV → shared resampling → float32 LFCC → LCNN → BCE/backward passed. |
| APK verifier regression suite | 10 tests passed, including malformed contracts, forged validation, changed/unhashed assets, forbidden permissions, ELF truncation and ZIP alignment. |
| Final debug and minified unsigned release APKs | Both passed merged network-permission, exact asset hash/contract and 16 KiB ELF/ZIP checks. Four 64-bit native entries across arm64/x86_64 were inspected per APK. |
| Signed release smoke copy | Passed the same APK gate. All 114 original unsigned APK entries have identical contents in the signed smoke copy; only signing metadata is added. |
| Strict distribution gate | Rejected the final legacy release as expected with `--require-validated-models`; the ordinary diagnostic APK gate passed. |
| Python/Keras export parity and actual full INT8 invocation | Passed for both networks. Random-weight FP32 maximum errors: image 2.98e-08; audio 5.96e-08. Both full INT8 conversions and finite TFLite invocations passed. Synthetic calibration data was used only to exercise the conversion chain. |
| NumPy/PyTorch LFCC and golden fixtures | Passed. Maximum full-tensor difference 1.094e-04; selected golden values 5.70e-05; neural output dtype float32. Shared resampler anti-alias, non-integer conversion, upsampling and silence checks passed. torchaudio's LFCC is deliberately not treated as interchangeable. |
| FFT implementation gates | Passed. FFT-core maximum error 5.23e-12; synthetic image score maximum error 1.55e-15. These Python parity checks complement JVM tests and do not establish native bitmap-decoding parity. |
| Source consistency | Passed across 56 production Kotlin files; all 10 ordered fusion features and legacy asset contracts checked. Expected warning: bundled models remain unvalidated. |
| Python syntax | All 72 Python source files parsed. |
| CI workflow syntax | All three workflows passed actionlint 1.7.12, including the custom provisioned runner label. The downloaded tool checksum was verified before execution. |
| Shell/PowerShell wrappers | Bash syntax passed for all four entry points. PowerShell syntax, space-containing argument preservation and nonzero native exit propagation passed. |
| Whitespace checks | `git diff --check` passed. |

The final device suite includes provider query timeout, blocked-pipe cancellation/cleanup, unknown-size byte-limit enforcement and simulated worker Error propagation. A standalone provider written in Java exercises the separate test-package process without relying on the target app's Kotlin runtime. No actual memory exhaustion was induced. The earlier incomplete provider run exposed this test-harness dependency problem; it was fixed before the final 12-test pass.

The earlier minified release completed a local 4-second tone scan through the real system file picker, private importer, LiteRT, Room and result UI in airplane mode. It displayed Inconclusive with a diagnostic score of 0.004; processing time was 276 ms for this one smoke observation, not a detector accuracy or general phone-performance benchmark. This exposed audio timestamps incorrectly labeled as sampled video frames. The wording was corrected and the new saved-audio regression passed on the final APK.

The final minified release, signed as a separate copy with the local debug key, cold-launched in airplane mode in 2,466 ms. Its repeated local-audio scan passed the file picker/import/decoder/LiteRT/history/result flow, showed Inconclusive with 4.0/4.0-second audio coverage, and showed no video-frame label. Processing time was 543 ms for that individual smoke scan. Neither timing is a latency benchmark for other media, models or phones.

Saved final-release evidence is under `android/app/build/validation/`: `release-last-cold-launch.txt`, `release-last-home-window.xml`, `release-last-result-window.xml` and `release-last-result.png`. The full instrumentation run, additional saved-audio UI regression and release smoke have distinct scopes; they are not accuracy tests for a replacement model.

An earlier UI inspection was interrupted by an emulator SystemUI ANR under host memory pressure; that attempt alone is not a completed UI smoke test. The final run uses a smaller transient framebuffer and captured/restorable display settings.

## Preserved artifacts and limits

Final build outputs, independently hashed after the last source rebuild:

| APK | Bytes | SHA-256 |
|---|---:|---|
| Debug (`android/app/build/outputs/apk/debug/app-debug.apk`) | 33,492,000 | `1c202811119eb12f9668dec4406b21d8dd8438039282cb31b3e0743d9fc512e7` |
| Minified unsigned release (`android/app/build/outputs/apk/release/app-release-unsigned.apk`) | 21,832,676 | `e0daf80bbd57f7b785055ccd88d1de55745d74afe1a7dc823c0e7e2b15e5c13d` |
| AndroidTest (`android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk`) | 1,211,975 | `927ab996a49a667562af0d708b3fc965cd59784ce8110181c2edff4af79ffb59` |
| Local-debug-key-signed smoke copy (`android/app/build/validation/app-release-local-smoke.apk`) | 21,850,319 | `dae15540e9cb9b9c3d9e4f5831c6fb63821ac3dbd9ffa7d8ee895430b81a3b1a` |

The debug APK is now 31.94 MiB versus 84.84 MiB in the original review. The minified unsigned release is 20.82 MiB. A separate copy can be signed with the local debug key for a release smoke test; that does not make it a production-signed release.

The neural assets were not overwritten:

| Asset | Bytes | SHA-256 |
|---|---:|---|
| Image TFLite | 99,824 | `5bcfefc15fdde3519f3bd1d077b3b61f39f2b5036b1adf86272af9a5f1a71818` |
| Audio TFLite | 503,368 | `2dd0c96a24da39c1b81398f8698e3c1b0e417692ea5e050ca7b57cf4695aff46` |

Local builds are diagnostic builds. The strict `--require-validated-models` distribution gate must reject this legacy bundle. JVM/unit success, conversion parity, emulator regression tests and native alignment cannot establish accuracy for Stable Diffusion, ChatGPT, Nano Banana or unseen audio generators.

The existing emulator was used without wiping or uninstalling user data. Its three animation scales were set to zero during the first test session without recording their prior values, so exact restoration of those settings cannot be claimed. The captured baseline was airplane mode disabled and physical density 480 with no override. A fresh boot without loading a snapshot on 8 October verified that those settings persisted: airplane mode disabled, density 480 with no override, and page size 16,384. Evidence is saved in `android/app/build/validation/emulator-restoration-verified.txt`; the test emulator was then stopped.

## Work needed for better accuracy

1. Obtain licensed diverse high-resolution image and multilingual speech corpora with immutable versions and source/group metadata. Use the [dataset research](COMPREHENSIVE_REVIEW_2026-10-06.md) for OpenFake/Community Forensics, ASVspoof 5, MLAAD/SpeechFake/AUDETER and partial-edit datasets; use separate music and environmental-audio data when those tasks are implemented. Multimillion-file metadata and source-balanced sampling require additional scaling work; lazy media loading alone does not bound manifest/index memory.
2. Retrain the corrected image and speech baselines, then compare compact students/cleared teachers on the same grouped holdouts. Current hardware is approximately 8 GB RAM and a 4 GB laptop GPU with CPU-only installed PyTorch; large teacher training needs a provisioned environment.
3. Test current commercial-platform samples, unseen generator families, legitimate edits, recapture and codec chains. Select low false-positive operating points using sufficient genuine examples, confidence intervals, subgroup metrics and accepted coverage.
4. Match exact media bytes, native frontend tensors, INT8 outputs and compiled fusion on representative phones. Measure package footprint, latency, peak memory, sustained energy and thermal behavior.
5. Calibrate audio routing/recording aggregation and video coverage separately, attach actual release-specific device evidence, then pass the strict APK gate before distributing a validated detector.

Commands and the complete staged-release workflow are in [TRAINING_GUIDE.md](TRAINING_GUIDE.md) and [DEPLOYMENT.md](DEPLOYMENT.md).
