# Detection, inference and provenance audit — 29 October 2026

## Scope and evidence

This is an analysis of the checked-out Android detection, inference, provenance and optional native VLM implementation. Production code and model assets were not changed. Source references use repository-relative paths and one-based line numbers from this checkout. The root review coordinates the Android build; this audit did not launch a second Gradle build.

Two executable proofs invoke the existing compiled application/test classes or the real Python verifier, rather than reimplementing the vulnerable logic:

- [Runtime proof source](test_artifacts/audit_detection_2026-10-29/AuditRuntimeProof.java), [runner](test_artifacts/audit_detection_2026-10-29/run-runtime-proof.ps1), [observed output](test_artifacts/audit_detection_2026-10-29/runtime-proof.txt).
- [Receipt proof source](test_artifacts/audit_detection_2026-10-29/AuditReceiptProof.py), [observed output](test_artifacts/audit_detection_2026-10-29/receipt-proof.txt).

The C2PA proofs use synthetic containers, a newly generated ephemeral CA, and the repository's existing test fixture builder. They demonstrate the implementation's trust-boundary failures. They are not a C2PA conformance test, a production-root compromise, or an executed attack against an independent real-world signed image. No private keys were saved.

## What actually runs

`ModelIntegrityChecker` distinguishes `release_manifest.validated` (device claim) from `desktop_validated`. Its public `Report.validated` means either usable validation path passed. Consequently, reading `validated: false` in the manifest alone does not establish that the app abstains.

Direct execution against the packaged assets produced:

| Requested model | allOk | Report.validated | deviceValidated | cascadeValidated | Threshold |
|---|---|---|---|---|---|
| SRM image | true | true | false | true | 0.57819986 |
| LCNN audio | true | true | false | true | 0.35546875 |

The current tree can therefore classify images, speech-like audio, and videos. README text asserting that all statistical decisions currently abstain is inconsistent with this behavior. ModelIntegrityChecker.kt:148-151 permits desktop use; the final report returns `validated=true` after the remaining checks. Independent device validation is still absent.

## Architecture traced from source

The main app imports media into a private file, computes the original-byte SHA-256, and dispatches to image, audio or video routes. Hilt supplies a shared LiteRT engine and modality analyzers. `ModelIntegrityGate` caches per-modality release reports. `LiteRTInferenceEngine` owns memory mappings, tensor-contract checks, finite input/output checks, model-level invocation locks, and an exclusive VLM memory lease.

The image route computes residual heatmap/document diagnostics, verifies the release, collects C2PA evidence, measures SRM rich and optionally poor texture mosaics, FFT and KL, optionally runs the SigLIP probe, and builds the ten-slot fusion vector. XGBoost produces the fusion score; the cascade can abstain on disagreement in its ambiguity band. An optional imported VLM pack can record diagnostic escalation and a head outcome. `VerdictPolicy` and `ScanResult` provide explicit SUCCESS/INCONCLUSIVE/failure contracts.

The audio route decodes a bounded 16 kHz window, optionally checks AudioSeal, extracts the named float32 LFCC frontend, invokes LCNN and applies a speech-content gate. Its default is one four-second window, typically from the file center. The public API also accepts several windows and pools their scores.

The video route samples up to ten frames, runs the probe on selected keyframes with score reuse across visually similar frames, computes SRM/FFT/KL and BioLip telemetry, produces per-frame image fusion scores, pools frames and optionally raises the score from an audio track. Its final aggregation currently produces classified verdicts without a distinct video calibration.

Provenance and statistical scores are separate data structures, which is a sound design choice. Hardware receipts bind a media digest and model metadata into a canonical payload signed through Android Keystore. Imported VLM packs are signed/hash checked before use and live outside immutable APK assets. Native multimodal inference is implemented through llama.cpp/mtmd JNI.

## Findings requiring correction before production claims

### D1 — Critical: verified C2PA evidence is not bound to all of the data it describes

**Executed reproduction:** a COSE payload signed for `digitalCapture` was reused unchanged. Replacing only the unsigned `clai` box with `trainedAlgorithmicMedia` returned `signed=true`, source `trainedAlgorithmicMedia`, and `confirmsAI=true`. Separately, a covered media byte was changed and the unsigned data-hash assertion was recomputed; the same untouched COSE signature still yielded `signed=true`.

Evidence:

- `C2PAManifestChecker.kt:237-248` prefers a separate `claim` box over the COSE payload when deriving `digitalSourceType`.
- `C2PAVerifier.kt:91-108` verifies a signature over the COSE payload but independently verifies whatever data hashes the scanner extracted.
- `C2PAManifestChecker.kt:320-325` finds data-hash assertions by scanning the entire store, without proving a signed claim reference authenticates their bytes.
- `C2PAVerifier.kt:118-129` combines these independent checks and publishes the unsigned source type as a signed claim.
- Existing fixtures explicitly sign a claim containing a state label rather than an assertion digest (`C2paTestFixtures.kt:134-141`); integration tests add the actual hash in another box (`ProvenanceIntegrationTest.kt:54-59`). The fixture shape therefore cannot test transitive assertion authentication.

**Impact:** the pipeline can elevate this result to `PROVENANCE_CONFIRMED` and a synthetic verdict (`DetectionPipeline.kt:109-145`). Hash verification, signature verification and chain verification must be connected into one authenticated graph. Three unrelated successful checks are insufficient.

**Completion criterion:** parse one active manifest structurally; derive semantics exclusively from the authenticated claim/assertions; resolve and verify every assertion reference and its digest; authenticate the hard binding before trusting its exclusions/digest. Add rejection tests for mismatched COSE/clai, replaced assertions, recomputed unsigned data hashes, transplanted signatures, duplicate/multiple manifests and unsupported assertion algorithms. Include independent known-good and known-bad fixtures produced outside this implementation. Until then, do not elevate this verifier's result to confirmed evidence.

### D2 — High: certificate validation shortcuts accept expired signer certificates

**Executed reproduction:** an expired leaf issued by the temporary trusted CA failed `X509Certificate.checkValidity()` but passed `C2PATrustStore.verifyChain()`.

`C2PATrustStore.kt:91-109` returns early after matching trusted keys or verifying issuer signatures. These branches bypass the PKIX validation at lines 114-121, including its date/constraints checks. The class comment promises current-clock validity checking, but the common shortcut does not enforce it.

**Completion criterion:** use one policy-complete certificate path validator, with explicit signer profile, validity, key usage, basic constraints and path constraints. Define and display the limitations of offline revocation/trust-list freshness separately. Cover expired, not-yet-valid, wrong-usage and non-CA intermediate certificates with negative tests.

### D3 — Critical: the receipt verifier accepts a software signature as hardware attestation

**Executed reproduction:** the repository's existing adversarial fixture generates a software EC key and self-signed certificate with no Android attestation extension. With a claimed `STRONGBOX` label, the real verifier returned `valid=true`, `hardware_backed=true`, and no reasons, even when supplied the correct media bytes and expected nonce.

`scripts/verify_forensic_receipt.py:129-171` checks leaf-key equality and adjacent issuer signatures but has no configured trusted hardware root requirement and accepts a one-certificate chain. At lines 201 and 231-236, a caller-controlled security-tier string supplies the hardware claim. No Android attestation extension is evaluated.

The signing side also does not request Android key attestation: `StrongBoxReceiptSigner.kt:230-244` never calls `setAttestationChallenge`. `tierOf()` at lines 197-200 labels every non-StrongBox key, including unknown/error cases, TEE. On older APIs the source itself documents an ambiguous hardware heuristic at lines 210-216.

**Completion criterion:** request attestation when enrolling the device key, bind the attestation challenge to a verifier-controlled enrollment flow, validate the chain to explicitly trusted roots, parse the Android attestation extension, enforce application identity and boot/security policy, and derive the accepted security level from authenticated evidence. Keep a receipt signature valid/signature integrity result separate from hardware attestation status. Add a mandatory rejection test for the existing software/self-signed fixture under enterprise mode. Do not describe present receipts as independently proven hardware-backed certificates.

### D4 — High: video verdicts bypass the calibrated score contract

Four reachable changes occur before a video classification:

1. `VideoForensicAnalyzer.kt:340` invokes `SRMForensicAnalyzer.analyze()`, which returns a hand-adjusted rich/poor effective score (`SRMForensicAnalyzer.kt:83-84,126-135`). The image route deliberately preserves the calibrated raw rich-mosaic score instead (`DetectionPipeline.kt:207-210`). The same fusion release receives different meanings in slot 0.
2. `VideoForensicAnalyzer.kt:332-375` routes measured landmark jitter into slot 9. The repository contract declares unsupported slots 7-9 fixed neutral values for this release. Changing a fixed feature into a measurement requires retraining and validation even if a particular tree currently ignores the slot.
3. `VideoForensicAnalyzer.kt:399-404` makes a new 50% mean + 50% top-k frame score but compares it against the image fusion threshold.
4. `VideoScorePolicy.kt:60-75` still raises a score when `audioValidatedSuccess=false`; it merely reduces the blend weight. Executing the real method with visual 0.55, threshold 0.57819986, audio 1.0 and invalid audio status produced **0.7075**, changing the decision side.

`VideoForensicAnalyzer.kt:424-430` calls its output provisional and `provisional-uncalibrated`, yet returns `SUCCESS` and `modelValidated=true`. `ScannerViewModel.kt:183` consumes those values through `VerdictPolicy`, so the warning does not prevent a classified verdict.

**Completion criterion:** restore the exact deployed per-frame feature contract, prevent failed/inconclusive/unvalidated audio from influencing the decision, and keep the whole video assessment inconclusive until the exact sampled routing/pooling/keyframe-reuse/multimodal policy has its own sealed video-level evaluation and threshold. Report diagnostic scores and analyzed coverage while it abstains. Test the full video Outcome-to-ScanResult path, not only the boundedness of a blend formula.

### D5 — Medium: image abstentions and failures lose the original media digest

`DetectionPipeline.analyzeMedia()` accepts `mediaSha256`, but its nested `result()` introduces a second nullable parameter with a default of null (`DetectionPipeline.kt:104-107`). Only the final success call passes the original digest (`429-431`). Inconclusive cascade results (`423-425`), unvalidated outcomes (`284-287`) and exception paths (`433-434`) lose it.

`ResultViewModel.kt:106-110` correctly refuses receipts without a real media digest. As a result, many otherwise useful forensic abstentions cannot be exported with content binding.

**Completion criterion:** capture the original digest in the result factory for every terminal outcome. Verify persistence and export for success, disagreement, unsupported input, inference failure and unvalidated results. An inconclusive receipt must still state that its verdict is inconclusive.

### D6 — Medium: AudioSeal evidence is discarded by an impossible condition

`AudioClassifier.kt:99-101` saves a watermark only when `WatermarkMark.confirmsAiInvolvement` is true. `ProvenanceEvidence.kt:64-65` correctly makes that property always false because authentic audio can also be watermarked. The audio classifier therefore discards every detected watermark, instead of displaying watermarking as a separate observation.

The current release does not declare an AudioSeal model; this missing optional capability must remain explicit. A future packaged model also needs its own release/hash/frontend/calibration gates rather than being loaded merely because its filename exists.

**Completion criterion:** preserve detected watermark evidence without converting it to AI-generation proof, and test AudioSeal-to-AudioClassifier-to-result persistence. Distinguish unavailable detector, evaluated absence and detected watermark.

### D7 — Medium: the public multi-window audio path has a coverage bug and no pooled calibration gate

Default UI operation currently uses one window (`AudioClassifier.kt:44`). For `windows>1`, lines 77-93 decode a center window first, then generate distributed offsets, but reuse the center window for offset index 0. A 60-second file with three four-second windows requests offsets 3, 28 and 53 seconds; the actual decoded windows become 28, 28 and 53 seconds. The leading window is missed and the center is counted twice.

Lines 117-151 pool several scores and still return SUCCESS using the single-window release threshold. The comment at lines 39-42 explicitly says that operating point was calibrated for one window.

**Completion criterion:** decode each requested offset exactly once, de-duplicate overlapping/identical windows, track the union of analyzed and usable time, and require a recording-level calibration identifier before producing a multi-window file verdict. Keep the supported default as one-window evidence until those gates exist. Do not imply a whole recording was authenticated by four seconds of sampled speech.

### D8 — High for optional VLM use: native close/import can race live inference

The Kotlin backend's extract, generate and close methods do not share a lifecycle lock (`LlamaCppVlmPrefillBackend.kt:129,182,220`). Model import/uninstall can close the same singleton backend (`VlmPackManager.kt:240,270`). JNI extraction checks a raw pointer before acquiring the native mutex (`llama_forensic_jni.cpp:192-227`); generation does the same (`371-382`). Freeing releases resources and deletes the raw pointer (`493-496`). The global mutex serializes much native work, but does not make those pre-lock raw-pointer checks or Kotlin handle access safe against deletion.

This is a source-confirmed race risk; no device crash was intentionally induced. Native prefill/generation also lacks a cancellation/deadline callback, so the image route's 30-second cooperative checks cannot interrupt a running native invocation.

**Completion criterion:** one owner must serialize initialization, invocation, cancellation, close and pack replacement. Use validated handle ownership rather than dereferencing possibly freed raw pointers, and a native abort mechanism for deadlines/cancellation. Stress-test scan/chat/import/uninstall overlap on a device with a real pack.

### D9 — Medium: imported VLM installation is destructive before commit and archive extraction is unbounded

`VlmPackManager.kt:237-253` closes the existing backend and deletes the installed pack before copying the new verified pack. A copy failure, storage exhaustion or process death leaves no working old pack. `extractZipStream()` at lines 281-305 has a useful canonical-path traversal defense but no expanded-byte, entry-count, individual-file or deadline budget. Extraction occurs before signature verification.

**Completion criterion:** enforce archive budgets while streaming, check available space, support cancellation and preserve the previous pack until a verified candidate directory is complete. Commit by a transactional directory swap and retain a recoverable rollback. Test malformed archives, duplicate filenames, oversized expansion, insufficient disk and process interruption.

### D10 — Medium: the LiteRT global close implementation does not obtain all model locks

`LiteRTInferenceEngine.kt:305-322` claims to take every model lock before closing, but actually loops over locks and closes **all** models while holding only the current single lock. Another model can still be in an invocation protected by its different lock. A concurrent new model load is also not blocked by a terminal engine-closed state.

No direct production engine-close call was found in the inspected application paths, so this is a lifecycle/API defect rather than an observed ordinary scan crash. The individual `evictModelWithReport()` implementation has much stronger ownership semantics.

**Completion criterion:** atomically enter a terminal closed/closing state, prevent new loads, await existing per-model owners, then close resources under a safe common lifecycle protocol. Add a concurrent predict/close regression before exposing the engine as an SDK lifecycle API.

## Strengths to preserve

- Explicit status/verdict types and a constrained ScanResult constructor provide useful central truthfulness checks.
- Model bytes, calibration thresholds, frontend identifiers, source snapshots and desktop reports are hash linked; per-model thresholds avoid image/audio operating-point mixups.
- Fusion missingness is explicit for the optional semantic slot instead of silently substituting a real numeric measurement.
- Image tensors are read from model contracts and checked against manifest shapes. Input and output numeric finiteness is enforced.
- Interpreter invocation is serialized per model, buffers belong to requests, and explicit model eviction/VLM leases are a sound basis for memory ownership.
- Audio decode, LFCC, resampling and signal-quality contracts are explicit and testable.
- Video returns sample timestamps, distinguishes audio and visual coverage, and releases owned frame bitmaps.
- Uncalibrated image ROI/contrast diagnostics are already kept outside the main image fusion score. Apply the same discipline consistently to video.
- Pack imports already check signatures, file hashes and canonical destination containment; transactional installation can extend this foundation.

## Practical completion order

1. Correct the C2PA authenticated graph, certificate validation and receipt hardware-attestation claims. These failures affect whether an observation can be trusted at all.
2. Restore exact video feature semantics and require validated aggregation before classified video decisions. Fix invalid-audio escalation immediately.
3. Preserve media digests and independent watermark evidence across every outcome; fix and gate multi-window audio.
4. Finish native/model lifecycle ownership and atomic bounded VLM installation before enabling pack-based features broadly.
5. Evaluate the exact packaged end-to-end policy on independent target-device data, including decoder/frontend parity, image/video/audio quality slices, out-of-distribution cases, sample coverage and selective abstention.
6. Measure cold/warm p50/p95 latency, peak RSS/native/Java memory, cancellation response and thermal behavior on named physical devices. The VFM source itself labels ARM64 140-180 ms as an unmeasured expectation (`VfmProbeAnalyzer.kt:50-52`); do not promote it to an observed shipping latency.

Passing algorithm unit tests proves bounded formulas and expected code behavior. It cannot substitute for authenticated evidence linkage, adversarial trust tests, exact deployed policy calibration or physical-device evaluation.
