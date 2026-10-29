# Product, Android application, and data audit — 2026-10-29

## Scope and evidence limits

This is a source-level audit of the current checkout's navigation, Compose screens, ViewModels, media import, history, voice vault, RAG, model manager, and receipt/certificate exports. It follows the current source and `CLAUDE.md`; earlier review documents and test-count claims are not treated as proof of current behavior. No Gradle task, Android emulator, real-device run, model download, or training was performed by this audit track. Findings marked confirmed are concrete source paths; their device timing and frequency remain unmeasured. No production source or model asset was changed.

The strongest architectural issue is inconsistent enforcement of the project's evidence contract. The normal scanner has a validated-result type and abstention policy. Voice verification and RAG construct separate verdict strings and bypass that contract. Passing tests for the scanner therefore cannot establish safe behavior for all visible app entry points.

## Application architecture and flows

| Area | Actual path | Assessment |
|---|---|---|
| Entry and sharing | `MainActivity` → `SharedMediaIntents` → `ONNNavGraph` → scanner route | Uses selected content URIs, consumes share extras, preserves Home below a shared scan. |
| Normal scan | `ScannerViewModel` → integrity check → bounded private import → SHA-256 → image pipeline/audio classifier/video analyzer → `ScanHistoryRepository` | Best-developed evidence and error handling path. The image decoder caps working size, applies EXIF orientation, requests sRGB, and composites alpha on white. |
| Result/history | Room v3 → `ResultViewModel` → `AssessmentDetails`, telemetry, receipts, certificate | Stores assessments rather than source media. Handles missing records, legacy status, database errors, unsaved assessments, and explicit history deletion. Historical release digest binding is incomplete. |
| Voice Vault | `VoiceEnrollmentScreen` → `VoiceEnrollmentViewModel` → audio classifier + DSP `SpeakerVerifier` → `VoiceVaultRepository` → Room v2 + Keystore AES-GCM | Cryptographic storage has meaningful defenses; verification verdict policy is unsafe. Current speaker matching is a custom acoustic feature algorithm, not a validated identity model. |
| RAG | `ChatRagViewModel` → `ForensicRagEngine` → independent model calls/heuristics → keyword retrieval → optional local language generation or deterministic prose | Duplicates detector orchestration, media ingestion and verdict logic. “Embeddings,” subject tracks and forensic root-cause explanations overstate what the implementation measures. |
| Model manager | document picker → `VlmPackManager` ZIP extraction → signature/hash checks → install under `filesDir/vlm_packs` | Good path traversal and hash defenses; missing resource budgets, transactional replacement and durable settings. |

Dependency injection and clear package separation provide a useful starting structure, but the application is one Android app module rather than a separately consumable Android SDK. Domain results directly refer to detection implementations, and feature screens invoke detectors themselves. A future SDK needs a public facade, explicit supported capability contract, and tests independent of the app UI.

## Prioritized findings

### P1 — Quick voice verification turns unavailable evidence into authentic speech

**Confirmed, high confidence.** `VoiceEnrollmentViewModel.kt:183–187` calls the audio classifier but ignores `status`, `modelValidated`, and the classifier's reason. Missing/non-finite evidence becomes `0.2f`. Because the default threshold is `0.35546875f`, failure takes the “authentic” branch. Lines 202–204 produce “VERIFIED FAMILY VOICE” or “AUTHENTIC SPEECH” based on that derived boolean.

This route is reachable from the visible Verify Clip button (`VoiceEnrollmentScreen.kt:58–63,163–175`). An intact but unvalidated release can also return a finite diagnostic score, which is classified here despite the core classifier abstaining. `AudioClassifier.kt:67,141–142` explicitly returns `MODEL_FAILURE`/`INCONCLUSIVE`; the visible voice workflow discards that distinction.

**Completion requirement:** use one typed verdict policy across scanner, vault and chat; never replace missing scores with plausible measurements. Require validated successful analysis before authenticity or clone language. Treat speaker similarity as a separate, explicitly unvalidated diagnostic until identity performance is measured.

**Regression cases:** missing model, unvalidated release with both low and high finite scores, silent/non-speech audio, failed decode, missing threshold, spoofed family-like audio, and corrupted stored embedding. All unavailable cases must remain inconclusive.

### P1 — RAG bypasses detector release, disagreement, and aggregation policies

**Confirmed, high confidence.** `ForensicRagEngine` calls SRM, FFT, KL, probe and XGBoost directly instead of consuming a `ScanResult`. Examples:

- Image scoring builds its own features and predicts at `ForensicRagEngine.kt:833–850`; its headline at line 902 compares to a hard-coded `0.578f` without `ModelIntegrityGate`, `VerdictPolicy`, or `CascadePolicy`.
- Audio at lines 730–732 ignores classifier status and substitutes `0.20f`; line 811 labels the result “AUTHENTIC HUMAN SPEECH” or “SYNTHETIC VOICE CLONE.”
- Video defaults failed SRM/KL/FFT observations to `0.25f` (598–602), injects measured jitter into a fusion slot that the repository contract says is unsupported (617), averages sampled frame predictions (688–690), and emits “LIKELY AUTHENTIC” at line 709 without validated sampled aggregation.
- Documents use heuristic forgery status or an SRM fallback (1017–1018) to produce “AUTHENTIC DOCUMENT”/“FORGED” (1051), despite no corresponding calibrated document authenticity release.

The currently shipped manifest explicitly has `validated: false` (`assets/constants/release_manifest.json:4`). These RAG routes can nevertheless issue definitive assessments. This is a completion blocker even if normal scans correctly abstain.

**Completion requirement:** make chat an explanation/retrieval layer over the same validated analysis service used by the scanner. Preserve unavailable signals, coverage, model IDs, thresholds, provenance, abstention reasons and diagnostic status in every indexed chunk. Do not allow natural-language rendering to upgrade an inconclusive result.

### P1 — Historical receipts bind old verdicts to current model hashes

**Confirmed, high confidence.** `ResultViewModel.releaseModelHashes()` (`ResultViewModel.kt:60–70`) reads the currently installed APK's manifest. `generateReceipt()` mixes those hashes with the historical entry's score and `modelVersion` (101,125–135). `ScanHistoryRepository.kt:32–35` stores version labels, but no release manifest/hash snapshot or exact set of model artifacts actually used.

After an app/model update, exporting an earlier assessment signs new hashes beside an old verdict. A valid signature cannot repair the incorrect evidence association. Optional imported model-pack hashes are also absent from this shipped-manifest-only path.

**Completion requirement:** persist an immutable analysis-time release descriptor with each assessment, including the exact model, preprocessing, calibration, fusion, optional-pack and policy identities used. Sign that snapshot. For old rows without binding information, reject attested export or clearly export an incomplete historical record rather than infer missing hashes.

**Regression case:** analyze with release A, install release B, export A's history entry; the exported signed payload must still name A's hashes.

### P1 — RAG and voice workflows bypass safe provider ingestion

**Confirmed, high confidence.** The normal scanner's `ProviderMediaImporter` bounds bytes, closes resources on cancellation and imposes a provider timeout. Chat opens provider streams directly:

- Full-size image decode without bounds/orientation/sRGB/white-alpha handling: `ForensicRagEngine.kt:828–829`.
- Unbounded PDF copy: lines 970–973.
- Unbounded text read: line 997.
- Another full-size document bitmap decode: line 1004.
- Voice enrollment reads/decodes an external URI multiple times and hashes it in an unbounded loop (`VoiceEnrollmentViewModel.kt:97,107,123–137`), without the scanner's importer, deadline or interruption wrapper.

Consequences include memory/disk exhaustion, stuck provider work, inconsistent predictions for the same oriented/transparent/wide-gamut image, and the possibility that a mutable provider returns different bytes for the voiceprint, anti-spoof sample and stored digest. Chat catches `Exception`, not `OutOfMemoryError`, so image allocation can terminate the app.

**Completion requirement:** one shared media-ingestion service must copy content once into bounded private storage, validate actual format, apply the same decoding contract, expose cancellation, and delete temporary files. Every consumer should use that immutable copy. Add byte/page/pixel/time limits for document parsing as well as media.

### P1 — Pack replacement destroys the installed pack before successful commit

**Confirmed, high confidence.** `VlmPackManager.kt:243–251` deletes the current installation and copies staged files into the final directory one by one. A full disk, provider interruption, process death or copy error leaves no working old pack and potentially a partial new one. Concurrent imports/uninstall are not serialized by the ViewModel or manager.

**Completion requirement:** stage and verify the full final directory, serialize lifecycle operations, close the active backend safely, atomically swap installation directories with rollback, and recover interrupted state at launch. Reject activation unless policy parsing, expected contents and backend compatibility checks all succeed.

### P2 — ZIP resource exhaustion happens before signature verification

**Confirmed, high confidence.** `VlmPackManager.extractZipStream()` (`281–305`) prevents path traversal but imposes no entry-count, expanded-byte, per-file, compressed-ratio, timeout or free-space budget. Extraction occurs before manifest signature checks (179–200), so an unsigned archive can fill storage. User cancellation does not check coroutine cancellation in the extraction/copy loops.

**Completion requirement:** enforce expanded-byte and entry budgets during extraction, limit JSON sizes, reject duplicate/unexpected entries, check free space for both staging and installation, and make provider reads cancellable. A malformed archive must not affect the working installed pack.

### P2 — Main-thread file hashing and matching can freeze scan controls

**Confirmed, high confidence; device latency unmeasured.** `ScannerViewModel.start()` runs in the default `viewModelScope.launch` dispatcher (58). After the importer suspends, `computeFileSha256(imported)` (74,114–127) reads the complete file on the UI thread, up to the allowed 512 MiB video input. The function has no cancellation check. The secondary vault decode and DSP comparison (`148–159`) also run after returning to the main coroutine. The 120-second timeout cannot interrupt a synchronous main-thread loop on schedule.

Receipt generation is also invoked synchronously from a Compose `remember` (`ResultScreen.kt:833`); PNG render/compress/MediaStore writes run directly from the click callback (`948–951`; `ForensicCertificateExporter.kt:241–261`). Key warm-up reduces one cost but does not move all export I/O off the UI thread.

**Completion requirement:** perform digest, decode, matching and export work on appropriate background dispatchers with cooperative cancellation and progress/error state. Verify with StrictMode and a large video/slow-storage device while pressing Cancel/Back.

### P2 — Exported certificate claims hardware attestation even when no receipt exists

**Confirmed, high confidence.** The PNG header unconditionally prints “100% OFFLINE HARDWARE-ATTESTED AUDIT” (`ForensicCertificateExporter.kt:126`). If signing fails and the receipt is null, it substitutes “Hardware Keystore (TEE)” and “RFC 8785 Canonical JCS” (`215–216`) in the proof area. The result card allows PNG export with that null receipt (`ResultScreen.kt:833,948–951`). Even a present chain is treated as a Google hardware root merely because it is nonempty (`835,894–897`); no chain validation is performed there.

The PNG includes only a truncated signature snippet, so it is a visual summary rather than independently verifiable proof. It also calls raw scores “Synthetic Probability” and its default operating point “Calibrated” even for unvalidated results (`ForensicCertificateExporter.kt:163–166`).

**Completion requirement:** display signed/unsigned and attested/unattested states from verified evidence, never from a fallback label. Export the full canonical receipt alongside the image or use a verifiable QR/attachment reference. Call uncalibrated values model scores. Explain what a signature attests: the assessment bytes, not correctness of authenticity detection.

### P2 — Voice vault authentication failures are silently reinterpreted as non-matches

**Confirmed, high confidence.** `VoiceVaultRepository.unsealedEmbedding()` deliberately throws for AES-GCM authentication failure. The scanner wraps it with `runCatching(...).getOrNull()` (`ScannerViewModel.kt:157`), and `SpeakerVerifier.matchAgainstVault()` skips null vectors (`SpeakerVerifier.kt:269–270`). If all profiles fail authentication, it returns ordinary `NO_MATCH`, with no corruption warning. This contradicts comments claiming failures are skipped loudly and must not become “no match.”

Legacy-row sealing runs only in `VoiceEnrollmentViewModel.init` and discards all failures (`74–77`). Opening an upgraded vault via a scan or RAG does not perform that sealing pass; plaintext legacy vectors can remain until the enrollment screen is opened. The schema migration intentionally carries plaintext forward (`VoiceVaultDatabase.kt:47–54`), and repository reads still accept it.

**Completion requirement:** initialize/migrate the repository centrally before any matching access; expose actionable vault-corruption/key-loss states; never turn unauthenticated stored data into a speaker mismatch. Complete legacy sealing transactionally and report unrecoverable rows. Define the desired treatment of plaintext remnants in database/WAL files for the threat model.

### P2 — RAG describes unsupported semantic observations and causal proof

**Confirmed, high confidence.** Subject detection is a skin-color pixel heuristic (`ForensicRagEngine.kt:170,182–198`) that assigns “boy” or “young person” tags without an age/identity detector. A global `bioLip.hasFace` value is passed to every sampled video frame (555,627–630), so detecting one face can mark unrelated frames as having a face. Deterministic output then calls the first marked frame the actual scene entry (329–330).

The hotspot answer says a residual variance cluster indicates in-painting/splicing “in that exact bounding zone” (362–364). There is no validated localization or causal attribution contract supporting that certainty. Retrieval is keyword/tag ranking (`213–261`), and the local model receives text with `bitmap = null` (1146–1149), despite UI descriptions about visual/acoustic embeddings. The streaming UI types out an already completed answer (1193–1203).

**Completion requirement:** label these features accurately as sampled heuristic telemetry/keyword retrieval; remove demographic inference and causal tamper claims; add real per-frame tracks and evaluated semantic retrieval only if those become intended product promises. Preserve time-sampling uncertainty in answers. Distinguish text generation from actual vision/audio understanding.

### P2 — Installed-pack enablement is lost across process restarts

**Confirmed, high confidence.** `experimentalVlmCascadeEnabled` defaults to false in static process memory (`OnDemandForensicVlmAnalyzer.kt:55`). Import sets it true (`VlmPackManager.kt:256`), and UI toggles only change that variable (`ModelManagerViewModel.kt:129–135`). `OnnDetectApplication.onCreate` does not restore the preference. After restart the installed files remain, but escalation silently returns to disabled. Custom-key trust preference has the same lifetime.

**Completion requirement:** persist explicit user settings, restore them at application startup, revalidate the installed pack before activation, and clearly distinguish installed, verified, compatible, enabled, and diagnostic states. Do not automatically preserve a development trust bypass as a production trust policy.

### P2 — RAG opens imported model weights outside the verified backend loader

**Confirmed, high confidence.** Pack import normally verifies a pinned Ed25519 publisher plus policy/language/vision SHA-256 digests (`VlmPackManager.kt:195–234`; `OnDemandForensicVlmAnalyzer.kt:554–592`). A developer setting explicitly permits self-signed custom packs. However, RAG obtains pack details by merely parsing the installed manifest and comparing its public-key string (`VlmPackManager.kt:124–159`), then constructs `LlamaCppVlmPrefillBackend` directly (`ForensicRagEngine.kt:1127–1151`). It does not invoke the verified runtime loader, inspect the current custom-key allowance, recheck digests, or honor the diagnostic-only flag. An imported custom pack remains available to chat even after the in-memory custom-key permission resets on restart.

This is local inference, not a remote service. A real language/vision GGUF pack must be obtained separately, imported through the system picker, fit device storage/RAM, and have a compatible native backend. The current chat invocation passes no image (`bitmap = null`), so an installed multimodal pack is used as a text generator over precomputed heuristic descriptions.

**Completion requirement:** expose one verified pack/session manager used by detection and chat. Revalidate trusted publisher, immutable artifact identities and compatibility at activation, explicitly enforce development/diagnostic status, and prevent model files being replaced while a session uses them.

### P2 — B2B SDK and camera injection protection are not delivered integration surfaces

**Confirmed, high confidence.** `android/settings.gradle.kts:18` includes only `:app`; no Android library module/public SDK contract exists. The UI's video path analyzes a selected prerecorded file, then calls `VirtualCameraDetector.inspect(appContext, o.temporalScore)` (`ScannerViewModel.kt:184–186`). That inspector reads the current phone's camera HAL, process hook indicators and emulator fingerprint (`VirtualCameraDetector.kt:49–95`). These measurements do not establish which camera or injection path created the selected video's bytes. There is no live camera capture/injection-boundary enforcement in this route.

**Completion requirement:** scope the current feature as local device/environment diagnostics. If an SDK/live KYC shield is a release goal, build a library API and sample host, bind checks to the actual capture session, define supported threats, and measure detection/false positives on adversarial device fixtures. Documentation must not assert that this selected-file app already defeats arbitrary hooking/camera injection or establishes regulatory compliance.

## Strong existing work to preserve

- Manifest explicitly removes network permissions and avoids broad storage/camera grants; backup is disabled.
- `ProviderMediaImporter` has byte limits, provider cancellation, deterministic temporary-file cleanup and a separate timeout.
- `ScanResult` and `VerdictPolicy` enforce finite validated classified results; legacy Room history defaults to inconclusive.
- The scanner preserves an unsaved assessment and offers retry instead of inventing a result after storage failure.
- History uses migrations rather than destructive recreation and requires user confirmation for clearing/deleting history.
- Vault sealing uses fresh GCM IVs and row-bound AAD; tests cover ciphertext/IV tampering. These defenses are valuable even though callers currently mask some failures.
- Pack import checks signatures, policy digest and model digests, with a path traversal guard. The missing protection is lifecycle/resource handling, not absence of all integrity defenses.

## Tests and release acceptance gaps

Existing RAG tests mostly exercise formatting, keyword retrieval, deterministic prose and a mock generation delegate. One test named `retrieveTopChunksPrioritizesPersonEntryForBoySceneQuery` does not invoke `retrieveTopChunks`; it checks a fixture field (`ForensicRagEngineTest.kt:24–78`). The tests do not exercise media ingestion or policy enforcement on failed/unvalidated analysis. No `VoiceEnrollmentViewModel`, `ModelManagerViewModel` or `ResultViewModel` test was found in the current Android test trees by name.

Before calling the product complete, require:

1. The same missing-model/unvalidated/silent/unsupported/failed-inference cases across normal scan, share import, vault verify and chat attachment; all abstain consistently.
2. An immutable analysis-time evidence snapshot surviving app upgrade and exporting the original model identities.
3. Slow/malicious provider, huge image, huge text/PDF, ZIP bomb, duplicate archive entries, disk-full, interrupted pack upgrade and process-death tests.
4. Real-device onboarding, share-sheet and scan recovery on rotation/background/process death, including large-font/TalkBack flows for new chat/vault/model-manager screens.
5. Identity evaluation before any “verified family” security promise, including impostors, similar relatives, replay/clone attacks, channel degradation and unsealed/corrupted profile rows.
6. Durable pack settings, transactional install, explicit supported-device matrix and cold/warm RAM/latency measurement.
7. Signed and unsigned export UX tests, full receipt verification, stale-release history tests and recoverable MediaStore failure behavior.

History scalability is a later concern: `HistoryViewModel` subscribes to `observeAll` even though the DAO has `observePage`; it loads every record and heatmap/evidence JSON into memory. Use paging after correctness and safety gaps are closed. Localization/accessibility also need deliberate completion: substantial new UI text is hard-coded English and some interactive icons lack useful spoken labels.

## Suggested implementation order

1. Unify verdict policy and safe ingestion; close the quick-verify and RAG fail-open routes.
2. Freeze per-assessment evidence and make receipts/certificates accurately represent it.
3. Add transactional, bounded, cancellable pack installation and durable settings.
4. Move blocking I/O and matching off the UI thread and make corruption/errors actionable.
5. Evaluate speaker matching, document localization and semantic timeline behavior, then either validate the capability or label it experimental.
6. Run real-device/end-to-end acceptance and align visible product claims with the measured release.
