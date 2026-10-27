# ONN-DETECT: Project Deliverables & TODO Status

**Release:** `onn-prod-v10` / `v10.1`
**Last Synchronized:** 2026-10-26 (Post-Milestone 1, 2, and 2-R2 Verification)
**Tracking Document:** `docs/STATE_OF_THE_ART_REVIEW_2026.md`

---

## 1. Completed Deliverables

### Milestone 1: Tooling, Dependencies & Cryptographic Hardening
- [x] **Lineage & Dependency Synchronization**: Added `cryptography>=50.0.2` to `ml/requirements.txt`. Reconciled calibration record in `evaluation_report.json` to 400 samples, synchronized SHA-256 digests in `release_manifest.json`, passing all 7 lineage checks in `onn_release_lineage_gate.py`.
- [x] **StrongBoxReceiptSigner Concurrency & Thread-Safety**: Hardened `StrongBoxReceiptSigner.kt` with double-checked locking on `keyLock`, `@Volatile testKeyPair`, and `@JvmOverloads constructor(context: Context? = null)`. Enforced fail-closed behavior on physical devices (`if (androidKeystoreAvailable()) throw e`). Verified under 16 concurrent worker threads in `StrongBoxReceiptSignerTest.kt`.
- [x] **VoiceVaultCrypto Value Equality**: Replaced referential equality in `SealedBlob` with content-based `Arrays.equals(iv, other.iv)`, `Arrays.equals(ciphertext, other.ciphertext)`, and `31 * Arrays.hashCode(iv) + Arrays.hashCode(ciphertext)`. Verified hashing, deduplication, and tamper detection in `VoiceVaultCryptoTest.kt`.

### Milestone 2: Video Forensics, BioLip & NaN Contract Hardening
- [x] **Video Resource Leak Prevention**: Enclosed `ParcelFileDescriptor` in `pfd.use`, moved `MediaMetadataRetriever.release()` into deterministic `finally` blocks, and enforced explicit bitmap recycling (`bitmap.recycle()`) during frame decoding in `PerioralLandmarkAnalyzer.kt`.
- [x] **MediaPipe 0.10+ Reflection**: Upgraded dynamic reflection to target `com.google.mediapipe.tasks.vision.facelandmarker.*` hierarchy with `models/face_landmarker.task`.
- [x] **BioLip & Malar rPPG Temporal Continuity**: Removed `.filterNotNull()` to preserve original timeline indices for acceleration finite differencing. Enforced contiguous run partitioning ($\ge 4$ frames) in `computeRppgBilateralPhaseAnomaly`, returning `Float.NaN` when cheek ROIs are occluded or lighting is inadequate.
- [x] **Elimination of Toxic Sentinels (`-1f` -> `Float.NaN`)**: Eliminated `-1f` sentinels across `VideoForensicAnalyzer.kt`, `ScanResult.kt`, `SignalBreakdownJson.kt`, and `ForensicCertificateExporter.kt`. Fixed IEEE 754 `Math.max(Float.NaN, x)` collapse in certificate export.

### Milestone 2 Round 2: VideoScorePolicy Remediation & Invariant Hardening
- [x] **IEEE 754 NaN Collapse in Multimodal Video Escalation**: Guarded audio escalation in `VideoScorePolicy.combineMultimodalScore` with finite range checks (`visualThreshold in 0f..1f`, `audioThreshold in 0f..1f`). Replaced `maxOf` with monotonic check `if (candidate.isFinite() && candidate > base) base = candidate`, preventing uncalibrated thresholds from destroying valid visual scores.
- [x] **Restoration of Strict Adversarial Invariants**: Re-established strict invariant checks in `VideoScorePolicyAdversarialTest.kt` across 2,001 fuzzing iterations and added deterministic regression tests for NaN threshold edge cases.
- [x] **UI & ViewModel Architecture Cleanup**: Migrated `ScannerViewModel.kt` to named arguments, safely defaulting unmeasured forensic channels to `Float.NaN` without in-memory `-1f` sentinels.

---

## 2. In-Progress & Staged SOTA Features

- [x] **SigLIP-B/16 Vision Foundation Probe**: 86M parameter ViT backbone (`google/siglip-base-patch16-224`, 89.96 MB INT8) with Tunable Attention Pooling (TAP) over 196 tokens, compiled via zero-overhead Keras mirror folding with zero additional ops.
- [x] **SRM PatchCraft 20-Channel CNN**: 12 SRM + 4 NPR + 4 RGB convolutional stem on native $4 \times 4$ texture mosaics (`-patchcraft-mosaic-tile32`, 142 KiB INT8).
- [x] **Audio Anti-Spoofing (LCNN-MFM + LFCC v3)**: 508 KiB INT8 model, quadratic-window sinc resampler, 128 linear filters, 40 DCT coefficients, voiced duration gate $\ge 300$ ms, Log-Sum-Exp multi-window logit pooling.
- [x] **Offline C2PA Content Credentials**: JUMBF/APP11 parser, COSE_Sign1 ES256 verification, fail-closed `signed = hashVerified && coseSignatureValid && certChainValid` binding against offline trust anchors.
- [x] **VirtualCameraDetector**: Camera2 HAL sensor audit, exposure time range verification, and dynamic hook detection (Xposed/LSPosed/Frida/Pine).
- [x] **AudioSeal Detector ONNX Export Pipeline**: Verified sample-level 1D SEANet export (`ml/scripts/export_audioseal_detector.py`) matching PyTorch with 0.000e+00 parity and exact binomial sign test ($k \ge 12, p < 10^{-4}$).

---

## 3. Remaining Deliverables & Action Items

### Physical Device Lineage Promotion (v10.2 Deliverable)
- [ ] **On-Device Reference Hardware Benchmark**: The current release `onn-prod-v10` is verified in its **desktop-validated** state (`desktop_validated: true`). To promote to `validated: true` per `docs/DEPLOYMENT.md`, attach an independent device measurement report (`measurement_kind: on_device_android`) captured from physical Android hardware (e.g., Google Pixel 8/9, Samsung Galaxy S24).

### Neural Watermarking Mobile Lowering (v10.2 Deliverable)
- [ ] **Direct Keras Mirror for AudioSeal 1D SEANet**: Standard `onnx2tf` fails on 1D convolutions with weight normalization. Implement a native Keras 1D mirror in `ml/src/export/keras_audioseal_mirror.py` to lower the AudioSeal detector weights into `assets/models/audioseal_detector_int8.tflite` with exact agreeing-bit parity.

### Training Corpus Shortcut Mitigation (v11.0 Deliverable)
- [ ] **B-Free Paired Real/Synthetic Twin Dataset Expansion**: Rebuild the visual training corpus using paired diffusion reconstructions (SDEdit / autoencoder twins) of real LAION, Pexels, and DocCI images so each real sample has an identical-content synthetic twin. This will break the acquisition source label correlation ($I(\text{source}; Y) = 0$) and reduce per-source real-image FPR below the required $5\%$ bound.
- [ ] **Acoustic Presentation Degradation Matching**: Route both authentic and spoofed audio corpora through identical telephone/VoIP channel chains (Opus, AAC, AMR-WB, ambient room impulse responses) to eliminate the in-the-wild noise shortcut.

---

## 4. Verification Harness Status

All 6 primary project verification gates are green:
1. `.\.venv\Scripts\python.exe scripts/check_android_consistency.py` — **PASS (89 Kotlin sources, 10 feature contract order verified)**
2. `.\.venv\Scripts\python.exe onn_release_lineage_gate.py --repo .` — **PASS (7/7 Lineage checks OK)**
3. `.\.venv\Scripts\python.exe -m unittest discover -s scripts/tests -v` — **PASS (62/62 tests OK)**
4. `.\.venv\Scripts\python.exe tests/e2e/run_e2e_tests.py` — **PASS (145/145 E2E tests OK)**
5. `.\.venv\Scripts\python.exe -m pytest ml/tests` — **PASS (168 passed, 1 skipped)**
6. `cd android && .\gradlew.bat testDebugUnitTest assembleRelease` — **PASS (326 Kotlin unit tests passed, signed 16 KiB-aligned APK generated)**
