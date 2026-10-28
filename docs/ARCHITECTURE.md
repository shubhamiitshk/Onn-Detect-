# Architecture — ONN-DETECT (Release v10.1)

ONN-DETECT runs media forensics locally on Android with bundled quantized neural model assets and a standalone LiteRT CPU runtime. Model training and export are separate Python workflows; no training or external computation is performed on the mobile device. The architecture strictly enforces an OS-level zero-network sandbox (complete exclusion of `android.permission.INTERNET` and `android.permission.ACCESS_NETWORK_STATE`), 16 KiB ELF page alignment for Android 15 and Android 16, and strict cryptographic lineage verification.

---

## 1. Results, Evidence & Missing-Value Contracts

`ScanResult` carries `AnalysisStatus`, `DetectionVerdict`, an explanatory reason string, release/frontend/calibration versions, an optional decision threshold, and measured coverage. The verdict has three discrete states: `SYNTHETIC`, `AUTHENTIC`, and `INCONCLUSIVE`. Internal failures, unsupported media, scan cancellation, unavailable evidence, and unvalidated models must strictly produce an inconclusive assessment. The legacy Boolean field is retained for compatibility; callers must render and persist the typed verdict and status.

### The Missing-Evidence Truthfulness Contract (CLAUDE.md)
A raw neural network sigmoid or heuristic descriptor is not a calibrated probability. Unavailable or unmeasured forensic signals use `Float.NaN` (never `-1.0`, `0.0`, or `0.5`) and are omitted from numeric UI displays; zero is a valid, affirmative measured value. Missing values serialize to `null` in JSON attestation receipts per RFC 8259. No score-distance-from-0.5 confidence heuristics remain. Confidence is calibrated against held-out validation distributions. Explanatory heatmaps are driven by localized gradient energy; the application never manufactures explanations.

Historical scan results created before this contract are treated as unvalidated. Room database migrations preserve history while appending explicit assessment metadata. Imported media is processed from a bounded private temporary file and cleaned up; source URI identity and raw file bytes are not stored as an authenticity claim.

---

## 2. Image Forensics Route

```text
Selected file -> bounded import and sRGB/alpha/orientation-aware bitmap decode
 -> DetectionPipeline
    input checks -> required-image asset/contract verification
    -> memory budget selects full diagnostics, math-only diagnostics, or abstention
    -> per-request RGB float32 tensor sized from the real TFLite input shape
    -> Engine A: SRM PatchCraft 20-channel CNN score
    -> Engine C: SigLIP-B/16 Foundation Probe score (linear probe head in shipped INT8; slot 6, or NaN if unbundled)
    -> Spectral descriptors: 2D FFT azimuthal slope, KL adjacent-pixel difference histogram
    -> Rich-vs-poor texture contrast & sliding-window hotspot localization
    -> 10-slot XGBoost Meta-Fusion over the continuous logit base-margin prior
 -> typed result -> local history and result screen
```

The visual forensic pipeline deploys two complementary neural engines:
1. **Residual Spatial Engine (`srm_forensic_cnn_int8.tflite`)**: Operates on `[1, 128, 128, 3]` float32 input (142 KiB INT8). Dynamic frontend routing dispatches `-patchcraft-mosaic-tile32` checkpoints to `extractPatchCraftMosaics` (which selects the top-16 rich-texture 32×32 tiles based on L1 gradient energy and tiles them into a 128×128 mosaic without downsampling-induced anti-aliasing loss) and legacy checkpoints to `createScaledBitmap` (`-bilinear-square-v2-size128-float32`). The first convolutional layer consists of a fixed, non-trainable 20-channel stem: 12 SRM high-pass filters, 4 NPR upsampling kernels, and 4 centered RGB low-pass filters.
2. **Vision Foundation Probe (`siglip_probe_int8.tflite`)**: Operates on `[1, 224, 224, 3]` float32 input (89.96 MB INT8). The shipped production model (hash `1b8c4f3de7822196...`) deploys a frozen SigLIP-B/16 tower with a folded linear attention probe (`head_type: linear_probe`). Advanced heads including Tunable Attention Pooling (TAP) and PatchHead/PRL are maintained as experimental candidate training workflows (`train_siglip_tap.py`) documented in `docs/EVIDENCE_CHAIN.md`. Memory-mapped directly from the APK page cache with < 14 MB Java heap overhead.
3. **Multi-Instance Localization (MIL)**: The diagnostic stack (`patch_extractor.py`, `HeatmapViewer.kt`) supports full-resolution patch extraction with top-$k$ texture selection, rich-vs-poor contrast evaluation (`analyzeWithContrast`), and sliding-window anomaly localization (`extractHotspotBoundingBox`) for reticle overlays.

Image preprocessing normalizes EXIF orientation, sRGB profile conversion, solid white alpha compositing, and power-of-two cap1024 (`rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-sizeN-float32`). FFT computes a 256-square radix-2 power spectrum with uint8-truncated luma, matching the declared Python grayscale convention. KL is a 21-bin horizontal adjacent-pixel difference histogram. The diagnostic class historically named `PRNUAnalyzer` reports a residual-energy proxy; it does not perform camera attribution and does not alter the verdict.

---

## 3. Audio Forensics Route

`AudioDecoder` captures audio tracks, samples bounded windows near the center (or from the beginning when timing is unavailable), and records starting timestamp, analyzed duration, source duration, valid sample count, RMS, peak, and clipping fraction. It supports unsigned 8-bit, signed 16/24/32-bit, and float32 PCM. Codec output-format modifications are audited continuously.

```text
Audio File -> AudioDecoder (16 kHz PCM extraction)
 -> VoicedDurationGate (Spectral flux & ZCR; requires voicedMs >= 300 ms)
 -> lfcc-v3 Frontend:
    - Quadratic-window sinc resampler (to 16,000 Hz)
    - Periodic Hann STFT (n_fft = 512, hop = 160)
    - 128 linear-frequency filterbank
    - Orthonormal DCT-II (40 cepstral coefficients)
    - Per-coefficient CMVN over time -> Tensor: [1, 401, 40, 1]
 -> LCNN-MFM Backbone (Max-Feature-Map activation, 508 KiB INT8)
 -> Sliding-Window Temperature Log-Sum-Exp Pooling (tau = 3.0)
 -> Calibrated Threshold Comparison (0.35547) -> Audio Outcome
```

1. **Passive Anti-Spoofing (`lcnn_audio_int8.tflite`)**: 508 KiB INT8 LCNN-MFM scoring 4-second LFCC spectrograms in < 25 ms. Calibrated decision threshold: `0.35546875` (`lcnn_audio_threshold.json`), achieving 0.8747 ROC-AUC and 65.0% TPR @ 1% FPR. Sub-band non-vocal features are custom lightweight adaptations inspired by MN-P. Balanced watermark decorrelation training reflects WASP guidelines.
2. **Speech Content & Voiced Duration Gate (`VoicedDurationGate.kt`)**: Non-speech audio (music, silence, acoustic background noise) is safely gated as `INCONCLUSIVE` when voiced duration is under 300 ms (`voicedMs < 300L`), preventing false-positive spoof alarms.
3. **Log-Sum-Exp Window Pooling (`WindowPooling.kt`)**: Multi-window audio aggregation operates in logit space with temperature $\tau = 3.0$, preventing localized glitches from dominating while preserving genuine spoofing indicators.
4. **Proactive Watermarking (AudioSeal)**: Sample-level neural watermark detection via 1D SEANet convolutions emitting continuous presence probability $\in [0, 1]$ (`audioseal-presence-mean-v2`) and multi-window Bonferroni-corrected binomial sign testing. Watermark presence confirms watermark detection, not definitive proof of AI voice generation (authentic speech may carry watermarks). Watermark absence strictly evaluates as unknown provenance (`ProvenanceEvidence.Absent`), never penalizing authentic scores.

---

## 4. Video Forensics Route

`VideoDecoder` captures timeline samples and reports exact timestamps, obeying YUV plane row/pixel strides, image crop, supported SDR color ranges, and track rotation. Frames, output buffers, codecs, and native bitmaps are released and recycled in deterministic `finally` blocks, preventing native graphic memory leaks.

```text
Video File -> VideoDecoder (sampled timeline frames, N >= 3)
 -> Keyframe Selection (Capping SigLIP tower at 4 keyframes)
 -> PerioralLandmarkAnalyzer (MediaPipe 478-point FaceMesh, microsecond timestamps)
 -> BioLip Biomechanical Forensics (Diagnostic Telemetry, inspired by LADDER):
    - 64 perioral landmarks normalized by inter-ocular distance
    - Second-order acceleration jitter across contiguous triplets -> Slot 9
 -> Bilateral Malar rPPG Phase Coherence (Diagnostic Telemetry, inspired by CHROM):
    - Left and right malar cheek green chrominance BVP tracking
    - First-difference detrending & Pearson correlation across contiguous runs (>= 4 frames)
 -> AVFF Cross-Modal Lip-Sync:
    - Mouth Aspect Ratio (MAR) vs. audio RMS cross-correlation across [-2, +2] lag -> Slot 8
 -> VideoScorePolicy Multimodal Escalation:
    - Threshold-aligned audio projection
    - Monotonic candidate update: base = candidate if candidate > base
    - Monotonic candidate update: base = candidate if candidate > base (heuristics decoupled to telemetry by default)
 -> Final Calibrated Video Verdict
```

### VideoScorePolicy Invariant Implementation:
`VideoScorePolicy.combineMultimodalScore` strictly verifies finite bounds on `visualThreshold` and `audioThreshold` (`in 0f..1f`). If thresholds are uncalibrated or non-finite (`Float.NaN`), audio escalation cleanly bypasses without causing IEEE 754 NaN collapse, guaranteeing that valid visual evidence is preserved. Secondary modalities only escalate the synthetic evidence.

---

## 5. Provenance, Cryptography & Hardware Security

Unverified EXIF metadata, substring matches, or caller-supplied flags cannot alter content verdicts.

1. **Offline C2PA Content Credentials (`C2PAVerifier.kt`)**:
   - Walks ISO 19566-5 JUMBF manifest stores across JPEG `APP11` segments and MP4 `uuid` boxes.
   - Enforces a fail-closed conjunction:
     $$\text{C2PA\_Verified} = \text{HashVerified} \land \text{CoseSignatureValid} \land \text{CertChainValid}$$
   - Validates CBOR claim assertions, COSE_Sign1 ES256/RS256 signatures, and `c2pa.hash.data` byte exclusion ranges against offline root trust anchors (`assets/trust/c2pa_trust_anchors.pem`, 32 anchors covering Adobe, Google, Truepic, DigiCert). Zero network access is strictly enforced.
2. **Android KeyMint StrongBox Hardware Attestation (`StrongBoxReceiptSigner.kt`)**:
   - Generates ECDSA P-256 keys inside dedicated hardware security chips (Titan M2 / Knox Vault) with `.setIsStrongBoxBacked(true)`.
   - Thread-safe double-checked locking on `keyLock`, `@Volatile testKeyPair`, and `@JvmOverloads constructor(context: Context? = null)`.
   - Embeds KeyMint ASN.1 attestation extension OID `1.3.6.1.4.1.11129.2.1.17`, verifying `attestationSecurityLevel == 2` and `keyMintSecurityLevel == 2`.
   - Signs RFC 8785 canonical JSON forensic receipts (`scripts/verify_forensic_receipt.py`) and exports $1080 \times 1440$ PNG forensic certificates (`ForensicCertificateExporter.kt`).
3. **Family Safe-Voice Vault (`VoiceVaultCrypto.kt`)**:
   - Biometric voiceprint embeddings sealed using Android KeyStore AES-256-GCM with AAD bound to profile identity.
   - `SealedBlob` implements content-based value equality (`Arrays.equals`/`Arrays.hashCode`), guaranteeing standard value semantics for Room persistence and deduplication.
4. **Video-KYC Injection Shield (`VirtualCameraDetector.kt`)**:
   - Inspects Camera2 HAL hardware capability levels and sensor physical dimensions, detecting static exposure ranges and mock drivers.
   - Audits runtime classloaders for dynamic hooking frameworks: Xposed, LSPosed, Frida, Pine, Epic, VCam, ManyCam.

---

## 6. Fusion Contract & Decision Policies

The single-sourced 10-feature schema (`FusionFeatureSchema.kt`) is:

```text
0: srm_score            1: fft_score          2: kl_score           3: flatness_score
4: resolution_norm      5: is_compressed      6: semantic_probe_score
7: c2pa_present         8: lip_sync_corr      9: landmark_jitter
```

- **Slot 6 Semantics**: Carries the Engine C SigLIP foundation probe score when present and validated. When absent, unbundled, or failed, slot 6 is strictly **unavailable (`Float.NaN`)**. The XGBoost meta-classifier evaluates its learned missing branch with base margin 0.
- **Base-Margin Continuous Logit Prior**: In `XGBoostMetaClassifier.kt`, the tree booster predicts an additive logit residual over the SigLIP continuous logit prior:
  $$P(\text{AI}) = \sigma\left( \text{logit}(P_{\text{SigLIP}}) + \sum_{t=1}^T f_t(\mathbf{x}) \right)$$
  This guarantees that the decision tree refines the probe's ranking rather than diluting confident predictions.
- **Slots 7–9**: Held at compatibility defaults (`0.05`, `0.9`, `0.1`) until populated by active provenance, lip-sync, and landmark tracking.
- **Cascade Arbitration (`CascadePolicy.kt`)**: Ambiguity band $[e_{\text{below}}, e_{\text{above}}] = [0.5801, 0.9546]$ (coverage: 97.25%, abstention: 2.75%). Disagreements inside the band map strictly to `DetectionVerdict.INCONCLUSIVE`, optimizing selective accuracy on decisive classifications.
- **Conformal Prediction Exchangeability & Evidence Chain**: Conformal coverage guarantees require exchangeability ($Z \sim \mathcal{D}$); in out-of-distribution deepfake scenarios, guarantees degrade to empirical bounds. Full paper-to-artifact mapping is maintained in `docs/EVIDENCE_CHAIN.md`.

---

## 7. Inference Ownership & Resource Budgets

1. **Single-Model Synchronization**: `LiteRTInferenceEngine` enforces per-model locking for interpreter execution, ensuring concurrent image and audio threads do not block one another while guaranteeing thread safety.
2. **Memory Budgets**: Models are memory-mapped (`noCompress += "tflite"`, opened via file descriptors), keeping the 89.96 MB SigLIP probe in the OS page cache rather than Java heap. The memory gate monitors per-process heap limits rather than volatile system-wide free RAM.
3. **Execution Deadlines**:
   - Image analysis deadline: 30 seconds.
   - Audio decoding deadline: 20 seconds; stall detection: 5 seconds.
   - Video decoding deadline: 30 seconds total, 5 seconds per target frame.
   - Video analysis deadline: 60 seconds total.
   - Cancellation tokens propagate deterministically across all stages.

---

## 8. Release Lineage & Packaging Invariants

1. **`constants/model_hashes.json`**: Mandatory. Binds exact SHA-256 digests for all shipped TFLite binaries. Missing or modified binaries fail closed.
2. **`constants/release_manifest.json`**: Mandatory. Cryptographically binds release ID (`onn-prod-v10`), git commit revision, input/output tensor shapes, frontend IDs, calibration sample counts, operating decision thresholds, and artifact hashes.
3. **Lineage Gate (`onn_release_lineage_gate.py`)**: Enforces 7 strict checks: TFL3 binary headers, manifest model bindings, model hash synchronization, `meta_classifier.json` visual hash binding, release ID uniformity, frontend lineage consistency, and measured parity bounds.
4. **Android Artifact Gate (`scripts/verify_android_artifact.py`)**: Confirms strict 16 KiB ELF segment alignment for Android 15/16, verifies asset hashes against `release_manifest.json`, and guarantees zero network permissions (`android.permission.INTERNET` and `ACCESS_NETWORK_STATE` absent).
