# Architecture

ONN-DETECT runs media analysis locally on Android with bundled model assets and a standalone LiteRT CPU runtime. Training and export are separate Python workflows; training does not need to happen on the phone. The architecture enforces zero network access (no `android.permission.INTERNET`), 16 KiB ELF page alignment for Android 15/16, and strict cryptographic lineage verification.

## Results and evidence

`ScanResult` carries `AnalysisStatus`, `DetectionVerdict`, a reason, release/frontend/calibration versions, an optional decision threshold, and measured coverage. The verdict has three states: `SYNTHETIC`, `AUTHENTIC`, and `INCONCLUSIVE`. Failures, unsupported media, cancellation, unavailable evidence, and unvalidated models must not produce an authentic or synthetic verdict. The legacy Boolean is compatibility data; callers must render and persist the typed verdict/status.

A raw network sigmoid or heuristic descriptor is not a calibrated probability. Unavailable signals use `-1` and are omitted from numeric UI displays; zero is a valid measured value. No score-distance-from-0.5 confidence rule remains. Confidence is calibrated against held-out validation distributions. Heatmaps are unavailable; the app does not manufacture explanations.

Historical results created before this contract are treated as unvalidated. Room migrations preserve history and add explicit assessment metadata. Imported media is processed from a bounded private temporary copy and cleaned up; source URI identity and raw file bytes are not stored as an authenticity claim.

## Image route

```text
Selected file -> bounded import and sRGB/alpha/orientation-aware bitmap decode
 -> DetectionPipeline
    input checks -> required-image asset/contract verification
    -> memory budget selects full diagnostics, math-only diagnostics, or abstention
    -> per-request RGB float32 tensor sized from the real TFLite input shape
    -> SRM model score, spectral descriptors, horizontal-difference histogram
    -> optional residual-energy texture proxy
    -> validated fusion only when its release and calibration contract passes
 -> typed result -> local history and result screen
```

The visual pipeline deploys two complementary neural engines:
- **Residual Spatial Engine (`srm_forensic_cnn_int8.tflite`)**: Operates on `[1, 128, 128, 3]` float32 input. Dynamic frontend routing dispatches `-patchcraft-mosaic-tile32` checkpoints to `extractPatchCraftMosaics` (which extracts 16 high-frequency 32x32 texture tiles assembled into a 128x128 mosaic) and legacy checkpoints to `createScaledBitmap` (`-bilinear-square-v2-size128-float32`).
- **Vision Foundation Probe (`siglip_probe_int8.tflite`)**: Operates on `[1, 224, 224, 3]` float32 input (89.96 MB INT8). Employs Tunable Attention Pooling (TAP) over 196 patch tokens folded directly into the pre-trained `_SiglipMAPAttention` head (`vm.head`), introducing zero new ops and 0 ms inference overhead.
- **Multi-Instance Localization (MIL)**: The diagnostic and training stack (`patch_extractor.py`) supports full-resolution patch extraction with top-$k$ gradient/texture selection, Log-Sum-Exp (LSE) pooling, and calibrated noisy-OR aggregation.

The scanner and shared Python decoder retain the original minimum image edge for the resolution feature while bounding the working image. New image checkpoints version EXIF orientation, sRGB profile conversion, white alpha compositing and power-of-two cap1024 in rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-sizeN-float32. Pillow and BitmapFactory may sample pixels differently; same-file native decoding/resize parity must still be measured before promoting a replacement release.

FFT uses a 256-square radix-2 power spectrum with uint8-truncated luma, matching the declared Python grayscale convention. Insufficient spectral content is unavailable rather than an authentic-side fallback. KL is a 21-bin horizontal adjacent-pixel difference histogram with positive finite baseline validation. Neither descriptor proves generative origin. The class historically named `PRNUAnalyzer` reports a residual-energy proxy; it does not perform camera fingerprint attribution and does not influence the verdict.

## Audio route

`AudioDecoder` finds an audio track, samples one bounded window near the center (or from the beginning when timing is unavailable), and records the actual starting timestamp, analyzed duration, source duration, valid sample count, RMS, peak, and clipping fraction. It honors PCM encoding: unsigned 8-bit, signed 16/24/32-bit, and float32. Unsupported or truncated PCM layouts are rejected. Codec output-format changes are checked instead of assuming every output buffer contains signed shorts.

The current clip contract is native PCM window selection followed by shared resampling and crop/pad to 64,000 samples at 16 kHz. The resampler is **quadratic-window sinc v1 with reflective/short-signal edge rules and per-sample renormalization**. It is not torchaudio's default Hann resampler. Training must use the same bounded-window/edge ordering; whole-file resampling followed by clipping can differ at boundaries. Real-file Python/Android parity remains a release requirement.

LFCC applies centered reflect padding, a periodic Hann STFT (`n_fft=512`, hop 160), 128 linear-frequency filters, power dB with no top-dB clipping, 40 orthonormal DCT coefficients, and per-coefficient time CMVN with sample-standard-deviation correction. The float32 interface is `[1,401,40,1]`. Runtime creates one contiguous tensor buffer rather than thousands of single-element arrays.

Empty, silent, heavily clipped, or too-short sampled material is inconclusive. The audio classifier incorporates a two-stage speech content and voiced duration gate (`VoicedDurationGate.voicedMs >= 300L` with zero-crossing rate and spectral flux checks). Non-speech audio (music, silence, acoustic background noise) is safely gated as `INCONCLUSIVE` rather than emitting erroneous spoofing verdicts. The training pipeline supports Dual-SSL distillation (`DualSSLToLCNNDistiller`) from `WavLM-Base` and `HuBERT-Base` teacher ensembles into the 0.72 MB LCNN-MFM student across Opus/AAC/MP3 telephony codecs, along with multi-window audio aggregation (`fit_multiwindow_audio_aggregator`).

## Video route

`VideoDecoder` captures up to ten default timeline samples and reports actual captured timestamps. It obeys YUV plane row/pixel strides, buffer positions, image crop, supported SDR color standard/range, and track rotation. Unsupported pixel formats and HDR requiring tone mapping are rejected. Null images do not count as captured frames. Output buffers, images, codecs, and intermediate bitmaps are released on failure/cancellation; the analyzer recycles returned frames after extracting diagnostics.

`VideoForensicAnalyzer` samples timeline frames, evaluates adjacent-frame cosine similarities, performs keyframe reuse for the SigLIP tower (capping at 4 keyframes per video), and extracts BioLip perioral facial landmarks (`BioLipLandmarkAnalyzer`, `PerioralLandmarkAnalyzer`) with MediaPipe FaceMesh. AVFF Audio-Visual Synchrony (`computeAvffLipSyncCorrelation`) evaluates maximum normalized cross-correlation across lag window `[-2, +2]` between per-frame Mouth Aspect Ratio (MAR) and audio RMS envelope. Strict polarity inversion (`1.0 - anomaly`) enforces alignment with slot 8 (`lip_sync_corr`), where authentic synchronized speech yields high positive correlation (neutral default `0.9`).

## Provenance and voice identity

Unverified EXIF fields, substring matches, and caller-supplied flags cannot raise or lower content scores.
- **Offline C2PA Content Credentials**: `C2PAManifestChecker`, `C2PATrustStore`, and `C2PAVerifier` extract and validate signed JUMBF manifest stores from JPEG (`APP11`) and MP4/ISO-BMFF (`uuid` boxes). Validates CBOR claim structures, COSE Sign1 ECDSA/RSA signatures, and Merkle leaf hash assertions against offline PEM root trust anchors (`c2pa_trust_anchors.pem`) with strict offline airgap enforcement (zero network access).
- **Voice Vault**: Biometric speaker enrollment and verification. Embeddings are encrypted at rest using hardware-backed Android KeyStore AES-256-GCM (`AndroidKeyStoreVoiceVaultCrypto`), supporting offline enrolled profile matching and StrongBox cryptographic attestation. No unencrypted biometric data touches persistent storage.

## Fusion contract

The retained ten-feature schema is:

```text
0 srm_score        1 fft_score       2 kl_score        3 flatness_score
4 resolution_norm 5 is_compressed   6 semantic_probe_score 7 c2pa_present
8 lip_sync_corr   9 landmark_jitter
```

Slot 6 carries the Engine C ("Lite C") semantic measurement and is
**unavailable (`NaN`)** when the probe did not run — never a neutral
constant, because a tree that splits on `0.1` cannot distinguish "low
anomaly" from "not measured". The booster learns a missing branch for that
case and the Kotlin transpiler emits it. Slots 7–9 stay fixed at `0.05`,
`0.9`, and `0.1`; they are compatibility constants, not evidence of
authenticity. A new fusion export must use the same extractor and
independently calibrated quantized base scores. Generated trees validate
feature count/finiteness (NaN permitted in the designated optional slot
only) and carry their release revision.

Engine C ("Lite C") is an **optional** second engine. A release may ship it
or not. When present it is held to the same standard as the required models:
hash-bound in the release manifest, calibrated on a declared split, and
covered by both the desktop parity report and the independent device report.
When absent, the pipeline behaves exactly as it did before the dual engine
existed. See `docs/DUAL_ENGINE_CASCADE.md`.

Neural training uses explicit independent `train`, `val`, `calibration`, and sealed `test` partitions. Visual stacking adds independent `stacking` and `fusion_val` partitions. Dataset manifests bind source/revision, semantic real=0/fake=1 labels, file hashes, and authoritative source/group identities. A missing split or corrupt sample is an error, never an implicit fallback to the dataset root or replacement silence/image.

## Inference ownership and resource limits

The singleton engine serializes interpreter loading, tensor inspection, invocation, and closing behind one lock. Each request owns its input buffer. Models are memory-mapped from uncompressed assets; mapping descriptors close deterministically after mapping. Failed interpreter construction releases resources. Input/output count, positive shape, scalar float32 output, exact buffer size, and finite input/output values are checked. CPU/XNNPACK is the current conservative baseline, not a measured fastest-delegate claim. No GPU dependency is included.

Image work has a 30-second stage budget. Audio decode has a 20-second deadline and five-second stall detection. Video decode has a 30-second total and five-second per-target limit; video analysis has a 60-second limit. Cancellation propagates instead of becoming a normal score. Native inference itself cannot be interrupted mid-kernel; cancellation is checked immediately before/after, so device latency tests are still necessary. The UI also bounds overall scan time and cancels work when leaving/resetting.

## Release integrity and validation

`constants/model_hashes.json` is mandatory. It declares both required neural models, but verification runs for the requested modality so a missing optional audio model does not silently break image-only diagnostics. Missing/malformed/empty manifests, missing required entries, invalid hashes, or changed model bytes fail closed.

`constants/release_manifest.json` is also mandatory. A validated release must bind:
- schema/version, release ID, source commit, score convention and dataset manifest hashes;
- neural hashes, exact tensor dtype/shapes, frontend IDs, and all staged artifact hashes;
- per-model/fusion calibration linked to the model/frontend/data split, at least 100 calibration examples, finite operating thresholds and AUC above chance;
- compiled fusion revision and threshold parity with constants;
- hash-bound evaluation and device reports for that release, with model/frontend parity and offline verification.

The release lineage gate (`onn_release_lineage_gate.py`) strictly verifies SHA-256 bindings between `meta_classifier.json`, `release_manifest.json`, and physical model files. Verification is covered across 417 automated unit and contract tests. The APK verifier (`verify_android_artifact.py`) confirms 16 KiB ELF segment alignment and zero internet permissions.
