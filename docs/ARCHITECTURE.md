# Architecture

ONN-DETECT runs media analysis locally on Android with bundled model assets and a standalone LiteRT CPU runtime. Training and export are separate Python workflows; training does not need to happen on the phone. The current bundled networks are **unvalidated legacy artifacts**. Their binary integrity and ability to run do not establish detection accuracy or class orientation.

## Results and evidence

`ScanResult` carries `AnalysisStatus`, `DetectionVerdict`, a reason, release/frontend/calibration versions, an optional decision threshold, and measured coverage. The verdict has three states: `SYNTHETIC`, `AUTHENTIC`, and `INCONCLUSIVE`. Failures, unsupported media, cancellation, unavailable evidence, and unvalidated models must not produce an authentic or synthetic verdict. The legacy Boolean is compatibility data; callers must render and persist the typed verdict/status.

A raw network sigmoid or heuristic descriptor is not a calibrated probability. Unavailable signals use `-1` and are omitted from numeric UI displays; zero is a valid measured value. No score-distance-from-0.5 confidence rule remains. Confidence is unavailable until a separate validated calibration policy exists. Heatmaps are unavailable; the app does not manufacture explanations.

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

The current SRM tensor is `[1,32,32,3]`. Runtime size is read from the interpreter and compared with the release manifest; it is no longer a hardcoded input-buffer size. This permits compatible square RGB replacement exports. It does **not** create a high-resolution detector: native patches, aspect-preserving global views, partial-edit learning, and their aggregation still require new training and matching preprocessing.

The scanner and shared Python decoder retain the original minimum image edge for the resolution feature while bounding the working image. New image checkpoints version EXIF orientation, sRGB profile conversion, white alpha compositing and power-of-two cap1024 in rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-sizeN-float32. Pillow and BitmapFactory may sample pixels differently; same-file native decoding/resize parity must still be measured before promoting a replacement release.

FFT uses a 256-square radix-2 power spectrum with uint8-truncated luma, matching the declared Python grayscale convention. Insufficient spectral content is unavailable rather than an authentic-side fallback. KL is a 21-bin horizontal adjacent-pixel difference histogram with positive finite baseline validation. Neither descriptor proves generative origin. The class historically named `PRNUAnalyzer` reports a residual-energy proxy; it does not perform camera fingerprint attribution and does not influence the verdict.

## Audio route

`AudioDecoder` finds an audio track, samples one bounded window near the center (or from the beginning when timing is unavailable), and records the actual starting timestamp, analyzed duration, source duration, valid sample count, RMS, peak, and clipping fraction. It honors PCM encoding: unsigned 8-bit, signed 16/24/32-bit, and float32. Unsupported or truncated PCM layouts are rejected. Codec output-format changes are checked instead of assuming every output buffer contains signed shorts.

The current clip contract is native PCM window selection followed by shared resampling and crop/pad to 64,000 samples at 16 kHz. The resampler is **quadratic-window sinc v1 with reflective/short-signal edge rules and per-sample renormalization**. It is not torchaudio's default Hann resampler. Training must use the same bounded-window/edge ordering; whole-file resampling followed by clipping can differ at boundaries. Real-file Python/Android parity remains a release requirement.

LFCC applies centered reflect padding, a periodic Hann STFT (`n_fft=512`, hop 160), 128 linear-frequency filters, power dB with no top-dB clipping, 40 orthonormal DCT coefficients, and per-coefficient time CMVN with sample-standard-deviation correction. The float32 interface is `[1,401,40,1]`. Runtime creates one contiguous tensor buffer rather than thousands of single-element arrays.

Empty, silent, heavily clipped, or too-short sampled material is inconclusive. Signal-energy gates do not constitute speech recognition. The current LCNN is a speech countermeasure: a validated speech/music/environment routing policy and file-level sampled-window calibration are unavailable. **Audio therefore remains inconclusive even if a future neural release's byte and model calibration checks pass.** It is not advertised as an AI music/environmental-sound detector. A central four-second sample does not assess a whole recording, and the UI reports that coverage.

## Video route

`VideoDecoder` captures up to ten default timeline samples and reports actual captured timestamps. It obeys YUV plane row/pixel strides, buffer positions, image crop, supported SDR color standard/range, and track rotation. Unsupported pixel formats and HDR requiring tone mapping are rejected. Null images do not count as captured frames. Output buffers, images, codecs, and intermediate bitmaps are released on failure/cancellation; the analyzer recycles returned frames after extracting diagnostics.

Per-frame image descriptors, a grayscale temporal-change proxy, and a sampled audio-track diagnostic remain separate. There is no trained container-level aggregator. Maximum modality score and OR threshold rules have been removed. **Every video result is inconclusive until a dedicated sampling/aggregation/quality protocol is trained and independently validated.** Sampled frames cannot exclude a short edit elsewhere in the recording.

## Provenance and voice identity

Unverified EXIF fields, substring matches, and caller-supplied C2PA/SynthID flags cannot raise or lower content scores. C2PA verification is unavailable; implementing it requires signed manifest/content-binding verification and a documented offline trust policy. No deployed SynthID verifier exists. The absence of credentials is neutral.

The simulated statistical voice embedding is disabled. `SpeakerVerifier` rejects enrollment/verification as unavailable, and the app exposes no completed biometric identity flow. A future licensed speaker encoder and consent/quality/deletion flow would be a separate feature; matching a voice identity does not prove speech was not cloned.

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

`constants/release_manifest.json` is also mandatory. The shipped legacy manifest has `validated:false`; this deliberately permits diagnostics but never a classified result. A validated release must bind:

- schema/version, release ID, source commit, score convention and dataset manifest hashes;
- neural hashes, exact tensor dtype/shapes, frontend IDs, and all staged artifact hashes;
- per-model/fusion calibration linked to the model/frontend/data split, at least 100 calibration examples, finite operating thresholds and AUC above chance;
- compiled fusion revision and threshold parity with constants;
- hash-bound evaluation and device reports for that release, with model/frontend parity and offline verification.

The staged installer checks the complete bundle, retains previous assets, installs the matching compiled Kotlin fusion source, and maps the root staging manifest into `constants/release_manifest.json`. The APK verifier checks actual merged permissions, bundled contracts, compiled-fusion source parity, and 64-bit native ELF/ZIP 16 KiB alignment. APK signing authenticates the package; internal hashes additionally catch stale copy/export mismatches.

Desktop math tests and device smoke tests demonstrate specific implementation behavior. They are not cross-generator accuracy, calibrated confidence, real-phone battery/latency, or universal detection evidence. Promotion needs independent datasets and actual device/file-decoding tests with the final shipped artifact.
