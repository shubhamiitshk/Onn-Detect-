# Android detection, inference, domain, and test audit

Review date: 6 October 2026. This is a review artifact; application code, model files, and calibration assets were not changed.

## Coverage and limits

Read every file in `android/app/src/main/java/com/onndetect/detection/` (15), `inference/` (7), `domain/` (5), and `android/app/src/test/` (6), and all six assets (four JSON constants plus two TFLite binaries): **39 files in the assigned scope**. Related dependency injection, scanner orchestration, signal UI, training feature extraction, LFCC implementations, exporter/calibration code, and parity scripts were read to trace behavior across boundaries. Every assigned file has an entry in the coverage table below.

The two binary models were parsed, allocated, and invoked with the already installed TensorFlow 2.21.0 CPU interpreter. SHA-256 hashes match the bundled manifest. This is useful binary/interface evidence; it does not certify Android's pinned TensorFlow Lite 2.14.0 runtime, particular codec vendors, latency, battery use, or cross-generator detection accuracy. No new datasets, large dependencies, or model weights were downloaded. No full device scan was run in this subreview. Android build/unit-test results belong to the companion app review.

Severity means: P0 blocks a trustworthy release; P1 materially affects correct verdicts, input coverage, or reliability; P2 should be addressed before broad production use. A **proven finding** is directly established by source inspection or a reproduced computation. A **validation risk** is a behavior that still requires real input/device/model evaluation to quantify.

## Most important conclusion

The shipped artifacts and the current training/runtime source do not form a reproducible, validated detector release. Replacing the backbone or adding a large dataset before fixing polarity, preprocessing, calibration, failure handling, and release provenance would obscure the source of any improvement. This is already an offline prototype with small genuine quantized networks, but its present scores cannot support a claim that it detects all modern generated photos or audio.

## P0: shipping and verdict correctness

### D01. The shipped audio threshold asset contains severely inverted evaluation results

**Proven artifact mismatch.** `android/app/src/main/assets/constants/lcnn_audio_threshold.json:3-7` records AUC **0.0150510204**, EER **0.9607142857**, `n_eval=280`, and threshold **0.2255557030**, with the rule `score >= threshold => flagged synthetic`. In contrast, `ml/evaluation_results.json:13-18` reports audio AUC **0.9760204082** and EER **0.05**. Current `ml/scripts/generate_android_constants.py:103-109` explicitly describes the formerly shipped inverted-polarity defect and now adds sanity checks, yet the committed Android asset still contains those old results. A source fix does not fix the packaged asset.

This proves the release metadata is inconsistent; it does **not** by itself prove the committed network's true AUC is 0.015, since evaluation logic, frontend, checkpoint, and release provenance may differ. Do not simply replace `p` by `1-p`: first validate real/fake labels and model score orientation on known held-out data using the exact shipped TFLite binary and Android-equivalent preprocessing. Recompute the operating threshold on a separate validation split, then lock the evaluation split. Bundle model hash, frontend ID, dataset/split manifest hash, calibration method, score convention, and release ID together. Fail release checks when AUC is inverted or inconsistent across artifacts.

### D02. Inference errors are converted into ordinary verdicts

**Proven defect.** `AudioClassifier.kt:41-51` catches inference exceptions, substitutes **0.5**, and compares it with the shipped **0.2256** threshold: an unavailable/incompatible model is classified **synthetic**. `SRMForensicAnalyzer.kt:32-39` also substitutes 0.5 on any failure; `DetectionPipeline.kt:146-193` still fuses the failed signal and constructs a normal result. A missing signal is not a calibrated neutral measurement, especially with a fusion model not trained for missing modalities.

Return structured statuses (`Success`, `NoUsableSignal`, `UnsupportedInput`, `ModelFailure`, `OutOfDistribution`, `Cancelled`) and an explicit reason. Produce an indeterminate verdict when required inputs fail. Optional-signal degradation needs a separately trained missingness-aware fusion model or a separately validated fallback, and the user should see which evidence was available. Capture diagnostic error codes without retaining user media.

### D03. Shipped XGBoost code cannot have been trained by the current extractor as written

**Proven source/export inconsistency; historical cause is an inference.** `XGBoostMetaClassifier.kt:18,48` splits on EXIF feature 6 at about **0.8** and `:310` splits on C2PA feature 7 at about **0.9**. Current `ml/src/meta_classifier/feature_extractor.py:231-235` defaults those features to fixed **0.1 / 0.05**, and both dataset loops in `ml/scripts/train_xgboost_meta.py:288-300` call the extractor without measured provenance. Constant columns cannot produce those tree splits in a freshly trained model. The extractor documentation `:27-30` acknowledges that older code synthesized metadata from SRM scores, creating shortcut/leakage risk. The committed tree may be from that older pipeline, but its exact historical training origin is not established by this review.

Runtime metadata is different again: `C2PAManifestChecker.kt:41` yields **0.8**, which never crosses the shipped C2PA **0.9** split. EXIF is a heuristic sum with only a narrow path to 0.8. Thus the documented ten-feature contract alone does not ensure semantic parity. Regenerate fusion from features extracted with the **shipped quantized network**, measured metadata, and the exact production image path. Record extractor and training revisions; use out-of-fold base scores for stacking and separate validation/test splits. Avoid class-derived artificial provenance values.

### D04. Aborted scans are represented and saved as authentic-side results

**Proven defect.** `DetectionPipeline.kt:197-213` emits `ScanResult(overallScore=0.5, isAiGenerated=false, confidence=0)` for an abort. `ScannerViewModel.kt:77-84` records every non-null result and routes it to the result page. The unsupported-mode path `DetectionPipeline.kt:88-94` also invokes this method with the misleading reason "insufficient memory".

Introduce a verdict enum and analysis status rather than relying on a Boolean. An abort should not count as a successful classified scan, nor become a "real" entry in history. Persist the failure/status only if history explicitly supports failed scans. Carry the actual failure reason. The UI review covers the resulting false-authenticity presentation.

## P1: preprocessing and media handling

### D05. The mobile image model sees a 32×32 stretched thumbnail

**Proven architecture/input limit; generalization impact requires evaluation.** `SRMForensicAnalyzer.kt:22-27,42-54` resizes every image to **32×32** and discards aspect ratio. Binary input shape confirms `[1,32,32,3]`. This removes the fine texture and local forensic traces the stated task relies on; a small edited or generated region is averaged into the rest of the image. It also differs from `feature_extractor.py:220-225`, which passes images at native resolution. The training/export calibration deliberately accepts only 32×32 images (`quantize_int8.py:71-90`), so the current artifacts are suitable for a thumbnail baseline rather than a high-resolution universal detector.

Choose a documented native-resolution patch route, e.g. 224/256-pixel RGB patches plus residual/frequency patches, with aspect-ratio-preserving global view and local views. Train with the same patch selection and aggregate at image level. Keep patch count/device-tier budgets explicit; calibrate the final image score using those budgets. Include partial edits, natural illustrations, camera denoise/HDR, and recaptured/screenshot images in evaluation. Do not just feed larger images into the current fixed-shape binary.

### D06. FFT preprocessing is not actually cross-platform equivalent

**Proven discrepancy.** `FFTSpectralAnalyzer.kt:38` uses continuous double luma. Python `feature_extractor.py:184-186` invokes `to_gray_u8`, which truncates luma to integer uint8 before FFT. The root review also ran `scripts/verify_fft_parity.py`: the FFT core itself differed only by about **5.23e-12**, but synthetic probe feature scores differed by as much as **0.0792** and spectral flatness by **1.47e-4**. Those probes simulate the Android arithmetic; they are not device decoding measurements.

Resize implementations and decode stages introduce further unverified differences: the scanner may first power-of-two-downsample to under 1024 pixels (`ScannerViewModel.kt:260-269`), then Android resizes to 256; training resizes the original image directly with Pillow. Different downsampling kernels cannot be assumed identical. Make preprocessing versioned and deterministic, preferably inside an exportable graph where appropriate. Test the same real file bytes through Python and instrumented Android, compare RGB pixels, FFT/KL feature values, model scores, fusion margins, and final decisions. Fail CI on meaningful mismatches.

### D07. The purported torchaudio-equivalent resampler uses a different window

**Proven numerical mismatch.** `AudioDecoder.kt:265-267` calls `1 - hk*hk` a Hann window; this is a quadratic/parabolic window, not Hann. It also normalizes weights per sample and uses reflective edges, while training resampling calls torchaudio defaults (`asvspoof_dataset.py:101`, `inthewild_audio.py:134`, and `export_all_models.py:108`). `lfcc_reference.py:223-226` copies the Kotlin algorithm, so comparing Kotlin with that oracle does not establish torchaudio equivalence. `verify_front_end_parity.py:175-178` merely prints a prewritten near-perfect-correlation assertion; it does not measure it.

Reproduction with installed NumPy/PyTorch/torchaudio: create `default_rng(7).normal(0,.1,src_rate//20).astype(float32)`, compare `lfcc_reference.resample_sinc(x,src_rate,16000)` against `torchaudio.functional.resample(torch.from_numpy(x),src_rate,16000)`, and discard 30 samples at each edge. Relative RMS errors were **4.97%** at 48 kHz, **5.53%** at 44.1 kHz, and **6.75%** at 8 kHz; correlations were **0.99889 / 0.99860 / 0.99784**. This is a preprocessing mismatch; the size of its detector-accuracy effect needs evaluation.

Either port the exact torchaudio kernel/phase/padding contract or deliberately use one shared custom algorithm in both training and Android and retrain. Add broadband, speech, silence, fractional-ratio, boundary, and 8-kHz fixtures, compare PCM and **downstream LFCC/model scores**, and test the actual Kotlin implementation. Tone-amplitude tests alone are insufficient.

### D08. PCM decoding always assumes signed 16-bit samples

**Proven input-format defect for supported non-16-bit PCM.** `AudioDecoder.kt:173-187` always reads `asShortBuffer()` and divides by 32768. Neither the raw-WAV path `:69-96` nor codec format update `:131-135` checks `KEY_PCM_ENCODING`. Float/24-bit/32-bit/unsigned-8-bit WAV samples can therefore be interpreted as unrelated shorts. Android documents the encoding field and says 16-bit is the default **when the field is missing**, not when another encoding is present. See [MediaFormat.KEY_PCM_ENCODING](https://developer.android.com/reference/android/media/MediaFormat#KEY_PCM_ENCODING).

Use a small encoding-aware PCM conversion layer, preserve channel layout/sample count, confirm output format after codec changes, and explicitly reject unsupported formats. The RAW path also fixes scratch storage at 64 KiB; query/read bounded sample sizes or verify the extractor cannot return a larger packet. Create golden WAV/FLAC/AAC/MP3/Opus fixtures with stereo, nonstandard rates, known phase/duration, empty and truncated data.

### D09. Empty, silent, very short, or non-speech audio can produce a normal authenticity verdict

**Proven missing gate; exact input frequency is unmeasured.** `AudioDecoder.kt:87-96,155-166` pads even zero decoded samples to 64,000 zeros. `LFCCExtractor.kt:53-54` then accepts the padded length, with no check of actual usable samples, speech duration, or decoder completeness. Invoking the shipped audio model with a zero LFCC tensor produced **0.01953125**, below the bundled threshold, so an all-zero feature input is assigned authentic-side. This is a model probe, not a full-device codec reproduction.

Return decoded duration, usable sample count, clip/peak/RMS statistics, and codec status. Require enough voiced material for a speech-spoof classifier, or route non-speech to an explicitly trained general-audio classifier. Report silence/no speech as unassessable. Use VAD conservatively and retain pause/context artifacts when training; it should not erase the evidence that distinguishes synthesis.

### D10. A single central four-second audio sample does not assess an entire recording

**Proven coverage gap.** `AudioDecoder.kt:26,62-65` seeks near the middle and materializes only four seconds. A 60-second file can contain a cloned voice at the beginning or end that is never examined. Sync seeking also does not discard pre-target decoded PCM by timestamp, so the actual analyzed window can be earlier than the intended centered interval.

Offer a fast sampled scan and a deeper streaming scan with reported coverage and analyzed timestamps. For deep mode, use bounded overlapping voiced windows, batch/sequence them within a memory budget, aggregate validated window evidence, and return localized time intervals. Keep real vs synthetic identity, generator, utterance, and original recording groups separate in train/test splits. An EER threshold learned on whole benchmark utterances is not automatically valid for a max over many windows: revalidate file-level false-positive rates.

### D11. Video YUV conversion ignores pixel stride and other essential image properties

**Proven conversion defect on interleaved chroma layouts.** `VideoDecoder.kt:151-174` reads U/V bytes at `rowOffset+x/2`; it uses row strides but never `pixelStride`. For chroma `pixelStride=2`, successive reads alternate valid and interleaved bytes and produce corrupted colors. Android states that each plane has independent row and pixel strides. See [Image](https://developer.android.com/reference/android/media/Image.html) and [Image.Plane](https://developer.android.com/reference/android/media/Image.Plane.html).

The same function ignores `image.cropRect`, plane buffer offsets, and actual color standard/range; it applies BT.601 limited-range coefficients to all media. It does not handle HDR/P010. Video track rotation is also not applied. Request/validate a supported output format, obey both strides and crop, apply color information and orientation, and verify frames against a trusted decoder on Qualcomm/Mali/emulator devices. Reject unsupported HDR formats or provide a tested tone-map path; do not silently clamp bytes into plausible images.

### D12. Decoder loops use very long iteration guards instead of cancellation/deadlines

**Proven robustness risk.** Audio permits 50,000 iterations with 10 ms input and output waits (`AudioDecoder.kt:112-130`); a broken codec can occupy a worker for many minutes. Video permits 5,000 iterations per target (`VideoDecoder.kt:80-97`) across up to ten targets. Neither checks coroutine cancellation or elapsed deadline. Audio's zero-size output branch releases the buffer but does not honor EOS (`:149-150`), potentially using the full guard after an empty EOS output.

Use monotonic per-stage deadlines, progress/byte counters to detect stalls, cancellation propagation, and explicit EOS handling regardless of buffer size. Always release a dequeued buffer/image in `finally`; `VideoDecoder.kt:104-107` leaks an acquired image if conversion throws, and successful-looking `hit=true` can be returned when `getOutputImage` was null. Count only actual captured frames. Clean up already-created frames on failure.

### D13. Video sampling and aggregation are not trained as a video detector

**Proven implementation limit; accuracy unmeasured.** `VideoForensicAnalyzer.kt:79-109` averages individual visual signals over sampled frames and applies the still-image meta model once. `VideoDecoder.kt:65-71` samples ten widely spaced timeline points. A short edited segment can be missed, or diluted by averaging. The code's own `VideoForensicAnalyzer.kt:28-32` acknowledges the CIFAKE/video domain shift.

Use a separate video validation/training protocol. Score global/frame/face-region patches and scene-adaptive temporal bursts; train aggregation for the exact sampling policy. Return examined timestamps and missing-modality status. A deep mode may use scene cuts plus random/audited coverage and an inexpensive temporal student. Keep entire source videos and identities grouped to prevent adjacent-frame leakage. Do not convert a photo model's AUC into a video accuracy claim.

## P1: numerical semantics, uncertainty, and provenance

### D14. Score distance from 0.5 is not validated confidence

**Proven semantic defect.** `DetectionPipeline.kt:216-218`, `VideoForensicAnalyzer.kt:138`, and `ScannerViewModel.kt:212` report `2*abs(score-.5)` as confidence. Yet decision thresholds are **0.6277** for images and **0.2256** for audio. A score just above the image decision boundary gives about 0.255 “confidence”; an audio score 0.2 gives 0.6 confidence despite being near its decision boundary. The expression ignores quality, available evidence, out-of-distribution inputs, model failure, and calibration. Even for a .5 threshold, a raw sigmoid is not automatically a population probability.

Fit calibration on a separate representative validation set (temperature/Platt scaling or isotonic calibration where justified), report calibration error/Brier score and reliability by source/compression/resolution/language, and expose a three-way decision with an abstention band selected for target false-positive rates. Distinguish model score, validated operating point, predictive uncertainty, and coverage. Do not label every heuristic score as a probability.

### D15. Maximum multimodal score and OR decisions disagree with probability semantics

**Proven inconsistency.** Video decides synthetic using `visual >= .6277 OR audio >= .2256` (`VideoForensicAnalyzer.kt:124-130`) but sets overall score to `max(visual,audio)` (`:133`) and computes confidence around .5. Example: visual 0.10 / audio 0.24 is flagged synthetic with overall score 0.24; visual 0.60 / audio 0.10 is not flagged with overall 0.60. The score no longer has one coherent operating threshold. OR also increases container-level false-positive risk as more evidence is sampled.

Preserve separate modality verdicts and thresholds. If an aggregate score is needed, learn/calibrate it on labeled combinations (real video+synthetic audio, synthetic video+real audio, both, neither) with missingness and quality inputs. Report the particular implicated modality. Reassess false-positive rates at **recording/container level**, including multiple-window scanning.

### D16. C2PA byte presence is not provenance validation or an AI label

**Proven limitation.** `C2PAManifestChecker.kt:24-41` searches only the first 64 KiB for ASCII `c2pa.org` and returns 0.8 if found. It verifies neither container structure, signature, media hard binding, signer/trust chain, nor assertion contents. It can miss legitimate manifests farther into JPEG/PNG/BMFF assets and can match unrelated/injected text. The claim in `:39` that presence usually indicates generation/strong edits is not a reliable labeling rule. Current C2PA specifies assertions for digital capture, generation, compositing, and editing and prescribes validation. See [C2PA specification](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html).

Build a real offline validation path using a maintained library only after measuring size/device cost; do not assume a Rust/JNI library inevitably crashes low-end devices as the file comment does. Keep states such as absent, malformed, signature valid/untrusted, trusted/asset-bound, AI-generation assertion, AI-edit assertion, digitally captured assertion, and revocation information unavailable/stale. Embedded credentials and a versioned packaged trust policy can be examined offline; remotely referenced or current online revocation data may be unavailable and should remain explicit. Metadata absence should contribute no assurance of either class.

### D17. Provenance adjustment is effectively disconnected, with no SynthID implementation

**Proven wiring fact.** `ScannerViewModel.kt:69-75` passes source URI but no `ProvenanceInfo`; all video/audio results set provenance null. The image pipeline computes scalar EXIF/C2PA features (`DetectionPipeline.kt:141-154`) but never populates `ProvenanceInfo`. A search of all main source found no construction setting `hasSynthID=true` or actual watermark detector. Thus `SignalFusion.kt:19,27` is tested but not populated by a production verification path.

The SynthID branch's unconditional 1.0 depends entirely on a future detector's trustworthiness; a comment at `SignalFusion.kt:14-18` appropriately warns that string matching is insufficient. Represent evidence with provider/method/verification status and validate the detector's supported media/providers. Treat watermark support as specific to participating generators; it cannot supply universal offline coverage. Remove unsupported claims from the product until this works.

### D18. EXIF heuristics confuse normal processing with synthetic origin

**Proven heuristic behavior; false-positive rate unmeasured.** `EXIFAnalyzer.kt:21-24` gives Photoshop, GIMP, Lightroom, and Canva the same manipulation penalty as generator names. `:35-43` penalizes absent dates and 512/1024-square dimensions, which are common normal image workflows and easy to spoof. Calling it `exif_anomaly` does not make it a verified generative signal.

Show measured metadata facts as context, and train/ablate any metadata inputs against source-balanced real and fake examples with identical export/compression/metadata patterns. Prefer validated explicit generation assertions over software-name hints. Ensure classification still works when all metadata is stripped. Use both present and missing metadata in held-out tests and adversarial metadata injection tests.

### D19. PRNU and temporal labels overstate what the code measures

**Proven mismatch between naming and computation.** `PRNUAnalyzer.kt:10-16` candidly says it is **not** PRNU, but `SignalBreakdown.kt:27` displays “PRNU sensor noise.” The implementation resizes to 512, computes Laplacian residual variance (`PRNUAnalyzer.kt:68-76,89-101`), and uses uncalibrated raw thresholds .8 and 8 (`:42-50`). Resizing changes sensor traces; no camera reference pattern, denoising estimator, or fingerprint correlation exists. It is correctly excluded from the calibrated meta score, which should continue until validated.

`FrameCosineDelta.kt:19-49,79-85` is a 16×16 mean-luma cosine heuristic. A stationary authentic scene, slideshow, or ordinary cuts can produce high anomaly, and black vs nonblack returns similarity 1 whenever either vector norm is zero (`:62`). Sample spacing is ignored, so the same motion can have different score at different clip lengths. It is correctly reported only as a heuristic rather than fused.

Rename these to measured texture residual/scene-change descriptors or remove them from verdict-facing explanations. Add validation before treating them as evidence. True camera attribution needs per-camera enrollment/reference and a tested forensic method; generic camera-noise features need natural low-light, blur, computational photography, and compression controls. Temporal modeling should use timestamps, scene cuts, and motion-aware normalization.

### D20. Missing evidence is encoded inconsistently

**Proven data-model problem.** `ScanResult.kt:24-35` uses floats with mixed sentinels: -1 for some unavailable signals, 0 for others. `DetectionPipeline.kt:174-176` sets absent audio/temporal signals to zero; video uses -1 for unavailable audio. `SignalBreakdown.kt:24-26` checks `!=0`, so **-1 is displayed as if measured**, while an actual 0 score is hidden. Visual rows at `:19-21` are always shown, even audio-only results contain -1 for those scores.

Use nullable typed measurement records containing score/value, status, method/version, availability reason, units and calibration meaning. Distinguish measured zero from unmeasured. Also preserve the raw visual fused score, provenance evidence, and final verdict separately; current `provenanceScore` is set to the overall fused score (`DetectionPipeline.kt:177`, scanner `:157,222`) and is not a distinct provenance measurement.

## P1/P2: concurrency, memory, runtime, and integrity

### D21. Shared interpreter and SRM input are not protected during inference

**Proven unsafe ownership; reproduction depends on concurrent calls.** The singleton `LiteRTInferenceEngine` synchronizes model creation only (`:32-44`), then calls `run` outside the lock (`:53`). `SRMForensicAnalyzer.kt:30` shares a mutable input tensor across scans, filled before each inference (`:35-36`). Hilt provides both as singletons (`PipelineModule.kt:34-43`), and scanner's busy flag is per ViewModel, not global. Separate scanner instances/background work can race. Closing the engine also can occur while `run` is active because `close` uses only the creation lock. The official [Interpreter reference](https://developers.google.com/edge/api/tflite/java/org/tensorflow/lite/Interpreter) explicitly says interpreter instances are not thread-safe.

Use one dedicated inference worker/actor or per-model mutex encompassing input fill, invocation, and close. Keep buffers request-owned or borrowed exclusively. If parallel models are desired, use independent interpreter ownership and measured CPU budgets. Add a concurrency regression test with distinct fixed inputs whose expected outputs differ.

### D22. Hardware degradation is declared but never implemented

**Proven defect.** `DetectionPipeline.kt:274-277` returns `MATH_ONLY` for heap <=256 MB or free RAM <400 MB, but `:112-115` only handles `ABSTAIN`; all other values proceed to SRM, FFT, KL, and PRNU. Partial visual/audio states are unused. A 128 MB heap is blanket-aborted although the bundled model weights total under 0.6 MB; feasibility must depend on peak allocations and device tests rather than an unexplained universal cutoff.

Measure per-stage native and Java peak memory, decode size, p50/p95 latency and thermal behavior on real low/mid/high-tier devices. Use capability-based budgets for patch count/audio coverage, release buffers promptly, and make degradation affect execution. Every degraded configuration needs calibration/accuracy reporting. Do not silently swap a validated detector for unvalidated math-only scores.

### D23. Bitmap and tensor allocation patterns need clear ownership

**Proven cleanup opportunities; OOM frequency unmeasured.** `PRNUAnalyzer.kt:27` creates a scaled bitmap and never recycles it when distinct from input. `VideoForensicAnalyzer` holds every sampled bitmap for the scan and never explicitly releases it after use or on failure. Scanner's decoded image is not released after collection. Android garbage collection eventually reclaims unreachable objects; these are transient allocation/ownership problems, not proof of a permanent leak.

`LFCCExtractor.kt:111` creates about **16,040 separate FloatArray(1) objects** for a 401×40 tensor, plus row arrays. This is disproportionate overhead for 64 KiB of payload. `FFTUtil.kt:108-113` allocates two arrays for every row transform; KL creates full pixel+gray arrays; PRNU rescales even already-small images to 512. Use contiguous native-order direct buffers or dense primitive arrays and reuse exclusive workspaces with size budgets. The existing `TensorBufferPool` is unused in production. Avoid upscaling low-resolution inputs for noise analysis.

### D24. The mmap loader and interpreter constructor lack exception-safe cleanup

**Proven resource-management omission.** `MemoryMappedModelLoader.kt:21-29` obtains an AssetFileDescriptor and opens a stream from its descriptor without closing the descriptor explicitly; no cleanup surrounds mapping failure. `LiteRTInferenceEngine.kt:35-42` does not close mapped resources/delegate if interpreter construction throws. Sharing the same descriptor makes ownership particularly unclear. The engine's normal `close` is better, but no main-source caller was found, so interpreters intentionally live for the process.

Map in nested `use` scopes if resource lifetime permits; mapping remains valid after closing the mapping FD on the tested platform, otherwise retain/close both required holders explicitly. Clean up on every constructor failure. Provide an app-managed unload policy if many models are added, and make loading/closing serialized with inference. Retest descriptor counts after repeated failed loads.

### D25. Sequential loading is unused and inaccurately described as low-memory fallback

**Proven dead path.** `SequentialModelLoader.kt:12-15` allocates a full heap byte array **and** a full direct copy. Engine constructor depends on concrete `MemoryMappedModelLoader` (`LiteRTInferenceEngine.kt:18`), so mapping failure throws rather than using the declared `ModelLoader` fallback. Packaging already sets `noCompress += "tflite"`.

Either remove unused fallback code or inject the interface and add a tested fallback with streaming/channel writes to avoid double weight storage. Treat fallback memory cost explicitly. `TensorBufferPool.kt:13-20` discards the first pooled buffer if too small even if a suitable later buffer exists; size-class pools with a byte cap are preferable once the pool is actually used. Neither issue presently changes model accuracy.

### D26. Runtime names/contracts should match the deployed artifact

**Proven implementation facts, performance recommendation.** `DelegateSelector.kt:18-22` always selects CPU, despite its “fastest supported” description; `availableProcessors().coerceIn(2,4)` is fixed heuristic. The comment that GPU delegates do not support quantized models is outdated as a general statement: [Google's Android GPU documentation](https://developers.google.com/edge/litert/android/gpu) says quantized model support is available, with dequantization/quantization overhead and thread-affinity considerations. CPU is a reasonable choice for these tiny networks; enabling GPU without benchmark proof may make them slower.

Benchmark a modern **bundled standalone** LiteRT runtime and CPU/XNNPACK path first, then GPU/NPU variants for larger replacement students. Choose by measured end-to-end latency and energy, operator support, initialization overhead, and native peak memory. Keep offline first-run operation without Play-services/network dependency. CPU fallback must work after delegate/interpreter creation failures; current GPU catch only wraps delegate construction/application, not interpreter initialization. If a GPU worker is used, create/run on its required same thread.

`LiteRTInferenceEngine.kt:48-57` hardcodes an output array from a nominal `outputShape` rather than verifying tensors; its two branches do not correctly construct arbitrary shapes. Both present model outputs are `[1,1]` and happen to fit the default. Validate shape/dtype/signatures against a versioned manifest before invoking; use explicit modality-specific adapters rather than a generic scalar API that silently assumes the first output is a fake probability.

### D27. Integrity checks cover binary bytes but not the detector contract

**Proven limitation.** `ModelIntegrityChecker.kt:32-37` treats missing, malformed, or empty manifests as success. It verifies only names in that manifest; omitted required models, thresholds, baseline histogram, Kotlin meta trees, score convention, and frontend version are not covered. `ModelIntegrityGate` caches the entire all-model result for the process. One bad audio binary blocks even visual-only scans via the shared all-model check.

APK signing protects release contents better than a mutable in-package digest from the same source; hashes here are still useful to catch stale export/copy errors. For production, require the exact per-mode model/artifact set, validate schema and expected filenames, and bind hashes for preprocessing, calibration, fusion, and dataset/model revision into a signed release manifest. Missing metadata should be a release failure, with a deliberate developer-only relaxation. Verify the needed modality so an optional unrelated model failure can degrade explicitly. Thread-safe one-time verification and invalidation on supported updates avoid duplicate expensive passes.

### D28. Non-finite output/constant validation is absent at runtime

**Proven omission.** Calls to `coerceIn` do not validate NaN; Kotlin NaN remains NaN. SRM, audio, meta features, and calibration values do not require finite values or valid thresholds in `[0,1]`. `KLCooccurrenceAnalyzer.kt:106-109` only normalizes a baseline sum, without checking positivity/finite values; public histogram scoring can hit `0*ln(0/q)` or invalid `q`. A corrupt-but-readable constant file is not covered by hashes and may propagate invalid verdicts or JSON-storage errors.

At loading and inference boundaries enforce finite scalars, known shapes, valid histogram length/positive normalized mass, threshold range, and known frontend/release IDs. Abort on invalid required contracts; expose optional signal failures. Unit-test NaN/infinity, zero/negative baseline entries, shape mismatches, and stale frontend IDs. For the present valid bundled baseline, no numerical failure was reproduced.

### D29. Heuristic math does not establish generative origin

**Validation risk rather than arithmetic defect.** FFT compares all images to a fixed slope -2 and maps deviation/2 directly to a score (`FFTSpectralAnalyzer.kt:50-55`); KL compares a single horizontal difference histogram to a 500-image baseline and saturates at divergence .3 (`KLCooccurrenceAnalyzer.kt:96-97`). This is a **one-dimensional adjacent-difference histogram**, not a full co-occurrence matrix. Resizing, crops, illustration style, texture, phone sharpening, compression and denoise can produce those same variations. FFT fallback returns the expected natural slope when frequency evidence is insufficient (`:116,124`), rather than an unavailable flag.

Keep interpretable frequency/residual inputs as measured features, not proof. Test real-only false-positive strata and generator-held-out real-world data; ablate whether these add value beyond the neural student. Learn fusion and quality conditioning on actual observations. Empty/degenerate/too-low-resolution spectra need a status, not an authentic score. Expand residual/co-occurrence directions or learned representations only after they improve independent tests.

### D30. SpeakerVerifier is a simulation, not voice identity or spoof detection

**Proven implementation limit.** `SpeakerVerifier.kt:57-70` fills a 192-vector with `(mean*i)%max` and normalizes it. There is no trained ECAPA-TDNN encoder; identity cannot be inferred from this. Its own caveat `:12-17` is correct, and no scan-path caller was found. Enrollment saves this vector and gives no explicit failure result (`:48-50,106-109`); stored vectors are loaded with no 192-length/finite validation (`:112-117`), so a malformed vector can cause index errors in `:77-80`.

Remove security/identity wording for the simulation. If speaker verification is a separate intended feature, use a trained, licensed speaker encoder with multiple enrollment utterances, quality/consent/deletion support, domain/language validation and presentation-attack detection. **A clone can sound like the enrolled speaker**, so a matching biometric embedding is not a substitute for a spoof countermeasure. Enrollment protection and optional matching scores should remain separate from generative-origin classification.

## Existing tests and required validation

There are **42 JUnit tests in six files**: FFT core, normalized KL/histogram behavior, grayscale temporal descriptors, fusion rules, LFCC frontend goldens/invariants, and PCM length/tone/anti-aliasing checks. These are useful regressions for several previously fixed math mistakes. They do not establish deployment accuracy or end-to-end file decoding correctness. `SignalFusionTest` even locks in a definitive SynthID result without a real watermark detector; it tests a conditional rule only.

Before a release, add a focused suite with actual media fixtures and artifact-linked outputs:

1. Known real/fake polarity fixture list for every model and exported artifact; reject inverted/constant scores and calibration mismatch.
2. Same file bytes through Python and real Android: decoded pixels/PCM, feature tensors, FP32/QAT/INT8 scores, meta predictions, thresholds, final status. Include image orientation/color, alpha compositing, high-resolution resize, audio PCM encodings/rates/stereo, video chroma stride/crop/rotation.
3. Failure cases: empty/truncated/unsupported files, no audio, no speech, decoder timeouts/EOS, malformed constants/manifests, wrong tensor shapes, nonfinite output, missing model, cancellation, concurrent requests. These must abstain or give explicit partial status.
4. Accuracy test matrix: unseen generator versions/platforms; generator/time-held-out and source-held-out splits; deduplicated prompt/source/identity/video/utterance groups; social recompression, crop, resize, screen capture, camera recapture, filters/denoise/blur; low/high resolution; Indian languages/accents and codecs for speech; partial AI edits; real illustrations/CGI/screenshots.
5. Device release checks: low/mid/high devices, offline cold first run, airplane mode after install, codec vendor diversity, p50/p95 latency/energy/Java+native peak memory, repeated scans/rotation/navigation/concurrency, quantized artifact operating point and false positives.

Keep thresholds and calibration outside locked test data. Report TPR at fixed FPR (e.g. 0.1%, 1%, 5%) alongside ROC-AUC/PR-AUC, EER where relevant, confidence intervals, sample counts per generator/source, and abstention/coverage. Video/audio multiple-instance aggregation should be evaluated at file level, not only window/frame level. The best mobile architecture is the one that improves these measured outcomes inside the app's actual budget.

## Practical offline architecture direction

These are proposed experiments, not promises of measured superiority in this repository:

- **Image:** retain the current SRM as a reproducible small baseline; compare a pretrained semantic teacher and a residual/frequency teacher offline during training, then distill to a mobile student (for example MobileNetV4/MobileViT/EfficientFormer-family candidates selected after operator/device benchmarking). Use 224/256 native patches plus a global aspect-preserving view, source-balanced mixed modern generators, deliberate cross-generator holdouts, and partial-edit examples. Test RGB-only, residual-only, and dual-branch students; multi-view aggregation must be learned/calibrated and benchmarked. Larger teachers and training datasets do not need to ship on the phone.
- **Speech audio:** first fix and retrain the LFCC/LCNN baseline with one versioned dtype/resampling frontend. Compare modern self-supervised audio teachers plus a graph/attention anti-spoofing head in training, then distill to a compact log-mel/LFCC or waveform student. AASIST-family systems are useful strong comparison baselines, but checkpoint size/operator support and out-of-domain behavior must be measured before deployment. Train with vocoder/TTS/voice-conversion diversity, codecs, room/noise/replay, languages/accents, short clips and partial splices. Separate voice synthesis, human replay, and general generated music/sound if the product promises all audio.
- **Video:** make this a distinct validated route: photo frame evidence plus face manipulation specialist where faces exist, temporal motion/scene-aware evidence, and an independently scored speech track. Expose examined time ranges. Keep small local models loaded sequentially or on an exclusive inference worker, with fast/deep tier budgets.
- **Provenance:** parse and verify supported embedded credentials offline as a separate evidence channel, with explicit trust and access status. It supplements content-based detection and is not assumed to be present. Include neither substring presence nor ordinary metadata omissions as decisive evidence.
- **Release:** version one manifest covering model, preprocessing/frontend, class orientation, quantization, calibration/fusion, supported input types and measured device budgets. Ship bundled weights/runtime and a deterministic offline path. Treat accuracy and unknown-generator behavior as measured scope, not universal certainty.

Current research and dataset choice are covered by the root review. This section is a deployable design direction; final backbone choice needs ablations and device results, not a nominal “best model” ranking.

## Per-file coverage ledger

All paths below are repository-relative. Findings refer to the numbered entries above; “no direct defect” means the file was read, not that it has been exhaustively proven safe.

| File | Lines / binary evidence | Review result |
|---|---:|---|
| `detection/audio/AudioClassifier.kt` | 53 | LFCC->model shape matches binary; failure fallback becomes synthetic (D02), invalid/silent coverage and single-window calibration (D09/D10), threshold source D01. |
| `detection/audio/AudioDecoder.kt` | 312 | Bounded decode avoids whole-file memory; PCM encoding assumed, target seeking/coverage, EOS/deadline/cancellation, actual resampler window mismatch (D07-D10/D12). |
| `detection/audio/LFCCExtractor.kt` | 164 | Read all STFT/filters/log/DCT/CMVN logic; correct per-coefficient time CMVN and n-1 variance; nested tensor overhead D23; finite contracts and supported-input status D28; parity still requires actual speech and bytes. |
| `detection/audio/SpeakerVerifier.kt` | 119 | Simulated statistics vector, no identity/security validity; encrypted storage is useful but silent enrollment failure and shape validation missing (D30). |
| `detection/fusion/XGBoostMetaClassifier.kt` | 573 | Read all 19 trees and sigmoid; emitted tree metadata splits incompatible with current extractor (D03); no shape/finite/missing-direction validation; scalar sigmoid uncalibrated (D14/D28). |
| `detection/provenance/C2PAManifestChecker.kt` | 51 | Bounded full 64 KiB read is sensible; substring only, .8 vs shipped .9 split, no validation, unsupported provenance certainty (D03/D16/D17). |
| `detection/provenance/EXIFAnalyzer.kt` | 53 | Stream cleanup via use is correct; software/date/square heuristics confound origin, only measured contextual evidence (D18). |
| `detection/temporal/FrameCosineDelta.kt` | 94 | 16×16 descriptor and cosine read completely; no timestamp/scene awareness; norm-zero pair returns identical; reported-only heuristic (D19). |
| `detection/temporal/VideoDecoder.kt` | 189 | Seek sampling bounded by count; YUV stride/crop/color/rotation defects, null-image hit, long guards, exception cleanup, allocations (D11-D13/D23). |
| `detection/video/VideoForensicAnalyzer.kt` | 151 | Still-trained features averaged before fusion; modality OR vs max score/threshold inconsistency, temporal diagnostic-only, audio error silence, frame release (D13/D15/D23). |
| `detection/visual/FFTSpectralAnalyzer.kt` | 151 | FFT core actual radix2; continuous luma vs truncated Python, insufficient-evidence natural-slope fallback, fixed -2 heuristic/flatness semantics (D06/D29). |
| `detection/visual/FFTUtil.kt` | 115 | Radix2 and fftshift logic read; core tests useful and root numerical parity excellent; per-row allocation; 2D public API should validate positive n and pixels size (D23/D28). |
| `detection/visual/KLCooccurrenceAnalyzer.kt` | 111 | Correct clamped horizontal diff histogram and calibrated asset loading; naming not full cooccurrence, arbitrary saturation, resolution drift, baseline finite/positive validation (D06/D28/D29). |
| `detection/visual/PRNUAnalyzer.kt` | 103 | Not PRNU despite UI name; residual thresholds unvalidated, scale changes evidence, temporary bitmap cleanup; excluded from meta appropriately (D19/D23). |
| `detection/visual/SRMForensicAnalyzer.kt` | 96 | 32×32 input matches binary; resizing destroys local forensic detail, shared mutable input/failure neutral; CalibrationStore silently defaults malformed threshold/constants (D02/D05/D21/D28). |
| `inference/DelegateSelector.kt` | 49 | Forced CPU can be sensible tiny-model baseline; misleading fastest/GPU INT8 comment, fixed thread count, constructor-failure fallback and GPU worker affinity (D26). |
| `inference/LiteRTInferenceEngine.kt` | 77 | Lazy caching and normal close present; run outside lock, constructor cleanup, concrete loader, output-shape assumptions, process-lifetime memory (D21/D24-D26). |
| `inference/MemoryMappedModelLoader.kt` | 31 | Assets uncompressed in build; descriptor/stream ownership and failure cleanup need explicit safe handling (D24). |
| `inference/ModelIntegrityChecker.kt` | 63 | Streaming SHA-256 correct and actual assets match; missing/malformed/empty manifest success, required models and contracts omitted (D27). |
| `inference/ModelIntegrityGate.kt` | 52 | Blocking gate consumed by scan paths is better than cosmetic badge; caches all-model report, unrelated modality blocked, no atomic verify or frontend/calibration checks (D27). |
| `inference/SequentialModelLoader.kt` | 19 | Unused fallback; full heap bytes+direct copy makes “low memory” description misleading (D25). |
| `inference/TensorBufferPool.kt` | 30 | Synchronized obtain/recycle; unused, first too-small buffer discarded, count instead of byte cap (D23/D25). |
| `domain/models/DeviceTier.kt` | 5 | Nominal RAM labels only; real measured memory/latency/capabilities should govern budgets (D22/D26). |
| `domain/models/ScanMode.kt` | 5 | Image/video/audio/cross-modal enums exist but pipeline supports modes inconsistently and misuse abort reason (D04/D13). |
| `domain/models/ScanResult.kt` | 50 | Boolean cannot represent abstention; score/sentinel/confidence/quality semantics and provenance need typed status/versioned evidence; FloatArray data-class equality caveat if used for comparison (D04/D14/D20). |
| `domain/pipeline/DetectionPipeline.kt` | 280 | Integrity and real source-resolution forwarding useful; ignored MATH_ONLY, normal abort result, failure fusing, confidence formula, disconnected provenance, constant temporal features, handcrafted quality values (D02-D06/D14/D17/D20/D22). |
| `domain/pipeline/SignalFusion.kt` | 44 | No unsafe reduction from unverified C2PA; still arbitrary .5 floor, future unverified SynthID certainty, disconnected production inputs; use explicit validated evidence (D16/D17). |
| `test/.../FFTUtilTest.kt` | 66; 4 tests | Core impulse/tone/DC/2D shift good; add dimensions/roundtrip/property input tests and feature-level actual Android parity (D06/D28). |
| `test/.../FrameCosineDeltaTest.kt` | 76; 6 tests | Heuristic consistency tests; no black/nonblack regression or real stationary/cut/timed clip validity (D19). |
| `test/.../KLCooccurrenceTest.kt` | 91; 6 tests | Baseline self-divergence, clamped extremes, degeneracy useful; add finite/positive baseline/actual calibrated asset fixtures and real preprocessing parity (D28/D29). |
| `test/.../LFCCExtractorTest.kt` | 196; 10 tests | Correct CMVN axis and shape/silence/golden regressions; tone goldens narrow; add actual speech/nonstationary/transient and real-rate PCM frontend/model fixtures (D07/D09). |
| `test/.../PcmUtilResampleTest.kt` | 132; 11 tests | Anti-alias/length/tone tests valuable; they cannot prove exact torchaudio kernel parity; add actual torchaudio broadband goldens and boundary/short/rate validation (D07). |
| `test/.../SignalFusionTest.kt` | 48; 5 tests | Tests current conditional rules correctly; SynthID certainty has no verified production detector; missing evidence/typed verdict/failure tests needed (D17/D20). |
| `assets/constants/lcnn_audio_threshold.json` | 8 | Inverted AUC/EER persists in shipped asset; threshold .2256 affects failures and current decision rule (D01/D02). |
| `assets/constants/meta_classifier.json` | 20 | Ten-feature order agrees nominally; test/calibration provenance stale vs source/tree and threshold .6277 differs confidence neutral point (D03/D14). |
| `assets/constants/model_hashes.json` | 7 | Both SHA-256 values match actual binaries; does not cover all detection artifacts or frontend/threshold semantics (D27). |
| `assets/constants/natural_diff_histogram.json` | 26 | 21 positive bins, 500 calibration images; high edge-bin mass plausible clamped diff histogram; native-resolution/source-dependent baseline and stale provenance risk (D06/D29). |
| `assets/models/lcnn_audio_int8.tflite` | 503,368 bytes | SHA256 `2dd0c96a24da39c1b81398f8698e3c1b0e417692ea5e050ca7b57cf4695aff46`; FLOAT32 input `[1,401,40,1]`, output `[1,1]`; 59 INT8 /22 INT32 /2 FLOAT32 tensors; quantized conv/MFM operations; allocated/invoked on local TF2.21. Zero feature score .01953125; constant .5 input .9921875 /1 input .66796875 (probes only). |
| `assets/models/srm_forensic_cnn_int8.tflite` | 99,824 bytes | SHA256 `5bcfefc15fdde3519f3bd1d077b3b61f39f2b5036b1adf86272af9a5f1a71818`; FLOAT32 input `[1,32,32,3]`, output `[1,1]`; 41 INT8 /14 INT32 /2 FLOAT32 tensors; quantized residual/CNN operators; allocated/invoked TF2.21. Constant RGB0 score0 /.5 score .6796875 /1 score .97265625 (probes only). |

The two binary models have quantize/dequantize float interfaces around genuine integer operators, so their `int8` filenames are consistent with internal quantization. They are not float-only fake placeholders. Binary validity and successful CPU invocation still do not establish forensic accuracy on modern generator platforms.

