# Repository development contract

Read README.md and docs/FIX_STATUS_2026-10-07.md for current behavior. Historical review documents describe the pre-implementation snapshot and are not acceptance evidence for the current tree.

## Truthful output

Keep real=0, synthetic=1 and model scores finite in [0,1]. Missing evidence is unavailable, not zero. Unvalidated releases, failed inference, unsupported media and unvalidated sampled aggregation must abstain. Never invert scores automatically because AUC is below chance. Diagnose labels, frontend and domain.

Do not label a model score as confidence or a calibrated posterior without evidence. Byte-pattern EXIF/C2PA/SynthID matches are not cryptographic attestation. Speaker verification and lip-sync are unavailable; do not synthesize observations for them. Research paper mapping, custom adaptation specifications, and model digests are tracked in docs/EVIDENCE_CHAIN.md. Video heuristics (LADDER-inspired jitter, CHROM-inspired rPPG) are diagnostic telemetry only and decoupled from production verdicts. Conformal prediction theoretical coverage bounds require data exchangeability, which does not hold under arbitrary out-of-distribution generator shifts.

## Shared contracts

The fusion feature order is:
srm_score, fft_score, kl_score, flatness_score, resolution_norm, is_compressed, semantic_probe_score, c2pa_present, lip_sync_corr, landmark_jitter.

Slot 6 carries the Engine C ("Lite C") semantic measurement when a validated probe ran, and is unavailable (NaN) when it did not — a neutral constant would read as a real low measurement to any tree that splits on it. Slots 7–9 are fixed unsupported neutral values in this baseline, not measured proof. Model/fusion retraining must version changes across both platforms.

The optional second engine is a distinct second opinion, not a replacement: Engine A reads high-frequency residual structure, Engine C reads appearance/semantics. When both are present, a disagreement between them is evidence of uncertainty and the cascade abstains rather than picking a winner.

Audio uses the versioned LFCC frontend (16kHz, periodic Hann, NFFT512, hop160,128 linear filters,40 coefficients, no top_db, coefficient CMVN over time with sample std), float32 features and quadratic_window_sinc_v1 resampling. A four-second window has401frames. Torchaudio LFCC/resample is not an interchangeable training shortcut.

Image inference size comes from a checked tensor contract. New image checkpoints use rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-sizeN-float32: orient EXIF, convert to sRGB, composite alpha on white, cap the working image by powers of two, preserve source dimensions, then square-resize. The old32px model is diagnostic only. Pillow and Android bitmap sampling still require independent device parity measurement; source-level arithmetic comparison cannot certify platform decoding.

## Model release

Preserve user datasets and trained binaries. Downloads require a named actual source, immutable revision, explicit destination and semantic labels. Never erase a dataset or repartition by counters. Validate source groups and all derived variants across independent splits.

Train from effective YAML config, select on validation, calibrate on calibration and evaluate on sealed test. Image fusion requires independent stacking and fusion_val. Export stages a complete bundle. Only a validated transaction installs assets together with its matching compiled Kotlin fusion source. Thresholds, preprocessing, weights, reports, dataset lineage and source revision are bound in the release manifest.

## Verification

Use JDK17, SDK36 and the pinned toolchain. Run relevant JVM tests and build/lint; the minified APK and merged permissions/native alignments need inspection. No INTERNET or ACCESS_NETWORK_STATE permission is permitted. Use bash gradlew on Linux (checkout mode may lack executable permission).

Root scripts and ML tests must fail on missing required dependencies and divergent numeric contracts. Synthetic fixtures test implementation behavior; they are not accuracy data. Preserve source changes that predate the task; never reset the worktree or quietly replace model assets.
