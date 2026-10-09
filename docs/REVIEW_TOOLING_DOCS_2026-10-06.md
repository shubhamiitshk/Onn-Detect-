# ONN-DETECT tooling and documentation review — 6 October 2026

This review covers the current working tree, including the user's uncommitted fixes. It is analysis only; existing source, workflows, and documentation were preserved. The main plan is in `COMPREHENSIVE_REVIEW_2026-10-06.md`. Paths and line references below are repository-relative.

## Findings that affect reproducibility

1. **The full evaluation workflow has an invalid job-level condition.** `.github/workflows/evaluate-models.yml:36` uses `hashFiles()` under `jobs.full-eval.if`. GitHub documents that function for step contexts, while job `if` only supports the status functions. Move detection to a step after checkout, or use an explicit workflow input/preceding job output. Checkpoints and datasets are ignored and are never downloaded by this workflow, so a fresh runner cannot perform full evaluation even after correcting the expression. Make a pinned artifact download part of the evaluation job. See [GitHub context availability](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts).

2. **Linux Android CI cannot execute the committed wrapper directly.** `git ls-files --stage android/gradlew` reports mode `100644`; `.github/workflows/build-android.yml:26` invokes `./gradlew`, with no chmod step. Commit its executable bit, or run `bash ./gradlew`. This is separate from successful Windows compilation.

3. **The training workflow cannot meet its own prerequisites.** `.github/workflows/train-models.yml:15` selects a fresh standard `ubuntu-latest` runner, checks for ignored local datasets, and later forces `--device cuda`. There is no dataset/artifact download or GPU runner configuration. Use a documented self-hosted GPU runner or a training service; run bounded CPU fixtures on PRs. Upload the complete release bundle: both models, constants, Kotlin fusion, export report, environment lock, model card, evaluation manifest. The current upload omits most of these and never runs the evaluator before uploading `evaluation_results.json`.

4. **The shell dataset downloader calls nonexistent files.** `scripts/download_all_datasets.sh:7,10,13` invokes `datasets/download_asvspoof.py`, `download_genimage.py`, and `download_openfake.py`, none of which exists. Replace it with the canonical, corrected downloader CLI. Dataset-specific label mapping must be fixed first (see ML review).

5. **Several checks can report success without establishing the advertised guarantee.** `scripts/check_android_consistency.py` checks source manifest resources and nonempty assets, but it does not check the merged APK manifest, model hashes, tensor I/O, feature order numerically, model/calibration compatibility, or actual prediction parity. It only prints whether the first four feature names occur in Kotlin. Its success is useful static hygiene, not an accuracy or offline certificate.

6. **FFT parity prints a mismatch and exits successfully.** `scripts/verify_fft_parity.py:131–156` reports errors without asserting limits. In this review, its FFT core matched NumPy to `5.23e-12`, while the synthetic full preprocessing chain differed by `0.0792` in the FFT score and `1.47e-4` in flatness. Fix grayscale/resize semantics and require a numerical tolerance. The script simulates Kotlin in Python; add actual JVM/device execution on shared image fixtures.

7. **The resampling parity claim is printed text, not a measured assertion.** `scripts/verify_front_end_parity.py:175–177` prints an almost-perfect correlation with torchaudio without computing it. Anti-alias rejection tests are valuable, but they do not prove equivalence. The NumPy and Kotlin replicas share a parabolic window, while training resamples through torchaudio's Hann implementation. Add measured PCM, LFCC, and score comparisons for speech, noise, edges, and all source rates. Preserve the existing CMVN checks, which passed here.

8. **Export verification can silently skip.** `scripts/verify_export_path.py:56–60` exits zero when TensorFlow cannot be imported. `gen_lfcc_golden.py --torch` also skips the explicitly requested cross-check when torch is absent. Have strict CI/release modes that fail on missing dependencies, while keeping optional local smoke modes explicit. The random-weight export checks establish graph/layout conversion, not trained-model accuracy.

9. **Windows orchestration does not reliably stop on native command failures.** `scripts/setup_and_train.ps1:1` sets `$ErrorActionPreference='Stop'`, but external `uv`/Python exit codes need explicit handling or the appropriate supported native-command preference. Check `$LASTEXITCODE` after every stage; use `Push-Location`/`Pop-Location` in `try/finally`. Fail before exporting if training failed. Preflight `uv`, actual CUDA availability, disk quota, dataset manifest, and environment versions. The file assumes an RTX A4000, while this machine reports an RTX 3050 Laptop GPU.

10. **Packaging permits incomplete releases.** `scripts/package_for_android.sh` warns and continues when one model is absent, selectively copies whichever constants exist, and only prints hashes. Stage a complete versioned bundle, validate all files, then atomically publish it. The packaged hash file should cover constants and generated fusion as well as model binaries; fingerprints establish consistency, while signed APK/release verification establishes origin.

11. **Historical diagnostics contain obsolete contracts.** `diag_accuracy.py` constructs six features, while production uses ten; it also models Android decoding with PIL rather than the real decoder. Other diagnostics use different checkpoint formats and legacy TensorFlow import paths. Turn their useful experiments into one configurable tool with `--model`, `--manifest`, and strict current schema handling. Do not treat their filename or printed output as a regression gate.

## Documentation corrections

The code has made progress on polarity, CMVN, anti-alias filtering, integrity blocking, and removing unused stub modalities. Preserve those changes, but distinguish source changes from retrained and shipped behavior.

- `docs/ARCHITECTURE.md` still describes C2PA lowering scores by `*0.1` and averaging audio/video, whereas current code uses different fusion. Hardware gating is also outdated.
- `docs/DEPLOYMENT.md` contains a wrong constants-generation path in the release checklist, describes integrity as a home badge, and claims source-manifest network restrictions physically guarantee offline behavior. The merged debug and release manifests contradict that claim.
- `README.md` links an absent `LICENSE`, claims cryptographic provenance although the C2PA scanner does not verify signatures, and labels raw classifier output calibrated. The Android 14+ badge disagrees with `minSdk 26`.
- `docs/MODEL_CARD.md` describes the changed audio head alongside an older packaged model without a version distinction. A model card must describe exact binary hashes and preprocessing contract.
- `docs/BENCHMARK_RESULTS.md` correctly marks metrics stale, but still lists deleted stubs as wired neutral modalities. Its claim that all relevant parity problems are fixed is too strong given the resampling and FFT findings.
- `CLAUDE.md` calls CIFAKE-derived data “GenImage thumbnails” and claims generated fusion parity despite no matching current artifact bundle. Keep instructions aligned with canonical dataset identities.
- `ROADMAP.md` is helpful but includes claims that need revision: CommunityForensics-Small has its own non-commercial share-alike/research terms, distinct from the full corpus card; RawTFNet-16's paper table reports 18.53% 2021-DF EER and RawTFNet-32 16.82%; NPR's official implementation uses a larger backbone, so `<5 MB` is an unproven student target; MFM is not generally nonnegative; training on an evaluation partition and then calling it held-out is invalid. Blanket bans on VAD, rotation, and pretrained normalization, mandatory Q90 recoding, fixed dataset percentages, and GPU-duration estimates should become measured experiments. Do not declare an entire research method nonexistent because it is unsuitable for general synthetic-photo detection.
- `TODO.md` records useful historical repairs but has duplicate sections and mixes source fixes with verified released capabilities. Separate code, unit verification, data preparation, training, export, and real-device evidence.

These source checks support the roadmap corrections: [CommunityForensics-Small card](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small), [RawTFNet paper, table 1](https://arxiv.org/html/2507.08227v1), [official NPR repository](https://github.com/chuangchuangtan/NPR-DeepfakeDetection).

## Executed checks

| Check | Outcome | What it establishes |
|---|---|---|
| `python scripts/check_android_consistency.py` | PASS; 53 main Kotlin files | Limited source consistency only |
| `python scripts/verify_front_end_parity.py` | All hard checks passed; NumPy/PyTorch LFCC max delta `1.094e-4` | Tested LFCC/CMVN contract and anti-alias behavior; does not prove runtime-model or resampler equivalence |
| `python scripts/verify_fft_parity.py` | Exit 0; feature-score mismatch `0.0792` | Correct FFT core but preprocessing discrepancy; missing failure assertion |
| Git wrapper mode inspection | `100644` | Linux executable-bit defect |

The local Python environment has Python 3.11.9, torch and TensorFlow, but no torchvision. Full trained evaluation was not possible: datasets, matching PyTorch checkpoints, and exported booster are absent. No dataset downloads or retraining were started.

## Every-file coverage for this review partition

All 28 files in this partition were read. The rest of the 171 original project files are accounted for in the three companion reviews. Generated build/cache files are excluded; relevant generated manifest and test evidence is inspected separately.

| File | Review conclusion/action |
|---|---|
| `.env.example` | Empty W&B key placeholder; document logging as optional, keep dataset/user media out of telemetry. |
| `.gitignore` | Appropriate model/data/cache exclusions; release CI must explicitly retrieve ignored artifacts. Ignore real `.env` secrets and generated egg metadata; add model-release policy. |
| `CLAUDE.md` | Useful feature-contract notes; correct dataset identity, runtime guarantees, stale artifact claims. |
| `README.md` | Honest stale-results caveat; correct network/provenance/calibration/platform/license claims. |
| `ROADMAP.md` | Strong direction on diversity; revise licensing, test/train boundaries, model sizes, blanket preprocessing rules and unmeasured timing. |
| `TODO.md` | Useful repair history; remove duplicate/outdated entries and distinguish code status from shipped validation. |
| `.github/workflows/build-android.yml` | Executable wrapper blocker; add lint, release manifest/permission and device tests, native alignment verification. |
| `.github/workflows/evaluate-models.yml` | Unsupported job `hashFiles`; no artifact retrieval; make critical tests run in pinned environment rather than silently skip. |
| `.github/workflows/train-models.yml` | Missing datasets/GPU, no fresh eval, incomplete artifact upload; use a real training runner with manifests. |
| `docs/ARCHITECTURE.md` | Outdated C2PA, video and hardware flow; generate from current contracts. |
| `docs/DEPLOYMENT.md` | Wrong command path and obsolete guarantees; release checks must inspect actual APK. |
| `docs/BENCHMARK_RESULTS.md` | Keep stale header until fresh end-to-end evaluation; remove references to deleted runtime stubs. |
| `docs/MODEL_CARD.md` | Version model source and shipped assets separately; include calibration and per-generator/domain failures. |
| `docs/TRAINING_GUIDE.md` | Commands require canonical loaders, dtype fix and pinned dependencies; explain actual larger-data adapters and split/calibration boundaries. |
| `scripts/check_android_consistency.py` | PASS is a limited static check; add schema/hash/merged-manifest enforcement. |
| `scripts/diag_accuracy.py` | Obsolete six-feature contract and approximate image decoder; modernize before use. |
| `scripts/diag_export_parity.py` | Useful graph stage isolation; temporary output, no current strict tolerances/CLI; unify diagnostic entrypoint. |
| `scripts/diag_isolate.py` | Useful historical native32 diagnosis; hardcoded absent assets and legacy checkpoint/import handling. |
| `scripts/diag_quant_repro.py` | Useful saturation inspection; calibration order can use only real samples; make balanced and representative. |
| `scripts/download_all_datasets.sh` | Broken nonexistent paths; replace with corrected canonical downloader. |
| `scripts/gen_lfcc_golden.py` | CMVN goldens useful; strict requested torch mode should fail if dependencies absent, test more than a two-tone probe. |
| `scripts/package_for_android.sh` | Partial packages allowed; require complete atomic model release and provenance manifest. |
| `scripts/setup_and_train.ps1` | Native exit codes/preflight/location restoration needed; hardware assumption wrong for this machine. |
| `scripts/train_everything.sh` | `set -e` helpful; cwd assumption and unpinned installation remain; evaluate staged assets before publishing. |
| `scripts/verify_audio_front_end.py` | Finite-score smoke only, despite “calibrated” wording; add model hash/frontend identity and actual Kotlin comparison. |
| `scripts/verify_export_path.py` | Random-weight layout tests valuable; required missing dependencies must fail, trained quantization tests separate. |
| `scripts/verify_fft_parity.py` | FFT math correct; measured preprocessing drift and no failure gate. |
| `scripts/verify_front_end_parity.py` | CMVN/anti-alias checks passed; static resampler-equivalence note is unsupported and dtype contract untested. |
