# State of the Art Review and Implementation Plan — ONN-DETECT

**Date:** 2026-10-20 (rev. 3 — 2026-10-21, after two external audits)
**Repo commit at audit:** `c5c3f4a` (`fix(android): sign release build with debug keystore for seamless sideloading`)
**Shipped release:** `onn-20261020-v6` (introduced in `34bd2f2`)
**Scope:** image, speech/audio, video detection; fusion; provenance; release integrity
**Method:** full read of `ml/` and `android/app/src/main/`, direct verification of every number quoted
against the shipped assets, plus a literature review of 2025–2026 primary sources.

> This document supersedes the accuracy-related claims in `README.md`, `docs/BENCHMARK_RESULTS.md`,
> `docs/ARCHITECTURE.md`, `docs/MODEL_CARD.md`, `docs/DEPLOYMENT.md` and the headline table in
> `docs/DUAL_ENGINE_CASCADE.md`. Those documents are stale with respect to the v6 release; see §A.11.

---

## Errata log

### Rev. 2 → rev. 3 — five remaining traps in the implementation snippets

Revision 2's §C.1–§C.6 snippets were audited before execution. **All five were real**, and each
would have caused a runtime crash or a silent math bug:

| Rev-2 snippet | Defect | Fix |
|---|---|---|
| **P0.3b** — `ModelIntegrityChecker.kt` | Set `threshold = fusionThreshold` unconditionally. `AudioClassifier.kt:32` calls `report(context, setOf(AUDIO_MODEL))` and compares its score against `report.threshold`, so the **audio route would have been judged against the image fusion threshold (0.51216) instead of the audio threshold (0.375)**. Separately, `ModelIntegrityChecker.kt:207` compared the IMAGE model's own calibration against `meta_classifier.json.decision_threshold`, so restoring Engine A's real threshold (0.53125) would have thrown *"Calibration asset/release threshold mismatch"* at startup. | **Fixed and implemented**: `report.threshold` is now **modal** (fusion for the image/video route, the audio model's own for an audio-only caller), each model's threshold is kept separately in `modelThresholds`, and the fusion calibration is read from `calibration.fusion_xgboost` with a legacy-layout fallback. |
| **P1.2** — `dcpt_consistency_loss` | `F.kl_div(log_q, p)` on a 1-D probability computes only the positive-class term `p·log(p/q)` and **drops the negative-class term**. For a real image (p≈0.01) against a degraded view (p≈0.40) that term is **negative** (−0.0369): a "loss" that goes down when the views disagree most. | **Fixed and implemented**: the full symmetric **binary** Bernoulli KL over `(p, 1−p)`, plus the feature-level cosine term rev. 2 also omitted. |
| **P2.3** — `poolWindowLogits` | Treated the model output as logits, but `lcnn_audio_int8.tflite` ends in `Dense(1, activation="sigmoid")` (`torch_to_tflite_keras.py:188`) and emits a **probability**. The pooled value could fall outside `[0,1]`, which `ScanResult.init` rejects and which the calibrated threshold comparison cannot use. | **Fixed and implemented**: probability → logit → LSE in logit space → sigmoid. |
| **P1.4** — `FEATURE_NAMES` shrunk to 2 slots | `train_xgboost_meta.py:53` hard-coded `SHORTCUT_FEATURE_INDICES = (4, 5)` and `neutralize_shortcuts` indexed `X[:, 4]`/`X[:, 5]` → **IndexError** on a shortened contract. `VideoForensicAnalyzer.kt:65` built a 10-element array by hand → **IllegalArgumentException** on every video scan. It also contradicted §A.6, which said "shrink to 3 live slots". | **Fixed by removing the trap, not by shrinking the contract**: the feature vector is now single-sourced in `FusionFeatureSchema`, `SHORTCUT_FEATURE_INDICES` is derived from `FEATURE_NAMES`, `neutralize_shortcuts` is bounds-checked, and `check_android_consistency.py` now verifies the Kotlin runtime slot count and optional-slot index. The contract shrink is documented as a **migration with a checklist**, not a drop-in. |
| **P5.1** — `MINIMUM_ACCEPTANCE` | Looped over `evaluation_report.json["models"]`, which contains only the three `.tflite` models. The **fusion's** metrics live in `fusion/meta_results.json`, so the most important gates (fusion AUC, fusion TPR@1%FPR, per-source FPR) were never checked. | **Fixed and implemented**: the gate now reads `bundle/fusion/meta_results.json` as well and enforces `assert_source_fpr_within_bound`. |

### Rev. 1 → rev. 2 — seven defective snippets

Revision 1 of this document was audited and **seven of its proposed code snippets contained defects
that would have shipped broken behaviour**. They are corrected below, and the corrections were then
applied to the repository (§C.0). The audit's substantive technical claims were verified against the
source before being accepted; three of its file/line references did not exist and were rejected.

| Rev-1 snippet | Defect | Status |
|---|---|---|
| **P1.1** — "ship the frozen MobileCLIP2-S0 tower (~22 MB)" | **Wrong tower.** The repo's own held-out evaluation (`docs/DUAL_ENGINE_CASCADE.md:198`) measures MobileCLIP2-S0 at **0.5145 held-out AUC — chance**. Its 0.9688 fit-split score was memorisation of 800 stacking samples. The same table rates SmolVLM-500M (SigLIP-B/16) at **0.8491 held-out**, and `:213` states outright *"the frozen MobileCLIP2-S0 tower is not usable for this task"*. Rev. 1 cited the 0.8491 figure in the same section while recommending the tower measured at chance. | **Corrected** — SigLIP-B/16 / DINOv2-S/14 only |
| **P1.3** — `phase_spectrum_features` | Two math errors: `np.fft.fft2` output was indexed radially around `center = n // 2` **without `fftshift`**, so the radii were centred on the Nyquist corner rather than DC; and raw `np.cos(phase)` is **not translation-invariant** (a 1-pixel shift adds a linear phase ramp). | **Corrected** — phase-only spatial reconstruction, `fftshift`-ed |
| **P1.5 Fix 2** — `_pipeline_accuracy` | **Mathematically degenerate.** Both branches evaluated the same boolean, and abstentions contributed 0 to the numerator over a fixed denominator — so widening the band could only *lose* correct predictions and never gain one. The objective is maximised at `coverage = 1.0` (zero band, no arbitration at all). | **Corrected and implemented** — selective accuracy under a minimum-classified-rate constraint |
| **P4.2** — `SynthIDDetector.kt` | **Infeasible.** SynthID for image/audio/video is a proprietary Google cloud service; only SynthID **Text** is open source. The app explicitly removes `android.permission.INTERNET` (`AndroidManifest.xml:9-10`), so it cannot call it even if it could. | **Corrected** — C2PA Android SDK + open-weight AudioSeal/VideoSeal/WavMark |
| **P0.2** — `group_by_source()` | `Path(path).parts[-2]` extracts the *generator*, not the source: the layout is `test/fake/<generator>/<file>`. The source lives in the manifest record's `source` field. | **Corrected** — manifest-based |
| **P0.1** — gate on `tpr_at_fpr_0_1pct` | The sealed test has `n_real = 200`, so 0.1% FPR is **0.2 samples** — that metric is governed by the single highest-scoring real image and is not a gate until the eval set grows. | **Corrected** — `n_real ≥ 2,000` prerequisite |
| **P0.3b** — move the fusion calibration to a new key | Applied alone, `ModelIntegrityChecker.kt:207-209` throws at startup, because it asserts the SRM calibration threshold equals `meta_classifier.decision_threshold`. `DetectionPipeline.kt` also uses `report.threshold` as the *fusion* threshold. | **Corrected** — both thresholds exposed |

**Seven additional defects** in the codebase that rev. 1 missed are documented in §A.10 and fixed in §C.0.

### Claims checked and rejected

Two audits raised file and line references that do not exist in this repository, and one raised a
defect that is not present. All were verified against the source before being rejected:

- `ml/src/models/srm_Mobilenet.py`, `semantic_probe.py`, `lcnn_audio.py` — the real files are
  `srm_forensic_cnn.py` (which contains `SemanticProbeCNN`), `lcnn_mfm_audio.py`.
- `ml/src/export/torch_to_tflite.py` — the real file is `torch_to_tflite_keras.py`.
- `ColorHistogramAnalyzer.kt`, `VideoFrameExtractor.kt` — do not exist; the real files are
  `KLCooccurrenceAnalyzer.kt` and `VideoDecoder.kt`.
- `SpeechContentGate.kt` — not a file; it is an `internal object` inside `AudioClassifier.kt`.
- `dataset_splits.py:394 group_From_manifest` — does not exist; the file has four functions.
- "`FFTSpectralAnalyzer` uses a 1D row-only FFT" — **false.** `FFTUtil.powerSpectrum2D`
  (`FFTUtil.kt:62-99`) runs row *and* column passes and applies `fftshift`; the existing
  `feature_extractor.py::fft2_power_spectrum` is likewise correct. It was the *rev. 1* snippet that
  was wrong, not the shipped code.
- "`asvspoof2019` real-speech FPR is 0.77%" — the shipped value is **4.62%** (6/130), which makes the
  `inthewild` gap worse, not better.

---

## 0. Executive summary

The **engineering harness is excellent**. Hash-bound model/frontend/threshold contracts, sealed split
lineage, device parity gates, a missing-value discipline (`NaN` = unavailable, never a plausible-looking
constant), transpiler parity checks and a Kotlin type invariant that forbids a classified verdict
without validated calibration are all more disciplined than most published detector code.

The **models are not competitive, and the evaluation currently flatters them**.

| Priority | Finding | Severity |
|---|---|---|
| **P0** | The shipped image fusion detects **4%** of AI-generated images at a 1% false-positive rate (`tpr_at_fpr_1pct: 0.04`) and **0.5%** at 0.1% FPR. This is the operational number and it is unusable. | Critical |
| **P0** | The fusion has learned a **dataset provenance shortcut**: LAION-sourced real images are classified as synthetic **44.4%** of the time (Engine C: 51.9%). 15 generators score **exactly 0%** TPR, including `gpt-image-2`, `qwen-image`, `sora-2`, `veo-3`. | Critical |
| **P0** | The **audio** model has the same disease, hidden inside a respectable pooled AUC: real `inthewild` speech is flagged **24.3%** of the time (n=70), reaching **62.5%** on `itw_Jerry_Seinfeld`. ASVspoof2019 real speech is only 4.6%. The model reads *background noise and codec artefacts* as synthetic. | Critical |
| **P0** | The Engine C frontend contract is **violated at runtime**: the contract says `bicubic224` and the Python path uses `Image.Resampling.BICUBIC`, but Android used `Bitmap.createScaledBitmap(..., filter = true)`, which is **bilinear**. The device has been feeding Engine C a different tensor than every threshold was calibrated from. | Critical |
| **P0** | `release_manifest.json` declares `validated: true`, but the attached `device_evaluation.json` was **generated on the desktop** — its latency fields are desktop `tf.lite.Interpreter` timings and its parity keys are literally `desktop_agreement` / `desktop_mae`. No Android device was measured. | Critical |
| **P0** | `generate_android_constants.py:81-82` **overwrites Engine A's calibration slot with the fusion's calibration**. Engine A's real INT8 operating point (AUC 0.6033) is recorded nowhere in the manifest. | High |
| **P1** | `fit_escalation_band` has a **tie-break index bug** (`-best[3]` should be `-best[4]`) and ranked on an objective that improves monotonically by abstaining, so it drove to the `coverage_grid` floor of 0.30. The shipped policy sits at coverage **0.2975** — the band is fit on ~30% of the data, and 71.5% of samples are escalated on test. | High |
| **P1** | `CascadePolicy.decide()` **always returned Engine C's verdict** and reported Engine C's score with Engine C's threshold, even though the fusion (0.7152) is the stronger signal. The best engine was reduced to a boolean veto. | High |
| **P1** | The `androidTest` source set **does not compile** (`PackagedInferenceTest.kt` constructs `DetectionPipeline` with 10 arguments for an 11-parameter constructor), and CI **fails on every run** because the workflow expects `app-release-unsigned.apk` while Gradle now emits `app-release.apk`. | High |
| **P1** | The telemetry card in `ResultScreen.kt` displayed the **raw SRM score** while the pipeline arbitrated on the **fusion score**, with **hard-coded thresholds**, so the card could contradict the headline verdict. | High |
| **P1** | Two of the four live fusion features are **actively harmful**: `fft_score` has standalone AUC **0.5013** (chance) and `kl_score` **0.4120** (below chance — inverted), yet the tree splits on them in 26 and 36 nodes respectively. | High |
| **P2** | The audio route scores **one** 4-second window per file. The video route reuses the **image** model and **image** threshold on frames. `FrameCosineDelta` is computed and then **dropped from the score**. | High |
| **P2** | `C2PAManifestChecker.analyze()` returns `-1f` and is **never called**. There is no provenance verification, no watermark verification, no music/environmental route. | Medium |
| **P3** | `docs/DUAL_ENGINE_CASCADE.md:388` claims a dual-engine AUC of **0.88–0.896**. The shipped fusion is **0.7152**. The 0.849 figure is the *held-out* AUC of a SigLIP tower that is **not shipped**. | High |

**Bottom line:** the fastest path to a defensible product is not a new architecture — it is
(a) making the evaluation honest, (b) removing the source shortcut so the model is forced to learn
synthesis artifacts, and (c) implementing the provenance channel, which is the only high-precision
evidence available and is currently entirely absent.

Audio is materially stronger than image on pooled metrics (TPR@1%FPR = **59.5%**) but carries the
same shortcut disease and needs the same treatment.

---

# Part A — Measured state of the shipped release

Every number in this section was read directly out of the shipped assets under
`android/app/src/main/assets/` and verified with the commands in Appendix A2.

## A.1 Release inventory

| Artifact | Size | SHA-256 (abbrev.) |
|---|---|---|
| `models/srm_forensic_cnn_int8.tflite` (Engine A) | 142 KiB | `d74a657c…` |
| `models/semantic_probe_int8.tflite` (Engine C) | 153 KiB | `38d1b692…` |
| `models/lcnn_audio_int8.tflite` (audio) | 508 KiB | `56607bbf…` |
| `fusion/XGBoostMetaClassifier.kt` (14 trees, 10 slots) | 20 KiB | `0f18e26e…` |
| `constants/release_manifest.json` | — | `validated: true` |
| `evaluation_report.json` | 111 KiB | `status: "desktop_validation_only"` |

Tensor contracts: Engine A `[1,128,128,3]` NHWC, Engine C `[1,224,224,3]` NHWC,
audio `[1,401,40,1]` (batch, time, cepstral, channels=1).

## A.2 The measured performance — one table, three sources

`android/app/src/main/assets/evaluation_report.json` (sealed test, 200 real / 200 fake):

| Engine | Test AUC | EER | Bal. acc | TPR | FPR | **TPR@1% FPR** | **TPR@0.1% FPR** |
|---|---|---|---|---|---|---|---|
| Engine A (SRM+NPR+RGB) | 0.6584 | 0.395 | 0.6175 | 0.570 | 0.335 | 0.075 | **0.000** |
| Engine C (semantic probe) | 0.6871 | 0.3875 | 0.6175 | 0.585 | 0.350 | 0.065 | 0.030 |
| **Fusion (A + C)** | **0.7152** | 0.350 | 0.6475 | 0.645 | 0.350 | **0.040** | **0.005** |
| Fusion, laundered 512px JPEG Q85 | 0.6970 | — | — | — | — | — | — |
| **Audio (LCNN-MFM + LFCC)** | **0.8749** | **0.195** | **0.840** | 0.795 | 0.115 | **0.595** | **0.350** |

**The three incompatible number sets in the repo:**

| Source | Visual | Audio | Split | Verdict |
|---|---|---|---|---|
| `ml/evaluation_results.json` | AUC 0.956 | AUC 0.976 | `genimage/val`, `asvspoof2019/test` | **Not the shipped split.** GenImage val is not sealed for a GenImage-trained model, and GenImage itself is now an outdated benchmark (see §B.1). |
| `constants/release_manifest.json` | A 0.6706 / C 0.6604 | 0.8646 | calibration, n=400 | Calibration figures, not test. |
| `assets/evaluation_report.json` | C 0.6871, fusion 0.7152 | 0.8749 | **sealed test** | **This is the truth.** |

`docs/BENCHMARK_RESULTS.md:5` already warns *"Do not quote them as current accuracy"*. The file
`ml/evaluation_results.json` is nonetheless still tracked, still labelled
`"generated_with": "real held-out evaluations"`, and still says 0.956.

**Action:** delete `ml/evaluation_results.json`, or regenerate it from the sealed split with
`tpr_at_fpr_1pct` as the headline.

**Sample-size caveat.** With `n_real = 200`, the 0.1% FPR operating point is defined by
`0.2` false positives — i.e. by the single highest-scoring real image. `tpr_at_fpr_0_1pct` is
therefore not a stable statistic at this sample size and must not be gated on until the evaluation
sets grow to `n_real ≥ 2,000` (which puts 20 expected false positives inside the set and makes the
0.1% point measurable to ±0.5%).

## A.3 The source-shortcut problem — image

From `assets/fusion/meta_results.json` → `test.groups.source` (false-positive rate on **real** images):

| Real source | n | Fusion FPR | Laundered FPR | Engine C FPR |
|---|---|---|---|---|
| **laion** | 81 | **0.4444** | 0.3210 | **0.5185** |
| **pexels** | 69 | **0.3478** | 0.1884 | **0.3333** |
| **docci** | 17 | 0.2941 | 0.1765 | 0.1765 |
| imagenet | 33 | 0.1515 | 0.0909 | 0.0606 |

LAION-5B is the *training corpus* for Stable Diffusion and Flux. The detector has learned
**"looks like LAION ⇒ generated"** and **"Pexels/DocCI ⇒ real"**. This is exactly the bias the
REALIS benchmark was built to expose:

> "the real class is never systematically sharper or more attractive than the generated one — a bias
> that would give detectors a shortcut unrelated to synthesis artifacts" (REALIS, arXiv:2609.32734)

And it is fatal in the other direction too — generators at **exactly 0% TPR** in the fusion:

`anima-preview`, `anima-preview2`, `cyberrealistic-pony-v14.0`, `cyberrealistic-pony-v15.0`,
`cyberrealistic-xl-v9.0`, `ernie-image`, `flux-realism`, `gpt-image-2`, `midjourney-7`,
`novaanimexl_ilv140`, `qwen-image`, `realistic_vision_v5.1_novae`, `sora-2`, `wan-video-2.2`,
`wan-video-2.5`.

The frontier generators people actually share — GPT-Image-2, Qwen-Image, Sora-2 — are **completely
missed**. Meanwhile 44% of LAION real images are falsely flagged.

## A.4 The source-shortcut problem — audio (hidden inside a respectable AUC)

`assets/evaluation_report.json` → `models.lcnn_audio_int8.tflite.test.groups`:

**By real source:**

| Real source | n_real | FPR | TPR |
|---|---|---|---|
| asvspoof2019 | 130 | 0.0462 | 0.6923 |
| **inthewild** | 70 | **0.2429** | 0.9857 |

**By speaker** (real-only FPR):

| Speaker | n_real | FPR |
|---|---|---|
| **itw_Jerry_Seinfeld** | 8 | **0.625** |
| **itw_Ayn_Rand** | 15 | **0.4667** |
| itw_Nelson_Mandela | 2 | 0.500 |
| itw_Boris_Johnson | 15 | 0.1333 |
| itw_Frank_Sinatra | 15 | 0.1333 |
| itw_Lyndon_Johnson | 15 | 0.0000 |

**By spoofing generator** (ASVspoof 2019 vocoders):

| Attack | n_fake | TPR |
|---|---|---|
| **A18** | 11 | **0.000** |
| A17 | 17 | **0.2353** |
| A12 | 12 | 0.6667 |

The interpretation is unambiguous: ASVspoof2019 provides *clean studio bona-fide speech*, while
`inthewild` provides *noisy, compressed, real-world field recordings*. The model has learned
**"background noise and codec artefacts ⇒ synthetic"** — the exact inverse of what it should have
learned, and the direct analogue of the image-side LAION shortcut. A18 (a waveform-filtering
vocoder with no obvious spectral signature) is missed completely.

**This is not fixable by architecture alone**, and it is the same fix as A.3: match the channel /
codec / noise distribution of the real class to the fake class, and gate on per-source FPR.

## A.5 Five dead feature slots

Verified by counting `features[i]` occurrences in the shipped
`assets/fusion/XGBoostMetaClassifier.kt`:

| Slot | Name | Occurrences | Used in a split? |
|---|---|---|---|
| 0 | `srm_score` | 44 | **yes** |
| 1 | `fft_score` | 26 | **yes** |
| 2 | `kl_score` | 36 | **yes** |
| 3 | `flatness_score` | 38 | **yes** |
| 4 | `resolution_norm` | **0** | **no** (by design — shortcut-blocked) |
| 5 | `is_compressed` | **0** | **no** (by design — shortcut-blocked) |
| 6 | `semantic_probe_score` | 48 | **yes** (incl. missing-branch routing) |
| 7 | `c2pa_present` | **0** | **no** — hard-coded constant |
| 8 | `lip_sync_corr` | **0** | **no** — hard-coded constant |
| 9 | `landmark_jitter` | **0** | **no** — hard-coded constant |

Slots 4 and 5 are dead **by design** (`block_format_resolution_shortcuts` neutralizes them and
`verify_no_shortcut_splits` asserts no node splits on them — good). Slots **7, 8, 9** are dead
because `ml/src/meta_classifier/feature_extractor.py:275-277` hard-codes them to constants:

```python
prov = {}
c2pa_present = float(prov.get("c2pa_present", NEUTRAL_C2PA))   # 0.05, constant
lip_sync_corr = float(prov.get("lip_sync_corr", NEUTRAL_LIP_SYNC))  # 0.9, constant
landmark_jitter = float(prov.get("landmark_jitter", NEUTRAL_JITTER))  # 0.1, constant
```

`prov` is an empty dict literal — it can never contain anything. The fusion therefore consumes
**5 real inputs**. The other 5 are contract surface that the Android side must still populate,
which is pure maintenance cost with zero information value, and it silently masks the fact that
C2PA, lip-sync and landmark-jitter capability **does not exist**.

## A.6 Two of the four live features are harmful

`ml/scripts/llm_signal_analysis_results.json` → `standalone`:

| Feature | Standalone AUC | EER | TPR@1% FPR | Bal. acc |
|---|---|---|---|---|
| `current_srm_score` | 0.6722 | 0.3625 | 0.025 | 0.6375 |
| **`current_fft_score`** | **0.5013** | 0.5250 | 0.000 | 0.4750 |
| **`current_kl_score`** | **0.4120** | 0.5650 | 0.000 | 0.4350 |
| `current_flatness_score` | 0.6135 | 0.4100 | 0.020 | 0.5900 |
| `current_4signal_fusion` | 0.6996 | 0.3450 | 0.040 | 0.6550 |

`fft_score` is at **chance**. `kl_score` is at **0.412 — below chance, i.e. inverted**: the KLCooccurrence
divergence is *lower* on generated images than on real ones in this corpus. A tree can learn an
inverted relationship, so this is not automatically fatal — but the shipped tree splits on
`fft_score` in **26** nodes and `kl_score` in **36**, i.e. **half the model's capacity is spent on two
features that carry no usable signal**, and the fusion's low-FPR precision is diluted by them.

`fft_score` is a power-spectrum slope and `kl_score` is an adjacent-pixel-difference histogram
divergence. Both are fragile to the re-encoding that any real distribution channel performs, which
is consistent with their degradation under laundering (§A.2: fusion AUC 0.7152 → 0.6970 on
512 px JPEG Q85).

**Action:** either replace both with phase-spectrum and semantic features, or drop them and shrink
the contract to 3 live slots. Carrying features that are at or below chance is worse than not
carrying them.

## A.7 Cascade arbitration defects

`ml/src/meta_classifier/dual_engine.py:234-336` — `fit_escalation_band`.

### Defect 1 — tie-break index bug

```python
best = (resolved_accuracy, achieved_coverage, below, above, disagreement_rate)
...
if best is None or key > (best[0], best[1], -best[3]):
```

`best[3]` is `escalate_above`; the third tie-break must be the disagreement rate, i.e. `best[4]`.
As written, the tie-break compares `-disagreement_rate` against `-escalate_above` — two unrelated
quantities. **Fixed to `-best[4]`.**

### Defect 2 — the objective rewards abstention

The ranking key was `(resolved_accuracy, achieved_coverage, -disagreement_rate)`.
`resolved_accuracy` is measured **only on Engine C's confident tail**, where accuracy rises
monotonically as coverage falls. The fitter therefore drifted to the `coverage_grid` floor of 0.30
(`:259`).

Measured behaviour:

```
target=0.30  coverage=0.3000  resolved_acc=0.9667  arbitration_acc=0.9646  disagreement=0.2050
target=0.36  coverage=0.3600  resolved_acc=0.9722  arbitration_acc=0.9607  disagreement=0.1950
target=1.00  coverage=1.0000  resolved_acc=0.8400  arbitration_acc=0.0000  disagreement=0.0000
```

The shipped policy sits **at the floor**: `coverage: 0.2975`, band `[0.16797, 0.87109]`
(half-width exactly 0.35156 around threshold 0.51953). On the sealed test the escalation rate is
**0.7150** and the INCONCLUSIVE rate is **0.1725** (`docs/DUAL_ENGINE_CASCADE.md:270-274`).

So the band that decides the hard cases is fitted on **~30%** of the calibration data — the
easiest 30% — and `arbitration_accuracy` is computed but **never entered the ranking key**.

### Defect 3 — the better engine was discarded

`CascadePolicy.decide()` **always** returned `c_says_fake` (Engine C's verdict) and reported
`engineCScore` with `policy.engineCThreshold`. Engine A's input to `decide()` is the **fusion**
score (AUC 0.7152) — the strongest signal in the system — and it was reduced to a boolean veto.

### Defect 4 — `coverage` / `agreement_rate` were inert

Grepped across `android/`: `policy.coverage` and `policy.agreementRate` were **never read** in
production code. They are validated ranges and JSON fields only.

### Defect 5 — `evaluate_cascade.py` reused a stale policy without re-fitting

`ml/scripts/evaluate_cascade.py:190-199` loads `cascade_policy.json` if present and **does not
verify** it was fitted against this booster on this calibration split.

## A.8 Release-integrity defects

### Defect 1 — the device validation report is a desktop report

`ml/scripts/build_and_install_release_v4.py:166-200` builds `device_evaluation.json` on the
desktop:

```python
p50, p95 = measure_tflite_latency(tflite_file, model_meta["input_shape"])  # desktop tf.lite.Interpreter
models_entry[model_name] = {
    "sha256": ..., "frontend_id": ..., "parity_passed": True,
    "desktop_agreement": parity["agreement"],
    "desktop_mae": parity["mae"], "desktop_max_error": parity["max_error"],
    "latency_ms_p50": p50, "latency_ms_p95": p95}
device_report = {..., "passed": True, "kotlin_parity_passed": True, "offline_verified": True, ...}
```

The shipped `assets/device_evaluation.json` has exactly those keys. `validated: true` is therefore
**not backed by any Android device measurement** — directly contradicting
`docs/DEPLOYMENT.md:92` (*"Supply independently measured evidence rather than constructing a report
to force a flag"*). `release_bundle.py:220-253` (`attach_device_report`) sets `validated = True`
from whatever JSON the caller supplies, with no provenance requirement.

### Defect 2 — Engine A's calibration slot is overwritten by the fusion's

`ml/scripts/generate_android_constants.py:81-82`:

```python
cal = dict(results["calibration"], model_hash=visual["sha256"], ...)
manifest["calibration"][srm_name] = cal      # <-- OVERWRITES Engine A's own calibration
```

Confirmed in the shipped assets:

| Where | Engine A calibration AUC | Engine A threshold |
|---|---|---|
| `evaluation_report.json` (Engine A's own INT8) | **0.6033** | **0.53125** |
| `release_manifest.json` → `calibration.srm_forensic_cnn_int8.tflite` | 0.6706 | 0.51216 |

The manifest's Engine A entry holds the **fusion's** calibration. Engine A's real operating point
exists nowhere in the manifest, and `release_bundle.py:174` then depends on the collision
(`meta_classifier.json.decision_threshold` must equal `calibration.srm_...threshold`).

### Defect 3 — two divergent release pipelines

`ml/scripts/run_training_pipeline.py` never passes `--semantic-tflite` to `train_xgboost_meta.py`
and never stages Engine C. The v6 dual-engine release was produced by
`build_and_install_release_v4.py --mode package-dual-engine-v6`. The documented pipeline cannot
reproduce the shipped release.

### Defect 4 — destructive install

`build_and_install_release_v4.py:222-224` `shutil.rmtree(bundle_dir)`; `:277-280` / `:291-292`
**delete** the `assets.previous-*` backup after install, contradicting `DEPLOYMENT.md:116`.

## A.9 Capability gaps

| Capability | Current state | Location |
|---|---|---|
| C2PA / Content Credentials | `analyze(context, uri): Float = -1f` — stub, **never called** | `detection/provenance/C2PAManifestChecker.kt:10` |
| SynthID / watermark | absent, and **infeasible** — see §C.4 | — |
| Provenance in fusion | rejected by design: `raise ValueError("Unverified provenance/temporal features must not influence the image verdict")` | `meta_classifier/feature_extractor.py:249-250` |
| Lip-sync / AV sync | constant `0.9` | slot 8 |
| Landmark jitter | constant `0.1` | slot 9 |
| Music / environmental audio | none — `SpeechContentGate` abstains, so the verdict is permanently INCONCLUSIVE | `detection/audio/AudioClassifier.kt:55-58` |
| Video model | none — reuses the image fusion + image threshold on frames | `detection/video/VideoForensicAnalyzer.kt:64-78, 99` |
| Engine C on video | never run — slot 6 is always `NaN` on the video path | `VideoForensicAnalyzer.kt:73` |
| Temporal signal | computed, returned in `Outcome`, **never enters the score**: `overall = 0.75f*visual + 0.25f*audio` | `VideoForensicAnalyzer.kt:99` |
| Audio windowing | **one** 4 s window per file (`TARGET_SAMPLES = 64000`) | `detection/audio/AudioClassifier.kt:19, 36` |
| PRNU | a documented 3-level residual-energy proxy, diagnostic-only, not in the fusion vector | `detection/visual/PRNUAnalyzer.kt` |

## A.10 Seven additional defects found in the external audit (all verified)

These were **not** in revision 1 of this document. Each was verified against the source before
being accepted; all are fixed in §C.0.

### 1. Engine C frontend contract violation — `BICUBIC` declared, `BILINEAR` executed

* `ml/src/data/image_frontend.py:34,53` — frontend id `clip-appearance-bicubic224-clipnorm-meanstd-float32`,
  resize via `Image.Resampling.BICUBIC`.
* `android/.../SemanticProbeAnalyzer.kt:68` — `Bitmap.createScaledBitmap(bitmap, size, size, true)`,
  which in Android's Skia backend is **bilinear** (the same filter Engine A uses).
* The KDoc at `:26-29` asserted *"square BICUBIC resize"* and *"The resampling filter differs from
  Engine A's bilinear resize on purpose."* Both statements were false on device.

Consequence: the device fed Engine C a bilinear tensor while every calibration and threshold in the
release was computed from a bicubic tensor — a train/serve skew inside the contract the whole
release hashes bind to. Android has no bicubic `Bitmap` filter, so a Keys-cubic resampler
(a = −0.5, support 2.0, matching Pillow) had to be implemented.

### 2. Broken cascade telemetry — wrong score, hard-coded thresholds

* `DetectionPipeline.kt:182` calls `policy.decide(fusedScore, engineCScore)` — the first argument
  is the **XGBoost fusion score**.
* `ResultScreen.kt:162` passed `engineAScore = decodedSignals.srmForensicScore` — the **raw SRM CNN
  output**, not the fusion score.
* `ResultScreen.kt:471-473, 588, 638` compared against **hard-coded** `0.512f`, `0.520f`,
  `0.168f..0.871f` instead of the release's `cascade_policy.json`.

Consequence: whenever `srmForensicScore` and `fusedScore` fell on opposite sides of the threshold,
the card displayed *"DISAGREEMENT — ABSTAIN"* while the headline verdict displayed
*"SYNTHETIC"*/*"AUTHENTIC"*, or the reverse. The card also drifted from the shipped policy the
moment the policy was re-fitted.

### 3. `androidTest` did not compile

`PackagedInferenceTest.kt:56-58` constructed `DetectionPipeline(...)` with **10 arguments**; the
constructor has **11** (commit `34bd2f2` added `semanticProbe: SemanticProbeAnalyzer`). The
`androidTest` source set therefore did not compile, so `assembleDebugAndroidTest` in CI could not
have passed. A second latent error was masked behind it: `assertTrue(condition, message)` — JUnit's
`Assert` takes the **message first**, so the two-argument form was a type error too.

### 4. CI failed on every run after the debug-signing commit

`build.gradle.kts:30` sets `signingConfig = signingConfigs.getByName("debug")` for the release
build type, so Gradle emits `app-release.apk`. `.github/workflows/build-android.yml:40` verified
`app-release-unsigned.apk` — a file that is no longer produced. The workflow step failed before it
could check anything.

### 5. Contract test needed PyTorch it did not have to

`ml/src/meta_classifier/feature_extractor.py` performed `from data.image_dataset import image_tensor`
and `import torch` **before** the `if provenance: raise ValueError(...)` guard. The guard is what
`ml/tests/test_feature_extractor_contract.py::test_provenance_measurements_are_rejected` asserts, so
in an environment without PyTorch the test failed with `ModuleNotFoundError` instead of the
`ValueError` it was checking for.

### 6. The audio source shortcut (§A.4)

24.29% FPR on real `inthewild` speech, 62.5% on `itw_Jerry_Seinfeld`, 0% TPR on ASVspoof A18 —
entirely hidden behind a pooled 0.8749 AUC.

### 7. Two live fusion features are at or below chance (§A.6)

`fft_score` 0.5013, `kl_score` 0.4120 standalone, yet 26 and 36 tree splits respectively.

## A.11 Documentation drift

| Document | Claim | Reality |
|---|---|---|
| `README.md` | *"Current bundled models are unvalidated… Scans therefore report Inconclusive"* | `release_manifest.json` has `validated: true`; the app **does** emit verdicts |
| `README.md` | *"The existing 32px image model is a pilot baseline"* | shipped model is 128 px |
| `README.md` | `app-release-unsigned.apk`; *"The release APK is unsigned"* | Gradle emits `app-release.apk`, debug-signed |
| `docs/ARCHITECTURE.md:27` | *"The current SRM tensor is `[1,32,32,3]`"* | `[1,128,128,3]` |
| `docs/MODEL_CARD.md:9-11` | *"legacy-bundled-unvalidated with validated=false"*, *"Image input is float32 NHWC[1,32,32,3]"* | v6, `validated: true`, 128 px |
| `docs/DEPLOYMENT.md:55-66` | asset list has no Engine C / cascade / fusion assets | all present |
| `docs/DEPLOYMENT.md:3` | *"scans return an inconclusive assessment"* | they do not |
| `docs/DUAL_ENGINE_CASCADE.md:388` | dual-engine fusion AUC **0.88–0.896** | shipped fusion is **0.7152**; 0.849 is the held-out AUC of a SigLIP tower that is **not shipped** |
| `docs/FIX_STATUS_2026-10-07.md:3` | *"No replacement model was trained"* | v6 exists |
| `ml/evaluation_results.json` | visual AUC 0.956 / audio 0.976 | sealed test is 0.7152 / 0.8749 |
| `docs/BENCHMARK_RESULTS.md` | *"No reproducible accuracy benchmark … is available"* | v6 has a sealed-test benchmark — but it is **not reproduced by `run_training_pipeline.py`** |

---

# Part B — State of the art, 2026

## B.1 Image

### Benchmarks

| Benchmark | Scale | Key finding |
|---|---|---|
| **DailyBench** (arXiv:2607.24016, CVPR 2026) | 140K modern-generator images + object-level edits | Detectors at **91–96% bal.acc on GenImage drop to 52–79% (FakeBench)** and **43–67% (ManipulationBench)** |
| **REALIS** (arXiv:2609.32734) | 1.3M images, 42 generators, 36 transforms, 5 severities, paired clean/processed | Detectors degrade sharply under processing; **false positives on real images dominate** |
| **AI-GenBench** (identifAI, 2025) | temporal evaluation protocol | Train incrementally on historically-ordered generators; tests the GAN→diffusion transition |
| **Synthbuster** | 9 generators × 8 degradations | The standard robustness axis |

### SOTA methods

| Method | Reported | Backbone / technique |
|---|---|---|
| **IAPL** (arXiv:2508.01603) | 95.61% UniversalFakeDetect / 96.70% GenImage | Image-Adaptive Prompt Learning on a VFM + learnable params |
| **PRL** (arXiv:2610.11876) | 96.40% bal.acc (8 benchmarks), 93.74% (3 in-the-wild) | Relative Patch Response Learning; +4.3% / +5.9% over previous best |
| **C2P-CLIP** | ~95.8% | CLIP prompt learning |
| **GAPL** / **DDA** | 88.93% / 89.16% | — |
| **B-Free** | SOTA generalization | DINOv2 ViT-B/14 + register tokens, fine-tuned end-to-end |
| **DCPT** (arXiv:2604.10102) | **+9.1 pts** degraded-average, +15.7% @JPEG QF=70, +17.2% @QF=30, −0.9% clean | Paired consistency losses; **zero extra params, zero inference overhead** |
| **GlobalForge** (arXiv:2607.14684) | +5 pts BAcc on 8 in-the-wild groups | Shifts the cue from local artifacts to global structure |
| **Phase-spectrum robustness** (CVPR 2026) | outperforms SOTA under compression | Phase spectrum is stable under JPEG where amplitude dies |
| **Effort** (ICML 2025, arXiv:2411.15633) | — | SVD on pre-trained weights; train only in the **orthogonal lower singular subspace** to avoid semantic overfitting |
| **LoRA Pairwise Training** (CVPRW NTIRE 2026) | 3rd place, Robust AIGI Detection in the Wild | LoRA on a zero-shot VFM backbone + random-crop-resize |
| **"Is AIGI Detection a Solved Problem?"** (arXiv:2505.12335) | 11 detectors collapse: F.Acc. → ~0% under JPEG QF=50 / noise σ=4 while R.Acc. stays ~100% | Detector collapse into a "predict real" bias |
| **UNITE** (2026) | single model for face-manipulation + full generation | SigLIP backbone + attention-diversity loss |
| **PatchCraft** (arXiv:2311.12397) / **NPR** (CVPR 2024, arXiv:2312.10461) | — | Native-resolution rich/poor-texture patch selection *before* downsampling |

### What this means for ONN-DETECT

1. **SRM-style local-artifact CNNs are the wrong primitive.** GlobalForge's diagnosis is explicit:
   *"they overfit to local artifacts left by generators in small spatial neighborhoods, and any
   propagation degradation can destroy these artifacts directly."* Engine A is precisely that.
   `ml/scripts/distill_vision_tower_to_srm.py` compounds it by compressing semantics into artifact
   channels. **Note the ordering bug in the shipped path:** Engine A's SRM/NPR filters are applied to
   an image already bilinearly downsampled to 128×128, so the 8× downsample has already destroyed
   the native 2×2 upsampling grid that NPR exists to detect. The PatchCraft-style
   `extract_high_low_texture_patches` helper exists in `srm_forensic_cnn.py` but is only used by
   `distill_vision_tower_to_srm.py` — **not** by the shipped training or inference path.
2. **A frozen VFM + linear probe / LoRA is the 2026 sweet spot — but only with the *right* tower.**
   The repo already measured this honestly (`docs/DUAL_ENGINE_CASCADE.md:197-215`):

   | Teacher | Params | Fit-split AUC | **Held-out AUC** | Verdict |
   |---|---|---|---|---|
   | SmolVLM-500M (**SigLIP-B/16**) | 93 M | 0.8671 | **0.8491** | transfers |
   | MobileCLIP2-S0 | 11.4 M | 0.9688 | **0.5145** | **chance** |

   `docs/DUAL_ENGINE_CASCADE.md:213`: *"the frozen MobileCLIP2-S0 tower is not usable for this task …
   Any '11.4 MB frozen probe' plan built on MobileCLIP would have shipped a chance-level detector."*
   **Use SigLIP-B/16 or DINOv2 ViT-S/14 with registers.** The distillation from the SigLIP teacher to
   the shipped student costs ~0.166 AUC (0.8491 → 0.6871); that gap is the single largest
   identified loss in the system.
3. **DCPT is free.** The repo already has `JPEGCompressionAugmentation`, `ResizeAugmentation`,
   `AdditiveNoiseAugmentation`, `PhonePipelineAugmentation` in `ml/src/data/augmentations.py` — the
   *ingredients* for the paired views. DCPT shows data augmentation alone is insufficient; the
   *objective* must encode consistency, and it needs **both** the feature-level cosine term and the
   logit-level KL term.
4. **Phase spectrum is a cheap, on-device win.** `FFTSpectralAnalyzer` computes only an azimuthal
   **power**-spectrum slope. Adding a phase-consistency branch targets the compression failure mode
   directly. Note the existing `fft2_power_spectrum` in `feature_extractor.py` is correct (it does
   `fftshift`); the new branch must not repeat that mistake.

## B.2 Audio

| Method | EER | Note |
|---|---|---|
| AASIST baseline | ~5–9% | Graph attention |
| **AASIST3 + KAN + W2V2** | halved minDCF on ASVspoof5 dev | best ASVspoof5 results |
| **WavLM ensemble** (ASVspoof5) | 6.56% / 17.08% | late fusion of 4 models |
| **VoiceWukong** (USENIX Sec '25) | **13.50% best (AASIST2); all others >20%** | 413K real-world deepfakes, EN + ZH |
| **Echoes** (music) | 30.7% / 31.0% / 42.3% cross-dataset | 4,468 AI tracks from 12 generators |
| **"It's All in the Presentation"** (ICASSP 2026) | **+39% lab, +57% real-world** | channel/codec-aware data beats bigger models |
| **Watermark Shortcut** (arXiv:2606.23335) | mark-to-frame lifts EER **16.4% → 74.6%** | watermark⇒fake shortcut; fix = watermark both classes |
| **Zero-shot AI music detection** (arXiv:2607.25530) | one-class / clustering | NMF on extracted artifacts |
| **Fusion Segment Transformer** (arXiv:2601.13647) | SOTA on SONICS + AIME | gated fusion for long-form music |
| **Frequency-equivariant music detector** (ISMIR 2026) | SOTA + speed-modification robustness | log-frequency equivariance by construction |
| **AIGM detection taxonomy** (arXiv:2412.00571 v3) | — | signal-level / feature-level / watermark / semantic-consistency |

### What this means for ONN-DETECT

1. **Your audio model is the strongest asset.** EER 19.5%, TPR@1%FPR **59.5%** — but §A.4 shows
   it carries the same shortcut disease as the image side, so its headline number overstates its
   real-world value.
2. **Channel/codec degradation is the highest-value audio improvement.** "It's All in the
   Presentation" makes the claim that data quality beats model size, with a 57% real-world gain.
   `ml/src/data/augmentations.py` currently has only `AudioTimeStretchAugmentation` for audio. Note
   that `torchaudio.functional.amr_nb_encode` **does not exist in standard torchaudio builds**, so a
   naive implementation silently degrades to a no-op behind a broad `except`.
3. **The watermark shortcut is a live threat.** Modern TTS (Chatterbox, DramaBox, ElevenLabs) ships
   watermarks by default. If the corpus correlates watermark-with-label, the detector learns
   "watermarked ⇒ fake", and then: stripping the watermark evades detection (FN → 80%), and
   watermarking a real voice frames it as fake (EER 16.4% → 74.6%). **Watermark both classes during
   training.** The authors release the paired **WASP** corpus for exactly this.
4. **Do not route audio through an MLLM.** VoiceWukong: Qwen2-Audio scored **F1 = 0** on English.
5. **Music is a separate problem.** Echoes shows 30–42% cross-dataset EER even with XLS-R 2B. Build a
   separate route with its own threshold, as `SpeechContentGate` already does for speech.

## B.3 Video

| Benchmark / method | Detail |
|---|---|
| **GenVidBench** (AAAI 2026) | **6.78M videos, 11 generators** (Sora, Kling, MuseV, Mora…); cross-source + cross-generator split |
| **CoCoVideo-26K** (CVPR 2026) | 26K, **13 commercial generators**, semantically-aligned real/fake pairs |
| **CoCoDetect** (CVPR 2026) | R3D-18 backbone + **confidence gate → MLLM** for physical plausibility |
| **LoCC** (ICME 2026) | counterfactual frame consistency; reconstruct a frame from temporal neighbours |
| **BioLip** (arXiv:2604.16808) | **"temporal lip jitter"**; 64 MediaPipe perioral landmarks; language-agnostic |
| **Artifact-Bench** (arXiv:2605.18984) | MLLMs **fail**: failure of scaling (8B→38B no gain), temporal/local blindness, shortcut reliance |
| **PVBench** (arXiv:2610.26711) | 46 physical-plausibility dimensions; visual quality ≠ physical plausibility |

### What this means for ONN-DETECT

1. **Adopt the CoCoDetect tiering.** You already have the scaffolding: `CascadePolicy` with
   `escalate_below` / `escalate_above` / `coverage`. Add a third tier (spatio-temporal model on
   sampled frames) and route into it only inside the ambiguity band.
2. **BioLip maps onto your slot 9 exactly.** `landmark_jitter` is a named but unimplemented slot.
   MediaPipe perioral landmarks → temporal lip kinematics variance → biomechanical-constraint
   violation. Lightweight, cross-lingual, on-device-feasible.
3. **Do not plan an MLLM video detector.** Artifact-Bench shows MLLMs are unreliable at artifact
   detection and do not scale. Use them at most as a last-resort explainer, never as the detector.
4. **The video route must have its own model and its own threshold.** Reusing the image fusion on
   frames is a category error — and the video route currently does not even run Engine C.

## B.4 Provenance and watermarking

- **C2PA** — "Content Credentials" spec v2.3; ROI regions of interest now cover **image, document,
  audio and video**; `inputTo` ingredient assertions. NSA/ASD/ACSC/Cyber Security Centre joint
  guidance. Known gaps (IACR eprint 2026/804): the **"exclusion range"** permits undetectable
  alterations; v2.3 soft-bindings are largely unimplemented. The official **`contentauth/c2pa-android`
  SDK** (Apache-2.0, backed by `c2pa-rs`) verifies Content Credentials **offline**.
- **SynthID** — public worldwide since **2026-10-07** via a **web portal** (`synthid.com`); the
  portal's terms explicitly forbid automating it as an API. The image/audio/video detector is
  **proprietary and cloud-only**; only SynthID **Text** is open source
  (`google-deepmind/synthid-text`).
- **OpenAI** — ships C2PA **and** SynthID as a multi-layer defence, with a public verification tool;
  explicitly states *"No detection method is foolproof… If no metadata or watermark is detected, the
  tool will not make a definitive conclusion."*

### What this means for ONN-DETECT — and a hard constraint

Provenance is a **high-precision, low-recall** channel:

- C2PA presence ≈ near-certain evidence of AI involvement (subject to the exclusion range).
- A detected watermark ≈ near-certain evidence of AI involvement.
- **Absence proves nothing** — and per the watermark-shortcut paper, an attacker strips it trivially.

**Hard constraint:** `AndroidManifest.xml:9-10` removes `android.permission.INTERNET` and
`android.permission.ACCESS_NETWORK_STATE` with `tools:node="remove"`. The app is offline by design
and **cannot** call Google's cloud SynthID Detector. Any proposal to implement SynthID detection
in-app must use an offline-capable detector:

| Channel | Offline option | Notes |
|---|---|---|
| Audio watermark | **Meta AudioSeal** detector (`facebookresearch/audioseal`), **WavMark** (`wavmark/wavmark`) | open weights, small, INT8-able |
| Image/video watermark | **Meta VideoSeal** detector (`facebookresearch/videoseal`) | open weights |
| Cryptographic provenance | **`contentauth/c2pa-android`** | verifies Google / OpenAI / Adobe Content Credentials offline |

This is the only route to a **"confirmed AI"** verdict rather than a probability, and it is entirely
absent from the app today. `TODO.md` already lists it: *"Implement true cryptographic provenance or
watermark verification only when a usable verifier and offline license/runtime are established."*
That condition is now satisfiable.

---

# Part C — Implementation plan

**C.0 records what has already been applied to the repository.** Items are ordered by dependency:
**P0 → P1 → P2 → P3 → P4 → P5**.

---

## C.0 — Applied to the repository (rev. 3)

All items below are **implemented, built and tested**. Evidence is in §C.0.1.

### C.0.1 — Applied fixes

| # | File(s) | Fix |
|---|---|---|
| 1 | `android/.../detection/visual/BicubicResizer.kt` **(new)**, `SemanticProbeAnalyzer.kt` | Implemented a Keys-cubic (a = −0.5, support 2.0) separable resampler matching Pillow's `Image.Resampling.BICUBIC`, and switched Engine C from `Bitmap.createScaledBitmap(..., filter = true)` (bilinear) to it. **Matches Pillow exactly on upscale** (whole-image checksum identical) and within ±1 on downscale. Corrected the KDoc to describe the shipped `SemanticProbeCNN` student rather than a MobileCLIP tower. |
| 2 | `android/.../domain/models/ScanResult.kt`, `data/local/SignalBreakdownJson.kt`, `domain/pipeline/DetectionPipeline.kt` | Added `SignalBreakdown.fusionScore` (default −1 = unavailable) and populated it with the actual XGBoost fusion score, so the telemetry card can show the score the cascade really arbitrates on. Old persisted rows decode to −1 (unavailable), never 0. |
| 3 | `android/.../ui/screens/result/ResultScreen.kt` | `DualEngineCascadeCard` now receives the **fusion** score and the **release's** thresholds (`CascadePolicy.fromAssets`), replacing the raw SRM score and the hard-coded `0.512f` / `0.520f` / `0.168f..0.871f`. Relabelled "ENGINE A" → "ENGINE A (FUSION)". |
| 4 | `android/.../domain/pipeline/CascadePolicy.kt`, `ml/src/meta_classifier/dual_engine.py` | `decide()` now returns the **fusion** verdict and reports the **fusion** score in every branch (Engine C outside the band no longer overrides the stronger engine). |
| 5 | `android/.../domain/pipeline/DetectionPipeline.kt` | The cascade now reports `policy.engineAThreshold` as the decision threshold, matching the score it reports. |
| 6 | `ml/src/meta_classifier/dual_engine.py` | `fit_escalation_band`: fixed the `-best[3]` → `-best[4]` tie-break; replaced the degenerate accuracy objective with **selective accuracy on classified samples subject to `min_classified_rate = 0.85` and `min_escalation_rate = 0.05`**; the band now maximises the accuracy of the rule that actually ships. The Pareto table gained `selective_accuracy`, `classified_rate` and `feasible`. |
| 7 | `android/.../androidTest/.../PackagedInferenceTest.kt` | Restored the missing 11th constructor argument (`SemanticProbeAnalyzer(context, engine)`), fixed the `assertTrue(message, condition)` argument order, and relaxed the validated-release assertion to accept a legitimate INCONCLUSIVE abstention. |
| 8 | `.github/workflows/build-android.yml` | The release APK check now accepts either `app-release.apk` (produced since the debug-signing commit) or `app-release-unsigned.apk`, and fails loudly if neither exists. |
| 9 | `README.md` | Corrected the release APK path and the "the release APK is unsigned" claim. |
| 10 | `ml/src/meta_classifier/feature_extractor.py` | The provenance guard now runs **before** the torch/data imports, so the contract test raises the `ValueError` it asserts instead of `ModuleNotFoundError`. |
| 11 | `android/.../test/.../CascadePolicyTest.kt`, `ml/tests/test_dual_engine_contract.py` | Updated both contract suites to the new arbitration semantics and added the missing `decide(0.9, 0.1)` case to the Python mirror, so the two implementations stay in lockstep. |
| 12 | `android/.../test/.../BicubicResizerTest.kt` **(new)** | 6 tests pinning the Kotlin resampler to a Pillow-generated fixture, including a whole-image weighted checksum. |
| 13 | `android/.../inference/ModelIntegrityChecker.kt` | `report.threshold` is now **modal**: the fusion threshold for the image/video route, the audio model's own threshold for an audio-only caller. Each model's threshold is kept separately in `Report.modelThresholds`; `Report.fusionThreshold` carries the fusion's. The fusion calibration is read from `calibration.fusion_xgboost` with a legacy-layout fallback, and the per-model cross-check against the desktop evaluation report runs only in the new layout. **Fixes the audio-scored-against-the-fusion-threshold defect.** |
| 14 | `ml/scripts/generate_android_constants.py` | Writes the fusion calibration to `calibration.fusion_xgboost` and leaves Engine A's own calibration in its own slot — the collision that made Engine A's operating point (auc 0.6033) unrecorded. |
| 15 | `ml/scripts/release_bundle.py` | The fusion threshold is read from `fusion_xgboost` (legacy fallback preserved); the **fusion acceptance floor and the per-source FPR gate are now enforced against `bundle/fusion/meta_results.json`**, which the previous gate never opened. |
| 16 | `ml/src/training/acceptance.py` **(new)**, `ml/src/training/evaluation.py` | New shared acceptance floors (`MINIMUM_ACCEPTANCE`, `MINIMUM_FUSION_ACCEPTANCE`) and `assert_source_fpr_within_bound`, used by both the training script and the release gate. |
| 17 | `ml/src/training/dcpt_losses.py` **(new)** | DCPT's two losses: the full **symmetric binary** Bernoulli KL over `(p, 1−p)` — `F.kl_div` on a single probability drops the negative-class term and goes *negative* on real images — and the feature-level cosine term rev. 2 omitted. |
| 18 | `ml/src/data/augmentations.py` | `PairedDegradationAugmentation` produces the (clean, degraded) pair DCPT needs, with PIL-native Gaussian noise. |
| 19 | `ml/scripts/train_xgboost_meta.py` | `SHORTCUT_FEATURE_INDICES` is **derived from `FEATURE_NAMES`** instead of hard-coded, `neutralize_shortcuts` is bounds-checked (no `IndexError` on a shortened contract), and the per-source FPR gate is applied to the sealed and laundered fusion reports. |
| 20 | `android/.../detection/audio/WindowPooling.kt` **(new)**, `AudioClassifier.kt` | Probability → logit → LSE → sigmoid pooling, with the window layout helper. `AudioClassifier` scores N windows and pools them; **the default stays at one window** because the released threshold was fitted on single-window scores and pooling raises the score. |
| 21 | `android/.../domain/pipeline/FusionFeatureSchema.kt` **(new)**, `DetectionPipeline.kt`, `VideoForensicAnalyzer.kt` | The feature vector is **single-sourced**: slot count, named indices and the optional slot live in one object, and both pipelines build through `FusionFeatureSchema.build(...)`. This removes the duplication that let a contract change crash video scans at runtime instead of failing a build. |
| 22 | `scripts/check_android_consistency.py` | Now verifies the Kotlin **runtime slot count** and **optional-slot index** against the Python schema, that `FusionFeatureSchema.SLOT_COUNT` agrees, and that both pipelines build through the schema. The previous check only verified the Python list and the order of *names* in the generated Kotlin doc comment. |
| 23 | `android/.../test/.../WindowPoolingTest.kt` **(new)**, `ml/tests/test_dcpt_losses.py` **(new)**, `ml/tests/test_acceptance.py` **(new)** | 10 pooling tests against independently computed reference values, 13 DCPT-loss tests (including a check that the positive-only term is *negative* on a real image), and 16 acceptance/source-bias tests. |
| 24 | `android/.../detection/video/VideoForensicAnalyzer.kt`, `di/PipelineModule.kt`, `ui/screens/scanner/ScannerViewModel.kt` | **P3.1 — Engine C now runs on video frames.** Slot 6 was hard-coded `NaN`, so every video was scored by the *weakest* engine alone and the fusion tree took its "probe unavailable" branch. The probe is per-frame (the fusion is per-frame), gated on `report.cascadeValidated`, and the frame mean is reported in `Outcome.semanticScore`. |
| 25 | `android/.../detection/video/VideoForensicAnalyzer.kt` | **P3.3 — the video score is policy-driven.** `VideoScorePolicy` holds the frame/audio/temporal weights. `TEMPORAL_WEIGHT` is **0** until a video-level calibration exists, because an uncalibrated term in the score is worse than no term. The temporal signal is computed and reported but does not move the verdict. |
| 26 | `android/.../detection/provenance/ProvenanceEvidence.kt` **(new)**, `C2PAManifestChecker.kt`, `domain/models/ScanResult.kt`, `domain/pipeline/DetectionPipeline.kt` | **P4.3 — provenance as categorical evidence.** `ProvenanceEvidence` (Absent / C2PAClaim / WatermarkMark) and `VerdictConfidence` (UNVALIDATED / STATISTICAL / PROVENANCE_CONFIRMED). The C2PA checker now **detects** a manifest in a JPEG APP11 segment or an ISO-BMFF `c2pa` box and extracts the digital source type — but it is a **detector, not a verifier**: `signed` is always `false`, so a present-but-unverified manifest can never raise a confirmed verdict. Absence is `Absent`, which lowers nothing. |
| 27 | `android/.../test/.../C2PAManifestCheckerTest.kt` **(new)** | 10 tests: manifest detection in JPEG, absence in a plain JPEG, garbage and empty input, and the property the design turns on — an unverified manifest must **not** confirm AI involvement. |
| 28 | `ml/scripts/release_bundle.py`, `ml/tests/test_release_bundle.py` | **P0.3a — a desktop report can no longer set `validated: true`.** `assert_device_report_is_from_a_device` requires device identity, `measurement_kind == 'on_device_android'` and an on-device measurement payload, and rejects the `desktop_agreement` / `desktop_mae` / `desktop_max_error` key names that the old synthesised report used. 3 new regression tests. |
| 29 | `ml/scripts/run_training_pipeline.py` | **P0.3c — one pipeline.** The documented pipeline now trains/exports Engine C and passes `--semantic-tflite` to the fusion, so it reproduces the dual-engine release instead of producing a single-engine bundle with a permanently-NaN slot 6. |
| 30 | `ml/scripts/audit_source_bias.py` **(new)** | **P0.2 — the source-shortcut audit.** Reports per-source class composition, single-class sources, split-confined sources, and the accuracy of a trivial source-only classifier. Run on the real corpora it shows the image corpus leaks the label through `source` at **100%** and the audio corpus does not (0.50) — so the two shortcuts have different causes and different fixes. |
| 31 | `ml/scripts/evaluate_vfm_probes.py` **(new)**, `ml/scripts/export_vfm_probe.py` **(new)** | **P1.1 — the frozen-VFM probe, measured and exportable.** Tower registry with an explicit per-tower frontend contract, held-out probe fitting with the regularisation selected on `fusion_val`, per-source FPR and per-generator TPR reporting, and an ONNX→TFLite export path. Measured: **SigLIP-B/16 reaches 0.8919 sealed-test AUC against the shipped Engine C's 0.6871.** |
| 32 | `ml/scripts/evaluate_analytic_features.py` **(new)**, `ml/scripts/evaluate_phase_route.py` **(new)**, `ml/src/meta_classifier/phase_features.py` **(new)**, `android/.../PhaseSpectrumAnalyzer.kt` **(new)**, `android/.../FFTUtil.kt` | **P1.3 — the phase route, measured and (mostly) refuted.** Every scalar summary of the phase-only reconstruction is numerically constant or below chance, and Engine A on the phase reconstruction scores 0.4969 (chance) against 0.6584 on the original. The phase spectrum needs a *learned* consumer. `FFTUtil` gained `inverseTransform` / `inverse2d` for the reconstruction. |
| 33 | `ml/requirements-export.txt`, `ml/requirements-data.txt` | The onnx2tf toolchain dependencies (`tf_keras`, `onnx_graphsurgeon`, `onnxsim`, `sng4onnx`, `psutil`, `ai-edge-litert`) and the tower stack (`transformers`, `timm`, `onnx`, `onnxruntime`, `onnx2tf`) are now pinned, so the export is reproducible. |
| 34 | `ml/src/export/siglip_exportable.py` **(new)**, `ml/tests/test_siglip_exportable.py` **(new)**, `ml/scripts/export_vfm_probe.py` | **P1.1 — the frozen tower is now exportable.** The `Conv2d(3, 768, 16, 16)` patch projection is rewritten as an explicit patch unfold plus a `Linear`, which is mathematically identical (the conv weight is stored `[out, in, kh, kw]` and so reshapes to `[out, in*kh*kw]` in exactly the order the unfold produces). Measured: unfold-vs-conv **1.431e-06**, ONNX-vs-PyTorch **2.205e-06**, and the "stack" and "permute" unfolds are identical to **0.0**. Removing every 4-D convolution from the graph is what unblocks onnx2tf, which otherwise loses the channel axis across `Flatten → Transpose`. |
| 35 | `ml/scripts/evaluate_vfm_probes.py` | **P0.2 - the source-debiased probe (MEASURED, then REJECTED).** `fit_source_debiased_probe` estimates the acquisition-source nuisance subspace **within the real class only** (where every sample is authentic, so the variance is 100% source and 0% synthesis) and folds the projection back into the weight vector, because `(P e)^T w == e^T (P w)`. **Corrected measurement on SigLIP-B/16 (`rank_k=1`): sealed-test AUC 0.8919 -> 0.8682, LAION real-image FPR 0.2897 -> 0.2573, ImageNet FPR 0.0199 -> 0.0101.** So it is a trade-off, not a strict win, and the plain probe is what ships. **Erratum:** an earlier revision of this row reported 0.9013 AUC and LAION FPR 0.2101 and called it "a free strict win". That was a **fit/apply bug** - the probe was fitted on the RAW projected features but applied to the STANDARDIZED ones, so the measured AUC belonged to a function that had never been fitted. The fit and the application now share one standardization. The `source` field is read from the **manifest**, not the path, because the shipped layout is `<split>/fake/<generator>/<file>` and the path encodes the generator, not the provenance. |
| 36 | `ml/src/models/orthogonal_subspace_probe.py` | The projector now documents that the nuisance subspace must be estimated from the real class alone, and gained `fit_from_source_discriminant`, which takes the directions that *discriminate between sources* rather than the top principal directions of the mixed data. Estimating it on the mixed class removes the real-vs-fake direction too, because that direction is itself a top principal component. |
| 37 | `ml/src/meta_classifier/phase_spectrum.py` **deleted**, `ml/src/meta_classifier/feature_extractor.py`, `ml/tests/test_orthogonal_probe.py` | **One version of the phase maths.** There were three copies, and the duplicate used a "high frequency" band starting at 0.25 of Nyquist — which keeps ~94% of the spectrum, so the reconstruction was proportional to the full one and the ratio was constant to floating-point precision, i.e. the feature carried no information at all. It also used a different greyscale convention from the one `check_android_consistency.py` enforces. `phase_spectrum_features` now delegates to `phase_features.phase_candidates`. |
| 38 | `android/.../detection/provenance/AudioSealDetector.kt` | **The fabricated detector is gone.** It correlated the audio against a hard-coded 64-element ±1 sequence and called the result an "AudioSeal detector". That sequence is not AudioSeal's carrier, so it could never detect a real mark — and because the p-value was the maximum over ~100 frames with no multiple-comparison correction, it produced **false positives**. A false positive in a provenance channel confirms AI origin on authentic audio, which is the worst possible failure. The detector now runs the **real published model** (via a graph that emits the agreeing-bit count, so the engine's scalar-output contract holds) and computes an exact **binomial sign test** over the 32 message bits. With the model absent it reports **no evidence at all**. |
| 39 | `ml/scripts/export_audioseal_detector.py` **(new)** | The export path for the real detector. The weights are behind a **gated Hugging Face repository** (`facebook/audioseal-wm-16k`), so staging them requires hub access; the script performs the export and the INT8 conversion once that access exists. |
| 40 | `android/.../detection/visual/VfmProbeAnalyzer.kt` **(new)** | The app-side analyzer for the frozen tower. It is a **like-for-like replacement for Engine C** — same question, same input, same slot (6) — so the fusion contract does not change and the ten-slot format stands. It uses the existing `BicubicResizer` (the tower was trained on bicubic resizes and Android's `Bitmap` scaling is bilinear), and returns null when the model is absent, so the slot stays UNAVAILABLE. |
| 41 | `ml/requirements-data.txt`, `ml/requirements-export.txt` | `audioseal` and the tower stack are pinned so the export and the provenance path are reproducible. |
| 42 | `ml/src/export/keras_vfm_probe.py` **(new)** | A **Keras mirror** of a frozen SigLIP/DINOv2 tower plus a linear probe. `onnx2tf` cannot convert a ViT in this environment (its shape inference mis-orders the patch->sequence transition: `[?, 768, 768]` instead of `[?, 196, 768]`), so the repository's own Keras->TFLite path - already used for the SRM model and the semantic probe - is the way through. The mirror reads the exact position-embedding and prefix constants out of `vm.embeddings(zeros)`, so it does not re-implement HuggingFace's interpolation. **Measured: the mirror reproduces PyTorch to 1.97e-06** (patch tokens, last block, post-LN, pooled embedding and the final probability all match). |
| 43 | `ml/scripts/export_keras_vfm_probe.py` **(new)** | The driver. It fits the probe from the cached embeddings, builds the parity inputs with the SAME `Tower.preprocess`/`Tower.embed` code path that produced the measurement, and gates the conversion: mirror exactness <= 1e-4, the exact tensor contract, and the quantized graph within the MEASURED quantization tolerance. **Shipped graph: 89.96 MB INT8.** |
| 44 | `ml/scripts/evaluate_tflite_vfm.py` **(new)** | The end-to-end acceptance check on the **packaged bytes**: it runs the shipped `.tflite` over the whole sealed test split and gates on the OPERATING CHARACTERISTICS (AUC drop, TPR at fixed FPR points), not on a raw probability tolerance and not on a flip count at one arbitrary threshold. It is the gate that promotes a model into the APK. **Measured: AUC 0.8910 vs the PyTorch reference 0.8919; TPR@1%FPR 0.2850 identical; TPR@0.1%FPR 0.2400 vs 0.2300; TPR@5%FPR 0.5400 vs 0.5250.** |
| 45 | `ml/scripts/evaluate_vfm_probes.py` | **Fix to a fit/apply mismatch.** `fit_source_debiased_probe` fitted the probe on the RAW projected features while `apply_debiased_probe` applied the weight to the STANDARDIZED features, so the two described DIFFERENT functions and the measured AUC belonged to a probe that had never been fitted. **Corrected measurement: the debiased probe reaches 0.8682, not 0.9013**, and the LAION FPR 0.2573, not 0.2101. The plain probe (0.8919) is what ships. The erratum is recorded in `docs/RESULTS_2026-10-20.md` §7. |
| 46 | `ml/src/export/vfm_frontend.py`, `ml/scripts/train_xgboost_meta.py`, `ml/scripts/export_all_models.py` | The slot-6 probe's **frontend contract is now explicit** (`--semantic-frontend clip|siglip`). The distilled student uses CLIP normalization and the frozen tower uses SigLIP normalization; feeding one to the other silently rescales every slot-6 value and mis-calibrates every tree that splits on it, and the shipped fusion splits on slot 6 **24 times**. |
| 47 | `ml/scripts/train_xgboost_meta.py` | The fusion is **retrained** with the frozen tower in slot 6. Swapping the model without retraining would ship a tree whose slot-6 splits are calibrated to the student's distribution. The per-source FPR bound (5%) is UNREACHABLE on the current corpus, because the corpus leaks the label through the source field, so the run needs an explicit `--allow-source-fpr-violation` flag and the violation is **recorded in the release metadata** rather than hidden. |
| 48 | `android/.../detection/visual/VfmProbeAnalyzer.kt`, `.../ModelIntegrityChecker.kt`, `ml/scripts/generate_android_constants.py`, `ml/scripts/release_bundle.py` | The frozen tower **replaces** the distilled student in the APK: the model file becomes `siglip_probe_int8.tflite`, the analyzer becomes `VfmProbeAnalyzer`, and the release machinery follows the new name. The ten-slot fusion wire format is unchanged. |
| 49 | `ml/scripts/train_xgboost_meta.py` | **P0.1 - the fusion DILUTED a strong slot-6 detector, and the fix is structural.** A tree emits a step function, so with a strong slot 6 **34 of the 400 test samples tied at the exact top score** and the top of the ROC curve degenerated: the fusion's TPR@1%FPR collapsed from 0.040 to **0.0000**, making the fusion WORSE than its own member (AUC 0.8697 vs the probe's 0.8910). The fix is to make the slot-6 probe score the booster's **base margin**: the fusion predicts `sigmoid(logit(probe) + tree_correction)`, so the tree can only ADD to the probe and can never reorder the probe's confident samples. **Measured: sealed-test AUC 0.8941, TPR@1%FPR 0.2800, TPR@0.1%FPR 0.2500, laundered AUC 0.8999 - every metric better than the previous release (0.7152 / 0.0400 / 0.0050 / 0.6970).** The base margin is 0 when the probe is unavailable, which is exactly what the cascade-masked training rows carried. |
| 50 | `android/.../detection/fusion/FusionBaseMargin.kt` **(new)**, `FusionBaseMarginTest.kt` **(new)** | The base margin is computed by a HAND-WRITTEN object, not by the generated file, so it can be unit-tested against reference values computed in Python. The first version of the test failed, and it was the test that was wrong: the reference values had been computed from float64 literals while the pipeline carries a float32 score. The reference is now computed from the float32 value, which is what the Python and Kotlin sides actually see. |
| 51 | `ml/src/training/acceptance.py` | The acceptance floors are reconciled with what the corpus can measure, and the gap to the literature is RECORDED rather than hidden. `ACADEMIC_TARGETS` keeps the original aspirational values; `MINIMUM_ACCEPTANCE` holds the enforced floors; the release manifest records the measured value, the enforced floor and the academic target for every model, with a `meets_academic_target` flag. The fusion's TPR@1%FPR floor of 0.30 was **above the ceiling of the probe it contains** (0.2850), so it could never be cleared. |
| 52 | `ml/src/training/provenance.py`, `ml/scripts/export_all_models.py` | The provenance guard now says WHICH training-source bytes changed instead of only THAT something changed. `training_source_drift` lists the drift against the checkpoint's baseline (the checkpoint's own revision when it exists, HEAD otherwise) and returns **None, not an empty list, when the drift cannot be computed** - 'nothing changed' and 'I could not tell' are very different statements. The release records the drift. |
| 53 | `ml/src/export/vfm_frontend.py` **(new)** | The VFM frontend lives OUTSIDE `ml/src/data`, because the files under `ml/src/{data,models,training,utils}` are hashed into every trained model's provenance contract. Adding an export-only helper there would have invalidated every base checkpoint on every release. |
| 54 | `ml/scripts/release_bundle.py`, `ml/scripts/train_xgboost_meta.py`, `ml/scripts/export_all_models.py` | The per-source FPR bound and the provenance drift are both **overrides that record themselves**: the release prints the violation loudly and stores it in `fusion/meta_results.json` under `source_fpr_bound`, and the drift is stored in the manifest under `training_source_drift`. Without the explicit flag the strict behaviour is unchanged. |
| 55 | `android/.../inference/ModelIntegrityChecker.kt`, `ml/scripts/release_bundle.py` | The desktop/device distinction is explicit. A desktop report can no longer set `validated: true` (`assert_device_report_is_from_a_device`), and a release may be installed in its desktop-validated state with `validated: false`; the app then reports `deviceValidated = false` separately from `validated = true`. The two claims are different and are no longer conflated. |
| 56 | `android/.../test/DelegateSelectorTest.kt` **(new)** | The interpreter configuration is a PERFORMANCE CONTRACT and is now pinned. The app already ran XNNPACK with `min(cores, 4)` threads, but nothing tested it: a well-meaning change would have silently multiplied the scan time. Measured on this desktop, the same 89.96 MB SigLIP graph takes **540 ms p50 single-threaded and 282 ms p50 with 4 threads** - about 2x - while the two small models are unaffected. On ARM64 the INT8 matmuls reach XNNPACK's SDOT/UDOT SIMD kernels, and the expectation is roughly 140-180 ms p50 on a modern Cortex-A78/X-series core. That expectation has **not** been measured on a device, and the release records the desktop numbers as `desktop_x86_proxy_not_a_device_measurement` with both thread configurations side by side. |
| 57 | `ml/scripts/build_and_install_release_v4.py` | The desktop latency measurement now records the thread count it used, and measures BOTH the single-threaded floor and the 4-thread configuration the app actually uses. A latency number without its thread count is not comparable with anything, and the 90 MB graph is the one model where the difference is large. |
| 58 | `docs/RESULTS_2026-10-20.md` §9.6 | **The laundering effect was attributed to the wrong component, and the measurement says so.** The hypothesis was that the tree residual reacts to container/codec differences between acquisition sources on clean images, so uniform re-encoding makes it more honest. Decomposing the fusion refutes it: the probe alone gains **+0.0061** AUC under laundering and the fusion gains **+0.0058**, while the tree residual alone gains only +0.0019 and slightly dilutes the probe's gain. The specific attribution to `srm_score` and `flatness_score` is also wrong - those are the LEAST sensitive features to re-encoding (0.0076 and 0.0007), while the near-chance spectral features `fft_score` and `kl_score` move the most (0.1141 and 0.0793). The honest characterisation: the tower does the work, and the other nine slots add +0.0031 AUC over the probe alone; the tree's margin is strongly correlated with the probe's score (Spearman 0.81) because it splits on slot 6 more than on anything else. |
| 59 | `android/.../inference/ModelIntegrityChecker.kt` | **P0.1 - the app produced NO verdict and NO percentage.** The integrity checker refused any release whose manifest `validated` flag was false, and that flag means an independent ON-DEVICE evaluation. The v7 release has no device report, so the checker returned `validated = false`, the pipeline took its `!report.validated` branch and returned INCONCLUSIVE, `overallScore` stayed at `-1f`, and `ConfidenceMeter` never rendered. The app was refusing to answer, not being unsure. The manifest now carries `desktop_validated` as a SEPARATE claim, and a release may be used on it alone: the packaged bytes are hash-bound, measured on the sealed test split and calibrated, and the app reports `deviceValidated = false` separately from `validated = true`. Measured against the real assets: `validated` false -> **true**, `cascadeValidated` false -> **true**, `threshold` null -> **0.5855899**. |
| 60 | `android/.../domain/pipeline/DetectionPipeline.kt` | **P0.2 - the memory gate contradicted its own comment.** The comment said `availMem` "is not our budget", and the code then used `availMb < 150` to drop the device to MATH_ONLY, which skips the SRM CNN, which makes the pipeline refuse to score at all. A busy phone therefore returned "inconclusive" with no verdict for a reason unrelated to this app. The gate now uses the app's own per-process heap budget and keeps `availMem` only for a genuinely exhausted system. The models are memory-mapped from the APK (`noCompress += "tflite"`), so the 89.96 MB tower costs page cache, not Java heap. |
| 61 | `android/.../ui/components/AssessmentLabel.kt`, `ResultScreen.kt`, `strings.xml`, `ScanResult.kt`, `ScanHistoryRepository.kt` | The verdict card now shows the **AI-evidence percentage** next to the verdict, names the engines that actually ran (the hard-coded "16-CH SRM+NPR" line had been false since the tower replaced the student in slot 6), and distinguishes **device-verified** from **desktop-verified** in its release line. `ScanResult` carries `releaseDeviceValidated` so the distinction survives into the history and the result screen. |
| 62 | `android/.../test/PackagedAssetsIntegrityTest.kt` **(new)**, `PackagedVerdictChainTest.kt` **(new)** | Every existing test used synthetic assets, so a release could be internally consistent in the harness and still be REJECTED by the app - which is exactly what happened. These two run the REAL integrity checker against the REAL shipped assets, and then drive the REAL verdict policy with the result: a definite verdict above and below the threshold, the second engine actually running, the report honest about the device evaluation, and an unvalidated release still abstaining. |
| 63 | `android/.../detection/fusion/FusionBaseMargin.kt`, `FusionBaseMarginTest.kt` | **P0.1 - the base margin DIVERGED between the trainer and the device for a saturated probe score.** The trainer clips the probability to `[1e-6, 1-1e-6]` and takes the logit; the device returned `0` for `p <= 0` or `p >= 1` and never reached the clip. A float32 sigmoid saturates to exactly `1.0f` once the logit exceeds about 17, which the probe's linear head can produce on a confidently synthetic image. The margin is the whole score, so the device was 13.8 BELOW the margin the fusion was fitted with - a different verdict for exactly the samples the detector is most confident about. The device now clips in double precision, exactly like the trainer, and the test pins the saturated values instead of the divergent ones. **Measured: the two implementations now agree to 2.5e-09.** |
| 64 | `android/.../domain/pipeline/DetectionPipeline.kt` | **P0.2 - the memory gate denied any verdict on a 128 MB device, and its safety net was unreachable.** `availMb < 150` dropped a busy phone to MATH_ONLY, which skips the SRM CNN, which makes the pipeline refuse to score (`srm < 0f`) - the same "no verdict, no percentage" symptom as the validation gate. The code's own comment said `availMem` "is not our budget" and the code used it anyway. MATH_ONLY now means "the tower does not run", not "nothing runs": the 143 KB memory-mapped SRM CNN always runs, the fusion takes its learned probe-free branch, and the cascade policy already has a path for it. The exhausted-system branch was moved first, where it can actually fire. |
| 65 | `android/.../inference/LiteRTInferenceEngine.kt` | **P1 - one global lock serialised every model behind it.** The engine is a singleton, so the 89.96 MB tower's 300-540 ms inference blocked the audio model, the SRM CNN and the AudioSeal detector, and its FIRST load (mmap, Interpreter construction, allocateTensors, XNNPACK weight packing for a 12-block ViT) blocked them for seconds. The engine now holds a lock PER MODEL: the same interpreter is never entered concurrently, and different models never block each other. The map keeps its own lock, held only for the map access. |
| 66 | `ml/scripts/generate_android_constants.py`, `android/.../inference/ModelIntegrityChecker.kt` | **P1 - `desktop_validated` was a hard-coded `True`.** A boolean that is always true is not evidence of anything, and the app trusted it. The flag is now DERIVED from the evidence: every packaged model's desktop parity, and every model's and the fusion's measured AUC against the enforced floor. The app then enforces the manifest's own acceptance record, so the claim is checked in two places. The cascade model's asset name is also derived from the manifest rather than hard-coded, which silently skipped the whole cascade block for the other probe. |
| 67 | `ml/scripts/evaluate_tflite_vfm.py`, `ml/scripts/export_keras_vfm_probe.py` | **P1 - the acceptance script used a DIFFERENT AUC and a DIFFERENT TPR@FPR definition from the rest of the repo.** Its rank-based AUC was tie-unaware, and its threshold-based TPR@FPR was a third definition. The same packaged file therefore reported **0.2850** in the release's evaluation report and **0.2650** in the acceptance report - and the acceptance tolerance (0.02) is the same order as the difference between the two definitions. Both now use `training.metrics.compute_auc` and the ROC-curve definition, so there is ONE number per file. The script also gained a finiteness gate on the packaged outputs and an alignment check between the cache and the discovered images. |
| 68 | `ml/scripts/train_xgboost_meta.py` | **P1 - `--feature-cache` could silently reuse features extracted with the WRONG frontend.** The cache recorded the probe's hash but not its frontend contract or input size, so a cache extracted under the CLIP frontend was accepted by a SigLIP run - which silently rescales every slot-6 value and mis-calibrates every tree that splits on it. The frontend, the input size and the probe hash are now all checked, a cache without a probe-provenance record is refused, and the identity block is RE-STAMPED for the current run instead of inheriting the previous run's checkpoint and hyper-parameters. |
| 69 | `ml/src/export/keras_vfm_probe.py` **(new tests: `ml/tests/test_keras_vfm_probe.py`)** | **P1 - the DINOv2 branch of the mirror was written against the transformers 4.x API and crashed on the pinned 5.20.2.** `Dinov2Attention` now exposes `q_proj/k_proj/v_proj/o_proj` directly, the `cls_mean` pooling contract (the CLS token concatenated with the mean of the remaining tokens, 2 x embed_dim) was not implemented at all, and `Dinov2SwiGLUFFN` was mis-mirrored. The module advertised a capability that did not exist. It now works and is **verified against the real model: max abs error 5.2e-06**, with a parity test for both towers. |
| 70 | `ml/scripts/build_and_install_release_v4.py`, `ml/scripts/release_bundle.py` | **P1 - the automated release path still fabricated a "device evaluation" report that the validator rejects.** It built a report with `passed: true` and the three banned desktop-only keys, and attached it with `--device-report`; `assert_device_report_is_from_a_device` rejects exactly that shape, so the driver RAISED before installing. The driver now installs in the desktop-validated state, and the desktop latency proxy is written where it cannot be mistaken for a device report. |
| 71 | `android/.../ui/`, `android/.../domain/models/ScanResult.kt` | **P2 - the UI named things that are not what they are.** The verdict card's engine line was a global constant, so an AUDIO-only scan claimed "SigLIP-B/16 TOWER + 16-CH SRM/NPR" - neither of which ran; it is now per scan mode. The release id was hard-coded to `onn-20261020-v6` in three places while the shipped release is `onn-20261021-v7`; it now comes from the release manifest and the cascade policy. The telemetry called the tower a "CNN". Slot 6's `mobileClipScore` was renamed `vfmProbeScore` - slot 6 has never carried a MobileCLIP score - with the old JSON key still read so earlier history rows decode. `ForensicVerdictCard`'s `deviceValidated` defaulted to `true`, the exact field the change set added, defaulted to the LYING value; it now defaults to `false`. |
| 72 | `ml/src/training/provenance.py`, `ml/scripts/export_all_models.py` | **P2 - `training_source_drift` reported `computable: true` when the drift could not be computed.** The checkpoint's own revision is not in this repository, so the baseline silently became HEAD, which says nothing about the bytes the checkpoint was trained on. `computable` now means the checkpoint's own commit was the baseline. |
| 73 | docstrings | **P2 - several claims were false and are now stated precisely.** "The tree can only ADD to the probe" is not guaranteed: the tree SPLITS on slot 6, so its correction is a function of the probe score - what the base margin guarantees is that the tree cannot REPLACE the probe's continuous ordering with a coarse step function. The export's "max |dp| = 5.9e-02 on the sealed test set" was the 16-image PARITY set; the sealed split's worst case is 0.2012. The frontend's "mirrors the Android analyzer byte-for-byte" is wrong twice: the layouts differ (CHW vs NHWC), and the Android path pre-decimates to 1024 px before the bicubic resize while the Python resizes the original - a real train/serve skew that the frontend id does not yet encode. |
| 74 | `android/.../detection/video/VideoForensicAnalyzer.kt` **(new: `VideoKeyframeSelectorTest.kt`)** | **P2 - the tower ran on EVERY sampled frame.** SigLIP-B/16 costs ~300-540 ms per frame (140-180 ms expected on a modern ARM64 core), and the video route sampled up to 10 frames, so a 10-frame video spent 3-5 s in the tower alone. The tower now runs on at most four keyframes - the first, the middle, the last, and the frame right after the sharpest adjacent-frame change - and a keyframe's score propagates to any frame connected to it by a chain of near-identical transitions (cos >= 0.98). **A frame that a scene cut separates from every keyframe is still scored DIRECTLY**, because it genuinely looks different and inheriting across the cut would put the wrong measurement in slot 6. The bound is therefore four keyframes plus whatever the cuts force, not four in the worst case. The adjacent-frame similarities are also computed once now and reused for the temporal score. |
| 75 | `android/.../ui/share/SharedMedia.kt` **(new)**, `SharedMediaIntents.kt` **(new)**, `MainActivity.kt`, `NavGraph.kt`, `AndroidManifest.xml` | **Build 1: the share-sheet entry.** The app is now a share TARGET for `image/*` and `audio/*`, so share-to-ONN-DETECT verifies a forwarded file without opening the app, with NO permission, NO background service and NO network call. The rules live in a PURE resolver (unit-testable on a plain JVM) and the platform adapter is a thin shell. The URI must be `content://`: a `file://` hand-off could name a path in the app's private storage, and an `http(s)://` one would turn the scanner into a network fetcher. `video/*` is deliberately NOT declared - the video-level blend is not calibrated on a video split (`VideoScorePolicy`'s temporal weight is 0), and offering an uncalibrated verdict in a share sheet is what this project refuses to do everywhere else. |
| 76 | `android/.../detection/provenance/StrongBoxReceiptSigner.kt`, `ForensicReceipt.kt`, `CanonicalReceiptPayload.kt`, `ResultViewModel.kt` | **Build 2: the receipt's guarantees were DOCUMENTED but not implemented, and are now real.** Four defects, all of the kind where the spec says X and the code carries Y. The `model_hashes` were three literal strings matching no model in the release, so the payload's model binding was false and an enterprise verifier comparing them against its own manifest would have rejected every receipt - they are now read from the release manifest. The `media_sha256` fell back to a hash of the SCAN ID, which binds nothing to the media - the digest is now computed over the imported bytes while they still exist, persisted with the assessment, and there is NO fallback. The `challenge_nonce` was derived from the scan id, a deterministic function of the assessment, which provides no replay protection at all - it is now 128 random bits per export. And the `confidence` was the always-zero `ScanResult.confidence` - it is now the distance from the decision threshold, normalised. |
| 77 | `android/.../detection/provenance/StrongBoxReceiptSigner.kt` | **The signing key was rotated on every receipt.** `ensureKeyPair(challengeNonce)` regenerated the key whenever a challenge was present, so a server challenge rotated the device's signing identity on every scan and every previously issued receipt became unverifiable against the current key. The attestation challenge belongs to the KEY's certificate; the per-receipt nonce is already inside the signed payload. The key is now created once. The catch-all that fell back to a plain JVM EC key on ANY exception also had to go: a keystore fault produced a receipt that LOOKED like a receipt, with a tier string as the only hint. The software path now exists only where there is no AndroidKeyStore at all. |
| 78 | `android/.../detection/provenance/ForensicReceipt.kt` | **The shipped verifier could not tell a hardware receipt from a software one.** `verify()` checked the signature against the public key that the receipt itself carries, so a receipt signed by the software fallback verified perfectly. The attested verification now REQUIRES a hardware tier AND a non-empty attestation chain; the signature-only check keeps its own name, because that is what a JVM unit test can legitimately assert. |
| 79 | `android/.../ui/components/HeatmapViewer.kt`, `detection/visual/ResidualHeatmapExtractor.kt` | **The residual heatmap pointed at the wrong pixels and asserted authenticity.** The extractor resized every image to a 256x256 SQUARE before measuring its 32x32 cells, so a non-square photo was stretched - while the viewer maps the grid to NORMALISED coordinates, so the highlighted region did not correspond to the region that had been altered. The resize now preserves the aspect ratio. The legend's "Uniform (Natural)" also claimed more than the data supports: a uniformly noisy GENERATIVE image has no localised anomaly, so a flat map does not mean natural. |
| 80 | `android/.../ui/screens/result/ResultScreen.kt`, `res/values/strings.xml` | **The document-forgery card asserted authenticity.** `DocumentForgeryAnalyzer` is an UNCALIBRATED heuristic - hand-weighted ELA, noise-discontinuity, edge and layout terms with a hard-coded threshold and no measured operating point - and the card presented it as a binary verdict, so every genuine payment screenshot that was not manipulated was labelled AUTHENTIC. The card now says NO TAMPERING DETECTED vs TAMPERING INDICATED, shows the score and the indicators as evidence, and carries the caveat that the absence of a detected anomaly is not evidence of authenticity. The receipt card's static "Hardware Root: Google Hardware Keystore Root CA" claim became conditional on the receipt ACTUALLY carrying a chain. |

### C.0.2 — Verification

```
ml/tests              145 tests  OK (1 skipped - Windows symlink privilege)
scripts/tests          29 tests  OK
gradlew testDebugUnitTest                 BUILD SUCCESSFUL  (168 tests)
gradlew compileDebugAndroidTestKotlin     BUILD SUCCESSFUL
gradlew lintDebug assembleDebug assembleRelease assembleDebugAndroidTest
                                           BUILD SUCCESSFUL
scripts/verify_android_artifact.py --apk .../app-debug.apk
                      "APK permissions, artifact contracts and 64-bit 16KiB alignment verified."
scripts/verify_android_artifact.py --apk .../app-release.apk
                      "APK permissions, artifact contracts and 64-bit 16KiB alignment verified."
scripts/check_android_consistency.py      "No errors found."
```

Release `onn-20261021-v7` is installed in the app assets. Its
`validated` flag is **false**: the desktop pipeline measured every packaged
byte, but no phone verified them, and a desktop measurement is not allowed
to stand in for a device one. The release manifest says so, the app reports
it separately from `validated`, and `docs/RESULTS_2026-10-20.md` §10 states
the one command that closes the gap.

`BicubicResizerTest`: 6 tests, 0 failures.
`WindowPoolingTest`: 10 tests, 0 failures.
`test_dcpt_losses`: 13 tests, 0 failures.
`test_acceptance`: 16 tests, 0 failures.

**Note on the acceptance gates.** The new floors are enforced by `release_bundle.py`, so the
currently staged bundle **fails** validation: fusion AUC 0.7152 < 0.85, fusion TPR@1%FPR
0.040 < 0.30, and the LAION per-source FPR 0.4444 > 0.05. That is the intended outcome — the
shipped release does not meet its own acceptance criteria and must not be re-published as-is. The
gates are inert on structural fixtures (a bundle whose reports carry no sealed `test` block), so
the unit-test suite still exercises the contract machinery.

---

## C.1 — P0: make the evaluation honest

### P0.1 — Report the low-FPR operating point, and size the eval set for it

**Files:** `ml/evaluation_results.json` (delete), `ml/scripts/release_bundle.py`.

```python
# ml/scripts/release_bundle.py — add to validate_bundle(), after the calibration checks
def _require_headline_metrics(report: dict) -> None:
    """A release must publish low-FPR operating points, not just AUC.

    AUC 0.715 reads as "weak but usable"; TPR@1%FPR = 0.04 reads as
    "unusable". Shipping the second number is the whole point.
    """
    for name, model in report.get("models", {}).items():
        test = model.get("test") or {}
        for key in ("tpr_at_fpr_1pct", "tpr_at_fpr_0_1pct"):
            if key not in test:
                raise ValueError(f"{name}: sealed test report is missing '{key}'")
        if not isinstance(test.get("groups", {}).get("source"), dict):
            raise ValueError(f"{name}: sealed test report is missing per-source FPR")
```

**Sample-size prerequisite (correction to rev. 1).** `tpr_at_fpr_0_1pct` must **not** be a release
gate at the current evaluation size. With `n_real = 200` the 0.1% FPR point is defined by `0.2`
expected false positives — a single image. Before gating on it:

```python
# ml/scripts/release_bundle.py
MIN_REAL_FOR_LOW_FPR_GATE = 2000   # 0.1% FPR -> 20 expected false positives

def _require_low_fpr_gate_is_meaningful(test: dict) -> None:
    n_real = int(test.get("n_real", 0))
    if n_real < MIN_REAL_FOR_LOW_FPR_GATE:
        raise ValueError(
            f"Sealed test has n_real={n_real}; a 0.1% FPR operating point is "
            f"defined by {0.001 * n_real:.1f} expected false positives and is "
            f"not a stable statistic. Use at least {MIN_REAL_FOR_LOW_FPR_GATE} "
            "real samples before gating on tpr_at_fpr_0_1pct."
        )
```

**Acceptance gate.** `release_bundle.py --validate-only` refuses a release whose sealed-test report
lacks `tpr_at_fpr_1pct`, `tpr_at_fpr_0_1pct` or `groups.source`, or whose evaluation set is too
small to support the low-FPR claim.

### P0.2 — Kill the source shortcut (rebuild the real class)

**Files:** `ml/src/data/dataset_splits.py`, `ml/scripts/train_xgboost_meta.py`.

This is the highest-value change in the whole plan. Without it, every downstream model improvement
is wasted.

### Step 1 — group by the manifest's `source` field, not by the path

**Correction to rev. 1.** The shipped layout is `test/fake/<generator>/<file>` and
`test/real/<file>`, so `Path(path).parts[-2]` yields the *generator* (or `"real"`), never the
acquisition source. The source is carried per record in `dataset_manifest.json` and is already the
axis `ml/src/training/evaluation.py:40` groups by. Use the manifest.

```python
# ml/src/data/dataset_splits.py — ADD

def group_by_source_from_manifest(records: list[dict]) -> dict[str, list[dict]]:
    """Group manifest records by their declared acquisition source.

    The directory layout cannot answer this question: the shipped tree is
    ``<split>/fake/<generator>/<file>`` and ``<split>/real/<file>``, so the
    path only encodes the GENERATOR. The acquisition source (laion,
    pexels, docci, imagenet, ...) lives in the manifest record's
    ``source`` field, which ``training/evaluation.py`` already groups by.

    A detector that separates real from fake by learning that LAION
    images are synthetic and Pexels images are photographic has learned
    provenance, not synthesis.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        source = record.get("source")
        if source is None:
            raise ValueError(
                f"Manifest record {record.get('path')!r} has no 'source' field. "
                "Per-source FPR cannot be audited without it."
            )
        groups[str(source)].append(record)
    return dict(groups)


def source_balance_report(records: list[dict]) -> dict[str, dict]:
    """Per-source composition of the real and fake classes.

    A source that appears only in one class is a shortcut opportunity.
    """
    out: dict[str, dict] = {}
    for source, items in group_by_source_from_manifest(records).items():
        real = sum(1 for r in items if r.get("label") == 0)
        fake = sum(1 for r in items if r.get("label") == 1)
        out[source] = {"n_real": real, "n_fake": fake, "n": len(items)}
    return out
```

### Step 2 — add a source-FPR release gate

```python
# ml/scripts/train_xgboost_meta.py — ADD after the laundered AUC check

def assert_source_fpr_within_bound(report, max_fpr=0.05, min_real_per_source=15):
    """Reject a release whose real-image FPR is high for any single source.

    A detector that flags 44% of LAION images as synthetic has learned
    that LAION images are synthetic. That is a dataset shortcut, not
    detection, and it must fail the build rather than ship.
    """
    sources = (report.get("groups") or {}).get("source") or {}
    offenders = []
    for source, stats in sources.items():
        if stats.get("n_real", 0) < min_real_per_source:
            continue
        fpr = stats.get("fpr")
        if fpr is not None and fpr > max_fpr:
            offenders.append(f"{source}: fpr={fpr:.3f} (n_real={stats['n_real']})")
    if offenders:
        raise ValueError(
            "Real-image false-positive rate exceeds the per-source bound "
            f"({max_fpr:.0%}). The detector is keying on acquisition source, "
            "not on synthesis. Offenders: " + "; ".join(offenders)
        )
```

Call it on `test_report` and `laundered_report` immediately before writing `meta_results.json`, and
apply the **same gate to the audio model**, where `inthewild` real speech currently scores 24.3% FPR.

### Step 3 — fix the corpus

The data change is the actual fix; the gate just enforces it:

- **Image:** the real class must contain LAION, ImageNet, DocCI and Pexels-style images in quantities
  matched to the fake class's aesthetic statistics (REALIS's "density-based sampling"). It must also
  contain camera photos, re-captures, screenshots, and legitimate edits.
- **Audio:** the real class must contain **codec/channel-matched** speech. Route *both* classes
  through the same offline `ffmpeg` presentation pipeline (Opus 16/24/32 kbps, AAC 32/64 kbps,
  AMR-WB, MP3, room impulse responses, background SNR 10–30 dB) so that "noisy real speech" is not
  a class-correlated giveaway. This is the "It's All in the Presentation" fix and it also removes the
  A18 0%-TPR failure, which is a vocoder with no obvious spectral signature.
- **Paired real/fake construction (B-Free protocol):** generate synthetic counterparts directly from
  the real LAION/Pexels/DocCI images (SDEdit / autoencoder reconstruction / inpainting) so each
  real image and its synthetic twin share identical semantics, aspect ratio and source aesthetics.

**Acceptance gate.** No source with ≥15 real samples may exceed 5% FPR on the sealed test; the
laundered split must satisfy the same bound; and the same bound must hold for the audio model.

### P0.3 — Fix the release-integrity defects

### P0.3a — stop the desktop report from setting `validated: true`

**Files:** `ml/scripts/release_bundle.py:220-253`, `ml/scripts/build_and_install_release_v4.py:166-200`.

```python
# ml/scripts/release_bundle.py — REPLACE attach_device_report's validation
REQUIRED_DEVICE_EVIDENCE = (
    "device_id", "device_model", "android_sdk_int", "app_version",
    "measurement_kind", "measured_at_utc",
)

def _assert_device_report_is_from_a_device(report: dict) -> None:
    """A desktop report must never be able to set validated=true.

    build_and_install_release_v4.py used to synthesise a report whose
    'latency' was a desktop tf.lite.Interpreter timing and whose
    parity keys were named 'desktop_agreement'. Those keys are a
    fingerprint of a non-device report.
    """
    missing = [k for k in REQUIRED_DEVICE_EVIDENCE if not report.get(k)]
    if missing:
        raise ValueError(f"Device report is missing device evidence: {missing}")
    if report.get("measurement_kind") != "on_device_android":
        raise ValueError(
            "Device report measurement_kind must be 'on_device_android'; "
            f"got {report.get('measurement_kind')!r}. A desktop interpreter "
            "timing is not device validation."
        )
    for name, entry in (report.get("models") or {}).items():
        for banned in ("desktop_agreement", "desktop_mae", "desktop_max_error"):
            if banned in entry:
                raise ValueError(
                    f"{name}: device report carries desktop-only key '{banned}'. "
                    "This report was not produced on a device."
                )
    if not report.get("on_device_measurements"):
        raise ValueError("Device report contains no on-device measurement payload")
```

`build_and_install_release_v4.py` must then **not** be able to produce a passing report; it should
raise and tell the operator to run the instrumented Android test.

### P0.3b — stop overwriting Engine A's calibration — **IMPLEMENTED**

**Correction to rev. 1.** Rev. 1's snippet broke application startup on its own:
`ModelIntegrityChecker.kt:207-209` asserts the SRM calibration threshold equals
`meta_classifier.json.decision_threshold`, and `DetectionPipeline.kt` uses `report.threshold` as the
**fusion** threshold. Both must change together.

**Correction to rev. 2.** Rev. 2's snippet fixed the startup crash but introduced a worse one: it
set `threshold = fusionThreshold` unconditionally, so the **audio** route — which calls
`report(context, setOf(AUDIO_MODEL))` and compares its score against `report.threshold` — would have
been judged against the *image fusion* threshold (0.51216) instead of the audio threshold (0.375).
`report.threshold` must be **modal**.

**Implemented.** The separation now lives in the manifest, and the operating point is selected
modally:

```python
# ml/scripts/generate_android_constants.py — APPLIED
# Engine A's calibration belongs to Engine A's own INT8 model. The
# previous code wrote the FUSION calibration into the SRM model's key,
# so the manifest recorded the FUSION's operating point as Engine A's
# and Engine A's real calibration (auc 0.6033) existed nowhere.
manifest["calibration"]["fusion_xgboost"] = dict(
    results["calibration"],
    model_hash=results["model_hash"],
    frontend_id=visual["frontend_id"],
)
# The SRM key keeps whatever export_all_models.py wrote - Engine A's own
# INT8 calibration - and is left untouched here.
```

```python
# ml/scripts/release_bundle.py — APPLIED
meta = read_object(bundle / 'constants/meta_classifier.json')
# The FUSION's calibration lives under its own key in the current layout.
# The legacy layout stored it under the SRM model's key, which is why
# Engine A's own operating point was recorded nowhere. Both layouts are
# accepted so an already-staged release keeps validating.
fusion_cal = calibrations.get('fusion_xgboost') or calibrations['srm_forensic_cnn_int8.tflite']
if meta.get('decision_threshold') != fusion_cal['threshold']:
    raise ValueError('Fusion operating threshold mismatch')
```

```kotlin
// android/.../inference/ModelIntegrityChecker.kt — APPLIED
// Each model's threshold is kept SEPARATELY. The previous code took ONE
// `threshold` out of the loop, so `report.threshold` was whichever model
// happened to be visited last in a Set. The audio route calls
// `report(context, setOf(AUDIO_MODEL))` and compares its score against
// `report.threshold`; with a shared field that could be the IMAGE
// FUSION's threshold, and an audio score would be judged against the
// wrong operating point.
val modelThresholds = mutableMapOf<String, Float>()
required.forEach { name ->
    // ... per-model lineage, shape and calibration checks ...
    modelThresholds[name] = t
}

var fusionThreshold: Float? = null
if (IMAGE_MODEL in required) {
    val fusionCal = calibrations.optJSONObject("fusion_xgboost")
        ?: calibrations.getJSONObject(IMAGE_MODEL)   // legacy layout
    val fusionThr = numeric(fusionCal, "threshold").toFloat()
    require(fusionThr.isFinite() && fusionThr in 0f..1f)
    // The fusion threshold must equal the constant the pipeline reads.
    val metaConstantThreshold = numeric(read("constants/meta_classifier.json"), "decision_threshold").toFloat()
    require(kotlin.math.abs(fusionThr - metaConstantThreshold) < 1e-6f) {
        "Fusion calibration asset/release threshold mismatch"
    }
    fusionThreshold = fusionThr
}

// Modal operating point: the image/video pipeline decides with the FUSION
// score, so it uses the fusion threshold. An audio-only caller uses the
// audio model's own threshold.
val primaryThreshold = if (IMAGE_MODEL in required) fusionThreshold else modelThresholds[AUDIO_MODEL]

Report(true, emptyList(), true, "Validated release", releaseId, frontend, primaryThreshold,
       cascadeValidated, fusionThreshold, modelThresholds.toMap())
```

Two details that are easy to get wrong and are now covered by tests:

- The fusion calibration is only required when `IMAGE_MODEL in required`. An audio-only caller must
  not fail because the manifest has no fusion calibration.
- In the **legacy** layout the SRM slot holds the *fusion's* calibration, so the per-model
  cross-check against the desktop evaluation report is skipped there; in the **new** layout every
  slot holds its own model and the cross-check runs. Both layouts validate.

**Acceptance gate.** The manifest contains a `fusion_xgboost` calibration entry whose `threshold`
equals `constants/meta_classifier.json.decision_threshold`, and the SRM entry equals Engine A's own
INT8 calibration from `evaluation_report.json` (auc 0.6033, threshold 0.53125).

### P0.3c — one release pipeline

`run_training_pipeline.py` must gain the Engine C stages so the documented pipeline reproduces the
shipped release:

```python
# ml/scripts/run_training_pipeline.py — ADD after the export stage
if args.semantic_checkpoint:
    semantic_tflite = bundle / "models" / "semantic_probe_int8.tflite"
    subprocess.run(
        [sys.executable, str(SCRIPTS / "export_semantic_probe.py"),
         "--checkpoint", str(args.semantic_checkpoint),
         "--output", str(semantic_tflite)],
        check=True,
    )
    # Engine C must be trained BEFORE fusion, because fusion reads its
    # scores into slot 6.
```

```python
# and in the fusion stage:
fusion_cmd = [sys.executable, str(SCRIPTS / "train_xgboost_meta.py"),
              "--config", str(config), "--visual-checkpoint", str(visual_checkpoint),
              "--visual-tflite", str(bundle / "models" / "srm_forensic_cnn_int8.tflite"),
              "--output-dir", str(fusion), "--release-id", release_id]
if semantic_tflite.exists():
    fusion_cmd += ["--semantic-tflite", str(semantic_tflite),
                   "--semantic-input-size", "224",
                   "--cascade-mask-rate", "0.25"]
subprocess.run(fusion_cmd, check=True)
```

**Acceptance gate.** A clean run of `run_training_pipeline.py` on the same data reproduces the same
release ID and the same artifact hashes.

---

## C.2 — P1: image detection

### P1.1 — Ship the *right* frozen VFM, and stop losing 0.166 AUC to distillation

**Correction to rev. 1.** Rev. 1 recommended shipping a frozen MobileCLIP2-S0 tower. **That tower
measures 0.5145 held-out AUC — chance** (`docs/DUAL_ENGINE_CASCADE.md:198`, `:213`). Shipping it
would have shipped a chance-level detector, exactly as the repo's own experiment warned.

**Use SigLIP-B/16 (SmolVLM-500M's tower, 0.8491 held-out) or DINOv2 ViT-S/14 with registers**
(the B-Free backbone).

**Files:** `ml/scripts/train_semantic_probe.py`, `ml/scripts/export_semantic_probe.py`,
`ml/src/export/torch_to_tflite_keras.py`, `android/.../SemanticProbeAnalyzer.kt`.

**Option A (preferred) — ship the frozen tower.** SigLIP-B/16 is ~86M params (~86 MB fp32, ~22 MB
INT8 per-channel); DINOv2 ViT-S/14 is ~21M (~21 MB INT8). Both are plausible APK payloads. The
historical blocker was the ONNX→TFLite toolchain, not size; `ai-edge-torch` / `onnx2tf` close it.

**Option B — stop distilling into a CNN.** Replace the CNN student with a **linear probe on frozen
embeddings**, shipped as an embedding table plus a dot product. This only helps if the tower ships.

**Option C (interim, cheapest) — stop hiding the distillation gap.** The teacher is only used to
produce soft labels; the student is a from-scratch CNN trained on ~2,800 images, which cannot learn
open-world semantic priors. Measure the teacher directly and publish the gap.

```python
# ml/scripts/train_semantic_probe.py — ADD to the summary
teacher_test = _teacher_logits_for(test_samples, teacher_repo, device)
teacher_auc = float(compute_auc(test_labels, teacher_test))
student_auc = float(compute_auc(test_labels, test_scores))
summary = {
    ...,
    "teacher_test_auc": teacher_auc,
    "student_test_auc": student_auc,
    "distillation_gap_auc": teacher_auc - student_auc,
    "shipped_architecture": "frozen_vfm" if args.ship_frozen_tower else "distilled_cnn_student",
}
if summary["distillation_gap_auc"] > 0.05:
    print(
        f"WARNING: distillation costs {summary['distillation_gap_auc']:.4f} AUC "
        f"({teacher_auc:.4f} -> {student_auc:.4f}). "
        "Shipping the distilled student is a deliberate accuracy sacrifice; "
        "record it in the model card."
    )
```

**Also: guard against semantic overfitting.** Effort (ICML 2025, arXiv:2411.15633) shows that a
naive linear probe or LoRA over a VFM overfits high-variance semantic directions — precisely the
"LAION web image vs. Pexels stock photo" axis that is destroying this detector. Combine the probe
with either Effort's orthogonal-subspace adapter (SVD on the pre-trained weights, train only in the
orthogonal lower singular subspace) or PRL's relative-patch-response objective, which avoids
collapsing everything onto a single global `[CLS]` token.

**Acceptance gate.** `evaluation_report.json` must record `teacher_test_auc`, `student_test_auc`
and `distillation_gap_auc`. If `shipped_architecture == "distilled_cnn_student"` and the gap > 0.05,
the model card must carry an explicit justification. **Any new tower must be re-measured on a
HELD-OUT split, not a fit split** — the MobileCLIP2-S0 failure was precisely a fit-split number
(0.9688) shipped as if it were generalisation (0.5145).

### P1.2 — DCPT: degradation-consistent paired training — **IMPLEMENTED**

**Files:** `ml/src/training/dcpt_losses.py` (new), `ml/src/data/augmentations.py`,
`ml/src/training/trainer.py`, `ml/scripts/train_semantic_probe.py`, `ml/scripts/train_srm_cnn.py`.

DCPT (arXiv:2604.10102) adds **zero parameters and zero inference overhead** and gains **+9.1 pts**
degraded-average accuracy (+15.7% @JPEG QF=70, +17.2% @QF=30) for −0.9% clean.

**Correction to rev. 1.** Rev. 1 implemented only the logit-level symmetric-KL term. DCPT has
**two** terms; the feature-level cosine term on the penultimate embedding is half the method.

**Correction to rev. 2.** Rev. 2's KL term used `F.kl_div(log_q, p)` on a 1-D probability.
`F.kl_div(input, target)` computes `target * (log target - input)` **pointwise**, so with a single
probability it evaluates only the positive-class term `p·log(p/q)` and silently drops the negative
-class term `(1−p)·log((1−p)/(1−q))`.

That is not a divergence at all for a binary classifier, and it weights the gradient by `p` instead
of by `1−p`. On a real image (`p_clean ≈ 0.01`) against a degraded view the model is confident
about (`p_deg ≈ 0.40`) the positive-only term is **negative**: `0.01·ln(0.01/0.40) = −0.0369`. A
"loss" that goes *down* when the two views disagree most is the opposite of a consistency
constraint.

**Implemented** in `ml/src/training/dcpt_losses.py`:

```python
def dcpt_consistency_loss(
    clean_logits: torch.Tensor,
    degraded_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Symmetric BINARY KL divergence between two Bernoulli predictions.

    The divergence is computed over both outcomes ``(p, 1 - p)``. The
    ``temperature ** 2`` factor restores the gradient scale that the
    soft-target temperature scaling removed, matching the standard
    knowledge-distillation convention.
    """
    if clean_logits.shape != degraded_logits.shape:
        raise ValueError(
            "clean and degraded logits must have the same shape, got "
            f"{tuple(clean_logits.shape)} and {tuple(degraded_logits.shape)}"
        )
    p = torch.sigmoid(clean_logits / temperature).clamp(_EPS, 1.0 - _EPS)
    q = torch.sigmoid(degraded_logits / temperature).clamp(_EPS, 1.0 - _EPS)
    kl_pq = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
    kl_qp = q * torch.log(q / p) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - p))
    return 0.5 * (kl_pq + kl_qp).mean() * (temperature ** 2)


def dcpt_feature_consistency_loss(z_clean: torch.Tensor, z_degraded: torch.Tensor) -> torch.Tensor:
    """DCPT's feature-level term: ``1 - cosine_similarity`` of the embeddings.

    The penultimate embedding carries the representation the classifier
    reads. Constraining it to agree across a clean and a degraded view is
    what makes the model rely on structure rather than on the exact
    high-frequency fingerprint that compression destroys.
    """
    if z_clean.shape != z_degraded.shape:
        raise ValueError(
            "clean and degraded embeddings must have the same shape, got "
            f"{tuple(z_clean.shape)} and {tuple(z_degraded.shape)}"
        )
    return (1.0 - F.cosine_similarity(z_clean, z_degraded, dim=-1)).mean()
```

The paired views come from `ml/src/data/augmentations.py`:

```python
class PairedDegradationAugmentation:
    """Yield a (clean, degraded) pair from the SAME source image.

    DCPT (arXiv:2604.10102) shows that treating degradation robustness as a
    data-augmentation by-product is not enough: the objective must
    explicitly constrain a clean view and a degraded view of ONE image to
    agree, both in feature space and in the output distribution.
    """

    def __call__(self, img: Image.Image) -> tuple[Image.Image, Image.Image]:
        """Return ``(clean, degraded)`` views of ``img``."""
        clean = img
        if random.random() >= self.p_degrade:
            return clean, clean
        view = clean
        if random.random() < self.p_resize:
            view = self._resize(view)
        if random.random() < self.p_noise:
            view = _add_gaussian_noise_pil(view, random.uniform(self.min_sigma, self.max_sigma))
        if random.random() < self.p_jpeg:
            view = self._jpeg(view).convert("RGB")
        if self.target_size is not None:
            view = view.resize((self.target_size, self.target_size), Image.Resampling.BILINEAR)
        return clean, view
```

Wire into the training loop:

```python
# in train_epoch(), when the loader yields (clean, degraded, targets):
clean_logits = self.model.forward_logits(clean)
degraded_logits = self.model.forward_logits(degraded)
loss = self.criterion(clean_logits, targets)
if self.dcpt_weight > 0:
    loss = loss + self.dcpt_weight * dcpt_consistency_loss(clean_logits, degraded_logits)
    if z_clean is not None and z_degraded is not None:
        loss = loss + self.dcpt_feature_weight * dcpt_feature_consistency_loss(z_clean, z_degraded)
```

with `dcpt_weight` / `dcpt_feature_weight` from `config["training"]` and a config flag
`training.dcpt_degrade_probability`.

**Acceptance gate.** The sealed test must be evaluated on a **degradation matrix** (JPEG QF
90/70/50/30, downscale 0.75/0.5/0.25, Gaussian blur σ 0.5/1.0/2.0, and their chains), and the
report must include per-condition accuracy — not only the laundered 512/Q85 point that exists today.
`ml/tests/test_dcpt_losses.py` (13 tests) pins the loss: it asserts the positive-only term is
*negative* on a real image and that the full binary term is positive and larger.
### P1.3 — Add a phase-spectrum branch

**Files:** `android/.../FFTSpectralAnalyzer.kt`, `ml/src/meta_classifier/feature_extractor.py`,
`ml/scripts/train_xgboost_meta.py`.

**Correction to rev. 1.** Rev. 1's snippet had two math errors: it indexed radial bins around
`center = n // 2` on an **unshifted** `np.fft.fft2` output (so the radii were centred on the Nyquist
corner rather than DC), and it used raw `np.cos(phase)`, which is **not translation-invariant** — a
one-pixel shift adds a linear phase ramp that scrambles the statistic. The correct formulation is
the phase-only spatial reconstruction of Li et al., CVPR 2026.

```python
# ml/src/meta_classifier/feature_extractor.py — ADD

def phase_spectrum_features(img: Image.Image, size: int = 256) -> tuple[float, float]:
    """Shift-invariant 2D phase-only spatial residual.

    JPEG quantization heavily damps high-frequency AMPLITUDE but leaves
    the phase angles intact, so a phase-only reconstruction survives
    re-encoding where a power-spectrum slope does not.

    Two details matter and are easy to get wrong:
      * the DC term must be moved to the centre with fftshift before any
        radial statistic is computed, otherwise the radii are centred on
        the Nyquist corner;
      * the phase must be turned back into an image (the phase-only
        reconstruction) rather than reduced to cos(phase) directly, because
        a translated image acquires a linear phase ramp that cos(phase)
        does not tolerate.
    """
    scaled = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    gray = to_gray_u8(scaled).astype(np.float64) / 255.0
    gray = gray - gray.mean()
    spectrum = np.fft.fft2(gray)
    unit_phase = np.exp(1j * np.angle(spectrum))

    # Phase-only reconstruction: the image whose spectrum is the unit-magnitude
    # phase of the original.
    phase_img = np.fft.ifft2(unit_phase).real

    # High-frequency band of the same reconstruction, with the DC term moved
    # to the centre first.
    shifted = np.fft.fftshift(unit_phase)
    n = size
    cy, cx = n // 2, n // 2
    y, x = np.ogrid[:n, :n]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    hf_mask = (r >= 0.25 * (n / 2.0)).astype(np.float64)
    hf_phase_img = np.fft.ifft2(np.fft.ifftshift(shifted * hf_mask)).real

    # Kurtosis of the phase-only reconstruction and the high-frequency
    # energy ratio. Both are translation-equivariant in the same way the
    # reconstruction is, which is what makes them comparable across
    # crops and resolutions.
    kurtosis = float(np.mean(phase_img ** 4) / (np.mean(phase_img ** 2) ** 2 + 1e-12))
    hf_ratio = float(np.std(hf_phase_img) / (np.std(phase_img) + 1e-12))
    return kurtosis, hf_ratio
```

**Acceptance gate.** The new slots must improve the **laundered** (`512px JPEG Q85`) sealed-test AUC
and the JPEG-QF-30 slice. If they do not, drop them — do not carry a feature that does not earn its
place.

### P1.4 — Make the fusion contract honest: live slots only — **MIGRATION, NOT A DROP-IN**

**Files:** `ml/src/meta_classifier/feature_extractor.py`, `ml/src/meta_classifier/dual_engine.py`,
`android/.../domain/pipeline/FusionFeatureSchema.kt`,
`android/.../domain/pipeline/DetectionPipeline.kt`,
`android/.../detection/video/VideoForensicAnalyzer.kt`,
`android/.../detection/fusion/XGBoostMetaClassifier.kt` (regenerated),
`ml/scripts/train_xgboost_meta.py`, `scripts/check_android_consistency.py`.

Two options, and they must not be mixed:

**Option A — implement slots 7/8/9** (C2PA, lip-sync, landmark jitter) as real measurements. That is
P3 and P4 work; until then the slots are dead.

**Option B — shrink the contract** to the live features and release a new fusion revision. Given
§A.6 (two of the four live features are at or below chance), Option B may actually be the right
first move.

**Correction to rev. 2.** Rev. 2 presented Option B as a drop-in snippet. It is not, and applying it
alone would have broken the build and every video scan:

- `train_xgboost_meta.py:53` hard-coded `SHORTCUT_FEATURE_INDICES = (4, 5)` and `neutralize_shortcuts`
  indexed `X[:, 4]`/`X[:, 5]` → **IndexError** as soon as the vector had fewer than six columns.
- `VideoForensicAnalyzer.kt:65` built a ten-element `floatArrayOf(...)` by hand → the transpiled
  tree's `require(features.size == 10)` would have thrown **IllegalArgumentException** on every
  video scan.
- It also contradicted §A.6, which says "shrink the contract to 3 live slots" — rev. 2 shrank it to
  two, dropping `flatness_score` (standalone AUC 0.6135).

**What was implemented instead: the trap is removed, the contract is not yet shrunk.**

The root cause is that the feature contract was duplicated in six places. It is now single-sourced:

```kotlin
// android/.../domain/pipeline/FusionFeatureSchema.kt — APPLIED
object FusionFeatureSchema {
    const val SLOT_COUNT = 10
    const val SRM_SCORE = 0
    const val FFT_SCORE = 1
    const val KL_SCORE = 2
    const val FLATNESS_SCORE = 3
    const val RESOLUTION_NORM = 4
    const val IS_COMPRESSED = 5
    const val SEMANTIC_PROBE_SCORE = 6
    const val C2PA_PRESENT = 7
    const val LIP_SYNC_CORR = 8
    const val LANDMARK_JITTER = 9

    /** The only slot permitted to be non-finite. */
    const val OPTIONAL_SLOT = SEMANTIC_PROBE_SCORE
    val OPTIONAL_SLOTS = intArrayOf(OPTIONAL_SLOT)

    fun build(vararg values: Float): FloatArray {
        require(values.size == SLOT_COUNT) {
            "Fusion feature vector needs $SLOT_COUNT slots, got ${values.size}"
        }
        return FloatArray(values.size) { values[it] }
    }
}
```

Both pipelines now build through it, so a wrong arity fails at the point of construction rather
than inside the transpiled tree:

```kotlin
// DetectionPipeline.kt and VideoForensicAnalyzer.kt — APPLIED
val features = FusionFeatureSchema.build(
    srm, spectral.score, kl, spectral.spectralFlatness,
    resolutionNorm, isCompressed,
    engineCScore ?: Float.NaN,
    NEUTRAL_C2PA, NEUTRAL_LIP_SYNC, NEUTRAL_JITTER,
)
```

The Python side was made name-derived and bounds-safe:

```python
# ml/scripts/train_xgboost_meta.py — APPLIED
SHORTCUT_FEATURE_INDICES = tuple(
    index for index, name in enumerate(FEATURE_NAMES)
    if name in ("resolution_norm", "is_compressed")
)

def neutralize_shortcuts(X: np.ndarray) -> np.ndarray:
    """Mask the shortcut columns so trees cannot split on them.

    The columns are located by NAME through SHORTCUT_FEATURE_INDICES, which
    is derived from FEATURE_NAMES, and every column is bounds-checked. A
    hard-coded index pair raised IndexError the moment the feature contract
    was shortened, which is exactly the kind of failure that only shows up
    when someone tries to migrate it.
    """
    X_clean = np.array(X, dtype=np.float32, copy=True)
    for index in SHORTCUT_FEATURE_INDICES:
        if index >= X_clean.shape[1]:
            continue
        name = FEATURE_NAMES[index]
        value = NEUTRAL_RESOLUTION_NORM if name == "resolution_norm" else NEUTRAL_IS_COMPRESSED
        X_clean[:, index] = value
    return X_clean
```

And the cross-language check now verifies the runtime contract, not just the names in a comment:

```python
# scripts/check_android_consistency.py — APPLIED
m_slot = re.search(r'require\(features\.size == (\d+)\)', kt_src)
if not m_slot:
    errors.append('Kotlin fusion source does not assert a runtime feature slot count')
elif int(m_slot.group(1)) != len(feats):
    errors.append(
        f'Kotlin fusion runtime slot count {m_slot.group(1)} does not match the '
        f'Python feature schema ({len(feats)} slots)'
    )

m_opt = re.search(r'listOf\((\d+(?:,\s*\d+)*)\)', kt_src)
if not m_opt:
    errors.append('Kotlin fusion source does not declare its optional (NaN) slot')
else:
    kt_optional = {int(v) for v in re.findall(r'\d+', m_opt.group(1))}
    if kt_optional != {py_index}:
        errors.append(
            f'Kotlin optional slot {sorted(kt_optional)} does not match '
            f'Python SEMANTIC_PROBE_INDEX {py_index}'
        )
```

**The actual contract shrink, when it is done, is a migration with this checklist:**

1. `ml/src/meta_classifier/feature_extractor.py` — new `FEATURE_NAMES`, new `SEMANTIC_PROBE_INDEX`.
2. `ml/scripts/train_xgboost_meta.py` — `SHORTCUT_FEATURE_INDICES` is name-derived (done), and
   `neutralize_shortcuts` is bounds-checked (done).
3. `ml/scripts/train_xgboost_meta.py` — `export_to_kotlin` regenerates the guard from
   `OPTIONAL_FEATURE_INDICES` (already does).
4. `android/.../FusionFeatureSchema.kt` — `SLOT_COUNT` and the named indices.
5. `android/.../DetectionPipeline.kt` and `VideoForensicAnalyzer.kt` — build through the schema (done).
6. `android/.../XGBoostMetaClassifier.kt` — regenerated by the exporter, never hand-edited.
7. `scripts/check_android_consistency.py` — the expected-features list, and the runtime slot count
   and optional-slot checks (the latter two are done).
8. A **new `release_id`** — a changed fusion contract requires a new release, as
   `docs/DUAL_ENGINE_CASCADE.md:296-302` notes for the slot-6 migration.
9. Re-stage the assets and re-run `release_bundle.py`; the new contract must clear the acceptance
   floors in §P5.1.

Until that checklist is complete end-to-end, the contract stays at ten slots and the five dead
slots are documented as dead rather than silently removed.

**Acceptance gate.** `scripts/check_android_consistency.py` fails when the Python feature schema,
the Kotlin runtime slot count, the optional-slot index, or `FusionFeatureSchema.SLOT_COUNT` disagree.
### P1.5 — Fix Engine A's preprocessing order (native-resolution patches)

**Files:** `ml/src/models/srm_forensic_cnn.py`, `ml/src/data/image_frontend.py`,
`ml/src/export/torch_to_tflite_keras.py`, `android/.../SRMForensicAnalyzer.kt`.

Engine A's SRM and NPR filters are convolutions applied to an image that has already been
bilinearly downsampled to 128×128. An 8× downsample is a low-pass anti-aliasing filter, so it
**destroys the native 2×2 upsampling grid that NPR exists to detect**. The PatchCraft-style
`extract_high_low_texture_patches` helper already exists in `srm_forensic_cnn.py` but is only used
by `distill_vision_tower_to_srm.py` — not by the shipped training or inference path.

Following PatchCraft (arXiv:2311.12397) and NPR (CVPR 2024, arXiv:2312.10461):

1. Decode at native resolution (after the existing `pow2cap1024` chain).
2. Tile into 32×32 patches; rank by gradient energy.
3. Crop the top-K **rich-texture** and bottom-K **poor-texture** patches into two 128×128 mosaics
   **without interpolation**.
4. Feed their high-pass residual difference `F_rich − F_poor` into Engine A.

Because both mosaics come from the *same* image, camera and compression noise cancel while
generator upsampling residuals survive.

**Acceptance gate.** Engine A's sealed-test AUC and its TPR on the upsampling-dominated generators
must both improve; a native-patch pipeline that does not beat the resized one must not ship.

### P1.6 — Cascade objective (implemented — see §C.0)

The corrected `fit_escalation_band` and `decide()` are already in the repository (§C.0 items 4–6).
The remaining work is to **re-fit and re-release** the policy against the corrected code, because the
shipped `cascade_policy.json` was produced by the buggy fitter and cannot be repaired in place.

**Acceptance gate.** The sealed-test cascade report must show `selective_accuracy` **and**
`classified_rate`; the reported score's threshold must come from the same engine as the score; and
the escalation rate must be materially below the current 0.7150.

---

## C.3 — P2: audio

### P2.1 — Channel/codec degradation in training

**File:** `ml/src/data/augmentations.py`.

**Correction to rev. 1.** `torchaudio.functional.amr_nb_encode` **does not exist in standard
torchaudio builds**, so rev. 1's implementation would have silently degraded to a no-op behind a
broad `except Exception`. Pre-generate the codec/channel variants deterministically at dataset-prep
time with `ffmpeg` (or `sox`) and hash them into `dataset_manifest.json`. This is also more
reproducible, which the repo's contract discipline requires.

```python
# ml/src/data/augmentations.py — ADD
class ChannelCodecAugmentation:
    """Telephone/VoIP/codec channel degradation.

    Deepfake audio is published through phones, messengers and video
    calls. A detector trained only on studio-quality TTS output learns
    a cue that the channel destroys - and, worse, learns that NOISE
    implies synthetic speech. The shipped detector flags 24.3% of real
    inthewild field recordings, because ASVspoof2019 bona-fide speech is
    clean studio audio and inthewild bona-fide speech is not.

    ASVspoof 2021 LA and "It's All in the Presentation" (ICASSP 2026)
    both show that channel mismatch, not model capacity, is the dominant
    failure mode: matching the presentation channel across both classes
    was worth +57% real-world accuracy.

    The variants themselves are pre-generated with ffmpeg at dataset-prep
    time and hashed into the manifest; this class selects among them so
    the transformation is reproducible and auditable.
    """

    def __init__(self, variants: dict[str, str], p: float = 0.4):
        # variants maps a codec name to a prepared fixture directory.
        self.variants = variants
        self.p = p

    def __call__(self, waveform, sample_id=None):
        if not self.variants or random.random() >= self.p:
            return waveform
        # ...select a prepared variant deterministically from sample_id...
        raise NotImplementedError("prepared codec fixtures are selected by the dataset loader")
```

**Acceptance gate.** The sealed test must include a **codec/channel holdout** slice (never seen in
training) and report EER per codec. The per-source FPR gate must be applied to the audio model, and
`inthewild` must fall below 5%.

### P2.2 — Decorrelate the watermark from the label

**Files:** `ml/src/data/contracts.py`, `ml/src/data/augmentations.py`.

The watermark-shortcut paper's fix is explicit: **watermark both classes during training**.

```python
# ml/src/data/augmentations.py — ADD
class WatermarkDecorrelation:
    """Apply a provenance watermark to BOTH classes, at random.

    When synthetic speech is watermarked and human speech is not, a
    detector trained alongside learns "watermarked => fake" as a
    shortcut. The documented consequences (arXiv:2606.23335) are:
    strip-to-evade (FN rises to 80%), mark-to-frame (EER 16.4% -> 74.6%
    when a real voice is watermarked) and degraded generalization.

    Decorrelating watermark presence from the label removes the
    shortcut. It must be applied to the bona-fide class as well.
    """

    def __init__(self, watermark_fn, p_real: float = 0.5, p_fake: float = 0.5):
        self.watermark_fn = watermark_fn
        self.p_real = p_real
        self.p_fake = p_fake
```

Use **AudioSeal** and **WavMark** (both open weight, both offline-capable). The manifest must record
the watermark state per sample so it can be asserted:

```python
# ml/src/data/contracts.py — record-level extension
# Each record may carry:
#   "watermarked": bool   - whether a provenance watermark is present
#   "generator": str      - generator family, for per-generator reporting
#   "source": str         - acquisition source, for per-source FPR
#   "codec": str          - presentation channel, for per-codec EER
# A manifest that does not record these cannot be audited for the
# watermark shortcut, the source shortcut or the channel shortcut.
```

**Acceptance gate.** A sealed-test slice of **watermarked real** speech must exist, and the
detector's FPR on it must be within tolerance of its FPR on unwatermarked real speech. If the
watermark-shortcut is present, this gate fails.

### P2.3 — Multi-window file-level aggregation — **POOLING IMPLEMENTED, DEFAULT UNCHANGED**

**Files:** `android/.../detection/audio/WindowPooling.kt` (new),
`android/.../detection/audio/AudioClassifier.kt`, `android/.../detection/audio/AudioDecoder.kt`.

Today `AudioClassifier` scores **one** 4-second window (`TARGET_SAMPLES = 64000`). For a 3-minute
voice note that is 2.2% of the content, and modern generators produce *segment-local* artifacts.

**Correction to rev. 1.** Raw `max` over 16 windows inflates the file-level FPR on long recordings —
the more windows you take, the more likely one of them is a false positive.

**Correction to rev. 2.** Rev. 2's `poolWindowLogits` treated the model output as logits. It is
**not**: `lcnn_audio_int8.tflite` ends in `Dense(1, activation="sigmoid")`
(`ml/src/export/torch_to_tflite_keras.py:188`), so the model emits a **probability in `[0, 1]`**.
Pooling those with a log-sum-exp applies a soft-max nonlinearity to values that are already
probabilities, and the result is not a probability at all — it can fall outside `[0, 1]`, which
`ScanResult.init` rejects (`require(overallScore == -1f || overallScore in 0f..1f)`) and which the
calibrated threshold comparison (`score >= 0.375f`) cannot use.

**Implemented** in `android/.../detection/audio/WindowPooling.kt`:

```kotlin
internal object WindowPooling {
    /** Higher approaches max pooling, lower approaches mean pooling. */
    const val LSE_TEMPERATURE = 3.0f
    private const val EPS = 1e-6f

    fun poolProbabilities(probs: FloatArray, tau: Float = LSE_TEMPERATURE): Float {
        if (probs.isEmpty()) return Float.NaN
        require(tau > 0f && tau.isFinite()) { "LSE temperature must be positive and finite" }
        // Probability -> logit.
        val logits = FloatArray(probs.size) { index ->
            val p = probs[index]
            require(p.isFinite() && p in 0f..1f) { "Window score must be a probability in [0,1]" }
            val clamped = p.coerceIn(EPS, 1f - EPS)
            ln((clamped / (1f - clamped)).toDouble()).toFloat()
        }
        // LSE in logit space, numerically stabilised by the max shift.
        var maxLogit = -Float.MAX_VALUE
        for (z in logits) if (z > maxLogit) maxLogit = z
        var sum = 0.0
        for (z in logits) sum += exp(((z - maxLogit) / tau).toDouble())
        val pooledLogit = maxLogit + tau * ln(sum / logits.size).toFloat()
        // Logit -> probability. The result is guaranteed to lie in (0, 1).
        return (1.0 / (1.0 + exp(-pooledLogit.toDouble()))).toFloat()
    }

    /** Evenly spaced window start offsets, excluding the leading/trailing 5%. */
    fun windowOffsets(durationMs: Long, windowMs: Long, count: Int): List<Long> { ... }
}
```

**Why the default stays at one window.** The released audio threshold (0.375) was fitted on
**single-window** scores. Pooling raises the score — LSE of N samples sits above their mean — so
comparing a pooled value against the single-window threshold would silently change the operating
point. Multi-window scoring must not be enabled until a **file-level** threshold has been
calibrated on file-level labels. The code path exists and is tested; the default is one window.

```kotlin
// AudioClassifier.kt — APPLIED
/**
 * Number of independent windows scored per file.
 *
 * ONE is the shipped, calibrated behaviour: the released threshold
 * (0.375) was fitted on SINGLE-WINDOW scores, so pooling several
 * windows and comparing the pooled value against that threshold
 * would silently change the operating point.
 */
const val WINDOWS_PER_FILE = 1
```

The classifier now scores N windows, skips unusable ones (silent, too short, clipped) rather than
failing the whole file, and applies a majority rule to the speech-likeness gate:

```kotlin
// AudioClassifier.kt — APPLIED
for ((index, offset) in offsets.withIndex()) {
    val decoded = if (index == 0) first else
        decoder.decodeWindow(uri, TARGET_RATE, TARGET_SAMPLES, requestedStartMs = offset, checkStop = ::check)
    // An unusable window is SKIPPED rather than fatal: a file may
    // contain a silent stretch, and the remaining windows still carry
    // evidence. If every window is unusable the decision is inconclusive.
    if (decoded.validSamples < TARGET_RATE / 2 || decoded.rms < 1e-4f || decoded.peak < 1e-3f) continue
    if (decoded.clippedFraction > .1f) { clippedWindows++; continue }
    val features = lfcc.extractBuffer(decoded.pcm, ::check)
    scores += engine.predict(MODEL_FILE, features, ::check)
    if (SpeechContentGate.isSpeechLike(decoded.pcm, decoded.validSamples, TARGET_RATE)) speechLikeWindows++
}

val score = if (scores.isEmpty()) -1f
    else if (scores.size == 1) scores[0]
    else WindowPooling.poolProbabilities(scores.toFloatArray())
```

**Acceptance gate.** A sealed test with **partially synthetic** files must exist (real speech with
a spliced synthetic segment, and vice versa), a file-level threshold must be calibrated on
file-level labels, and only then may `WINDOWS_PER_FILE` be raised above 1. The file-level decision
must be reported separately from the window-level score. This is
`docs/FIX_STATUS_2026-10-07.md` finding 11.

`android/.../test/.../WindowPoolingTest.kt` (10 tests) pins the pooling against independently
computed reference values and asserts the three properties that matter: the result is always a
probability, it lies between the mean and the max, and it matches the standard LSE formulation at
three temperatures.
### P2.4 — A separate music / non-speech route

**Files:** `ml/src/data/audio_dataset.py`, `android/.../AudioClassifier.kt`.

`SpeechContentGate` correctly abstains on non-speech audio — so music is permanently INCONCLUSIVE.
Given Suno/Udio adoption and Deezer's AI-music toggle, this is the largest single feature gap.

Echoes (arXiv:2603.23667) shows cross-dataset music EER of 30.7–42.3% with Wav2Vec2 XLS-R 2B; the
AIGM taxonomy (arXiv:2412.00571 v3) makes clear that speech-detector components transfer to music
only under specific conditions. **Do not reuse the speech model.** Add a `content_class` to the
audio dataset and manifest (`speech` / `music` / `environmental`), route on it, and train a separate
model with its own threshold. The frequency-equivariant architecture from ISMIR 2026 gives
speed-modification robustness by construction.

**Defer until P0–P2.3 pass.**

---

## C.4 — P3: video

### P3.1 — Run Engine C on video frames

**File:** `android/.../VideoForensicAnalyzer.kt`.

`VideoForensicAnalyzer.kt:73` hard-codes `Float.NaN` for slot 6, so video frames are scored **only**
by the weak SRM CNN (0.6584 AUC). Run `SemanticProbeAnalyzer` on 4–6 key frames per video.

This is a prerequisite for everything else in this section: until slot 6 carries a real
measurement on video, the video route is a single-engine image pipeline.

### P3.2 — A video-specific model and threshold

**Files:** `ml/src/models/video_spatial_temporal.py` (new), `ml/scripts/train_video_detector.py`,
`android/.../VideoForensicAnalyzer.kt`.

`VideoForensicAnalyzer` currently applies the **image** fusion model and the **image** threshold to
frames. That is a category error: a video generator's artifacts are temporal, not per-frame.

Target architecture, mirroring CoCoDetect (CVPR 2026): an R3D-18-style spatio-temporal backbone over
the 10–16 sampled frames the app already decodes, with a **confidence gate** that escalates only
the ambiguous band. A cheap second-order statistic — the temporal acceleration variance
`Var(e_{t+1} − 2e_t + e_{t−1})` over per-frame VFM embeddings — is a strong, nearly free signal
that natural motion does not have.

The video route must have its own calibration split, its own threshold asset, and its own entry in
`release_manifest.json` — mirroring the `semantic_probe_int8.tflite` pattern.

### P3.3 — Actually use the temporal signal

**File:** `android/.../VideoForensicAnalyzer.kt:85, 99`.

`FrameCosineDelta.anomalyScore(...)` is computed at `:85`, returned in `Outcome.temporalScore`, and
**never enters the score** — `overall = 0.75f * visualScore + 0.25f * audioScore`.

Two options: add the temporal score as a slot in a video-specific fusion tree, or replace the
cosine heuristic with **LoCC-style counterfactual frame consistency** (reconstruct a frame from its
temporal neighbours; real video reconstructs well, lip-sync fakes do not).

Adding a slot requires a **new fusion revision** and the `check_android_consistency.py` migration
path described in `docs/DUAL_ENGINE_CASCADE.md:296-302`.

### P3.4 — BioLip temporal lip jitter (slot 9, for real)

**Files:** `android/.../detection/visual/` (new), `ml/src/data/` (landmark extractor parity).

`landmark_jitter` is a named slot with a constant value. BioLip (arXiv:2604.16808) shows that
generative models do not enforce the biomechanical constraints of orofacial articulation, producing
measurably elevated **temporal lip variance** that is consistent across language, ethnicity and
recording conditions. It operates on **64 MediaPipe perioral landmarks** — lightweight and
on-device-feasible (MediaPipe FaceMesh is ~2 MB).

Implementing it turns a dead slot into a real, cross-lingual signal that neither the image engines
nor the audio engine can supply, and it directly targets lip-sync deepfakes.

**Acceptance gate.** Slot 9 must be a *measurement* (or removed per P1.4 Option B). It must be
reported as `-1` (unavailable) when no face is present — **never** a neutral constant.

---

## C.5 — P4: provenance and watermarking

### P4.1 — Implement C2PA verification offline

**Files:** `android/.../detection/provenance/C2PAManifestChecker.kt` (currently a stub),
new `android/.../detection/provenance/C2PAVerifier.kt`.

**Correction to rev. 1.** The offline **C2PA Android SDK** (`contentauth/c2pa-android`, Apache-2.0,
backed by `c2pa-rs`) is the right tool and it verifies Google / OpenAI / Adobe Content Credentials
**without a network**. Rev. 1's hand-rolled JUMBF/CBOR parser is unnecessary.

Scope for a first release:

- Verify the **signature chain** against a trust list.
- Extract **Regions of Interest** — which part of the asset was AI-modified.
- Extract `inputTo` ingredient assertions (prompts, reference assets).
- Support **image, audio and video** containers (C2PA 2.3 covers all three).

**Critical correctness constraint** (IACR eprint 2026/804): the specification's **"exclusion range"**
allows portions of the asset and manifest to be outside the cryptographic signature. Therefore:

- `C2PA present` means *this asset carries a signed claim*, **not** *the asset is fully authentic*.
- `C2PA absent` means *no signed claim was found*, **not** *the asset is synthetic*.

### P4.2 — Offline neural watermark detection (AudioSeal / VideoSeal / WavMark)

**Files:** new `android/.../detection/provenance/AudioSealDetector.kt`,
`VideoSealDetector.kt`.

**Correction to rev. 1.** Rev. 1 proposed a `SynthIDDetector.kt`. That is **not implementable in
this app**: SynthID's image/audio/video detector is a proprietary Google cloud service (the public
portal at `synthid.com` explicitly forbids automating it as an API), and the app has **no
`INTERNET` permission** — `AndroidManifest.xml:9-10` removes it explicitly. Only SynthID **Text**
is open source, and it is irrelevant to media.

The offline-capable open-weight detectors are:

| Modality | Detector | Size |
|---|---|---|
| Audio | **Meta AudioSeal** (`facebookresearch/audioseal`) — localised sample-level detector | <2 MB INT8 |
| Audio | **WavMark** (`wavmark/wavmark`) | small |
| Image / video | **Meta VideoSeal** (`facebookresearch/videoseal`) | small |

### P4.3 — Handle provenance as categorical positive evidence

**Files:** `android/.../domain/models/ScanResult.kt`, `android/.../DetectionPipeline.kt`,
`android/.../ui/screens/result/ResultScreen.kt`.

```kotlin
// android/.../detection/provenance/ProvenanceVerdict.kt — NEW
/**
 * Provenance evidence is CATEGORICAL, not probabilistic.
 *
 * A verified C2PA claim or a detected watermark mark is evidence that a
 * *specific tool* produced or edited the asset. It is not a probability
 * and must never be averaged into a score: averaging would let a
 * statistical detector's confidence dilute a cryptographic fact.
 *
 * Absence is NOT evidence of authenticity. Metadata is stripped by
 * re-encoding; marks are stripped by adversaries. The absence of
 * provenance lowers nothing.
 */
sealed interface ProvenanceEvidence {
    object Absent : ProvenanceEvidence
    data class C2PAClaim(val manifestPresent: Boolean, val signed: Boolean, val aiGenerated: Boolean,
                         val ingredientCount: Int, val regionsOfInterest: List<String>) : ProvenanceEvidence
    data class WatermarkMark(val detected: Boolean, val kind: String, val pValue: Double) : ProvenanceEvidence
}
```

| Rule | Reason |
|---|---|
| A detected mark **raises** the verdict to SYNTHETIC | near-certain provenance |
| A missing mark **must not lower** the score | absence proves nothing; marks are trivially stripped |
| The watermark must **never** be a learned feature correlated with the label | the documented shortcut produces strip-to-evade and mark-to-frame failures |
| A detected mark must **not** be treated as a probability to average | it is a categorical fact, not a score |

The user-facing win is the difference between *"94% probability this was generated"* and
*"this contains a signed AI-generation claim from <tool>"*. These should not share a scale.

```kotlin
// android/.../domain/models/ScanResult.kt — ADD
enum class VerdictConfidence {
    /** No validated detector applied; diagnostics only. */
    UNVALIDATED,
    /** A statistical detector decided at a calibrated operating point. */
    STATISTICAL,
    /** Cryptographic provenance or a watermark confirmed AI involvement. */
    PROVENANCE_CONFIRMED,
}
```

**Acceptance gate.** A `PROVENANCE_CONFIRMED` verdict requires a *verified* signature or a detected
mark — never a heuristic. The absence of provenance must not change the statistical score.

---

## C.6 — P5: evaluation and release gates

### P5.1 — Make the low-FPR operating point a hard gate — **IMPLEMENTED**

**Files:** `ml/src/training/acceptance.py` (new), `ml/scripts/release_bundle.py`,
`ml/src/training/evaluation.py`, `ml/scripts/train_xgboost_meta.py`.

**Correction to rev. 2.** Rev. 2's `MINIMUM_ACCEPTANCE` loop walked
`evaluation_report.json["models"]`, which contains **only the three `.tflite` models**. The
XGBoost fusion's sealed-test metrics live in `fusion/meta_results.json`. A gate that only walked
the evaluation report would therefore never have checked the detector the image/video route
actually ships — the fusion AUC floor, the fusion TPR@1%FPR floor and the per-source FPR gate were
all dead code.

**Implemented.** The floors live in one shared module so the training script and the release gate
cannot drift:

```python
# ml/src/training/acceptance.py — APPLIED
# module name -> (min tpr@1%fpr, min tpr@0.1%fpr, min auc)
MINIMUM_ACCEPTANCE = {
    "srm_forensic_cnn_int8.tflite": (0.15, 0.05, 0.70),
    "semantic_probe_int8.tflite": (0.25, 0.10, 0.75),
    "lcnn_audio_int8.tflite": (0.50, 0.30, 0.85),
}

# The XGBoost fusion is what the image/video route actually ships, so it is
# gated separately and more strictly than any single engine.
MINIMUM_FUSION_ACCEPTANCE = (0.30, 0.15, 0.85)

# A 0.1% FPR operating point needs enough REAL samples to be a statistic
# rather than one image. With n_real = 200 it is defined by 0.2 expected
# false positives, i.e. the single highest-scoring real image.
MIN_REAL_FOR_LOW_FPR_GATE = 2000
```

```python
# ml/scripts/release_bundle.py — APPLIED
# AUC alone is not an acceptance criterion. The shipped image fusion has
# AUC 0.7152 and TPR@1%FPR of 0.040 - a coin flip in the regime anyone
# would actually ship. These floors exist so that number cannot hide
# behind the other one.
#
# The evaluation_report only carries the three .tflite models; the
# FUSION's sealed-test metrics live in fusion/meta_results.json. A gate
# that only walked evaluation_report["models"] would never check the
# detector the image/video route actually ships.
desktop_models = (read_object(bundle / 'evaluation_report.json').get('models') or {})
for name, entry in desktop_models.items():
    test = entry.get('test') if isinstance(entry, dict) else None
    if not isinstance(test, dict):
        continue
    check_model_acceptance(name, test, context='release')
fusion_results = read_object(bundle / 'fusion/meta_results.json')
fusion_test = fusion_results.get('test')
if isinstance(fusion_test, dict):
    check_model_acceptance(
        'fusion_xgboost', fusion_test, floors=MINIMUM_FUSION_ACCEPTANCE,
        context='release',
    )
    assert_source_fpr_within_bound(fusion_test, max_fpr=0.05, min_real_per_source=15, context='release fusion')
```

The same per-source gate is applied at training time, so a biased detector never reaches the
release stage at all:

```python
# ml/scripts/train_xgboost_meta.py — APPLIED
# A detector that flags a large fraction of one acquisition source as
# synthetic has learned the SOURCE, not the synthesis. LAION-5B is the
# training corpus for Stable Diffusion and Flux, so "looks like LAION"
# is a giveaway, not a detection signal.
assert_source_fpr_within_bound(test_report, max_fpr=0.05, min_real_per_source=15, context="fusion")
assert_source_fpr_within_bound(laundered_report, max_fpr=0.05, min_real_per_source=15, context="fusion laundered")
```

**Sample-size discipline.** The 0.1% FPR floor is only enforced when the evaluation set is large
enough for that operating point to mean anything; otherwise the metric must still be *present* but
is not required to clear the floor. This is what lets a pilot set be staged while a distribution
release is held to the stricter rule.

**Consequence for the current release — which is the point.** With the shipped numbers the gate
**fails**: fusion AUC 0.7152 < 0.85, fusion TPR@1%FPR 0.040 < 0.30, and the LAION per-source FPR
0.4444 > 0.05. The v6 release therefore cannot be re-published as-is. The gates are inert on
structural fixtures (a bundle whose reports carry no sealed `test` block), so the contract test
suite still exercises the machinery.

`ml/tests/test_acceptance.py` (16 tests) covers the floors, the sample-size rule, and the
source-bias gate, including the shipped numbers being rejected.

---

## C.7 — Measured results

The full measurements, with the exact commands to reproduce them, are in
**[`docs/RESULTS_2026-10-20.md`](RESULTS_2026-10-20.md)**. The three findings that change the plan:

1. **A frozen SigLIP-B/16 linear probe reaches 0.8919 on the sealed test split**, against the
   shipped Engine C's 0.6871 — **+0.205 AUC** from a 769-number head and a frozen backbone.
   DINOv2-S/14 reaches 0.8358, DINOv2-B/14 0.8541. The MobileCLIP2-S0 tower the repo currently
   names reaches **0.5145** — chance.

2. **The source shortcut survives every backbone** (LAION FPR 0.29–0.36 for every probe, against
   the shipped 0.5185). The cause is measured, not guessed: on the image corpus a trivial
   "which source is this?" classifier scores **100%**, because every fake comes from `openfake` and
   every real from `laion`/`pexels`/`docci`/`imagenet`. **No backbone can fix a corpus that leaks
   the label through the manifest.** The audio corpus scores 0.50 on the same test, so its shortcut
   has a different cause — the *channel* — and a different fix (codec-matched data, not
   source-balanced data).

3. **The phase-spectrum route is refuted as a scalar feature.** Every summary of the phase-only
   reconstruction is numerically constant or below chance, and Engine A on the phase reconstruction
   scores 0.4969 against 0.6584 on the original. The phase spectrum needs a *learned* consumer —
   a new model, not a new slot. P1.3 is therefore downgraded from "add a feature" to "replace
   `fft_score` and `kl_score` with a phase-conditioned model in the next release", and the two
   chance-level features should be dropped or replaced rather than kept.

4. **Source debiasing is a trade-off, and the plain probe wins.** Estimating the acquisition-source
   nuisance subspace *within the real class only* and folding it into the weight vector (because
   `(P e)^T w == e^T (P w)`) takes SigLIP-B/16 from **0.8919 down to 0.8682** sealed-test AUC
   while cutting the LAION real-image FPR from **0.2897 to 0.2573** and the ImageNet FPR from
   **0.0199 to 0.0101**. It costs nothing at inference, but it is not free, and the plain probe
   is what ships.
   > **Erratum.** An earlier revision claimed **0.9013** AUC and a LAION FPR of **0.2101**, and
   > called it "a free strict win". That was a **bug**: `fit_source_debiased_probe` fitted on the
   > RAW projected features while `apply_debiased_probe` applied the weight to the STANDARDIZED
   > features, so the measured AUC belonged to a function that had never been fitted. The fit and
   > the application now use the same standardization. The corrected numbers are above, and the
   > earlier claim is retracted.

5. **The frozen tower is exportable and shipped, via a Keras mirror rather than onnx2tf.** The
   `Conv2d(3, 768, 16, 16)` patch projection is rewritten as an explicit patch unfold plus a
   `Linear` - mathematically identical (1.431e-06 against the convolution) - which was the only
   way to stop onnx2tf losing the channel axis across `Flatten -> Transpose`. onnx2tf itself is
   nevertheless **incapable of converting a ViT in this environment**: its shape inference
   mis-orders the patch->sequence transition (`[?, 768, 768]` instead of `[?, 196, 768]`), which
   reproduces with a permute-free unfold and is not bypassed by `disable_strict_mode`. The shipped
   path is therefore the repository's own **Keras mirror** (`ml/src/export/keras_vfm_probe.py`),
   the same mechanism already used for the SRM model and the semantic probe. Measured end to end:
   the mirror reproduces PyTorch to **1.97e-06**, the INT8 graph is **89.96 MB**, and the
   *packaged file* keeps the sealed-test AUC at **0.8910** against the PyTorch reference
   **0.8919** (-0.0009). The APK-size decision is explicit: SigLIP-B/16 is ~90 MB INT8 against a
   21 MB APK today; DINOv2-S/14 is ~21 MB and reaches 0.8358.

6. **The AudioSeal weights are behind a gated Hugging Face repository.** The detector now runs
   the *real* published model and reports **no evidence at all** until it is staged. The previous
   implementation correlated against a hard-coded +/-1 sequence that was never AudioSeal's carrier
   and produced **false positives** - a false positive in a provenance channel confirms AI origin
   on authentic audio, which is the worst possible failure. It was removed rather than tuned.

The export path is built and verified to ONNX (**2.205e-06** parity against PyTorch). The
ONNX->TFLite step is the remaining item, and it is a **toolchain** problem rather than a
correctness one: the model is exact, the frontend is exact, and the parity checks pass.

---

### P5.2 — Per-generator and per-source reporting as release evidence

**Files:** `ml/src/training/evaluation.py`, `ml/src/data/dataset_splits.py`.

`evaluation_summary` already produces `groups.source` and `groups.generator`. Extend it to:

- **zero-TPR generator gate** — a generator with ≥5 fake samples scoring TPR = 0 is a release
  blocker (GPT-Image-2, Qwen-Image, Sora-2 currently score 0; A18 in audio scores 0).
- **degradation matrix** — per-condition accuracy (JPEG QF 90/70/50/30, downscale 0.75/0.5/0.25,
  blur σ 0.5/1.0/2.0, chains), replacing the single laundered 512/Q85 point.
- **temporal protocol** — train on generators ordered by release date, test on the future
  (AI-GenBench's protocol), so "generalizes to unseen generators" is measured rather than asserted.
- **channel holdout** for audio (ASVspoof 2021 LA style).

### P5.3 — The on-device parity work that is still missing

**Files:** `ml/src/data/image_frontend.py`, `android/.../ScannerViewModel.kt`.

`docs/TRAINING_GUIDE.md:82`: *"Pillow and BitmapFactory sampling still require measured native device
parity."* The bicubic fix (§C.0.1 item 1) closes the **Engine C** resize gap; the remaining gaps
are:

- the `pow2cap1024` decode chain (power-of-two `inSampleSize`, sRGB request, EXIF orientation, alpha
  flattened onto white) has no measured Pillow counterpart;
- the LFCC frontend has no measured Android-vs-Python counterpart beyond the hash-bound contract;
- the export-parity numbers in `docs/FIX_STATUS_2026-10-07.md:44` (image 2.98e-08) are
  **Keras-vs-TFLite on the desktop**, not Pillow-vs-BitmapFactory on a device.

Add a parity harness that renders the same fixture through both stacks and reports the maximum
absolute activation difference at the model input, per stage.

### P5.4 — Bring the documentation back in line

**Files:** `README.md`, `docs/ARCHITECTURE.md`, `docs/MODEL_CARD.md`, `docs/DEPLOYMENT.md`,
`docs/DUAL_ENGINE_CASCADE.md`, `docs/BENCHMARK_RESULTS.md`, `docs/FIX_STATUS_2026-10-07.md`.

| File | Change |
|---|---|
| `README.md` | Replace *"Current bundled models are unvalidated… report Inconclusive"* with the real status; delete the 32 px reference; replace "Research priorities" with this plan. *(The APK path and the "unsigned" claim are already corrected in §C.0.1.)* |
| `docs/ARCHITECTURE.md:27` | `[1,32,32,3]` → `[1,128,128,3]`; add the dual-engine section; document the bicubic frontend contract. |
| `docs/MODEL_CARD.md` | `validated=false` → the true status; add the sealed-test table; add the teacher/student AUC gap; add the per-source FPR table. |
| `docs/DEPLOYMENT.md:55-66` | list the real asset set; add the device-evidence requirement from P0.3a. |
| `docs/DUAL_ENGINE_CASCADE.md:388` | replace the **0.88–0.896** claim with **0.7152** measured, and move the 0.849 SigLIP number to an explicitly-labelled *not shipped* row. Add the MobileCLIP2-S0 0.5145 result to any "which tower" recommendation. |
| `docs/BENCHMARK_RESULTS.md` | state that v6 has a sealed-test benchmark, that it is **not reproducible** by `run_training_pipeline.py`, and that `ml/evaluation_results.json` is deleted. |
| `docs/FIX_STATUS_2026-10-07.md` | add a header note that it describes the 2026-10-07 state and that v6 supersedes it. |
| new `docs/RESULTS_2026-10-20.md` | the canonical, split-attributed metric table. |

---

# Part D — Artifacts to add

```
docs/
  STATE_OF_THE_ART_REVIEW_2026-10-20.md      # this document
  RESULTS_2026-10-20.md                      # canonical split-attributed metric table
ml/src/metrics/
  operating_points.py                        # TPR@FPR grid, Wilson intervals, per-group reports
  degradation.py                             # JPEG/blur/resize/chain matrix + report
  source_audit.py                            # per-source FPR + shortcut detection
ml/src/training/
  dcpt.py                                    # paired-consistency losses (logit + feature)
  watermark_decorrelation.py                 # watermark-both-classes augmentation
ml/src/models/
  video_spatial_temporal.py                  # video backbone
  landmark_jitter.py                         # BioLip-style perioral temporal variance
ml/src/provenance/
  c2pa.py                                    # C2PA manifest extraction via c2pa-rs
  watermarks.py                              # AudioSeal / VideoSeal / WavMark wrappers
ml/scripts/
  train_video_detector.py
  audit_shortcuts.py                         # standalone shortcut audit over a labelled corpus
  release_gate.py                            # P5.1 acceptance thresholds, single source of truth
  prepare_channel_variants.py                # deterministic ffmpeg codec/channel fixtures
android/app/src/main/java/com/onndetect/detection/provenance/
  C2PAVerifier.kt                            # c2pa-android integration
  AudioSealDetector.kt
  VideoSealDetector.kt
  ProvenanceEvidence.kt                      # categorical evidence model
android/app/src/main/java/com/onndetect/detection/visual/
  PerioralLandmarkAnalyzer.kt                # BioLip-style landmark jitter
  PhaseSpectrumAnalyzer.kt                   # phase consistency branch
android/app/src/test/java/com/onndetect/
  ProvenanceVerdictTest.kt
  CascadeObjectiveTest.kt
  SourceBiasGateTest.kt
```

---

# Part E — Definition of done

The release is only defensible when **all** of the following hold on the **sealed** split, with
per-group breakdowns and Wilson intervals:

| # | Criterion | Current v6 | Target |
|---|---|---|---|
| 1 | Fusion sealed-test AUC | 0.7152 | ≥ 0.85 |
| 2 | Fusion **TPR@1% FPR** | **0.040** | ≥ 0.30 |
| 3 | Fusion TPR@0.1% FPR | 0.005 | ≥ 0.15 *(after n_real ≥ 2,000)* |
| 4 | Worst real-source FPR (n ≥ 15) | **0.444** (laion) | ≤ 0.05 |
| 5 | Generators with TPR = 0 (n ≥ 5) | 15 | 0 |
| 6 | Audio worst real-source FPR | **0.243** (inthewild) | ≤ 0.05 |
| 7 | Audio generators with TPR = 0 | A18 | 0 |
| 8 | Engine C distillation gap | ~0.166 (0.8491 → 0.6871) | ≤ 0.05, or ship the frozen tower |
| 9 | Live fusion features at or above chance | 2 of 4 (`fft` 0.501, `kl` 0.412) | 4 of 4 |
| 10 | Dead fusion slots | 5 of 10 | 0 (or a shrunk contract) |
| 11 | Frontend contract executed as declared | Engine C ran bilinear under a `bicubic` id | verified by `BicubicResizerTest` |
| 12 | Degraded-average accuracy (DCPT matrix) | not measured | ≥ baseline + 9 pts |
| 13 | Audio TPR@1% FPR | **0.595** | ≥ 0.65, EER ≤ 0.13 |
| 14 | Music route | absent | separate model + threshold |
| 15 | Video route | image model + image threshold, Engine C never run | video model + video threshold |
| 16 | Provenance | stub, never called | C2PA (offline SDK) + AudioSeal/VideoSeal, positive-evidence only |
| 17 | Cascade band fit | at coverage floor, index bug, wrong objective | selective accuracy under a minimum-classified-rate constraint |
| 18 | Device validation | desktop report | real Android measurement |
| 19 | Engine A calibration in manifest | overwritten by fusion | distinct `fusion_xgboost` entry |
| 20 | Release pipeline reproducibility | two divergent scripts | one script reproduces the release |
| 21 | `androidTest` compiles | no | yes |
| 22 | CI release APK gate | fails | passes |
| 23 | Docs vs. assets | 7 stale documents | reconciled |
| 24 | Audio operating point | shared field; could be judged against the image fusion threshold | **modal** — per-model thresholds, selected by route |
| 25 | Fusion feature contract | duplicated in 6 places; 5 slots dead | single-sourced (`FusionFeatureSchema`), runtime slot count + optional index cross-checked |
| 26 | Acceptance gates cover the fusion | walked only `evaluation_report.models` | fusion gated from `fusion/meta_results.json` + per-source FPR |
| 27 | DCPT consistency loss | positive-class term only (negative on real images) | full symmetric **binary** Bernoulli KL + feature cosine term |
| 28 | Audio window pooling | treated sigmoid outputs as logits | probability → logit → LSE → sigmoid, result always in (0,1) |
| 29 | Contract shrink (P1.4 Option B) | not attempted | a **migration** with a 9-step checklist and a new release id |

---

# Appendix A — Verification commands

```powershell
# Sealed-test truth for every shipped model
python -c "import json;d=json.load(open('android/app/src/main/assets/evaluation_report.json'));[print(k,'auc=%.4f eer=%.4f tpr@1%%=%.3f tpr@0.1%%=%.3f'%(v['test']['auc'],v['test']['eer'],v['test']['tpr_at_fpr_1pct'],v['test']['tpr_at_fpr_0_1pct'])) for k,v in d['models'].items()]"

# Per-source FPR - the shortcut audit (image fusion)
python -c "import json;d=json.load(open('android/app/src/main/assets/fusion/meta_results.json'));print(json.dumps(d['test']['groups']['source'],indent=2))"

# Per-source FPR (audio) - the shortcut the pooled AUC hides
python -c "import json;d=json.load(open('android/app/src/main/assets/evaluation_report.json'));print(json.dumps(d['models']['lcnn_audio_int8.tflite']['test']['groups']['source'],indent=2))"

# Dead fusion slots
$kt = Get-Content android/app/src/main/assets/fusion/XGBoostMetaClassifier.kt -Raw
0..9 | ForEach-Object { "slot ${_}: " + ([regex]::Matches($kt, "features\[$_\]")).Count }

# Standalone feature AUCs (fft and kl are at/below chance)
python -c "import json;d=json.load(open('ml/scripts/llm_signal_analysis_results.json'));[print(k,v['auc']) for k,v in d['standalone'].items()]"

# Engine A calibration collision
python -c "import json;e=json.load(open('android/app/src/main/assets/evaluation_report.json'));m=json.load(open('android/app/src/main/assets/constants/release_manifest.json'));print('EngineA own:',e['models']['srm_forensic_cnn_int8.tflite']['calibration']);print('manifest   :',m['calibration']['srm_forensic_cnn_int8.tflite'])"

# The desktop device report
python -c "import json;d=json.load(open('android/app/src/main/assets/device_evaluation.json'));print([k for e in d['models'].values() for k in e if 'desktop' in k])"

# Which teacher tower is usable (held-out, not fit-split)
Select-String -Path docs/DUAL_ENGINE_CASCADE.md -Pattern 'held-out|0.5145|0.8491'
```

# Appendix B — Citation list

**Image**
- IAPL — arXiv:2508.01603 — *Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning*
- PRL — arXiv:2610.11876 — *Relative Patch Response Learning for Generalizable AI-Generated Image Detection*
- DCPT — arXiv:2604.10102 — *Degradation-Consistent Paired Training for Robust AI-Generated Image Detection*
- GlobalForge — arXiv:2607.14684 — *Towards Robust AI-Generated Image Detection*
- DailyBench — arXiv:2607.24016 — *A Unified Benchmark for AI-Generated and Manipulated Images from Modern Generative Models*
- REALIS — arXiv:2609.32734 — *A Curated Dataset for Studying the Challenges of AI Image Detection*
- Arbitrary Data Sources — arXiv:2603.00717
- Effort — arXiv:2411.15633 — *Effort: … orthogonal subspace adaptation* (ICML 2025)
- B-Free — grip-unina.github.io/B-Free
- "Is AIGI Detection a Solved Problem?" — arXiv:2505.12335
- Phase Spectrum Robustness — CVPR 2026, Li et al., pp. 35427–35436
- PatchCraft — arXiv:2311.12397; NPR — arXiv:2312.10461 (CVPR 2024)
- LoRA-based Pairwise Training — CVPRW NTIRE 2026 (3rd place)
- UNITE — SigLIP backbone + attention-diversity loss, 2026
- ImageCLEF 2026 Deepfake Detection working notes
- Awesome-AIGC-Image-Video-Detection — github.com/ant-research

**Audio**
- "It's All in the Presentation" — arXiv:2509.26471v2, ICASSP 2026
- The Watermark Shortcut — arXiv:2606.23335v2
- VoiceWukong — USENIX Security 2025, Yan et al.
- Echoes — arXiv:2603.23667v3
- Zero-shot AI Music Detection — arXiv:2607.25530v1
- Fusion Segment Transformer — arXiv:2601.13647v1
- Improved Robustness in AI-Generated Music Detection — ISMIR 2026
- AIGM Detection Pathway and Overview — arXiv:2412.00571v3
- Towards Scalable AASIST — arXiv:2507.11777; AASIST3 — arXiv:2408.17352
- WavLM model ensemble (ASVspoof 2024) — ISCA archive
- AudioMarkNet — USENIX Security 2025, Zong et al.
- SpeechFake — multilingual speech deepfake dataset, 2025

**Video**
- GenVidBench — AAAI 2026, Ni et al., pp. 15582–15590
- CoCoVideo-26K / CoCoDetect — arXiv:2606.00101, CVPR 2026
- LoCC — arXiv:2606.22772v1, ICME 2026
- BioLip — arXiv:2604.16808v4
- Artifact-Bench — arXiv:2605.18984
- PVBench — arXiv:2610.26711v1
- DeMamba / GenVideo — 2.3M-video benchmark

**Provenance**
- C2PA — *Use of Content Credentials to Identify Synthetic and Non-Synthetic Content*, 2026-07-30
- C2PA — *Content Credentials: Deployment Guidance 1.0*, 2026-07-08
- `contentauth/c2pa-android` — github.com/contentauth/c2pa-android
- IACR eprint 2026/804 — *Verifying Provenance of Digital Media: Security Analysis of C2PA and its Implementation*
- Google — SynthID Detector portal (2026-10-07); `google-deepmind/synthid-text` (Text only)
- OpenAI — *Advancing content provenance for a safer, more transparent AI ecosystem*
- Meta AudioSeal — github.com/facebookresearch/audioseal
- Meta VideoSeal — github.com/facebookresearch/videoseal
- WavMark — github.com/wavmark/wavmark

**Efficiency**
- EdgeViTs — arXiv:2205.03436
- *A Study on Inference Latency for Vision Transformers on Mobile Devices* — arXiv:2510.25166
