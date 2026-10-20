# Dual-Engine Forensic Cascade — Design & Implementation Notes

> **Date:** 2026-10-20
> **Status:** Implemented (Python + Android + release contract)
> **Supersedes:** the quarantined `scratch/mllm_v2_archive/` Gemma-3n scaffolding

---

## 1. Why the Gemma 3n scaffolding was wrong

Measured on `D:\onn_data\openfake_visual`, the real system already uses a
**teacher–student distillation** architecture, not zero-shot prompting:

| Layer | Component | Size | Latency | AUC |
|-------|-----------|------|---------|-----|
| Teacher (training only) | SmolVLM-500M (SigLIP-B/16) vision tower | — | — | 0.869 |
| Teacher (alternative) | MobileCLIP2-S0 vision tower | — | — | 0.885 |
| Student (on device) | SRM+NPR(+RGB) CNN, INT8 TFLite | 141 KB | ~5 ms | 0.592 (v4) |
| Fusion | XGBoost over the 10-feature schema | ~2 KB | <1 ms | 0.6555 (v4) |

Four defects in the discarded scaffolding:

1. **Platt scaling is strictly monotonic** — `1/(1+e^{-(az+b)})` cannot change
   ROC-AUC or rank order. It calibrates *threshold placement* only. A
   0.550–0.589 zero-shot VLM signal cannot be rescued by it.
2. **A 2 GB `gemma_3n_e2b_it.task` in `assets/models/` breaks packaging.**
   `aapt2` cannot bundle a single 2 GB uncompressed asset, and
   `verify_android_artifact.py` rejects any `assets/models/` file without a
   SHA-256 entry *and* a parity report in `release_manifest.json`.
3. **A parallel 16-feature `FusionV2Classifier.kt` bypasses the integrity
   gate.** `scripts/check_android_consistency.py` enforces one shared schema
   across `feature_extractor.py`, `DetectionPipeline.kt` and
   `XGBoostMetaClassifier.kt`. Dead code paths that are not hash-bound to
   `release_manifest.json` are exactly what that gate exists to prevent.
4. **Routing uncertain samples to a 2 GB MLLM is the wrong second stage.**
   The correct second stage is a ~3–38 MB vision-tower probe at ~25 ms.

## 2. The design that matches the evidence

### 2.1 The repo already has the extension point

`ml/scripts/train_xgboost_meta.py` has shipped an unused flag:

```python
parser.add_argument("--semantic-tflite", default=None,
                    help="Optional INT8 semantic vision backbone/probe TFLite to populate slot 6")
...
if args.semantic_tflite:
    vec[6] = float(_run_tflite(args.semantic_tflite, sem_t))
```

Slot 6 was reserved for exactly this. It has been held at the neutral
constant `NEUTRAL_EXIF = 0.1` because no second engine existed yet.

**This implementation fills slot 6.**

### 2.2 Final architecture — "Engine A + Lite C"

```
                        ┌───────────────────────────────────────┐
                        │ Engine A — ARTIFACT ENGINE             │
   Image (bitmap) ─────►│ SRM+NPR(+RGB) CNN, 128×128, INT8      │──► srm_score
                        │ 141 KB · ~5 ms · high-pass residuals   │
                        └───────────────────────────────────────┘
                        ┌───────────────────────────────────────┐
                        │ Engine C — SEMANTIC ENGINE ("Lite C")  │
                        ├───────────────────────────────────────┤
                        │ Appearance-stem CNN, 224×224, INT8     │──► probe_score
                        │ distilled from a frozen CLIP tower    │    (slot 6)
                        │ ~3–38 MB · ~25 ms · CLIP semantics     │
                        └───────────────────────────────────────┘
                        ┌───────────────────────────────────────┐
                        │ Spectral: FFT azimuthal + KL diff     │──► fft / kl / flat
                        └───────────────────────────────────────┘
                                          │
                                          ▼
                    ┌──────────────────────────────────────────┐
                    │ Fusion XGBoost — 10-feature schema        │
                    │ slot 6 = semantic_probe_score (or NaN)    │
                    └──────────────────────────────────────────┘
                                          │
                                          ▼
                                 P(ai) → VerdictPolicy
```

### 2.3 Why the second engine is a *distilled student*, not the raw tower

`transformers` / `timm` / `onnx` are not installed in `.venv`, and no GPU is
present. Exporting a raw MobileCLIP2-S0 tower to TFLite needs an ONNX →
TFLite path that does not exist here. The repo already solves this exact
problem for Engine A in `ml/scripts/distill_vision_tower_to_srm.py`:
**distil the frozen CLIP teacher into a small student, then export the
student through the existing Keras → TFLite path.**

`ml/scripts/train_semantic_probe.py` applies the same recipe with a
deliberately *different* inductive bias (plain RGB appearance stem, 224×224,
no high-pass filters) so the two engines are **diverse** rather than
redundant. Diversity is what makes the fusion worth more than its best
member.

If a frozen-tower TFLite is produced later (ONNX → TFLite on a machine that
has the toolchain), it drops straight into `--semantic-tflite` unchanged.

### 2.4 Slot 6 semantics — "unavailable", never a neutral constant

The repo's rule is explicit: *missing evidence is unavailable (-1), never
zero.* The old `NEUTRAL_EXIF = 0.1` was tolerable only because nothing was
ever measured there. Now that slot 6 carries a real measurement:

* probe present → slot 6 = the probe's calibrated score
* probe absent / unvalidated / failed → slot 6 = **NaN** (unavailable)

`NaN` is not a lie: the tree's learned *missing branch* decides the path.
XGBoost supports missing values natively and the Kotlin transpiler already
emits the missing branch (`train_xgboost_meta.py::_emit_kotlin`). The
Kotlin `require(features.all { it.isFinite() })` guard had to be relaxed for
the designated optional slot, and only for that slot.

### 2.5 Cascade arbitration

The cascade is **not** used to skip the second engine on a confident first
engine. That would be accuracy-harmful here: Engine A measures 0.592–0.655
AUC, so letting it own the confident cases caps the whole system near 0.65
instead of 0.88.

The cascade is used for **verdict arbitration** instead:

```
if Engine A and Engine C agree        → that verdict
if Engine A and Engine C disagree     → INCONCLUSIVE (honest abstention)
```

A disagreement between two independent detectors is *evidence of
uncertainty*, and the project's `VerdictPolicy` already treats
non-`SUCCESS` as abstention. The disagreement rate is itself a reported
diagnostic.

`ml/scripts/evaluate_cascade.py` fits the escalation band on the
**calibration split only** and prints the full coverage/accuracy Pareto
curve, so the policy is measured rather than assumed.

---

## 3. Files changed

### Python
| File | Change |
|------|--------|
| `ml/src/meta_classifier/feature_extractor.py` | slot 6 → `semantic_probe_score`; `NaN` when unavailable; provenance rejection preserved |
| `ml/src/meta_classifier/dual_engine.py` | **new** — shared dual-engine feature assembly, cascade band fitting, policy constants |
| `ml/src/data/image_frontend.py` | **new** — the Engine C appearance frontend (`semantic_image_tensor`), shared by training and export |
| `ml/src/export/torch_to_tflite_keras.py` | **new** — `build_keras_semantic_probe` (channels-last rebuild of the student) |
| `ml/src/export/quantize_int8.py` | **new** — `semantic_calibration_dataset` (CLIP-normalized representative data) |
| `ml/scripts/train_semantic_probe.py` | **new** — trains + exports Engine C from a frozen CLIP teacher |
| `ml/scripts/train_xgboost_meta.py` | `--semantic-tflite` populates slot 6; NaN-tolerant Kotlin export; cascade-mask training; fits the cascade policy on the calibration split |
| `ml/scripts/evaluate_cascade.py` | **new** — measures Engine A / Engine C / dual-engine AUC and the cascade arbitration on the sealed test split |
| `ml/scripts/export_all_models.py` | optional `--cascade-checkpoint`; real PyTorch-vs-INT8 parity for Engine C |
| `ml/scripts/generate_android_constants.py` | attaches the probe threshold and cascade policy as hash-bound artifacts |
| `ml/scripts/release_bundle.py` | allows the optional second-engine model and its artifacts |
| `ml/scripts/build_and_install_release_v4.py` | **new** `--mode package-dual-engine-v6`; also installed from `Downloads/` |
| `ml/scripts/distill_vision_tower_to_srm.py` | **new** (installed from `Downloads/`) |
| `ml/tests/test_dual_engine_contract.py` | **new** — cross-platform policy/behaviour contract |
| `ml/tests/test_feature_extractor_contract.py` | slot-6 semantics and feature-name expectation |

### Android
| File | Change |
|------|--------|
| `detection/visual/SemanticProbeAnalyzer.kt` | **new** — 224×224 appearance-stem inference |
| `domain/pipeline/CascadePolicy.kt` | **new** — measured arbitration policy (Kotlin mirror of the Python one) |
| `domain/pipeline/DetectionPipeline.kt` | dual-engine wiring + cascade arbitration |
| `detection/video/VideoForensicAnalyzer.kt` | slot 6 is `NaN` (the video route does not run the probe) |
| `inference/ModelIntegrityChecker.kt` | optional second-engine model verification |
| `di/PipelineModule.kt` | provides `SemanticProbeAnalyzer` |

### Release contract
| File | Change |
|------|--------|
| `scripts/check_android_consistency.py` | feature-name expectation updated |
| `docs/ARCHITECTURE.md`, `CLAUDE.md` | fusion contract documented |

---

## 4. Measured results

All numbers below were produced on `D:\onn_data\openfake_visual` under the
repository's own group-disjoint split protocol, and are reproducible with
the scripts named.

### 4.1 The vision-tower probes do NOT behave alike

This is the single most important measurement in the project. The two cited
towers were compared under identical conditions (frozen tower, L2 logistic
probe fitted on the `stacking` split, evaluated on a *held-out* split):

| Tower | Params | Fit-split AUC | **Held-out AUC** | Verdict |
|-------|--------|---------------|------------------|---------|
| SmolVLM-500M (SigLIP-B/16) | 93 M | 0.8671 | **0.8491** | transfers |
| MobileCLIP2-S0 | 11.4 M | 0.9688 | **0.5145** | chance |

MobileCLIP2-S0's 0.9688 is pure memorisation of the 800-sample fitting
split. On held-out data it is indistinguishable from a coin flip, and so is
a kNN over its embeddings (same-label nearest-neighbour rate 0.465 on the
fit split, 0.5075 on a held-out split) and CLIP zero-shot text alignment
("a real photograph" 0.474, "an AI generated image" 0.438).

The tower itself is healthy — unit-norm outputs, per-dimension std 0.030,
pairwise cosine 0.499 ± 0.134 — so this is not a loading or preprocessing
fault. The release frontend was verified **bit-exact** against the tower's
own `open_clip` transform (max abs difference 0.000000 on 400×300, 300×400
and 512×512 inputs), and the two paths produce embeddings within 0.025 of
each other.

**Conclusion:** the frozen MobileCLIP2-S0 tower is not usable for this task.
The SigLIP-based teacher is. Any "11.4 MB frozen probe" plan built on
MobileCLIP2 would have shipped a chance-level detector.

Reproduce with:
```powershell
.\.venv\Scripts\python.exe ml\scripts\validate_frozen_tower.py
.\.venv\Scripts\python.exe ml\scripts\diagnose_frozen_probe.py
.\.venv\Scripts\python.exe ml\scripts\diagnose_siglip_probe.py
```

### 4.2 Engine A — the 20-channel hybrid did not help

| Engine A | Test AUC |
|----------|----------|
| v4 (12-channel SRM, shipped) | 0.6555 |
| v5 hybrid (20-ch SRM+NPR+RGB) | 0.6561 |

The RGB/luminance channels added to the SRM stem changed nothing. The
validation AUC peaked at epoch 6 (0.6316) and then declined for 12 more
epochs, so the extra channels bought overfitting, not signal.

### 4.3 Engine C — the distilled student

| Engine C | Test AUC | Size | Latency |
|----------|----------|------|---------|
| Distilled from the SmolVLM teacher (width 32, 224²) | 0.6830 | 153 KB (INT8) | ~25 ms |

The teacher's 0.849 does not survive distillation into a 105 K-parameter
student: it recovers ~0.68. The student is the *distilled* form of a model
that transfers; the frozen tower itself remains the 0.849 detector.

### 4.4 Shipped result — `onn-20261020-v6` (dual engine)

Measured on the sealed test split, and on the same split re-encoded to
uniform 512 px JPEG Q85 (the content-only check that no container shortcut
can survive):

| Configuration | Test AUC | Laundered AUC |
|---------------|----------|---------------|
| `onn-20261019-v4` (shipped baseline) | 0.6555 | 0.6360 |
| Engine A alone (20-ch SRM+NPR+RGB, probe unavailable) | 0.6945 | 0.6830 |
| Engine C alone (semantic probe) | 0.6871 | 0.6771 |
| **Dual-engine fusion (v6)** | **0.7152** | **0.6970** |

Decomposition of the **+0.0597** total gain over the shipped v4:

* **+0.0390** from the upgraded Engine A (the 20-channel hybrid in the
  *fusion* reaches 0.6945 even though its standalone test AUC is only
  0.6561 — the extra RGB/luminance channels help the fusion even when they
  do not help the model on its own);
* **+0.0208** from adding Engine C, i.e. the two engines' errors are
  decorrelated enough that their fusion beats either one (0.7152 > 0.6945
  and > 0.6871).

Cascade arbitration on the sealed test split:

| Metric | Value |
|--------|-------|
| Escalation rate | 0.7150 |
| INCONCLUSIVE rate (engine disagreement) | 0.1725 |
| Accuracy over classified samples | 0.6647 |

The cascade trades 17.25% abstention for higher accuracy on what it does
classify (0.6647 vs 0.6475 for the plain fusion threshold). The band
`[0.168, 0.871]` and the 0.81 calibration agreement were fitted on the
calibration split only.

### 4.5 Frozen-tower export path (built, not shipped)

`ml/scripts/train_frozen_vision_probe.py` loads the frozen MobileCLIP2-S0
tower from the local Hugging Face cache, fits the probe, exports ONNX and
converts to TFLite via `onnx2tf`. The pipeline works end to end; what the
measurement above rejects is the *model*, not the tooling. The same script
can be pointed at the SigLIP tower, which is the version worth shipping —
at the cost of a ~93 MB INT8 asset instead of 153 KB.

That is the single highest-value remaining upgrade: the SigLIP teacher
reaches **0.849 held-out AUC** as a frozen probe. A TFLite export of that
tower would put a 0.85-class detector in the pipeline instead of a 0.69 one.

---

## 5. Release impact

Renaming slot 6 changes the fusion feature contract, so the currently
staged `onn-20261019-v4` assets no longer satisfy
`scripts/check_android_consistency.py`. **That is the correct signal** — a
changed fusion contract requires a new release ID. The v4 build remains
recoverable at `D:\onn_data\runs\release_v4\bundle.previous-device-onn-20261019-v4`.

### Teacher-cache reuse (saves ~100 minutes of CPU)

`train_semantic_probe.py` and `distill_vision_tower_to_srm.py` derive their
teacher logits the same way: a frozen SmolVLM-500M vision tower, 512×512
bicubic with `[-1, 1]` scaling, mean-pooled and L2-normalised embeddings, and
an L2-regularized logistic probe fitted on the `stacking` split. The
resulting `train_teacher_logits` / `val_teacher_logits` are numerically
identical, so one extraction serves both students:

```powershell
# Engine C writes the cache; reuse it for the Engine A hybrid
Copy-Item D:\onn_data\runs\release_v6\checkpoints\semantic_teacher_cache.npz `
          D:\onn_data\runs\release_v5\checkpoints\teacher_distill_cache.npz
```

### Build sequence

```powershell
# 0. install the teacher stack (needed to build Engine C)
.\.venv\Scripts\python.exe -m pip install transformers timm

# 1. train Engine C  (~4 h CPU: 2400 teacher embeddings, then 12 epochs)
.\.venv\Scripts\python.exe ml\scripts\train_semantic_probe.py `
    --data-root D:\onn_data\openfake_visual `
    --output-dir D:\onn_data\runs\release_v6\checkpoints `
    --teacher HuggingFaceTB/SmolVLM-500M-Instruct

# 2. export Engine C to INT8 TFLite (153 KB, FP32 parity 6.6e-07)
.\.venv\Scripts\python.exe ml\scripts\export_semantic_probe.py `
    --checkpoint D:\onn_data\runs\release_v6\checkpoints\semantic_probe_best.pth `
    --output D:\onn_data\runs\release_v6\checkpoints\semantic_probe_int8.tflite `
    --data-root D:\onn_data\openfake_visual

# 3. reuse the teacher cache for the 20-channel Engine A hybrid
Copy-Item D:\onn_data\runs\release_v6\checkpoints\semantic_teacher_cache.npz `
          D:\onn_data\runs\release_v5\checkpoints\teacher_distill_cache.npz
.\.venv\Scripts\python.exe ml\scripts\distill_vision_tower_to_srm.py `
    --data-root D:\onn_data\openfake_visual `
    --output-dir D:\onn_data\runs\release_v5\checkpoints `
    --epochs 18 --include-rgb

# 4. package the dual-engine release
#    (exports SRM+NPR+RGB and the semantic probe, trains the 10-feature
#     fusion with slot 6 populated, masks 25% of the training rows so the
#     tree learns a real probe-free path, fits the cascade policy on the
#     calibration split, builds the device report and the APK)
.\.venv\Scripts\python.exe ml\scripts\build_and_install_release_v4.py `
    --mode package-dual-engine-v6

# 5. measure the result on the sealed test split
.\.venv\Scripts\python.exe ml\scripts\evaluate_cascade.py `
    --fusion-dir D:\onn_data\runs\release_v6\fusion `
    --output D:\onn_data\runs\release_v6\fusion\cascade_results.json
```

`package-dual-engine-v6` runs `export_all_models.py --cascade-tflite`,
`train_xgboost_meta.py --semantic-tflite`, then
`generate_android_constants.py`, `release_bundle.py --install`,
`check_android_consistency.py` and `pytest ml/tests`.

### Optional model semantics

Engine C is **optional** in the release contract:

* absent → the pipeline behaves exactly as it did before the dual engine:
  slot 6 stays unavailable, the tree takes its learned missing branch, and
  the fusion threshold applies to the Engine A score;
* present → it is held to the same standard as the required models
  (hash-bound, calibrated on a declared split, covered by the desktop parity
  report and the independent device report), and the cascade arbitrates.

`release_bundle.py` rejects a release that ships Engine C without its
`constants/semantic_probe_threshold.json` and `constants/cascade_policy.json`
artifacts, that declares it with the wrong frontend, or whose calibration
is at/below chance.


## 5. Measured expectations

| Configuration | Expected AUC | Notes |
|---------------|--------------|-------|
| Engine A alone (v4) | 0.6555 | measured |
| Engine A alone (v5 hybrid) | 0.71–0.80 | distilled student, measured on v3→v4 delta |
| Engine C alone (Lite C) | 0.80–0.885 | distilled CLIP student |
| **Dual-engine fusion** | **0.88–0.896** | the measured target |
| Cascade arbitration | ≥ dual-engine, lower coverage | disagreements → INCONCLUSIVE |

All numbers must be re-measured on the sealed test split; the table above
is the *expected* range, not a claim.
