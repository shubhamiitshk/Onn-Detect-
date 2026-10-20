# MLLM Integration Guide — ONN-DETECT v2

> **Complete integration guide for adding on-device MLLM (Gemma 3n E2B)  
> to the ONN-DETECT forensic detection pipeline.**

---

## Quick Start

### 1. Training Pipeline (Python)

```bash
# Step 1: Score calibration split with MLLM
python -m ml.scripts.score_with_mllm \
    --dataset-manifest datasets/openfake/manifest.json \
    --split calibration \
    --model-id google/gemma-3n-E2B-it \
    --output releases/onn-20261020-v5/mllm_scores_calibration.json

# Step 2: Calibrate MLLM scores (Platt scaling)
python -m ml.scripts.calibrate_mllm \
    --scores releases/onn-20261020-v5/mllm_scores_calibration.json \
    --dataset-manifest datasets/openfake/manifest.json \
    --output releases/onn-20261020-v5/mllm_calibration.json

# Step 3: Score test split with calibrated MLLM
python -m ml.scripts.score_with_mllm \
    --dataset-manifest datasets/openfake/manifest.json \
    --split test \
    --model-id google/gemma-3n-E2B-it \
    --calibration-params releases/onn-20261020-v5/mllm_calibration.json \
    --output releases/onn-20261020-v5/mllm_scores_test.json

# Step 4: Train fusion v2 with expanded 16-feature schema
python -m ml.src.meta_classifier.fusion_trainer_v2 \
    --config ml/configs/fusion_v2.yaml \
    --mllm-scores releases/onn-20261020-v5/mllm_scores_test.json \
    --output-dir releases/onn-20261020-v5
```

### 2. Android Integration

```kotlin
// In your Application class or Activity:
val modelManager = MllmModelManager(context)
val status = modelManager.checkModelStatus()

if (status.isAvailable) {
    val mllmAnalyzer = MllmForensicAnalyzer.create(
        context = context,
        modelPath = status.modelPath!!,
        config = MllmConfig.DEFAULT,
    )

    // Use in DetectionPipelineV2
    val pipeline = DetectionPipelineV2(
        context = context,
        srmAnalyzer = srmAnalyzer,
        fftAnalyzer = fftAnalyzer,
        klAnalyzer = klAnalyzer,
        audioClassifier = audioClassifier,
        modelIntegrityChecker = modelIntegrityChecker,
        mllmAnalyzer = mllmAnalyzer,  // New MLLM analyzer
    )

    // Analyze image
    val result = pipeline.analyzeImage(bitmap)
    println("Fused score: ${result.score}")
    println("MLLM reasoning: ${result.mllmResult?.reasoning}")
}
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Training Pipeline                      │
│                                                          │
│  Dataset → MLLM Scorer → Platt Scaling → 16-Feature     │
│  Vectors → XGBoost v2 → Kotlin Export → Android        │
│                                                          │
├─────────────────────────────────────────────────────────┤
│                  Android Inference                       │
│                                                          │
│  Image → SRM + FFT + KL → Uncertainty Router            │
│                                     │                    │
│                    ┌────────────────┼──────────────┐    │
│                    ▼                ▼              ▼    │
│              SRM Only      Spectral Only     + MLLM    │
│              (~50ms)        (~100ms)       (~2000ms)  │
│                    │                │              │    │
│                    └────────────────┼──────────────┘    │
│                                     ▼                    │
│                           FusionV2Classifier            │
│                           (16-feature XGBoost)          │
│                                     │                    │
│                                     ▼                    │
│                           Verdict + Explanation          │
└─────────────────────────────────────────────────────────┘
```

---

## File Structure

### Python (Training)
```
ml/
├── src/
│   ├── models/
│   │   ├── mllm_forensic_scorer.py    # MLLM scorer + Platt scaling
│   │   ├── srm_forensic_cnn.py         # Existing SRM model
│   │   └── lcnn_mfm_audio.py           # Existing audio model
│   └── meta_classifier/
│       ├── feature_extractor.py       # Original 10-feature schema
│       └── fusion_trainer_v2.py       # New 16-feature fusion trainer
├── scripts/
│   ├── score_with_mllm.py             # Score datasets with MLLM
│   └── calibrate_mllm.py              # Fit Platt scaling
└── configs/
    ├── fusion_v2.yaml                 # v2 fusion config
    └── xgboost_meta.yaml              # Original v1 config
```

### Android (Inference)
```
android/app/src/main/
├── java/com/onndetect/
│   ├── detection/
│   │   ├── mllm/
│   │   │   ├── MllmForensicAnalyzer.kt    # MLLM inference via LiteRT-LM
│   │   │   └── MllmModelManager.kt         # Model download + verification
│   │   ├── fusion/
│   │   │   ├── FusionV2Classifier.kt      # 16-feature XGBoost (auto-generated)
│   │   │   └── UncertaintyRouter.kt       # Uncertainty-based routing
│   │   ├── visual/
│   │   │   ├── SRMForensicAnalyzer.kt     # Existing
│   │   │   ├── FFTSpectralAnalyzer.kt     # Existing
│   │   │   └── KLCooccurrenceAnalyzer.kt   # Existing
│   │   └── audio/
│   │       └── AudioClassifier.kt         # Existing
│   └── domain/pipeline/
│       ├── DetectionPipeline.kt           # Original v1 pipeline
│       └── DetectionPipelineV2.kt         # New v2 pipeline with MLLM
└── assets/
    ├── models/
    │   ├── srm_forensic_cnn_int8.tflite   # Existing
    │   ├── lcnn_audio_int8.tflite         # Existing
    │   └── gemma_3n_e2b_it.task           # NEW: MLLM model (~2GB)
    └── constants/
        ├── meta_classifier.json           # v1 schema
        └── fusion_v2_classifier.json      # NEW: v2 schema
```

---

## Configuration

### Gemma 3n E2B Model Settings

```json
{
  "model": {
    "id": "google/gemma-3n-E2B-it-litert-lm",
    "format": "task",
    "size_bytes": 2147483648,
    "quantization": "int4",
    "vision_encoder": "MobileNetV5",
    "max_resolution": 768,
    "context_window": 32768
  },
  "memory": {
    "minimum_mb": 2560,
    "recommended_mb": 4096,
    "fallback_behavior": "skip_mllm_use_spectral_only"
  },
  "inference": {
    "temperature": 0.1,
    "top_k": 1,
    "max_new_tokens": 512,
    "enable_vision": true,
    "enable_audio": false
  }
}
```

### Router Configuration

```json
{
  "uncertainty_thresholds": {
    "low": 0.15,
    "high": 0.40
  },
  "model_weights": {
    "srm": 0.6,
    "spectral": 0.4
  },
  "agreement_threshold": 0.25,
  "mllm_confidence_boost": 0.3
}
```

---

## Performance Expectations

### Latency by Device Tier

| Device Tier | SRM Only | +MLLM | Router Avg |
|-------------|----------|-------|------------|
| Flagship (Snapdragon 8 Elite) | ~30ms | ~800ms | ~200ms |
| Mid-range (Snapdragon 7 Gen) | ~50ms | ~2000ms | ~400ms |
| Entry (Snapdragon 6 Gen) | ~80ms | ~4000ms | ~700ms |
| Insufficient memory | ~50ms | N/A | ~50ms |

### Accuracy Improvement

| Metric | v1 (SRM only) | v2 (SRM + MLLM) | Delta |
|--------|---------------|-----------------|-------|
| Image AUC | 0.592 | 0.72–0.78 | +0.13–0.19 |
| Audio AUC | 0.864 | 0.87–0.90 | +0.01–0.04 |
| Calibration (ECE) | High | Low | Significant |
| OOD generalization | Poor | Good | Major |
| Explainability | None | Full | New feature |

---

## Troubleshooting

### OOM on Mid-Range Devices

The Gemma 3n E2B model requires ~2GB RAM. On devices with less:
- `MllmModelManager.checkModelStatus()` returns `isAvailable=false`
- `DetectionPipelineV2` automatically falls back to spectral-only analysis
- No crashes — graceful degradation

### MLLM Returns Invalid JSON

The `MllmForensicAnalyzer.parseResponse()` method handles this:
- Tries direct JSON parse first
- Falls back to regex extraction of `ai_probability`
- Returns neutral score (0.5) with LOW confidence on total failure

### Slow Inference

The `UncertaintyRouter` skips MLLM invocation for clear cases:
- Only ~30-40% of images need MLLM analysis
- Average latency with routing: 300-500ms (vs 2000ms always-on)
- Configure `RouterConfig.AGGRESSIVE` to skip more often

### Model Download

The 2GB model file should be:
1. Bundled in APK assets (for offline-first)
2. Or downloaded via Play Asset Delivery (for smaller APK)
3. Verified via SHA-256 against release manifest

---

## References

- Gemma 3n: https://ai.google.dev/gemma/docs/gemma-3n
- LiteRT-LM: https://github.com/google-ai-edge/LiteRT
- DFBench: https://github.com/IntMeGroup/DFBench
- MoA-DF: ACM MM 2025
- FakeVLM: NeurIPS 2025
