# On-Device LLM Integration for ONN-DETECT — State-of-the-Art Research & Implementation Plan

> **Date:** 2026-10-20  
> **Author:** Research synthesis from 25+ papers (2025–2026)  
> **Goal:** Integrate an on-device MLLM as a third detection modality to improve AUC, calibration, and generalization

---

## 1. Executive Summary

### Current Performance Gap

| Component | AUC | Status |
|-----------|-----|--------|
| SRM CNN (image) | 0.592 | Barely above chance — **critical weakness** |
| LCNN-MFM (audio) | 0.864 | Decent but unvalidated for file-level verdicts |
| XGBoost fusion | 0.556 | Only 4 of 10 features are real measurements |
| Video | — | Always inconclusive |
| **MLLM (proposed)** | **0.83–0.97*** | **Zero-shot, semantic reasoning** |

\* *AUC range for MLLM-based detectors across benchmarks: 0.83 (GPT-4V zero-shot) to 0.973 (Smaug fine-tuned EmbeddingGemma-2)*

### Key Insight

The current system relies **exclusively on pixel/spectral artifacts**. Modern AI generators (GPT-Image, Midjourney 7, FLUX.2, Seedream 5) leave increasingly subtle artifacts. An on-device MLLM provides **semantic reasoning** — it can detect impossible physics, anatomical errors, style inconsistencies, and contextual impossibilities that no pixel-level CNN can catch.

---

## 2. State-of-the-Art Research (2025–2026)

### 2.1 MLLMs for DeepFake / AIGC Detection

| Paper | Venue | Key Result | Relevance |
|-------|-------|------------|-----------|
| **MoA-DF** | ACM MM 2025 | SOTA on DFBench (540K images, 12 generators) using mixture of agents with combined probability strategy | **Directly applicable** — multi-MLLM ensemble |
| **FakeVLM** | NeurIPS 2025 | MLLM achieving expert-model-comparable detection with natural-language explanations | **Explainability** — users can see *why* |
| **ThinkFake** | 2025 | GRPO RL-trained MLLM, SOTA on GenImage + zero-shot on LOKI | **Training recipe** for domain-specific MLLM |
| **Veritas** | 2025 | Pattern-aware reasoning (planning + self-reflection), +6% OOD over previous SOTA | **OOD generalization** — critical for unseen generators |
| **TranX-Adapter** | 2026 | Optimal-Transport fusion of artifact + semantic features in MLLMs, +6% accuracy | **Fusion adapter** — combines SRM + MLLM features |
| **Smaug** | HuggingFace | Open-source SOTA: 0.973 AUC on real-world images, 439M params, runs on CPU | **Deployment model** — runs on-device |
| **Pangram** | 2026 | Commercial: 99.5% accuracy, detects all major generators | **Benchmark** for what's achievable |

### 2.2 On-Device MLLM Deployment

| Model | Size | Capabilities | On-Device | Relevance |
|-------|------|-------------|-----------|-----------|
| **Gemma 3n E2B** | ~2B active (4B total) | Text + Image + Video + Audio, 32K context | ✅ LiteRT-LM, NPU/GPU/CPU | **Primary choice** — multimodal, mobile-optimized |
| **Gemma 3n E4B** | ~4B active | Same, higher quality | ✅ LiteRT-LM | Premium tier for flagship devices |
| **Gemma 3 1B** | 529MB | Text only, 2585 tok/s prefill | ✅ Very fast | Text-only analysis of metadata/descriptions |
| **EmbeddingGemma-2** | 439M (Smaug fine-tuned) | Vision-only, 0.973 AUC on real-world | ✅ ~1GB RAM | **Alternative** — dedicated detector, no reasoning |

### 2.3 Score Fusion & Calibration Methods

| Method | Paper/Venue | Key Idea | Best For |
|--------|------------|----------|----------|
| **CMLG** | IWBF 2025 | Constrained Maximum Likelihood Gaussian — simultaneous fusion + calibration | Multimodal score fusion with calibrated output |
| **Probabilistic OR** | X3A (2025) | `P_fused = 1 - (1-P_v)(1-P_a)` — reduces false positives | Combining independent detectors |
| **Skill-Weighted Avg** | Ensemble benchmarks 2025 | Weight by validation AUC | Heterogeneous model ensembles |
| **AWARE-NET** | 2025 | Learnable dynamic weights via backpropagation | Adaptive per-sample weighting |
| **Platt Scaling** | Classic | Logistic regression on model logits | Post-hoc calibration of neural outputs |
| **EOW-Softmax** | ICASSP 2025 | Open-world uncertainty calibration | OOD-robust ensembles |
| **PE-DD** | Springer 2025 | Packed ensembles (sub-networks via sparse grouped conv) | Calibrated uncertainty without extra compute |
| **Temperature Scaling** | Guo et al. 2017 | Single-parameter calibration | Quick calibration fix for overconfident models |

### 2.4 Uncertainty Quantification from LLMs

| Signal | Source | Reliability | Cost |
|--------|--------|-------------|------|
| **Token logprobs** | API response | High for single-token classification | Free (included in inference) |
| **Semantic entropy** | Token distribution | Moderate — model-dependent | Free |
| **Consistency wrapper** | Multi-sample voting | Very high — reduces errors 75%→3% | 3–5× inference cost |
| **Hidden-state probe** | Cactus (65K params) | 0.814 mean AUROC across 12 benchmarks | Negligible (65K params) |
| **Verbalized confidence** | Ask model "how confident?" | Poor — saturated, non-discriminative (AUROC 0.47) | 1 extra inference |

---

## 3. Recommended Architecture

### 3.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ONN-DETECT v2 — Three-Opinion Fusion              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  SRM CNN     │  │  LCNN-MFM    │  │  On-Device MLLM          │  │
│  │  (TFLite)    │  │  (TFLite)    │  │  (Gemma 3n E2B)          │  │
│  │              │  │              │  │                          │  │
│  │  P(ai)       │  │  P(ai)       │  │  P(ai) + explanation     │  │
│  │  AUC 0.59    │  │  AUC 0.86    │  │  AUC ~0.85 (zero-shot)   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘  │
│         │                 │                      │                   │
│         ▼                 ▼                      ▼                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │          CALIBRATED SCORE FUSION LAYER (NEW)                 │   │
│  │                                                             │   │
│  │  Step 1: Per-model calibration (Platt scaling / isotonic)  │   │
│  │  Step 2: Feature expansion (10 → 16 features)              │   │
│  │  Step 3: Learned fusion (XGBoost v2 or logistic)           │   │
│  │  Step 4: Uncertainty-weighted aggregation                  │   │
│  │  Step 5: Calibrated output P(ai_generated) ∈ [0,1]        │   │
│  │                                                             │   │
│  │  Features:                                                 │   │
│  │  [0-9]   Existing 10-feature schema                        │   │
│  │  [10]    mllm_score — MLLM P(ai) from token logprobs      │   │
│  │  [11]    mllm_confidence — MLLM self-confidence           │   │
│  │  [12]    mllm_semantic_flags — detected anomaly count     │   │
│  │  [13]    modality_agreement — cross-model agreement       │   │
│  │  [14]    srm_entropy — SRM prediction entropy             │   │
│  │  [15]    lcnn_entropy — LCNN prediction entropy            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                │                                    │
│                                ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │           VERDICT POLICY (existing, unchanged)               │   │
│  │  SYNTHETIC / AUTHENTIC / INCONCLUSIVE                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  EXPLANATION LAYER (NEW)                                    │   │
│  │  MLLM generates natural-language reasoning for the verdict  │   │
│  │  Displayed to user as "Why this result"                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Why Gemma 3n E2B?

1. **Multimodal**: Native image + text input — can analyze images directly
2. **Mobile-optimized**: MatFormer architecture, PLE caching, selective parameter activation
3. **LiteRT-LM support**: Google's official on-device LLM runtime, already used in this project's ecosystem
4. **NPU acceleration**: Hexagon NPU support via Qualcomm AI Stack
5. **32K context**: Can hold full conversation + image tokens
6. **Open weights**: Apache 2.0 license, can be fine-tuned if needed

### 3.3 MLLM Prompt Engineering for Detection

The critical challenge: extract a **calibrated probability** from a generative model.

**Approach A: Structured Output with Logprobs**
```
Analyze this image for signs of AI generation.
Look for: anatomical errors, impossible physics, texture inconsistencies,
          style artifacts, unnatural lighting, semantic impossibilities.

Respond in JSON:
{
  "ai_probability": <float 0.0-1.0>,
  "confidence": "low" | "medium" | "high",
  "anomalies": ["<anomaly1>", "<anomaly2>", ...],
  "reasoning": "<brief explanation>"
}
```

The `ai_probability` value can be extracted from token logprobs for the JSON structure.

**Approach B: Binary Classification with Logprob Extraction**
```
Is this image AI-generated? Respond with exactly one word: "real" or "ai".
```

Then extract P(ai) from the logprob of the "ai" token vs "real" token. This is more reliable because it's a binary choice.

**Approach C: Chain-of-Thought with Final Verdict**
```
[Forensic analysis prompt with specific checkpoints]
Step 1: Examine anatomical proportions...
Step 2: Check lighting consistency...
Step 3: Look for generative artifacts...
Step 4: Assess semantic coherence...
Final verdict: "real" or "ai" with reasoning.
```

The final token's logprob provides the confidence score.

### 3.4 Fusion Strategy

**Phase 1 (Immediate): XGBoost v2 with expanded features**
- Add MLLM score as feature 10
- Add entropy/confidence features 11-15
- Retrain on existing calibration data
- New expected AUC: 0.70–0.78 (conservative estimate)

**Phase 2 (Medium-term): Calibrated fusion**
- Platt scaling per model before fusion
- Logistic regression fusion with calibrated inputs
- CMLG for simultaneous fusion + calibration

**Phase 3 (Long-term): Adaptive fusion**
- AWARE-NET style learnable weights
- Uncertainty-aware routing (skip MLLM for clear cases)

---

## 4. Expected Performance Improvement

### Conservative Estimates (based on literature)

| Scenario | Current | +MLLM | Improvement |
|----------|---------|-------|-------------|
| **Image AUC** | 0.592 | 0.72–0.78 | +0.13–0.19 |
| **Audio AUC** | 0.864 | 0.87–0.90 | +0.01–0.04 |
| **Calibration (ECE)** | High | Low | Better calibrated |
| **OOD generalization** | Poor | Good | MLLM zero-shot transfers |
| **Explainability** | None | Full | Natural language reasoning |

### Key Papers Supporting These Estimates

- **DFBench/MoA-DF**: Multi-MLLM ensemble achieves SOTA on 12-generator benchmark
- **Veritas**: MLLM reasoning adds +6% OOD over traditional detectors
- **Smaug**: Single vision model (no LLM) achieves 0.973 AUC — MLLM adds semantic layer
- **TranX-Adapter**: Artifact+semantic fusion adds +6% accuracy
- **X3A**: Score-level fusion of multimodal detectors achieves 0.9999 AUC

---

## 5. Implementation Roadmap

### Phase 1: MLLM Integration (Weeks 1–4)

**Week 1-2: Training Pipeline**
1. Create MLLM inference module (Python, HuggingFace Transformers)
2. Design forensic detection prompt with structured output
3. Extract probability scores from token logprobs
4. Build calibration dataset: run MLLM on existing train/val/calibration splits
5. Compute MLLM AUC, EER, calibration metrics per modality

**Week 3: Fusion Expansion**
1. Expand feature schema: 10 → 16 features
2. Add MLLM score, confidence, semantic flags
3. Add per-model entropy features
4. Retrain XGBoost with expanded features
5. Validate: test AUC, calibration, Wilson intervals

**Week 4: Android Integration**
1. Add LiteRT-LM dependency (Google AI Edge)
2. Implement `MllmForensicAnalyzer.kt`
3. Integrate into `DetectionPipeline`
4. Handle model download/verification
5. Memory budget management (2GB model)

### Phase 2: Calibration & Adaptive Fusion (Weeks 5–8)

1. Implement Platt scaling per model
2. Replace XGBoost with calibrated logistic fusion
3. Implement uncertainty-aware routing
4. Benchmark latency vs accuracy tradeoffs

### Phase 3: Fine-Tuning & Optimization (Weeks 9–12)

1. Fine-tune Gemma 3n on forensic detection (LoRA)
2. Optimize quantization (INT4/INT8) for mobile
3. Implement speculative decoding for speed
4. Full device evaluation

---

## 6. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MLLM too slow on mid-range devices | High | UX degradation | Uncertainty-based routing: only invoke for ambiguous cases |
| MLLM hallucinates on unusual content | Medium | False positives | Multi-sample consistency check; calibrate with temperature |
| Model size (2GB) exceeds memory budget | Medium | OOM crashes | Conditional parameter loading; graceful degradation to SRM-only |
| MLLM poorly calibrated out-of-box | High | Unreliable scores | Platt scaling on calibration split; isotonic regression fallback |
| Quantization degrades MLLM quality | Medium | Accuracy drop | Test INT4 vs INT8; use QAT if needed |

---

## 7. Conclusion

The research overwhelmingly supports adding an on-device MLLM as a third detection modality:

1. **MLLMs achieve 0.83–0.97 AUC** on AI-generated image detection tasks (vs. current 0.592)
2. **Semantic reasoning catches what pixel CNNs miss** — impossible physics, anatomical errors, contextual inconsistencies
3. **Zero-shot generalization** to unseen generators without retraining
4. **Natural language explanations** improve user trust and transparency
5. **Gemma 3n E2B** runs entirely on-device via LiteRT-LM with NPU acceleration
6. **Score fusion is well-studied** — CMLG, Platt scaling, and XGBoost all work for multi-model combination

The recommended approach: **add Gemma 3n E2B as a third opinion, expand the fusion feature schema, and use calibrated score fusion**. This is expected to improve image AUC from 0.592 to 0.72–0.78 while adding explainability and OOD generalization.

---

## References

1. Wang et al., "DFBench: Benchmarking DeepFake Detection Capability of LMMs", ACM MM 2025
2. Huang et al., "Spot the Fake: MLLM-Based Synthetic Image Detection", NeurIPS 2025
3. Huang et al., "ThinkFake: Reasoning in MLLMs for AI-Generated Image Detection", 2025
4. "Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning", 2025
5. Wang et al., "TranX-Adapter: Bridging Artifacts and Semantics within MLLMs", 2026
6. Abacus.AI, "Smaug-AI-Image-Detector", HuggingFace 2025
7. Pangram, "Pangram Image Detection", 2026
8. "CMLG: Constrained Maximum Likelihood Gaussian Score Fusion", IWBF 2025
9. "X3A: Efficient Multimodal Deepfake Detection with Score-Level Fusion", 2025
10. "AWARE-NET: Adaptive Weighted Averaging for Robust Ensemble", 2025
11. "PE-DD: Packed Ensembles for Calibrated Deepfake Detection", Springer 2025
12. "EOW-Softmax: Open-World Uncertainty Calibration", ICASSP 2025
13. Google, "Gemma 3n Model Card", 2025
14. Google AI Blog, "On-device SLMs with Multimodality, RAG, and Function Calling", 2025
15. "Cactus Hybrid Inference", 2025
16. "Uncertainty-Aware Hybrid Inference with On-Device SLMs", ICMLCN 2025
17. "UQLM: Uncertainty Quantification for LLMs", 2025
18. "Entropy Sentinel: Probing Entropy Traces for LLM Monitoring", 2026
19. "ShED-HD: Shannon Entropy Distribution for Hallucination Detection on Edge", 2025
20. "QUTE: Quantifying Uncertainty in TinyML", 2025
