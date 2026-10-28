# State of the Art Review and Technical Specification — ONN-DETECT (2025–2026 Edition)

**Title:** On-Device Zero-Cloud Media Forensics: SOTA Review, Mathematical Foundations, and Verified Architectural Alignment  
**Document Version:** 10.1-SOTA-2026  
**Target Engine Release:** `onn-prod-v10` / `v10.1`  
**Execution Environment:** Android 15/16 (64-bit ARM64, 16 KiB ELF alignment) & Standalone Python Verification Harness  
**Network Invariant:** Strict Zero-Network Isolation (`android.permission.INTERNET` and `ACCESS_NETWORK_STATE` excluded)  
**Date:** 2026-10-26  

---

## Table of Contents

1. [Chapter 1: Executive Summary & Paradigm Shifts in 2025–2026 Media Forensics](#chapter-1-executive-summary--paradigm-shifts-in-20252026-media-forensics)
   - 1.1 The Generative Transition: From GAN Artifacts to Latent Diffusion & Autoregressive Foundations
   - 1.2 The Dataset Provenance Shortcut Crisis (REALIS, DailyBench, and VoiceWukong)
   - 1.3 Architectural Invariants of ONN-DETECT: Zero-Cloud, Multi-Layer Edge Cascade
2. [Chapter 2: Audio Forensics & Neural Watermarking](#chapter-2-audio-forensics--neural-watermarking)
   - 2.1 Proactive vs. Passive Audio Forensics
   - 2.2 Meta AudioSeal: 1D SEANet Architecture & Sample-Level Temporal Localization
   - 2.3 Exact Binomial Sign Test & Mobile Runtime Graph Lowering
   - 2.4 The Watermark Shortcut Crisis & WASP Decorrelation Protocol
   - 2.5 Passive Speech Anti-Spoofing: LCNN-MFM & Versioned LFCC Frontend
   - 2.6 Dual-Stage Voiced Duration Gating & Multi-Window Logit Pooling
3. [Chapter 3: Vision Foundation Probes](#chapter-3-vision-foundation-probes)
   - 3.1 ViT Representations vs. Handcrafted Spatial Residuals
   - 3.2 SigLIP-B/16 Architectural Backbone & Token Extraction
   - 3.3 Tunable Attention Pooling (TAP) Across 196 Spatial Patch Tokens
   - 3.4 Zero-Overhead Keras Mirror Folding
   - 3.5 Degradation-Consistent Paired Training (DCPT) & Laundered Social Media Robustness
   - 3.6 Evaluation of Foundation Backbones: Why MobileCLIP Fails and SigLIP/DINOv2 Generalize
4. [Chapter 4: Spatial-Frequency Residuals](#chapter-4-spatial-frequency-residuals)
   - 4.1 The Anti-Aliasing Destruction Problem in Standard Downsampling
   - 4.2 Native PatchCraft 4x4 Texture Mosaics
   - 4.3 20-Channel Fixed Residual Kernel Stem (SRM + NPR + RGB)
   - 4.4 Rich-vs-Poor Texture Contrast & Sliding-Window Anomaly Hotspot Localization
5. [Chapter 5: Biomechanical & Physiological Dynamics](#chapter-5-biomechanical--physiological-dynamics)
   - 5.1 Category Errors in Video Forensics: Why Per-Frame Aggregation Fails
   - 5.2 BioLip: Biomechanical 40-Point Perioral Jitter on MediaPipe 478-Point FaceMesh
   - 5.3 Bilateral Malar rPPG Chrominance Phase Coherence
   - 5.4 Audio-Visual Lip-Sync Cross-Correlation (AVFF MAR vs. Audio RMS Envelope)
   - 5.5 Multimodal Video Score Escalation Policy (`VideoScorePolicy`)
6. [Chapter 6: Calibration, Conformal Prediction & Missing Modalities](#chapter-6-calibration-conformal-prediction--missing-modalities)
   - 6.1 The Mathematics of Missing Evidence (CLAUDE.md Truthfulness Contract)
   - 6.2 Mondrian Conformal Prediction Under Dynamic Missingness Masks
   - 6.3 Dual-Engine Cascade Arbitration & Selective Accuracy Optimization
   - 6.4 Temperature Scaling, Base-Margin Boosting & Platt Calibration
7. [Chapter 7: Cryptographic Provenance & Enterprise Attestation](#chapter-7-cryptographic-provenance--enterprise-attestation)
   - 7.1 C2PA Content Credentials v2.1/v2.3: JUMBF, APP11, and MP4 Box Architecture
   - 7.2 The Exclusion Range Vulnerability & Fail-Closed Cryptographic Binding
   - 7.3 Android KeyMint StrongBox Hardware Attestation (OID `1.3.6.1.4.1.11129.2.1.17`)
   - 7.4 RFC 8785 Canonical JSON Forensic Receipts & High-Resolution Verification Certificates
   - 7.5 Video-KYC (RBI V-CIP) Injection Defense: Camera2 HAL & Dynamic Hook Detection
8. [Chapter 8: Codebase Audit Changelog & Empirical Benchmark Matrix](#chapter-8-codebase-audit-changelog--empirical-benchmark-matrix)
   - 8.1 Itemized Changelog of Resolved Defects (Milestone 1, Milestone 2, Milestone 2 Round 2)
   - 8.2 Empirical Benchmark Matrix: Clean vs. Laundered Performance
   - 8.3 Low-FPR Operating Points (TPR @ 1% FPR & TPR @ 0.1% FPR)
   - 8.4 Resource Footprint, Mobile Latency & 16 KiB ELF Alignment
   - 8.5 Primary Literature Citations (2024–2026)

---

# Chapter 1: Executive Summary & Paradigm Shifts in 2025–2026 Media Forensics

## 1.1 The Generative Transition: From GAN Artifacts to Latent Diffusion & Autoregressive Foundations

Between 2020 and 2024, digital media forensics relied overwhelmingly on detecting the structural imperfections of Generative Adversarial Networks (GANs), such as StyleGAN, StarGAN, and CycleGAN. These synthesis engines operated via transposed convolutions that left rigid, periodic upsampling artifacts in the frequency domain (visible as distinct spectral spikes in 2D FFT magnitude spectra) and unnatural color co-occurrence patterns in RGB color space.

By 2025–2026, the state of the art in generative media shifted decisively. Frontier text-to-image, text-to-audio, and text-to-video synthesis are now driven by:
1. **Flow-Matching and Latent Diffusion Models (LDMs)**: Architectures such as FLUX.1 (Black Forest Labs), Stable Diffusion 3/3.5, Midjourney v6/v7, and AuraSR utilize continuous-time flow-matching formulations and rectified flow ODEs. They eliminate traditional transposed convolution grids by employing patchified Vision Transformer (DiT) backbones operating on continuous latent spaces.
2. **Autoregressive and Diffusion Video Foundations**: Sora-2 (OpenAI), Wan-2.1/2.5 (Alibaba), Kling 1.5, Gen-3 Alpha (Runway), and Luma Dream Machine synthesize temporally consistent, photorealistic video at $1080\text{p}$ and $4\text{K}$ resolutions. These models maintain coherent 3D geometry and realistic lighting across hundreds of frames, defeating naive frame-differencing and optical flow disparity metrics.
3. **Neural Speech & Audio Synthesizers**: Systems such as ElevenLabs Multilingual v2, Chatterbox, VALL-E 2, CosyVoice, Suno v3.5, and Udio 1.5 utilize neural audio codecs (EnCodec, DAC) driven by discrete speech language models or continuous diffusion decoders. They synthesize human speech complete with natural prosody, breathing pauses, glottal pulses, and subtle emotional micro-pitch inflections.

As a consequence of this paradigm shift, classical forensic indicators have largely lost their discriminative power:
- **Spatial Residual Filters**: High-pass SRM (Spatial Rich Models) filters applied to downsampled images fail because diffusion latents produce continuous naturalistic noise profiles rather than discrete grid lines.
- **Power Spectrum Slope**: The high-frequency spectral roll-off of modern flow-matching models matches physical optical lenses within standard measurement bounds.
- **Color Histograms**: Modern cross-attention mechanisms match natural sensor chrominance distributions, rendering simple color co-occurrence metrics ineffective.

## 1.2 The Dataset Provenance Shortcut Crisis (REALIS, DailyBench, and VoiceWukong)

The most alarming development uncovered by 2025–2026 forensic research is the **Dataset Provenance Shortcut Crisis**. Detectors evaluated on early benchmark suites (e.g., GenImage, FaceForensics++, ASVspoof 2019) achieved flattering headline metrics ($\text{AUC} > 0.95$), yet collapsed catastrophically when deployed in real-world environments.

Rigorous independent benchmarks have exposed the root causes:
1. **The Image Domain: REALIS (arXiv:2609.32734) & DailyBench (arXiv:2607.24016)**:
   - Modern benchmarks demonstrated that when training sets pair synthetic images from web generators with real images from standard photography repositories (e.g., LAION-5B, Pexels, DocCI, ImageNet), classifiers do not learn generative synthesis artifacts. Instead, they learn **dataset provenance shortcuts**.
   - Specifically, models learn the heuristic:
     $$\text{LAION-5B web crawl artifacts} \implies \text{Synthetic}$$
     $$\text{Pexels pristine DSLR photography} \implies \text{Authentic}$$
   - When audited on balanced held-out data, detectors exhibiting $96\%$ balanced accuracy on GenImage dropped to **$52\%\text{--}79\%$** on DailyBench (FakeBench) and **$43\%\text{--}67\%$** on ManipulationBench.
   - On the REALIS benchmark (1.3M images across 42 generators), real LAION images were misclassified as synthetic by legacy detectors over **$44\%$** of the time, while commercial frontier generators (GPT-Image-2, Qwen-Image, Sora-2) scored **$0\%$ True Positive Rate**.
2. **The Audio Domain: VoiceWukong (USENIX Security 2025) & "It's All in the Presentation" (ICASSP 2026)**:
   - In audio anti-spoofing, standard ASVspoof datasets supply real speech recorded in pristine, sound-dampened studio environments, while synthetic speech contains vocoder artifacts.
   - When tested against in-the-wild audio (VoiceWukong, 413K real-world deepfakes in English and Chinese), detectors evaluated at $98\%$ AUC on ASVspoof collapsed. Classifiers learned:
     $$\text{Background room noise, acoustic reverberation, telephony codecs} \implies \text{Synthetic}$$
   - On real field recordings (e.g., `itw_Jerry_Seinfeld`), the false-positive rate climbed to **$62.5\%$**, proving that classifiers were flagging ambient background noise rather than vocal tract anomalies.

## 1.3 Architectural Invariants of ONN-DETECT: Zero-Cloud, Multi-Layer Edge Cascade

To survive in real-world operational environments without leaking user biometric data or relying on vulnerable remote cloud APIs, **ONN-DETECT** (`onn-prod-v10` / `v10.1`) enforces three non-negotiable architectural invariants:

1. **Strict Zero-Network Airgap**:
   - The Android application manifest strictly excludes `android.permission.INTERNET` and `android.permission.ACCESS_NETWORK_STATE`. The app cannot make outbound network calls, transmit telemetry, query cloud foundation models, or leak private biometric voiceprints.
   - All neural inference, cryptographic verification, and biomechanical tracking execute strictly on the local device application processor.
2. **Deterministic Missing-Evidence Contract (`CLAUDE.md`)**:
   - An unmeasured or absent forensic signal must be strictly represented as **unavailable (`Float.NaN`)**, never coerced to a plausible neutral constant (e.g., $0.0$, $0.5$, or $-1.0$).
   - Coercing missing evidence to arbitrary numbers introduces catastrophic train/serve skew in gradient boosted decision trees. In contrast, XGBoost and LightGBM handle `NaN` natively via learned missing-branch routing.
   - In JSON attestation receipts, `Float.NaN` serializes strictly to `null` per RFC 8259, ensuring unambiguous downstream interpretation.
3. **16 KiB ELF Alignment & Mobile Edge Budgets**:
   - Starting with Android 15 and required on Android 16, all native shared libraries (`.so` files embedded in the APK) must adhere to 16 KiB ELF page boundaries to support modern ARM64 page translation tables.
   - Memory-mapped model storage (`noCompress += "tflite"`, accessed via direct file descriptor offsets) ensures that large neural weights (e.g., the 89.96 MB SigLIP probe) reside in the operating system's unified clean page cache rather than consuming JVM heap space, completely eliminating out-of-memory (OOM) aborts.

---

# Chapter 2: Audio Forensics & Neural Watermarking

## 2.1 Proactive vs. Passive Audio Forensics

Audio forensics is partitioned into two complementary technical methodologies:
1. **Proactive Forensics (Neural Watermarking)**: The generator embeds an imperceptible, cryptographically bound pseudo-random carrier directly into the synthesized audio waveform at generation time. Detection is framed as carrier extraction and statistical hypothesis testing.
2. **Passive Forensics (Speech Anti-Spoofing)**: Operates without generator cooperation. Inspects physical acoustic properties, glottal pulse shapes, phase spectrum consistency, and linear frequency cepstral coefficients (LFCC) to identify vocoder anomalies and neural synthesis signatures.

## 2.2 Meta AudioSeal: 1D SEANet Architecture & Sample-Level Temporal Localization

AudioSeal (Meta AI, ICML 2024; arXiv:2401.17264) is the state of the art in localized neural watermarking. Unlike global clip-level watermarks that output a single scalar per audio file, AudioSeal provides sample-level temporal localization at $16\text{ kHz}$.

```
Input Audio x ∈ ℝ^(1 × T)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ AudioSeal Detector: 1D SEANet Convolutional Backbone   │
│ - 1D Strided Convolutions (Downsampling factors 2,4,5,8)│
│ - Dilated 1D Residual Blocks with LeakyReLU           │
│ - Transposed 1D Convolutions (Upsampling to 16 kHz)   │
└────────────────────────────────────────────────────────┘
       │
       ├────────────────────────────────────────┐
       ▼                                        ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│ Temporal Localization Head   │ │ Message Decoding Head        │
│ 1D Conv1D(1) -> Sigmoid      │ │ Global AvgPool + Linear(16)  │
│ Output: m̂ ∈ [0, 1]^(1 × T)   │ │ Output: b̂ ∈ ℝ^(1 × 16)      │
│ (Sample-level probability)   │ │ (16-bit message carrier)     │
└──────────────────────────────┘ └──────────────────────────────┘
```

The generator embeds a watermark $w \in \mathbb{R}^T$ into raw audio $x \in \mathbb{R}^T$ yielding watermarked audio:
$$\tilde{x} = x + \alpha w$$
where $\alpha \approx 10^{-3}$ is an imperceptibility scaling parameter chosen such that the Signal-to-Distortion Ratio (SDR) exceeds $35\text{ dB}$.

The detector $D(\tilde{x})$ processes mono $16\text{ kHz}$ PCM audio through a 1D SEANet encoder-decoder architecture:
1. **Encoder**: Progressive temporal downsampling via four 1D convolutional stages (strides 2, 4, 5, 8; total downsampling factor 320, corresponding to $20\text{ ms}$ frames).
2. **Residual Bottleneck**: Dilated 1D convolutions with dilation factors $d \in \{1, 3, 9, 27\}$ capturing multi-scale acoustic context without expanding parameter counts.
3. **Decoder**: Mirrored transposed 1D convolutions reconstructing sample-level features.
4. **Dual Output Heads**:
   - **Localization Head**: Emits temporal mask $\hat{m}(t) \in [0, 1]$ indicating whether sample $t$ is watermarked.
   - **Message Head**: Decodes a multi-bit payload $\hat{b} \in [0, 1]^B$, where $B = 16$ bits for the standard AudioSeal model.

## 2.3 Exact Binomial Sign Test & Mobile Runtime Graph Lowering

### The Agreeing Bit Counter
On mobile platforms, `LiteRTInferenceEngine` enforces a strict execution contract:
$$\text{require}(\text{output.numElements}() == 1)$$
To satisfy this contract and avoid cross-language floating-point decoding divergences between Python and Kotlin, the 16-bit message logits $\hat{b} \in \mathbb{R}^{16}$ are mapped to a discrete scalar statistic within the model graph via `AgreeingBitCounter`:
$$k = \sum_{i=1}^{B} \mathbb{I}\left( \sigma(\hat{b}_i) > 0.75 \;\lor\; \sigma(\hat{b}_i) < 0.25 \right)$$
A bit is defined as agreeing if the detector is decisively confident in state $0$ or state $1$.

### Exact Two-Tailed Binomial Sign Test
Under the null hypothesis $H_0$ (authentic audio without an AudioSeal watermark), decoded bit errors follow a symmetric binomial distribution with success probability $p_0 = 0.5$. For an observed count of $k$ agreeing bits out of $N = 16$ independent trials, the exact two-tailed $p$-value is defined mathematically as:
$$P(X \ge k) = \sum_{j=k}^{N} \binom{N}{j} (0.5)^N$$
$$p\text{-value} = 2 \cdot \min\left( P(X \ge k), \; 1 - P(X \ge k) + \binom{N}{k}(0.5)^N \right)$$

Evaluating across the discrete support of $N = 16$ independent trials:
- $k = 16$: $p = 2 \cdot (0.5)^{16} \approx 3.0518 \times 10^{-5}$ ($p < 10^{-4} \implies$ High-Confidence Watermark Assertion).
- $k = 15$: $p = 2 \cdot [16 + 1] \cdot 2^{-16} \approx 5.188 \times 10^{-4}$.
- $k = 14$: $p = 2 \cdot [120 + 16 + 1] \cdot 2^{-16} \approx 4.181 \times 10^{-3}$ ($p < 0.005$).
- $k = 12$: $p = 2 \cdot [1820 + 560 + 120 + 16 + 1] \cdot 2^{-16} = \frac{5034}{65536} \approx 0.0768$ ($p \approx 0.08$).
- $k = 8$: $p = 1.0$ (Expected under random chance, zero evidence).

**Detection Criterion & Statistical Discrepancy Note**:
On $N = 16$ trials, the condition $p\text{-value} < 10^{-4}$ is satisfied **exclusively** when all 16 bits agree ($k = 16$, $p \approx 3.05 \times 10^{-5}$). An operating threshold of $k \ge 12$ corresponds to a relaxed significance level $\alpha \approx 0.08$ (or $k \ge 14$ for $\alpha \approx 0.005$).

Furthermore, in the Android codebase (`AudioSealDetector.kt:122`), detection asserts:
$$\text{detected} = k \ge 8 \quad \land \quad p\text{-value} < \text{ProvenanceEvidence.SIGNIFICANT\_P\_VALUE}$$
where `PRESENCE_THRESHOLD = 0.50f` and discrete significance uses $\alpha = 0.05$ with Bonferroni multiple testing correction over $M$ analyzed windows: $p_{\text{adj}} = \min(1.0, p \times M)$. Furthermore, the export contract has been upgraded to emit normalized continuous presence probability $\in [0, 1]$ (`audioseal-presence-mean-v2`) via temporal mean of the presence detector, separating it from the multi-bit payload decoder. Crucially, watermark detection indicates watermark presence rather than conclusive deepfake proof (authentic audio can be watermarked), so `WatermarkMark.confirmsAiInvolvement` returns `false` while absence remains strictly neutral (`ProvenanceEvidence.Absent`).

### The Absence Invariant
A fundamental requirement of forensic truthfulness is the **Absence Invariant**:
$$\text{WatermarkDetected} = \text{False} \implies \text{Evidence} = \text{Absent}$$
The absence of a watermark provides **zero negative evidence** against authenticity. Millions of authentic recordings exist without neural watermarks. Therefore:
$$P_{\text{final}} = P_{\text{statistical}}$$
An unwatermarked sample is judged solely by passive acoustic and visual models without receiving any artificial score penalty.

## 2.4 The Watermark Shortcut Crisis & WASP Decorrelation Protocol

Modern voice synthesis services (ElevenLabs, PlayHT, DramaBox) embed watermarks by default. If a forensic training corpus compiles synthetic samples directly from commercial APIs while using clean, unwatermarked speech for the authentic class, a lethal dataset vulnerability emerges: the **Watermark Shortcut** (Müller & Debus, arXiv:2606.23335).

When an acoustic model (e.g., LCNN or AASIST) trains on such data, the gradient optimizer discovers that detecting the high-frequency watermark carrier minimizes cross-entropy loss far more efficiently than modeling complex vocal tract physics.

This shortcut induces two catastrophic failure modes:
1. **Adversarial Framing Attack**: An adversary takes an authentic speech recording of an innocent victim and injects a low-amplitude AudioSeal or ElevenLabs watermark. The detector flags the audio as deepfake with $99.9\%$ confidence, raising the Equal Error Rate (EER) from $16.4\%$ to **$74.6\%$**.
2. **Watermark Stripping Evasion**: Applying a narrow notch filter or lossy compression (Opus at $12\text{ kbps}$) strips the watermark carrier. The detector immediately misclassifies the synthetic speech as authentic (False Negative Rate $> 80\%$).

**The WASP Mitigation Protocol**: To eliminate this shortcut, ONN-DETECT enforces the runtime absence-neutrality invariant inspired by WASP (Watermark-Aware Speech Processing, Müller & Debus, arXiv:2606.23335), and defines the following target decorrelation training protocol for future acoustic retrainings:
$$\mathcal{D}_{\text{train}} = \{ (x_i, y_i, w_i) \}$$
where watermark carrier $w \sim \text{AudioSeal}$ is injected into both classes with balanced marginal probabilities:
$$\mathbb{P}(w = 1 \mid y = 0) = 0.5, \quad \mathbb{P}(w = 1 \mid y = 1) = 0.5$$
Because the mutual information $I(W; Y) = 0$, the acoustic classifier is mathematically prevented from utilizing the watermark carrier as a predictive feature for the class label $Y$.

## 2.5 Passive Speech Anti-Spoofing: LCNN-MFM & Versioned LFCC Frontend

For unwatermarked audio, ONN-DETECT deploys a quantized Light Convolutional Neural Network with Max-Feature-Map activation (LCNN-MFM, $508\text{ KiB}$ INT8, `models/lcnn_audio_int8.tflite`).

```
Raw Audio (16 kHz PCM)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Versioned LFCC Frontend (`lfcc-v3`)                    │
│ - Quadratic-Window Sinc Resampling (to 16,000 Hz)     │
│ - Periodic Hann STFT (NFFT = 512, Hop = 160)          │
│ - 128 Linear-Frequency Filterbank                     │
│ - Orthonormal Discrete Cosine Transform (40 LFCCs)    │
│ - Cepstral Mean and Variance Normalization (CMVN)     │
└────────────────────────────────────────────────────────┘
       │
       ▼ Output Tensor: [1, 401, 40, 1]
┌────────────────────────────────────────────────────────┐
│ LCNN-MFM Backbone (Max-Feature-Map)                    │
│ - Conv2D -> Max-Feature-Map (MFM splits & max)        │
│ - Network-in-Network (NIN) Dimensionality Reduction    │
│ - Bi-directional Residual Connections                  │
│ - Dense(1, activation="sigmoid")                       │
└────────────────────────────────────────────────────────┘
       │
       ▼
 Calibrated Spoofing Probability P(spoof) ∈ [0, 1]
```

### The LFCC Frontend (`lfcc-v3`) Specification
1. **Input Resampling**: Audio is resampled to $16\text{ kHz}$ using quadratic-window sinc interpolation with reflective boundary handling.
2. **Short-Time Fourier Transform (STFT)**:
   - Window size: $N_{\text{FFT}} = 512$ samples ($32\text{ ms}$).
   - Hop length: $R = 160$ samples ($10\text{ ms}$).
   - Window function: Periodic Hann window.
   - Power spectrum: $P(f, t) = |X(f, t)|^2$ computed in double precision (`float64`).
3. **Linear Filterbank Integration**:
   - 128 triangular filters spaced linearly from $0\text{ Hz}$ to the Nyquist frequency $8000\text{ Hz}$.
   - Logarithmic energy compression in decibels (`AmplitudeToDB('power', top_db=None)` matching `LFCCExtractor.kt:111`):
     $$S(m, t) = 10 \log_{10}\left( \max\left( \sum_k P(k, t) H_m(k), \; 10^{-10} \right) \right)$$
4. **Discrete Cosine Transform (DCT-II Orthonormal)**:
   - 40 static cepstral coefficients extracted via orthonormal DCT-II (`LFCCExtractor.kt:173–184`):
     $$c_n(t) = \alpha_n \sum_{m=0}^{M-1} S(m, t) \cos\left( \frac{\pi n (2m + 1)}{2M} \right), \quad n \in \{0, \dots, 39\}$$
     where $M = 128$ filters and the orthonormal normalization factor $\alpha_n$ is:
     $$\alpha_n = \begin{cases} \sqrt{\frac{1}{M}} = \frac{1}{\sqrt{2}} \sqrt{\frac{2}{M}}, & n = 0 \\ \sqrt{\frac{2}{M}}, & n \in \{1, \dots, 39\} \end{cases}$$
5. **Cepstral Mean and Variance Normalization (CMVN)**:
   - Evaluated per-coefficient across the time axis:
     $$\hat{c}_n(t) = \frac{c_n(t) - \mu_n}{\sigma_n + \epsilon}$$
   - The resulting tensor shape for a 4-second audio window is $[1, 401, 40, 1]$ float32.

### Max-Feature-Map (MFM) Activation
Standard ReLU activations propagate positive unbounded noise. In contrast, MFM acts as a continuous competitive selection filter:
$$\text{MFM}(x)_k = \max\left( x_{2k-1}, \; x_{2k} \right)$$
Halving channel dimensions via competitive maximum suppresses out-of-distribution acoustic anomalies and separates genuine vocal resonances from vocoder synthesis artifacts.

## 2.6 Dual-Stage Voiced Duration Gating & Multi-Window Logit Pooling

### Dual-Stage Voiced Duration Gate (`VoicedDurationGate.kt`)
Detectors evaluated on non-speech audio (acoustic noise, background music, office chatter) frequently generate false alarms. To prevent misclassification, incoming audio must clear a strict two-stage speech gate:
1. **Spectral Flux & Zero-Crossing Rate (ZCR)**: Audio frames ($20\text{ ms}$) are analyzed for energy distribution within human vocal fundamental frequencies ($85\text{--}255\text{ Hz}$).
2. **Voiced Duration Threshold**:
   $$\text{VoicedMs} = \sum_{t=1}^{T} \mathbb{I}(\text{FrameVoiced}(t)) \times 20\text{ ms}$$
   $$\text{VoicedMs} \ge 300\text{ ms} \implies \text{Proceed to LCNN}$$
   $$\text{VoicedMs} < 300\text{ ms} \implies \text{Emit } \text{DetectionVerdict.INCONCLUSIVE}$$

### Log-Sum-Exp Multi-Window Pooling (`WindowPooling.kt`)
For audio recordings exceeding 4 seconds, sliding-window analysis extracts probabilities $\{ p_1, p_2, \dots, p_W \}$. To prevent single-frame vocoder glitches from dominating while preserving high-confidence localized spoofing signals, probabilities are pooled in logit space using temperature-scaled Log-Sum-Exp (`WindowPooling.kt:58–59`):
$$z_i = \text{logit}(p_i) = \ln\left( \frac{p_i}{1 - p_i} \right)$$
$$z_{\text{pooled}} = \tau \ln\left( \frac{1}{W} \sum_{i=1}^{W} \exp\left( \frac{z_i}{\tau} \right) \right)$$
$$P_{\text{pooled}} = \sigma(z_{\text{pooled}}) = \frac{1}{1 + \exp(-z_{\text{pooled}})}$$
Numerically evaluated with subtraction of $z_{\max} = \max_j z_j$ to prevent floating-point overflow:
$$z_{\text{pooled}} = z_{\max} + \tau \ln\left( \frac{1}{W} \sum_{i=1}^W \exp\left( \frac{z_i - z_{\max}}{\tau} \right) \right)$$
With temperature $\tau = 3.0$: as $\tau \to 0^+$, $z_{\text{pooled}} \to \max_i(z_i)$ (concentrating on the most spoofed segment); as $\tau \to \infty$, $z_{\text{pooled}} \to \frac{1}{W} \sum_{i=1}^W z_i$ (arithmetic mean). The setting $\tau = 3.0$ acts as a soft maximum that remains sensitive to brief spliced voice clones while resisting single-frame boundary false alarms.

---

# Chapter 3: Vision Foundation Probes

## 3.1 ViT Representations vs. Handcrafted Spatial Residuals

In early deepfake literature, spatial convolutional networks (e.g., MesoNet, Xception, ResNet-50) were trained end-to-end to classify real versus synthetic imagery. As established in §1.1, these networks heavily overfit to generator-specific artifact frequencies and perform poorly when confronted with unseen generative pipelines.

Vision Foundation Models (VFMs)—specifically Vision Transformers trained via contrastive language-image pre-training (SigLIP) or self-supervised feature alignment (DINOv2)—learn semantically rich, invariant visual manifolds. By freezing the foundational ViT backbone and attaching a specialized forensic probe head, the detector leverages pre-trained semantic priors (lighting consistency, physical anatomy, reflection symmetry) while retaining high computational efficiency on mobile devices.

## 3.2 SigLIP-B/16 Architectural Backbone & Token Extraction

ONN-DETECT utilizes `google/siglip-base-patch16-224` (86M parameters, quantized to $89.96\text{ MB}$ INT8, `models/siglip_probe_int8.tflite`).

1. **Input Tensor**: Normalized RGB tensor $X \in \mathbb{R}^{1 \times 224 \times 224 \times 3}$, scaled with mean $[0.5, 0.5, 0.5]$ and standard deviation $[0.5, 0.5, 0.5]$.
2. **Patch Projection**: Non-overlapping $16 \times 16$ convolutional stem mapping spatial inputs to $14 \times 14 = 196$ patch tokens:
   $$Z_0 = \text{Unfold}(X) W_{\text{patch}} + E_{\text{pos}} \in \mathbb{R}^{196 \times 768}$$
3. **Transformer Encoder Blocks**: 12 stacked Multi-Head Self-Attention (MHSA) and MLP layers equipped with pre-LayerNorm and residual connections:
   $$\hat{Z}_\ell = \text{MHSA}(\text{LN}(Z_{\ell-1})) + Z_{\ell-1}$$
   $$Z_\ell = \text{MLP}(\text{LN}(\hat{Z}_\ell)) + \hat{Z}_\ell, \quad \ell \in \{1, \dots, 12\}$$

## 3.3 Tunable Attention Pooling (TAP) Across 196 Spatial Patch Tokens

Standard classification heads either perform unweighted average pooling across all patch tokens or extract the single `[CLS]` token. Both approaches lose spatial anomaly localization:
- **Average Pooling**: Dilutes localized synthesis anomalies (e.g., distorted iris, inconsistent teeth, warped earlobes) across 196 tokens.
- **`[CLS]` Token**: Overfits to global scene semantics rather than fine-grained generation boundaries.

**Tunable Attention Pooling (TAP, CVPRW 2026)** introduces a learnable forensic query probe vector $q_{\text{tap}} \in \mathbb{R}^{1 \times 768}$ that selectively queries the final patch token sequence $Z_{12} \in \mathbb{R}^{196 \times 768}$:
$$Q = q_{\text{tap}} W_Q, \quad K = Z_{12} W_K, \quad V = Z_{12} W_V$$
As implemented in the Multi-Head Attention Pooling head (`_SiglipMAPAttention`, `ml/src/export/keras_vfm_probe.py:196–224`), cross-attention operates across $H = 12$ attention heads with per-head projection dimension $d_k = d_{\text{head}} = 768 / 12 = 64$:
$$A^{(h)} = \text{Softmax}\left( \frac{Q^{(h)} (K^{(h)})^T}{\sqrt{d_k}} \right) \in \mathbb{R}^{1 \times 196}, \quad h \in \{1, \dots, 12\}$$
where $d_k = 64$ and the query scaling factor folded at export time is $\frac{1}{\sqrt{d_k}} = \frac{1}{\sqrt{64}} = \frac{1}{8} = 0.125$. The multi-head context vectors are concatenated:
$$h_{\text{tap}} = \text{Concat}\left( A^{(1)} V^{(1)}, \; \dots, \; A^{(12)} V^{(12)} \right) W_O \in \mathbb{R}^{1 \times 768}$$
$$P(\text{AI}) = \sigma(h_{\text{tap}} W_{\text{head}} + b_{\text{head}})$$
The attention distribution $A$ dynamically assigns higher weights to spatial patches containing unnatural generation artifacts.

## 3.4 Zero-Overhead Keras Mirror Folding

In `ml/src/export/keras_vfm_probe.py`, the native SigLIP architecture implements Multi-Head Attention Pooling (`_SiglipMAPAttention`) over patch sequences.

Because the query probe $q_{\text{tap}}$, key projection $W_K$, and value projection $W_V$ are completely frozen post-training, the query representation is folded directly into the pre-trained weights at export time:
$$Q_{\text{const}} = q_{\text{tap}} W_Q \in \mathbb{R}^{1 \times 768}$$
*Deployment Status Note*: In the shipped release (`onn-prod-v10`), `siglip_probe_int8.tflite` (hash `1b8c4f3de7822196...`) packages the pre-trained SigLIP-B/16 tower with a folded linear attention probe (`head_type: linear_probe`). The zero-overhead TAP architecture and PatchHead/PRL spatial layers are implemented in `train_siglip_tap.py` as candidate research configurations documented in `docs/EVIDENCE_CHAIN.md`.

## 3.5 Degradation-Consistent Paired Training (DCPT) & Laundered Social Media Robustness

Detectors deployed in consumer messaging apps (WhatsApp, Telegram, Instagram) must process heavily recompressed, downscaled media. Standard data augmentation during training modifies individual samples independently, which fails to teach the model invariance across degradation boundaries.

**Degradation-Consistent Paired Training (DCPT; Yang, Hou & Yang, arXiv:2604.10102)** enforces consistency across clean image $x$ and degraded image $\tilde{x} = \text{Aug}(x)$ using a joint multi-task objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(z_{\text{clean}}, y) + \lambda_{\text{logit}} \mathcal{L}_{\text{KL}}(z_{\text{clean}}, z_{\text{deg}}) + \lambda_{\text{feat}} \mathcal{L}_{\text{feat}}(e_{\text{clean}}, e_{\text{deg}})$$

```
Clean Image x ────────► SigLIP Backbone ──► Features e_clean ──► Logit z_clean ──► BCE Loss
       │                                            │                   │
  Augmentation                                      │                   │
       ▼                                            ▼                   ▼
Degraded Image x̃ ─────► SigLIP Backbone ──► Features e_deg   ──► Logit z_deg   ──► KL Div Loss
                                                    │
                                                    ▼
                                            Cosine Dist Loss
```

### 1. Symmetric Binary Bernoulli KL Divergence
A common mathematical defect in naïve implementations is using single-output KL divergence $F.kl\_div(p, q)$, which computes only $p \ln(p/q)$ and discards the negative class outcome $(1-p)$. This can yield negative loss values when $p \approx 0.01$ and $q \approx 0.40$.

DCPT enforces the full symmetric Bernoulli divergence over both outcomes $\{p, 1-p\}$:
$$p = \sigma\left(\frac{z_{\text{clean}}}{\tau}\right), \quad q = \sigma\left(\frac{z_{\text{deg}}}{\tau}\right)$$
$$\text{KL}(p \parallel q) = p \ln\frac{p}{q} + (1-p) \ln\frac{1-p}{1-q}$$
$$\mathcal{L}_{\text{KL}} = \frac{1}{2} \left[ \text{KL}(p \parallel q) + \text{KL}(q \parallel p) \right] \cdot \tau^2$$
The $\tau^2$ scaling factor restores gradient magnitude under temperature scaling ($\tau = 1.0$).

### 2. Feature-Level Cosine Consistency
To force internal representations to ignore compression block boundaries:
$$\mathcal{L}_{\text{feat}}(e_{\text{clean}}, e_{\text{deg}}) = 1 - \frac{e_{\text{clean}} \cdot e_{\text{deg}}}{\|e_{\text{clean}}\|_2 \|e_{\text{deg}}\|_2}$$
DCPT yields **$+9.1\text{ percentage points}$** gain in degraded-average accuracy with zero inference overhead.

## 3.6 Evaluation of Foundation Backbones: Why MobileCLIP Fails and SigLIP/DINOv2 Generalize

During empirical evaluation (`docs/RESULTS_2026-10-20.md`), multiple foundation backbones were tested under identical cross-generator protocols:

| Teacher Backbone | Parameters | INT8 Size | Stacking Fit AUC | Sealed-Test Held-Out AUC | Verdict |
|---|---|---|---|---|---|
| **MobileCLIP2-S0** | 11.4 M | 11.4 MB | 0.9688 | **0.5145** | **Chance Level (Memorization)** |
| **DINOv2-S/14 (+ registers)** | 21.0 M | 21.2 MB | 0.9780 | **0.8358** | **Transfers Well** |
| **DINOv2-B/14** | 86.0 M | 86.5 MB | 1.0000 | **0.8541** | **Strong Generalization** |
| **SigLIP-B/16 (224px)** | 86.0 M | **89.96 MB** | 1.0000 | **0.8919** | **SOTA Transfer (Shipped)** |
| Distilled Student CNN | 0.15 M | 0.15 MB | — | 0.6871 | Distillation loss $-0.205$ AUC |

**Analysis**: MobileCLIP2-S0 memorized the 800-sample fitting split ($0.9688$ AUC) but collapsed to chance ($0.5145$) on held-out test data. In contrast, SigLIP-B/16 achieved **$0.8919$ held-out test AUC** (packaged INT8: **$0.8910$**), proving that large-scale contrastive pre-training on web-scale image-text pairs creates robust forensic representations.

---

# Chapter 4: Spatial-Frequency Residuals

## 4.1 The Anti-Aliasing Destruction Problem in Standard Downsampling

When an Android application ingests a high-resolution camera photograph or social media image ($1024 \times 1024$ or $4000 \times 3000$ pixels) and resizes it to match a neural network's input tensor (e.g., $128 \times 128$), standard graphics libraries utilize bilinear or bicubic downsampling filters.

These downsampling filters function mathematically as **anti-aliasing low-pass spatial filters**:
$$I_{\text{down}}(x, y) = \sum_{i} \sum_{j} I(x \cdot s + i, y \cdot s + j) \cdot W(i, j)$$
This decimation operation obliterates the fine $2 \times 2$ pixel upsampling checkerboard residuals left by diffusion decoders and GAN transposed convolutions. Downsampling an image before applying high-pass spatial filters destroys the exact forensic evidence the filters were designed to measure.

## 4.2 Native PatchCraft 4x4 Texture Mosaics

To preserve high-frequency sampling residues without exceeding mobile compute and memory budgets, ONN-DETECT integrates the **PatchCraft** native mosaic extraction algorithm (Zhong et al., arXiv:2311.12397; `-patchcraft-mosaic-tile32`):

```
Native Image I ∈ ℝ^(H × W × 3) (e.g., 2048 × 1536)
       │
       ▼
Non-overlapping 32×32 Tile Partitioning: {T_1, T_2, ..., T_M}
       │
       ▼
Compute L1 Gradient Energy: E(T_i) = ∑ (|Δ_x| + |Δ_y|)
       │
       ▼
Select Top-16 Maximum Energy Tiles: S_rich = Top16_E(T_i)
       │
       ▼
Assemble into 4×4 Grid (Uninterpolated):
┌────────┬────────┬────────┬────────┐
│ Tile 1 │ Tile 2 │ Tile 3 │ Tile 4 │
├────────┼────────┼────────┼────────┤
│ Tile 5 │ Tile 6 │ Tile 7 │ Tile 8 │
├────────┼────────┼────────┼────────┤
│ Tile 9 │ Tile 10│ Tile 11│ Tile 12│
├────────┼────────┼────────┼────────┤
│ Tile 13│ Tile 14│ Tile 15│ Tile 16│
└────────┴────────┴────────┴────────┘
  128×128 Native Texture Mosaic M_rich
```

### Algorithmic Execution:
1. Decode image at native resolution: $I \in \mathbb{R}^{H \times W \times 3}$.
2. Subdivide into non-overlapping $32 \times 32$ pixel patches $\{ T_i \}_{i=1}^M$, where $M = \lfloor H/32 \rfloor \times \lfloor W/32 \rfloor$.
3. Compute L1 gradient energy for each patch:
   $$E(T_i) = \sum_{x, y} \left( |I(x+1, y) - I(x, y)| + |I(x, y+1) - I(x, y)| \right)$$
4. Rank patches by texture richness and extract the top-16 patches: $S_{\text{rich}} = \text{Top16}_{E}(T_i)$.
5. Assemble the 16 patches without resizing into a $4 \times 4$ grid yielding a single $128 \times 128 \times 3$ mosaic tensor $M_{\text{rich}}$.
6. Feed $M_{\text{rich}}$ directly into `srm_forensic_cnn_int8.tflite`. High-frequency synthesis artifacts are preserved at native sensor resolution.

## 4.3 20-Channel Fixed Residual Kernel Stem (SRM + NPR + RGB)

The input layer of `srm_forensic_cnn_int8.tflite` (142 KiB) consists of a fixed, non-trainable 20-channel convolutional stem with shape $(20, 5, 5, 3)$:

1. **Channels 0–11 (12 SRM High-Pass Kernels)**:
   Derived from Spatial Rich Models (Fridrich & Kodovsky, IEEE TIFS). Includes 1st-order, 2nd-order, 3rd-order, square, and edge Laplacians:
   $$K_{\text{SRM1}} = \begin{bmatrix} 0 & 0 & 0 & 0 & 0 \\ 0 & -1 & 2 & -1 & 0 \\ 0 & 2 & -4 & 2 & 0 \\ 0 & -1 & 2 & -1 & 0 \\ 0 & 0 & 0 & 0 & 0 \end{bmatrix}, \quad K_{\text{SRM2}} = \begin{bmatrix} -1 & 2 & -2 & 2 & -1 \\ 2 & -6 & 8 & -6 & 2 \\ -2 & 8 & -12 & 8 & -2 \\ 2 & -6 & 8 & -6 & 2 \\ -1 & 2 & -2 & 2 & -1 \end{bmatrix}$$
2. **Channels 12–15 (4 NPR Upsampling Kernels)**:
   Neighboring Pixel Relationship kernels (CVPR 2024, arXiv:2312.10461) specifically tuned to detect $2 \times 2$ transposed convolution periodicity.
3. **Channels 16–19 (4 RGB/Luma Low-Pass Kernels)**:
   Centered Gaussian and identity stems providing baseline illumination context to prevent pure noise false positives on high-ISO authentic photography.

## 4.4 Rich-vs-Poor Texture Contrast & Sliding-Window Anomaly Hotspot Localization

- **Rich-vs-Poor Contrast (`analyzeWithContrast`)**: Evaluates the ratio between the top-16 rich-texture mosaic score and the bottom-16 poor-texture mosaic score. Authentic images show high residual variance across texture boundaries, whereas generative diffusion images exhibit uniform synthetic residual energy across both flat and textured regions.
- **Sliding-Window Hotspot Localization (`extractHotspotBoundingBox`)**: An $8 \times 8$ sliding patch window (stride $s=16$) computes localized anomaly energy across the image. The peak energy coordinate is projected onto the UI as an interactive **Tampering X-Ray Reticle** (`HeatmapViewer.kt`), localizing deepfake inpainting and face-swaps.

---

# Chapter 5: Biomechanical & Physiological Dynamics

## 5.1 Category Errors in Video Forensics: Why Per-Frame Aggregation Fails

A pervasive architectural failure in video forensics is evaluating independent per-frame image classifiers and calculating an arithmetic mean:
$$S_{\text{video}} = \frac{1}{N} \sum_{t=1}^N S_{\text{image}}(I_t)$$
This approach commits a fundamental category error:
1. **Flicker Sensitivity**: Authentic video frames subject to motion blur or high compression are falsely flagged by static spatial models.
2. **Temporal Blindness**: Autoregressive and diffusion video generators can produce individually pristine frames that are completely desynchronized in the time domain, exhibiting physical velocity and acceleration discontinuities that per-frame spatial models cannot detect.

## 5.2 BioLip: Biomechanical 40-Point Perioral Jitter on MediaPipe 478-Point FaceMesh

Human facial articulators (lips, jaw, buccinator muscles) are governed by physical biomechanics: tissue elasticity, muscle contraction latency, and inertial mass. Deepfake talking-head generators (Wav2Lip, SadTalker, Diff2Lip, HeyGen) synthesize mouth movements frame-by-frame, causing unnatural micro-jitter and acceleration spikes (adapted from the LADDER landmark-kinematics framework; Chen & Xu, arXiv:2604.16808).

```
Video Frame Sequence I_t (N ≥ 3)
       │
       ▼
MediaPipe 478-Point Face Mesh Tracking
       │
       ▼
Extract 40 Canonical Lip Landmarks p_t^(i) (LIP_INDICES)
       │
       ▼
2D Pose Normalization via Eye Corners (33/263) & Nose Tip (1):
p̂_t^(i) = R_t (p_t^(i) - p_t^(1)) / || p_t^(33) - p_t^(263) ||_2
       │
       ▼
Contiguous Triplet Acceleration & MAR Jerk:
a_t^(i) = p̂_(t+1)^(i) - 2 p̂_t^(i) + p̂_(t-1)^(i)
MARJerk_t = | MAR_(t+1) - 2 MAR_t + MAR_(t-1) |
       │
       ▼
Linear Combination & Clamped Normalization:
Jitter = clamp( (0.6 * meanAccel + 0.4 * meanMarJerk) / 0.08, 0.0, 1.0 ) ──► Slot 9
```

### Algorithmic Execution (`PerioralLandmarkAnalyzer.kt:38–140`):
1. **Facial Tracking**: Extract 478 MediaPipe landmarks $\{ p_t^{(i)} \}_{i=1}^{478}$ for frame $t$.
2. **Canonical Lip Landmarks**: Extract exactly 40 canonical inner and outer lip contour and commissure landmarks (`LIP_INDICES`):
   $$\{61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,$$
   $$78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415\}$$
3. **Pose Normalization**: Normalize for head rotation (roll angle $\alpha$) and distance using inter-ocular distance $D_{\text{IOD}}(t) = \| p_t^{(33)} - p_t^{(263)} \|_2$ relative to nose tip $p_t^{(1)}$:
   $$\hat{p}_t^{(i)} = \frac{R(\alpha) (p_t^{(i)} - p_t^{(1)})}{D_{\text{IOD}}(t)}, \quad i \in \text{Lip40}$$
4. **Discrete Second-Order Kinematics**: Evaluated strictly across contiguous triplets $(t-1, t, t+1)$:
   $$a_t^{(i)} = \hat{p}_{t+1}^{(i)} - 2\hat{p}_t^{(i)} + \hat{p}_{t-1}^{(i)}, \quad \bar{a}_t = \frac{1}{40} \sum_{i=1}^{40} \| a_t^{(i)} \|_2$$
   $$\text{MARJerk}_t = | \text{MAR}_{t+1} - 2\text{MAR}_t + \text{MAR}_{t-1} |$$
5. **Linear Normalized Jitter Score**:
   $$\text{raw} = 0.6 \cdot \left( \frac{1}{T} \sum_{t=1}^T \bar{a}_t \right) + 0.4 \cdot \left( \frac{1}{T} \sum_{t=1}^T \text{MARJerk}_t \right)$$
   $$S_{\text{jitter}} = \text{clamp}\left( \frac{\text{raw}}{0.08}, \; 0.0, \; 1.0 \right)$$
   Assigned directly to feature slot 9 of `XGBoostMetaClassifier` (or `Float.NaN` if fewer than 3 contiguous valid frames exist). *Production Decoupling Note*: In accordance with empirical validation findings, perioral landmark jitter is decoupled from the final production decision (`enableExperimentalHeuristicBoost = false`) and recorded strictly as diagnostic telemetry in `Outcome` to prevent uncalibrated heuristic inflation.

## 5.3 Bilateral Malar rPPG Chrominance Phase Coherence

Remote photoplethysmography (rPPG) tracks the physiological blood volume pulse (BVP) caused by cardiovascular cardiac contractions pumping oxygenated hemoglobin through facial capillary beds.

In an authentic human face, blood reaches the left cheek malar vascular bed and right cheek malar vascular bed synchronously, exhibiting high bilateral phase coherence ($\rho_{\text{malar}} \ge 0.55$). Deepfakes and face-swaps synthesize cheek skin patches independently, destroying this biological phase synchrony.

### Algorithmic Formulation (`PerioralLandmarkAnalyzer.kt:47–48, 275–365, 411–415`):
1. **Cheek ROI Landmark Indices**: Malar capillary patches are tracked using canonical MediaPipe face mesh landmarks:
   - `LEFT_MALAR_INDICES` = `[116, 117, 118, 123, 187, 205]` (6 points)
   - `RIGHT_MALAR_INDICES` = `[345, 346, 347, 352, 411, 425]` (6 points)
2. **Green-Ratio Chromaticity Extraction**:
   For each landmark, a $3 \times 3$ pixel spatial neighborhood is sampled. For valid skin pixels ($R + G + B > 15.0$):
   $$C_g(t) = \frac{G(t)}{R(t) + G(t) + B(t)}$$
   Averaged across malar points to yield bilateral traces $C_{g, L}(t)$ and $C_{g, R}(t)$.
3. **Contiguous Run Timeline Partitioning**:
   The video timeline is partitioned into contiguous uninterrupted tracked runs. Sequences with fewer than 4 contiguous frames abstain truthfully by returning `Float.NaN`.
4. **First-Difference Detrending**:
   On the longest contiguous run of length $N \ge 4$:
   $$\Delta C_{g, L}(t) = C_{g, L}(t+1) - C_{g, L}(t), \quad \Delta C_{g, R}(t) = C_{g, R}(t+1) - C_{g, R}(t)$$
   Detrending eliminates slow ambient lighting fluctuations and automatic exposure drift.
5. **Bilateral Pearson Correlation Across Contiguous Runs ($\ge 4$ frames)**:
   $$\rho_{\text{malar}} = \text{Corr}(\Delta C_{g, L}, \; \Delta C_{g, R}) \in [-1.0, +1.0]$$
6. **Physiological Phase Anomaly**:
   $$\text{Anomaly}_{\text{rPPG}} = \text{clamp}\left( \frac{0.55 - \rho_{\text{malar}}}{0.85}, \; 0.0, \; 1.0 \right)$$
   Authentic synchronized pulses ($\rho \ge 0.55$) yield anomaly $\approx 0.0$; desynchronized deepfakes ($\rho \le -0.30$) yield anomaly $\to 1.0$. *Production Decoupling Note*: Green chromaticity rPPG is an experimental lightweight adaptation inspired by CHROM; it is decoupled from the production decision (`enableExperimentalHeuristicBoost = false`) and recorded strictly as diagnostic telemetry in `Outcome`.

## 5.4 Audio-Visual Lip-Sync Cross-Correlation (AVFF MAR vs. Audio RMS Envelope)

To detect dubbed speech and decoupled voice clones, ONN-DETECT evaluates cross-modal synchrony:
1. **Mouth Aspect Ratio (MAR)**:
   $$\text{MAR}(t) = \frac{\| p_t^{(13)} - p_t^{(14)} \|_2}{\| p_t^{(78)} - p_t^{(308)} \|_2}$$
2. **Audio RMS Energy Envelope**: Audio PCM sampled within frame time window $[t - \Delta t, t + \Delta t]$:
   $$\text{RMS}(t) = \sqrt{\frac{1}{K} \sum_{k=1}^K x(k)^2}$$
3. **Lag Window Cross-Correlation**: Evaluated across lag window $\tau \in [-2, +2]$ frames:
   $$\rho_{\text{AVFF}} = \max_{\tau \in [-2, +2]} \text{Corr}(\text{MAR}(t), \; \text{RMS}(t + \tau))$$
   Authentic speech yields high positive correlation ($\rho \approx 0.9$); desynchronized audio yields low correlation ($\rho \approx 0.0$). Slot 8 receives $\rho_{\text{AVFF}}$ directly.

## 5.5 Multimodal Video Score Escalation Policy (`VideoScorePolicy`)

In video analysis, combining multimodal signals requires safeguarding against false alarms while ensuring high-confidence anomalies escalate the verdict:

```kotlin
// VideoForensicAnalyzer.kt:31–89 — VideoScorePolicy Invariant Implementation
object VideoScorePolicy {
    const val FRAME_WEIGHT = 0.65f
    const val AUDIO_WEIGHT = 0.20f
    const val TEMPORAL_WEIGHT = 0.15f

    fun combineMultimodalScore(
        visualScore: Float,
        visualThreshold: Float,
        audioScore: Float = Float.NaN,
        audioValidatedSuccess: Boolean = false,
        audioThreshold: Float = 0.35546875f,
        temporalScore: Float = Float.NaN,
        lipSyncAnomaly: Float = 0f,
    ): Float {
        if (!visualScore.isFinite() || visualScore !in 0f..1f) return Float.NaN
        var base = visualScore.coerceIn(0f, 1f)

        // 1. Audio escalation: synthetic audio projects across visual threshold
        // Guard against IEEE 754 NaN contamination: thresholds must be valid finite probabilities in [0, 1]
        if (visualThreshold.isFinite() && visualThreshold in 0f..1f &&
            audioThreshold.isFinite() && audioThreshold in 0f..1f &&
            audioScore.isFinite() && audioScore in 0f..1f &&
            audioScore >= audioThreshold
        ) {
            val mappedAudio = if (audioThreshold < 0.99f) {
                visualThreshold + ((audioScore - audioThreshold) / (1f - audioThreshold)) * (1f - visualThreshold)
            } else {
                audioScore
            }
            if (mappedAudio.isFinite()) {
                val weight = if (audioValidatedSuccess) 0.70f else 0.35f
                val candidate = (1f - weight) * base + weight * mappedAudio.coerceIn(0f, 1f)
                if (candidate.isFinite() && candidate > base) {
                    base = candidate
                }
            }
        }

        // 2. Temporal and lip-sync anomaly boost (capped at +0.30)
        val tempAnomaly = if (temporalScore.isFinite() && temporalScore > 0.30f) {
            ((temporalScore - 0.30f) / 0.70f).coerceIn(0f, 1f)
        } else 0f
        val lipAnomaly = if (lipSyncAnomaly.isFinite() && lipSyncAnomaly > 0f) {
            lipSyncAnomaly.coerceIn(0f, 1f)
        } else 0f

        val boost = (TEMPORAL_WEIGHT * tempAnomaly + 0.15f * lipAnomaly).coerceIn(0f, 0.30f)
        return (base + (1f - base) * boost).coerceIn(0f, 1f)
    }
}
```

### Key Mathematical Guarantees:
- **Zero NaN Collapse**: If `visualThreshold` or `audioThreshold` is uncalibrated (`NaN`), audio escalation cleanly bypasses without poisoning `base`.
- **Authentic Immunity**: Authentic audio ($audioScore < audioThreshold$) cannot drag down a high visual synthetic score.
- **Monotonic Candidate Update**: `candidate > base` guarantees that secondary modalities only escalate the synthetic evidence.

---

# Chapter 6: Calibration, Conformal Prediction & Missing Modalities

## 6.1 The Mathematics of Missing Evidence (CLAUDE.md Truthfulness Contract)

A central invariant of ONN-DETECT is the strict distinction between **zero evidence** and **absent evidence**:
$$\text{Measured Evidence} = 0.0 \implies \text{Affirmative proof of authenticity}$$
$$\text{Unmeasured Evidence} = \text{NaN} \implies \text{Modality was not evaluated}$$

If an unmeasured modality (e.g., audio score on a silent video, or lip-sync on an image) is set to a constant (e.g., $0.0$ or $0.5$), gradient boosted decision trees split on that value, creating false branches. Under ONN-DETECT:
1. Missing features are stored as `Float.NaN`.
2. XGBoost routes `Float.NaN` through its mathematically learned missing branch.
3. Attestation receipts serialize `Float.NaN` to JSON `null`.
4. If critical required models fail, the engine abstains by emitting `DetectionVerdict.INCONCLUSIVE`.

## 6.2 Mondrian Conformal Prediction Under Dynamic Missingness Masks

To provide distribution-free finite-sample coverage guarantees ($1 - \alpha$) on mobile devices, ONN-DETECT formalizes **Mondrian Conformal Prediction** (Angelopoulos & Bates, 2021; arXiv:2107.07511) stratified across dynamic modality missingness masks $m \in \{0, 1\}^K$:

```
Modality Mask m = [m_srm, m_siglip, m_audio, m_biolip]
       │
       ▼
Stratified Calibration Set D_cal(m) = {(x_i, y_i)}_i=1^(n_m)
       │
       ▼
Non-Conformity Scores: s_i(m) = 1 - P̂(Y = y_i | x_i, m)
       │
       ▼
Conformal Quantile:
q̂_(1-α)(m) = Quantile( ⌈(n_m + 1)(1 - α)⌉ / n_m ; {s_i(m)} )
       │
       ▼
Prediction Set Construction for Input X:
C(X, m) = { y ∈ {0, 1} : 1 - P̂(Y = y | X, m) ≤ q̂_(1-α)(m) }
       │
       ├─────────────────┬─────────────────┬─────────────────┐
       ▼                 ▼                 ▼                 ▼
C(X) = {0}        C(X) = {1}        C(X) = {0, 1}     C(X) = ∅
       │                 │                 │                 │
       ▼                 ▼                 ▼                 ▼
  AUTHENTIC          SYNTHETIC       INCONCLUSIVE      INCONCLUSIVE
 (Guaranteed)       (Guaranteed)       (Abstain)         (Abstain)
```

### Mathematical Formulation:
1. **Missingness Regimes**:
   - Mask 0 ($m = [1, 0, 0, 0]$): Image with spatial SRM only.
   - Mask 1 ($m = [1, 1, 0, 0]$): Dual-engine image (SRM + SigLIP).
   - Mask 2 ($m = [1, 1, 0, 1]$): Silent video (SRM + SigLIP + BioLip).
   - Mask 3 ($m = [1, 1, 1, 1]$): Full multimodal video with audio.
2. **Finite-Sample Coverage Guarantee**:
   For any user-selected significance level $\alpha \in (0, 1)$ (e.g., $\alpha = 0.05 \implies 95\%$ confidence):
   $$\mathbb{P}\left( Y \in \mathcal{C}(X, m) \right) \ge 1 - \alpha$$
   Over non-abstained predictions ($|\mathcal{C}(X, m)| = 1$), the classification error rate is strictly bounded by $\alpha$.
   *Critical Exchangeability Condition*: This finite-sample coverage guarantee holds mathematically under the assumption that calibration samples and runtime evaluation samples are **exchangeable** ($Z_i \sim \mathcal{D}$ i.i.d.). In open-world media forensics, out-of-distribution deepfake generators (unseen architectures, adversarial post-processing, social media compression) violate exchangeability; in such regimes, conformal prediction provides empirical uncertainty estimation rather than absolute frequentist guarantees. See `docs/EVIDENCE_CHAIN.md` for complete paper-to-artifact traceability.

## 6.3 Dual-Engine Cascade Arbitration & Selective Accuracy Optimization

In the visual pipeline, Engine A (SRM PatchCraft CNN) and Engine C (SigLIP-TAP Probe) operate concurrently:
- Engine A: Sensitive to high-pass spatial residuals ($128 \times 128$).
- Engine C: Sensitive to high-level semantic inconsistencies ($224 \times 224$).

### Base-Margin Prior Formulation
In `XGBoostMetaClassifier.kt`, the fusion booster predicts an additive logit correction over the SigLIP continuous logit base-margin prior:
$$P(\text{AI}) = \sigma\left( \text{logit}(P_{\text{SigLIP}}) + \sum_{t=1}^T f_t(\mathbf{x}) \right)$$
If SigLIP is unavailable (slot 6 is `NaN`), the base margin defaults to $0.0$, and the trees evaluate the learned probe-free branch.

### Cascade Decision Policy (`CascadePolicy.kt`)
The cascade arbitrates within an ambiguity band $[e_{\text{below}}, e_{\text{above}}] = [0.5801, 0.9546]$:
$$\text{Verdict} = \begin{cases}
\text{AUTHENTIC} & \text{if } (P_{\text{SigLIP}} \notin [e_{\text{below}}, e_{\text{above}}] \land P_{\text{fusion}} < \tau_A) \lor (P_{\text{SigLIP}} \in [e_{\text{below}}, e_{\text{above}}] \land \text{BothAgreeAuthentic}) \\
\text{SYNTHETIC} & \text{if } (P_{\text{SigLIP}} \notin [e_{\text{below}}, e_{\text{above}}] \land P_{\text{fusion}} \ge \tau_A) \lor (P_{\text{SigLIP}} \in [e_{\text{below}}, e_{\text{above}}] \land \text{BothAgreeSynthetic}) \\
\text{INCONCLUSIVE} & \text{if } P_{\text{SigLIP}} \in [e_{\text{below}}, e_{\text{above}}] \land \text{EngineDisagreement}
\end{cases}$$
Trading $2.75\%$ abstention on difficult borderline cases achieves $97.25\%$ coverage and $97.25\%$ cross-engine agreement.

## 6.4 Temperature Scaling, Base-Margin Boosting & Platt Calibration

To guarantee that the emitted score represents an empirical probability of generative origin:
1. **Temperature Scaling**: Logits are scaled by temperature $\tau$ fit via negative log-likelihood on the calibration set:
   $$P_{\text{cal}} = \sigma\left( \frac{z}{\tau} \right)$$
2. **Platt Scaling**: Fits two parameters $a, b \in \mathbb{R}$ such that $P = \frac{1}{1 + \exp(-(az + b))}$.
3. **Threshold Calibration**: Operating thresholds are calibrated on sealed splits to achieve target false-positive rates ($\text{FPR} \le 1.0\%$).

---

# Chapter 7: Cryptographic Provenance & Enterprise Attestation

## 7.1 C2PA Content Credentials v2.1/v2.3: JUMBF, APP11, and MP4 Box Architecture

The Coalition for Content Provenance and Authenticity (C2PA) defines open standards for binding digital provenance metadata directly to media files.

```
Media Container (JPEG APP11 / MP4 uuid Box)
       │
       ▼
ISO 19566-5 JUMBF Manifest Store (`c2pa`)
       │
       ├────────────────────────────────────────┐
       ▼                                        ▼
Claim Box (`c2pa.claim`)              Signature Box (`c2pa.signature`)
- Assertions (`c2pa.actions`, etc.)   - COSE_Sign1 (RFC 9052)
- Hard Bindings (`c2pa.hash.data`)    - ECDSA P-256 / SHA-256 Signature
- Soft Bindings                       - X.509 Attestation Cert Chain
       │                                        │
       └───────────────────┬────────────────────┘
                           ▼
  C2PAVerifier Conjunction:
  Signed = HashVerified ∧ CoseSignatureValid ∧ CertChainValid
```

### JUMBF Container Architecture:
- **JPEG**: Wrapped in `APP11` marker segments (`0xFFEB`) prefixed with JUMBF box headers (`jumb`).
- **MP4 / ISO-BMFF**: Wrapped in top-level `uuid` boxes with C2PA UUID GUID.
- **Claim Structure**: Encoded in Concise Binary Object Representation (CBOR, RFC 8949). Contains generation assertions (`c2pa.actions`), ingredient references, and cryptographically signed hashes.

## 7.2 The Exclusion Range Vulnerability & Fail-Closed Cryptographic Binding

### The Exclusion Range Threat Model (Golaszewski et al., IACR eprint 2026/804)
To allow editing container metadata without invalidating digital signatures, the C2PA specification defines `c2pa.hash.data` exclusion ranges:
$$\text{PayloadHash} = \text{SHA256}\left( \text{Bytes}(I) \setminus \bigcup_{j} [\text{offset}_j, \; \text{offset}_j + \text{length}_j] \right)$$
An adversary can exploit this by extracting a valid signed JUMBF manifest from an authentic camera photograph and transplanting it into a deepfake video or synthetic image, leaving the outer container structure valid while the content is completely fake.

### Fail-Closed Cryptographic Conjunction (`C2PAVerifier.kt`)
To guarantee complete immunity against manifest transplantation, ONN-DETECT enforces a strict fail-closed conjunction:
$$\text{C2PA\_Verified} = \text{HashVerified} \land \text{CoseSignatureValid} \land \text{CertChainValid}$$

1. **Byte Hash Verification**:
   The computed SHA-256 over media bytes (excluding the manifest segment) must match the signed hash in `c2pa.hash.data` bit-for-bit.
2. **COSE_Sign1 (RFC 9052) Verification**:
   Extracts `Sig_structure = ["Signature1", protected_bytes, external_aad, payload]` and verifies ECDSA P-256 ($r \parallel s$) against the leaf certificate public key.
3. **Air-Gapped Trust Store Chain Validation**:
   The leaf certificate must chain up to pre-installed root trust anchors (`assets/trust/c2pa_trust_anchors.pem`, covering Adobe, Google, Truepic, DigiCert, Microsoft).
4. **Zero-Network Invariant**: Online revocation checking (OCSP / CRL) is strictly disabled, ensuring zero network leakage. If any element in the conjunction fails, the verifier fails closed and returns `ProvenanceEvidence.Absent`.

## 7.3 Android KeyMint StrongBox Hardware Attestation (OID `1.3.6.1.4.1.11129.2.1.17`)

For enterprise compliance and legally defensible forensic audits, ONN-DETECT binds scan results to physical hardware using Android StrongBox KeyMint.

### Hardware Architecture:
StrongBox runs on a discrete hardware security chip (e.g., Google Titan M2, Samsung Knox Vault, NXP Secure Element) with dedicated CPU, secure internal memory, and hardware TRNG.

### Attestation Certificate Extension OID `1.3.6.1.4.1.11129.2.1.17`
When `StrongBoxReceiptSigner.kt` generates an ECDSA P-256 signing key with `.setIsStrongBoxBacked(true)`, the hardware chip generates an X.509 certificate chain rooted in the Google Hardware Attestation Root. The leaf certificate contains ASN.1 extension `1.3.6.1.4.1.11129.2.1.17`:

```asn.1
KeyDescription ::= SEQUENCE {
    attestationVersion         INTEGER,         -- Version 300+ (KeyMint)
    attestationSecurityLevel   SecurityLevel,   -- Must be 2 (StrongBox)
    keyMintVersion             INTEGER,
    keyMintSecurityLevel       SecurityLevel,   -- Must be 2 (StrongBox)
    attestationChallenge       OCTET_STRING,   -- 128-bit challenge nonce
    uniqueId                   OCTET_STRING,
    softwareEnforced           AuthorizationList,
    hardwareEnforced           AuthorizationList -- Hardware-bound parameters
}
SecurityLevel ::= ENUMERATED {
    Software           (0),
    TrustedEnvironment (1),
    StrongBox          (2)
}
```

The server-side verification script (`scripts/verify_forensic_receipt.py`) parses this ASN.1 extension, asserting:
$$\text{attestationSecurityLevel} == 2 \quad \land \quad \text{keyMintSecurityLevel} == 2$$
$$\text{attestationChallenge} == \text{ServerChallengeNonce}$$
Receipts signed with software keys or compromised emulators are rejected.

## 7.4 RFC 8785 Canonical JSON Forensic Receipts & High-Resolution Verification Certificates

### Canonical JSON Serialization (RFC 8785)
To prevent signature invalidation caused by key order differences or floating-point formatting variances across platforms, payloads signed by StrongBox are serialized in strict accordance with RFC 8785:
- Deterministic lexicographical sorting of JSON keys.
- Floating-point numbers formatted strictly to 4 decimal places via `_format_float_4dp`.
- No unescaped whitespace or control characters.

$$\text{Payload} = \text{RFC8785}(\text{challenge\_nonce}, \text{media\_sha256}, \text{model\_hashes}, \text{verdict}, \text{overall\_score}, \text{signals})$$
$$\text{Signature} = \text{ECDSA\_Sign}_{K_{\text{StrongBox}}}(\text{SHA256}(\text{Payload}))$$

### High-Resolution PNG Forensic Certificates (`ForensicCertificateExporter.kt`)
The engine exports shareable $1080 \times 1440$ PNG forensic certificates embedding:
- Verified verdict, confidence meter, and timestamp.
- High-pass residual tampering reticle.
- Cryptographic SHA-256 digests of media and model assets.
- QR code encoding the canonical JSON receipt and StrongBox signature.

## 7.5 Video-KYC (RBI V-CIP) Injection Defense: Camera2 HAL & Dynamic Hook Detection

Under Reserve Bank of India (RBI) V-CIP and SEBI digital KYC guidelines, video customer onboarding must be protected against virtual camera injection attacks and face-swap hooks.

`VirtualCameraDetector.kt` audits the Android execution environment across two layers:
1. **Camera2 HAL Sensor Audit**:
   - Queries `CameraCharacteristics` for physical hardware level. `HARDWARE_LEVEL_LEGACY` on modern devices indicates camera proxy emulation.
   - Audits physical sensor dimensions, focal lengths, and exposure time ranges. Virtual camera drivers (OBS, ManyCam, VCam) declare static or zero exposure ranges.
2. **Dynamic Runtime Hook Detection**:
   - Inspects the loaded runtime classloader for memory hooks and bytecode injection frameworks: Xposed, LSPosed, Frida, Pine, Epic, Substrate.
   - Inspects `/proc/self/maps` for unmapped native `.so` injections.
   - Any detected hook triggers an immediate `INJECTION_INDICATORS_PRESENT` report, preventing forged KYC video sessions.

---

# Chapter 8: Codebase Audit Changelog & Empirical Benchmark Matrix

## 8.1 Itemized Changelog of Resolved Defects (Milestone 1, Milestone 2, Milestone 2 Round 2)

| Issue ID | Module / Component | Severity | Defect Description | Root Cause | Engineering Resolution & Verification |
|---|---|---|---|---|---|
| **M1-01** | `ml/requirements.txt` | Medium | Missing cryptographic dependency | `cryptography` library missing from `.venv` requirements | Pinned `cryptography>=50.0.2`; verified receipt verification test suite |
| **M1-02** | `evaluation_report.json` & `release_manifest.json` | High | Lineage calibration sample mismatch | Calibration sample size was declared as `300` instead of `400` | Reconciled calibration records; synchronized SHA-256 hashes in `release_manifest.json` |
| **M1-03** | `StrongBoxReceiptSigner.kt` | Critical | Concurrency race condition during key generation | Multi-threaded calls to `sign()` generated conflicting keypairs | Implemented double-checked locking with `@Volatile testKeyPair`; added `@JvmOverloads` constructor |
| **M1-04** | `VoiceVaultCrypto.kt` | High | Referential equality bug in `SealedBlob` | Kotlin `ByteArray` properties in data classes compared references rather than byte content | Implemented `Arrays.equals` and `Arrays.hashCode` for IV and ciphertext; added 6 regression tests |
| **M2-01** | `PerioralLandmarkAnalyzer.kt` | Critical | Premature MediaMetadataRetriever release | `mediaMetadata.release()` was called before the frame extraction loop | Relocated `release()` to a deterministic `finally` block; wrapped descriptors in `pfd.use` |
| **M2-02** | `PerioralLandmarkAnalyzer.kt` | High | Native graphic bitmap memory leak | Extracted bitmaps in `analyzeVideo` were never explicitly recycled | Enclosed frame processing in `try/finally` with `if (!bitmap.isRecycled) bitmap.recycle()` |
| **M2-03** | `PerioralLandmarkAnalyzer.kt` | High | Obsolete MediaPipe reflection package paths | Targeted obsolete `com.google.mediapipe.tasks.vision.face.*` hierarchy | Updated reflection to 0.10+ `com.google.mediapipe.tasks.vision.facelandmarker.*` |
| **M2-04** | `PerioralLandmarkAnalyzer.kt` | High | Timestamp scaling factor error | Microsecond timestamp formula was scaled by an extra 1000x | Fixed timestamp computation to `(frameIdx * 1000000L / 30).toLong()` |
| **M2-05** | `computeJitterScore` | High | Temporal gap collapse in BioLip | `.filterNotNull()` collapsed frame drops, creating false acceleration spikes | Preserved timeline indices; evaluated second differences only across contiguous triplets |
| **M2-06** | `computeRppgBilateralPhaseAnomaly` | High | Discontinuous phase array concatenation | Cheek ROI green chrominance traces appended across discontinuous frames | Partitioned timeline into contiguous uninterrupted runs $\ge 4$ frames; returns `Float.NaN` if $<4$ |
| **M2-07** | `ScanResult.kt` & `VideoForensicAnalyzer.kt` | High | Toxic `-1f` sentinels violating NaN contract | Sentinels used `-1f`, causing exceptions when `Float.NaN` was passed | Replaced `-1f` with `Float.NaN`; updated `SignalBreakdownJson` and `ScanResult.init` |
| **M2-08** | `ForensicCertificateExporter.kt` | Medium | IEEE 754 Math.max collapse on NaN | `Math.max(Float.NaN, x)` returns `Float.NaN`, hiding valid measured signals | Replaced with `listOfNotNull(...).filter { it.isFinite() }.maxOrNull() ?: Float.NaN` |
| **M2R2-01** | `VideoScorePolicy.kt` | Critical | IEEE 754 NaN collapse in audio escalation | Uncalibrated `visualThreshold` evaluated to `NaN`; `maxOf(base, NaN)` destroyed valid visual scores | Added finite bounds checks (`in 0f..1f`); replaced `maxOf` with monotonic `if (candidate > base) base = candidate` |
| **M2R2-02** | `VideoScorePolicyAdversarialTest.kt` | High | Fuzzing invariant relaxation | Test assertion had been relaxed to tolerate `NaN` on valid visual inputs | Restored strict invariant: valid visual input guaranteed to yield finite score in $[0, 1]$ |
| **M2R2-03** | `ScannerViewModel.kt` | Medium | In-memory positional `-1f` sentinels in UI | Positional constructor calls passed literal `-1f` for unmeasured channels | Migrated to named arguments; defaulted unmeasured forensic channels to `Float.NaN` |

## 8.2 Empirical Benchmark Matrix: Clean vs. Laundered Performance

Evaluated on sealed benchmark splits (`evaluation_report.json`, `release_manifest.json`, and `meta_results.json` for release `onn-prod-v10`):

| Model / Pipeline Component | Input Tensor Contract | Asset Size | Clean Test ROC-AUC | Laundered Test ROC-AUC (JPEG Q85) | EER | Latency (ARM64 XNNPACK) |
|---|---|---|---|---|---|---|
| **SigLIP-B/16 TAP Probe (Standalone)** | `[1, 224, 224, 3]` float32 | 89.96 MB (INT8) | **0.9921** | 0.8972 | **0.0421** | ~130 ms (4 threads) |
| **SRM + PatchCraft Mosaic CNN** | `[1, 128, 128, 3]` float32 | 142 KiB (INT8) | **0.9655** | 0.9391 (WhatsApp) / 0.8949 (Mean) | **0.0971** | ~15 ms |
| **10-Slot XGBoost Meta-Fusion** | 10-slot feature vector | 20 KiB (Kotlin) | **0.9974** | **0.9972** | **0.0250** | < 1 ms |
| **LCNN-MFM Speech Anti-Spoofing** | `[1, 401, 40, 1]` float32 | 508 KiB (INT8) | **0.8747** | 0.8620 (Opus 16 kbps) | **0.2225** | < 25 ms |
| **AudioSeal Watermark Detector** *(Ref)* | `[1, 16000]` float32 PCM | 4.8 MB (ONNX ref) | 0.9999 *(Meta Ref)* | 0.9850 (Opus 24 kbps) | 0.0001 *(Ref)* | ~35 ms *(Host)* |
| **BioLip Perioral Jitter** *(Exp)* | Sequence of 478-pt mesh | Dynamic | 0.9240 *(Baseline)* | 0.9180 (H.264 CRF 28) | 0.1420 *(Baseline)* | ~8 ms / frame |
| **C2PA Manifest Verifier** | Byte stream container | < 50 KiB code | N/A *(Deterministic)* | N/A *(Lossless Binding)* | N/A *(Crypto)* | < 5 ms |

*Component Status Designations*:
- **AudioSeal Watermark Detector**: Reference implementation / unreleased prototype. Not evaluated in on-device `evaluation_report.json`; watermark absence is treated as absent evidence, not authentic.
- **BioLip Perioral Jitter Analyzer**: Biomechanical landmark analyzer (`PerioralLandmarkAnalyzer.kt`). Metrics represent synthetic lip-sync baseline estimates, not frozen neural models in `release_manifest.json`.
- **C2PA Manifest Verifier**: Deterministic cryptographic signature and hash verification. Statistical classification metrics (ROC-AUC, EER) are mathematically inapplicable.

## 8.3 Low-FPR Operating Points (TPR @ 1% FPR & TPR @ 0.1% FPR)

Operating decision thresholds and True Positive Rates strictly aligned with `evaluation_report.json` and `release_manifest.json`:

| Forensic Pipeline Component | Operating Decision Threshold | TPR @ 5% FPR | TPR @ 1% FPR | TPR @ 0.1% FPR |
|---|---|---|---|---|
| **Image Meta-Fusion (`onn-prod-v10`)** | **0.57819986** | **0.9850 (98.5%)** | **0.9400 (94.0%)** | **0.9150 (91.5%)** |
| SigLIP-B/16 TAP Probe Standalone | **0.76734054** | 0.9200 (92.0%) | 0.7243 (72.4%) | **0.5500 (55.0%)** |
| SRM PatchCraft CNN Standalone | **0.50000000** | 0.8100 (81.0%) | **0.5314 (53.1%)** | **0.2657 (26.6%)** |
| **LCNN-MFM Audio Anti-Spoofing** | **0.35546875** | **0.7850 (78.5%)** | **0.6500 (65.0%)** | **0.4500 (45.0%)** |
| AudioSeal Watermark Detector *(Ref)* | $k = 16, \; p = 3.05 \times 10^{-5}$ | 0.9990 (99.9%) | 0.9990 (99.9%) | 0.9990 (99.9%) |

## 8.4 Resource Footprint, Mobile Latency & 16 KiB ELF Alignment

| Metric / Requirement | Target Constraint | Measured Production Value | Verification Status |
|---|---|---|---|
| **Universal Release APK (`app-release.apk`)** | Packaging Artifact | **166.71 MB** (158.99 MiB, bundles all 4 ABIs) | INFO (Multi-ABI bundle) |
| **Single-ABI Release Slice (`arm64-v8a`)** | $< 150\text{ MB}$ | **110.93 MB** (includes 89.96 MB SigLIP INT8) | PASS (Target distribution) |
| **Java / ART Heap Bloat** | $< 32\text{ MB}$ | **< 14 MB** (Models mapped via page cache) | PASS |
| **Total Image Scan Latency** | $< 250\text{ ms}$ | **~140–180 ms** (ARM64 XNNPACK, 4 threads) | PASS |
| **Audio Stream Chunk Latency** | $< 50\text{ ms}$ | **< 25 ms** (4-second LFCC + LCNN) | PASS |
| **ELF Segment Page Alignment** | Strict $16\text{ KiB}$ ($16,384\text{ bytes}$) | **Aligned** (`verify_android_artifact.py`) | PASS |
| **Android Permissions Invariant** | Zero Network Access | `INTERNET` & `ACCESS_NETWORK_STATE` ABSENT | PASS |

## 8.5 Primary Literature Citations (2024–2026)

1. **REALIS**: He, P., et al. (2026). *REALIS: A Large-Scale In-The-Wild Benchmark for Realistic AI-Generated Image Detection*. arXiv preprint arXiv:2609.32734.
2. **DailyBench**: Zhang, Y., et al. (2026). *DailyBench: Exposing Dataset Shortcuts in Modern Generative Media Forensics*. CVPR 2026 / arXiv:2607.24016.
3. **AudioSeal**: San Roman, R., et al. (2024). *Proactive Detection of Voice Cloning with Localized Neural Watermarking*. International Conference on Machine Learning (ICML 2024). arXiv:2401.17264.
4. **Watermark Shortcut**: Müller, N. M., & Debus, P. (2026). *The Watermark Shortcut: How Provenance Marking Sabotages Audio Deepfake Detection*. arXiv preprint arXiv:2606.23335.
5. **DCPT**: Yang, Z., Hou, Y., & Yang, X. (2026). *Degradation-Consistent Paired Training for Robust AI-Generated Image Detection*. arXiv preprint arXiv:2604.10102.
6. **BioLip / LADDER**: Chen, H., & Xu, J. (2026). *LADDER: Landmark-Only Lip-Sync Deepfake Detection Across Generators and Languages*. arXiv preprint arXiv:2604.16808.
7. **VoiceWukong**: Chen, T., et al. (2025). *VoiceWukong: Benchmarking In-The-Wild Audio Deepfake Detection Across Codecs and Acoustic Environments*. USENIX Security Symposium 2025.
8. **GenVidBench**: Zhao, L., et al. (2026). *GenVidBench: A Multi-Generator Temporal Benchmark for Video Deepfake Forensics*. AAAI Conference on Human Computation and Artificial Intelligence (AAAI 2026).
9. **CoCoDetect**: Sun, K., et al. (2026). *CoCoDetect: Confidence-Gated Multi-Tier Video Deepfake Analysis*. CVPR 2026.
10. **PatchCraft**: Zhong, N., Xu, Y., Li, S., Qian, Z., & Zhang, X. (2023). *PatchCraft: Exploring Texture Patch for Efficient AI-generated Image Detection*. arXiv preprint arXiv:2311.12397.
11. **NPR**: Tan, C., et al. (2024). *Rethinking the Up-Sampling Artifacts in Generative Diffusion Models*. CVPR 2024. arXiv:2312.10461.
12. **Mondrian Conformal Prediction**: Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv preprint arXiv:2107.07511.
13. **C2PA Security Analysis**: Golaszewski, E., Krawetz, N., Sherman, A. T., et al. (2026). *Verifying Provenance of Digital Media: Security Analysis of C2PA and its Implementation*. IACR Cryptology ePrint Archive, Report 2026/804.
14. **CHROM rPPG**: de Haan, G., & Jeanne, V. (2013). *Robust Pulse Rate from Chrominance-Based rPPG*. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
15. **RFC 8785**: Rundgren, A., et al. (2020). *JSON Canonicalization Scheme (JCS)*. Internet Engineering Task Force (IETF) RFC 8785.
