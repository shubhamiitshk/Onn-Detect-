# ONN-DETECT — z21 Startup Acceleration Program Pitch Deck
**Host Institution:** IIT Kharagpur (KGP Forge)  
**Domain:** Deep Tech / AI Cybersecurity / Digital Forensics  
**Product:** On-Device Zero-Cloud AI Media Forensics Platform & Mobile SDK  
**Stage:** Pre-Seed | **TRL 6–7 (System Demonstration & Prototyping)**  
**Release:** `onn-20261023-v8`  
**Founder:** Shubham Kumar (IIT Kharagpur)  
**Repository:** [`github.com/shubhamiitshk/Onn-Detect-`](https://github.com/shubhamiitshk/Onn-Detect-)  
**Associated Presentation File:** [`ONN_DETECT_PITCH_DECK_Z21_KGP_FORGE.pptx`](file:///c:/Users/PLEXTEK/Desktop/Onn-Detect-/ONN_DETECT_PITCH_DECK_Z21_KGP_FORGE.pptx)

---

## 1. Verified Application Form Responses (Post-Patch v8)

### Page 1 — What is your startup about? *
> ONN-DETECT builds a 100% on-device, zero-cloud AI media forensics platform, consumer mobile security app, and enterprise B2B SDK that detects AI-generated images, voice clones, forged payment receipts, and deepfake videos directly on smartphones in under 450 ms. Powered by a quantized 5-layer cascade combining a Vision Foundation Model (INT8 SigLIP-B/16), a 20-channel spatial-frequency residual CNN (12 SRM + 4 NPR + 4 RGB stem), speech anti-spoofing (LCNN-MFM), perioral biomechanical lip-jitter analysis (BioLip on a 478-point face mesh), and offline C2PA Content Credentials verification against the official production trust list inside a strict zero-network sandbox, our v7/v8 release achieves 0.894 image AUC (0.900 under compression laundering) and 0.875 audio AUC (59.5% TPR at 1% FPR). Our shipped suite includes a dual-gate Family Safe-Voice Vault (LCNN-MFM anti-spoofing below 0.375 plus AES-256-GCM-sealed, window-averaged voiceprint matching), 1-tap WhatsApp Instant Forward Guard with 32×32 tamper heatmaps, ELA document/receipt forgery detection, Camera2 HAL virtual-camera KYC shielding, and StrongBox ECDSA P-256 hardware-signed forensic receipts—delivering privacy-compliant protection at $0 cloud inference cost.

---

### Page 2, Q1 — The Core Problem (max 150 words) * *(138 words)*
> Generative AI has made photorealistic voice clones, deepfake videos, and forged documents effortless to produce, creating three critical bottlenecks:
> 1. **Voice-Clone Extortion & Receipt Fraud:** Scammers forward synthetic voice notes impersonating family members or officials to extort instant UPI transfers, while merchants and insurers lose billions to AI-edited payment screenshots and manipulated claim photos.
> 2. **Video-KYC & Virtual-Camera Spoofing:** Attackers bypass bank and fintech onboarding (RBI V-CIP) by injecting pre-rendered deepfakes and audio-driven lip-sync videos into camera HAL streams.
> 3. **The Cloud Privacy, Cost & Compression Bottleneck:** Commercial detectors require uploading private voiceprints and biometric KYC video to cloud GPU servers—violating privacy laws (DPDP Act, GDPR), collapsing under WhatsApp JPEG Q85 or VoIP compression, and costing $0.02–$0.10 per scan, making high-volume screening economically unviable.

---

### Page 2, Q2 — Your Solution (max 150 words) * *(146 words)*
> **ONN-DETECT** eliminates these bottlenecks with a 100% on-device, zero-cloud multimodal forensics engine and Android SDK running in <450 ms at $0 server cost:
> * **Compression-Hardened Dual-Engine Core:** Pairs an INT8 Vision Foundation Model (SigLIP-B/16) base-margin prior with a 20-channel spatial-frequency CNN (SRM/NPR/RGB), LCNN-MFM speech anti-spoofing, and offline C2PA verification against the official production trust list (fail-closed: chain + COSE signature + data-hash), achieving 0.894 image AUC (0.900 under 512px JPEG Q85 laundering) and 0.875 audio AUC (59.5% TPR at 1% FPR).
> * **Dual-Gate Family Safe-Voice Vault:** LCNN-MFM anti-spoofing (score < 0.375) plus AES-256-GCM-sealed window-averaged voiceprints flag synthetic voice clones of relatives on shared audio (CAM++ embedding upgrade scheduled next).
> * **Instant Forward Guard & ELA Document Shield:** 1-tap WhatsApp Share Sheet check with 32×32 tamper heatmaps and in-memory ELA catching edited UPI receipts.
> * **Enterprise KYC & Claims SDK:** Camera2 HAL virtual-camera detection, BioLip 64-point perioral jitter, and StrongBox ECDSA P-256 signed forensic receipts.

---

### Page 3 — Core Technological Innovation (Max 200 words) *(193 words)*
> Replicating ONN-DETECT requires solving four hard ML-systems problems on mobile edge hardware:
> 1. **Base-Margin Residual Cascade & Selective Arbitration:** Naive tree fusion over neural scores discretizes outputs into step-function plateaus, collapsing low-FPR recall (`TPR@1%FPR → 0`). We inject an INT8 Vision Foundation Model probe (`SigLIP-B/16`) as a continuous logit `base_margin`, while a 20-channel spatial-frequency CNN (`12` locked 5×5 SRM high-pass filters, `4` NPR upsampling kernels, `4` centered RGB low-pass stem) and `NaN`-aware XGBoost trees learn an additive residual correction, governed by a selective-accuracy abstention band.
> 2. **ViT-to-LiteRT Mirror & Zero-Heap Runtime:** Standard `onnx2tf` toolchains corrupt ViT patch-to-sequence layouts. Our custom channels-last Keras-to-LiteRT compiler folds static Multihead-Attention-Pooling queries with **1.97e-06 FP32 layer parity**, `mmap`-ing the 94.3 MB INT8 tower directly from the APK into OS page cache (`282 ms` @ 4 threads, zero Java heap bloat).
> 3. **Bit-Exact Frontend Parity & Laundering Immunity:** We engineered a two-pass Keys-cubic (`a = -0.5`) resampler with intermediate 8-bit saturation matching Python `Pillow` within ±1 level, paired with Degradation-Consistent Paired Training—achieving **0.900 AUC under JPEG Q85 laundering**.
> 4. **Hardware-Bound Trust & Biomechanical Forensics:** Combines StrongBox/TEE ECDSA P-256 RFC 8785 receipts, Camera2 HAL hook detection, BioLip 64-point perioral jitter on a 478-point face mesh, and AES-256-GCM voiceprint sealing.

---

### Question: Technology Readiness Level (TRL)
> **Select:** **System Demonstration & Prototyping (TRL 6 – TRL 7)**  
> *Justification:* Production-ready signed Android release APK (`app-release.apk`, 110.93 MB) running INT8 SigLIP-B/16, 20-ch SRM/NPR/RGB CNN, LCNN-MFM, and BioLip models on physical ARM64 hardware in an operational mobile environment, verified by 365 automated test gates (220 JVM unit tests + 145 Python ML tests + APK 16KiB ELF alignment checks).

---

## 2. Slide-by-Slide Pitch Deck Structure (Release onn-20261023-v8)

### Slide 1: Title & Hook
* **Headline:** ONN-DETECT: Zero-Cloud AI Media Forensics
* **Subtitle:** On-device deepfake, voice-clone, and KYC fraud defense in under 450 ms.
* **Metadata:** Release: onn-20261023-v8 | Stage: Pre-Seed (TRL 6–7) | IIT Kharagpur
* **Speaker Hook:**
  > *"Every day in India, citizens and businesses lose crores to AI voice-cloning kidnapping scams, digital-arrest threats, and spliced UPI payment screenshots. Existing cloud forensic tools are too slow, cost ₹2 to ₹8 per scan, and violate the DPDP Act. ONN-DETECT executes enterprise-grade forensic neural networks directly on consumer smartphones in under 450 milliseconds—100% offline, at $0 server cost."*

---

### Slide 2: The Core Problem
* **Header:** Cloud Costs, Privacy Laws, and Compression Break Legacy Detection
* **Column 1 — Voice & Receipt Fraud:**
  * **Voice-Clone Scams:** Forwarded AI voice notes impersonate family members to extort instant UPI transfers.
  * **Forged Screenshots:** AI-edited payment receipts and fake invoices defraud merchants and insurers daily.
  * **No Local Verification:** Consumers lack offline identity + authenticity verification for shared media.
* **Column 2 — Video-KYC & HAL Spoofing:**
  * **Virtual-Camera Hooks:** Attackers inject pre-rendered deepfakes into bank Video-KYC (V-CIP) sessions.
  * **Lip-Sync Synthesis:** Audio-driven talking-head fakes bypass single-frame face liveness checks.
  * **Compression Collapse:** Academic CNNs fail after WhatsApp JPEG Q85 or VoIP Opus re-encoding.
* **Column 3 — Cloud Unit Economics:**
  * **Privacy Violations:** Uploading voiceprints and biometric KYC video violates DPDP Act and GDPR.
  * **Unsustainable Cost:** Cloud GPU APIs cost $0.02–$0.10 per scan, blocking high-volume screening.
  * **Unverified Provenance:** Stripped or forged C2PA metadata cannot be trusted without offline crypto checks.

---

### Slide 3: The ONN-DETECT Solution
* **Header:** Zero-Cloud Forensics at $0 Server Cost
* **Core Pillars:**
  * **Zero-Network Sandbox:** Strips `INTERNET` permission at the OS manifest level; voiceprints and KYC media never leave the phone.
  * **5-Layer Edge Cascade:** Unifies INT8 SigLIP-B/16 (94.3 MB), 20-ch SRM/NPR/RGB CNN, LCNN-MFM audio, BioLip, and C2PA.
  * **Play-Compliant B2B2C Suite:** Runs in 282 ms (4 threads) across our consumer Forward Guard + Voice Vault app and enterprise KYC SDK.

---

### Slide 4: Consumer & Forensic Capabilities
* **Header:** Play-Compliant Consumer & Forensic Suite Verified by 220 JVM Tests
* **Family Safe-Voice Vault:**
  * **Dual-Gate Verdict:** Gates on LCNN-MFM (<0.375) + 3s/1.5s window voiceprints.
  * **AES-256-GCM Sealed:** Seals voiceprints in Room with Keystore AAD binding.
  * **Clone Alert Path:** Flags synthetic audio matching a relative (CAM++ next).
* **Instant Forward Guard:**
  * **1-Tap Share Target:** Handles `ACTION_SEND` & `SEND_MULTIPLE` for WhatsApp media without invasive accessibility scraping.
  * **32×32 ROI Heatmap:** `ResidualHeatmapExtractor` pinpoints local tamper energy.
  * **ELA Document Guard:** `DocumentForgeryAnalyzer` catches spliced UPI receipts.
* **Offline Provenance:**
  * **32 Production Roots:** Verifies official C2PA anchors (Adobe, Google, DigiCert).
  * **Full Container Walk:** Reassembles JPEG APP11 (En, Z) & ISO-BMFF JUMBF boxes.
  * **Fail-Closed Crypto:** Requires chain + COSE_Sign1 + c2pa.hash.data match.

---

### Slide 5: Enterprise B2B Monetization
* **Header:** Offline SDK for Video-KYC & Claims
* **Enterprise Features:**
  * **V-CIP KYC Injection Shield:** `VirtualCameraDetector` audits Camera2 HAL hardware levels, silicon sensor bounds, and Xposed hooks.
  * **Hardware Forensic Receipts:** `StrongBoxReceiptSigner` signs RFC 8785 canonical JSON + media SHA-256 via StrongBox/TEE ECDSA P-256.
  * **BioLip 478-Pt Mesh Defense:** MediaPipe `FaceLandmarker` extracts 64 perioral points (inter-ocular normalized) to catch lip-sync fakes.

---

### Slide 6: Core Technological Innovation
* **Header:** Four ML-Systems Breakthroughs Make ONN-DETECT Hard to Replicate
* **1. Base-Margin Cascade:**
  * **Continuous Tail Order:** Injects INT8 SigLIP-B/16 logit as XGBoost `base_margin` to prevent tree step-ties.
  * **20-Ch SRM + NPR Stem:** Combines 12 locked 5×5 SRM high-pass filters, 4 NPR kernels, and 4 RGB channels.
  * **Selective Arbitration:** Escalation band maximizes selective accuracy under a $\ge 85\%$ classified-rate floor.
* **2. ViT-to-LiteRT Compiler:**
  * **1.97e-06 FP32 Parity:** Custom channels-last Keras mirror bypasses broken onnx2tf ViT shape inference.
  * **Static MAP Queries:** Pre-computes frozen attention-pooling queries, saving 589K weights at runtime.
  * **Zero-Heap APK mmap:** Maps the 94.3 MB INT8 tower into OS page cache (282 ms @ 4 threads, 0 MB heap).
* **3. Keys-Bicubic & Crypto:**
  * **Bit-Exact Resampler:** Custom two-pass Keys-cubic ($a=-0.5$) filter with 8-bit clip matches Pillow $\pm 1$.
  * **DCPT Paired Loss:** Symmetric Bernoulli KL + penultimate cosine alignment hardens Q85 accuracy.
  * **Hardware Crypto Binding:** StrongBox ECDSA P-256 + 32-anchor C2PA COSE_Sign1 + AES-256-GCM AAD vault.

---

### Slide 7: Measured Benchmark Validation
* **Header:** v7 Tower Architecture Lifts Sealed Image AUC to 0.894 (+17.9 pts)
* **Key Empirical Benchmarks:**
  * **7× Low-FPR Recall Gain:** Base-margin VFM fusion lifts image TPR@1%FPR from 4.0% to 28.0% and TPR@0.1%FPR to 25.0%.
  * **0.900 Laundered AUC:** Re-encoding test images to 512px JPEG Q85 raises v7 AUC to 0.900, proving social-feed resilience.
  * **0.875 Speech & 365 Gates:** LCNN-MFM reaches 59.5% TPR@1%FPR, backed by 220 JVM unit tests and 145 Python ML tests.
* **Source:** ONN-DETECT Sealed Benchmark (3,600 Image / 68 Generators & 2,400 Audio Records)

---

### Slide 8: Competitive Landscape & IP Status
* **Header:** TRL 6–7 System Prototype with Structural Privacy and Cost Moats

| Capability / Metric | **ONN-DETECT (v8)** | Cloud Deepfake APIs | Academic CNNs |
|---|---|---|---|
| **Execution & Privacy** | **100% On-Device (0 Cloud)** | Cloud Upload (DPDP Risk) | Desktop GPU Only |
| **Marginal Cost / Scan** | **₹0.00 ($0 Server Cost)** | ₹1.70 – ₹8.40 / Scan | N/A (Not Deployed) |
| **512px JPEG Q85 Laundering** | **0.900 AUC (Hardened)** | 0.72 – 0.81 AUC | Collapses (<0.65 AUC) |
| **Hardware Receipts & HAL KYC** | **StrongBox P-256 + Camera2 HAL** | No Device Attestation | No Mobile Runtime |
| **Readiness & IP Status** | **TRL 6–7 \| Trade Secret + Patent** | Commercial Cloud SaaS | TRL 2–3 Lab Papers |

---

### Slide 9: Market Size & Commercial Model
* **Header:** Targeting a ₹6,800 Cr On-Device SAM and ₹85 Cr 3-Year SOM
* **Market Dimensions:**
  * **Global TAM (₹62,000 Cr):** $7.4B global deepfake defense, Video-KYC anti-spoofing, and mobile voice-fraud market.
  * **B2B SDK (₹52.5 Cr SOM):** 35 fintech, NBFC, and insurance clients at ₹1.5 Cr avg ACV (₹20–₹40 per KYC/claim).
  * **B2C & Traction (₹32.5 Cr SOM):** 1.1M users at ₹299/yr (Safe-Voice Vault & Forward Guard); TRL 6–7 APK ready for pilots.

---

### Slide 10: Funding Ask, Runway & KGP Forge Support
* **Header:** ₹25 Lakh Seed Grant Fuels a 12-Month Runway to Enterprise Pilots
* **The Ask:** **₹25,00,000 (₹25 Lakhs)** for 12 months of runway.
* **Runway & P1–P3 Milestones:** ₹2.08L/mo burn (12 mos); ships CAM++ speaker encoder, 400-video calibration, & device gates.
* **Budget Allocation:**
  * **CapEx (35% / ₹8.75L):** GPU compute for B-Free twins & 400-video split (₹5.25L) + Android NPU test matrix (₹3.50L).
  * **OpEx (65% / ₹16.25L) & Support:** Eng stipends (₹9.5L), pilots (₹3.5L), patents/audit (₹3.25L) + KGP Forge mentorship.

---

### Slide 11: Closing
* **Tagline:** **Authenticity at the Edge. Zero Cloud. Zero Compromise.**
* **Footer:** Thank You — ONN-DETECT | z21 Startup Acceleration Program (KGP Forge)
* **Founder:** Shubham Kumar (IIT Kharagpur)
* **Repository:** `https://github.com/shubhamiitshk/Onn-Detect-`
