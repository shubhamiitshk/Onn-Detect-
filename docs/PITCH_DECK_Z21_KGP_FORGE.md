# ONN-DETECT — z21 Startup Acceleration Program Pitch Deck
**Host Institution:** IIT Kharagpur (KGP Forge)  
**Domain:** Deep Tech / AI Cybersecurity / Digital Forensics  
**Product:** On-Device Zero-Cloud AI Media Forensics Engine & Mobile SDK  
**Stage:** Pre-Seed | **TRL 6–7 (System Demonstration & Prototyping)**  
**Founder:** Shubham Kumar (IIT Kharagpur)  
**Repository & Verified Release:** [`github.com/shubhamiitshk/Onn-Detect-`](https://github.com/shubhamiitshk/Onn-Detect-)  
**Associated Presentation File:** [`ONN_DETECT_PITCH_DECK_Z21_KGP_FORGE.pptx`](file:///c:/Users/PLEXTEK/Desktop/Onn-Detect-/ONN_DETECT_PITCH_DECK_Z21_KGP_FORGE.pptx)

---

## 1. Executive Summary & Application Form Responses

### Question: What is your startup about? (150–200 Words)
> **ONN-DETECT** builds an on-device, zero-cloud AI media forensics engine and mobile SDK that detects AI-generated and manipulated images, voice clones, and deepfake videos directly on smartphones in **under 180 ms** (sub-25 ms for streaming audio). Unlike cloud-based detectors that require uploading sensitive personal media, incur recurring per-scan server costs (₹1.70–₹8.50/scan), and fail offline, our quantized edge cascade combines compact Vision Foundation Models (INT8 SigLIP-B/16), spatial-frequency residual networks (36-channel SRM/NPR), speech anti-spoofing (LCNN-MFM), and cryptographic provenance verification (C2PA & Meta AudioSeal) inside a strict zero-network sandbox.
>
> On our sealed benchmark spanning 68 image generators and in-the-wild speech datasets, our v7 release achieves **0.894 AUC (0.900 under social media JPEG Q85 compression)** for images and **0.875 AUC (59.5% TPR at 1% FPR)** for voice deepfakes. Our patent-pending pipeline defends against India's fastest-growing cyberfraud threats: **real-time UPI coercion & digital arrest call interception**, **family voice-clone kidnapping extortion**, **forged payment confirmation screenshots**, and **virtual camera deepfake injection for RBI Video-KYC (V-CIP)**.

### Question: The Core Problem (Max 150 Words)
> Generative AI enables weaponized deepfakes, automated voice cloning, and synthetic identity fraud at near-zero marginal cost. Existing forensic tools suffer from three fatal bottlenecks:
> 1. **Cloud Dependency & Privacy:** Uploading private consumer calls or biometric KYC video to remote cloud APIs violates strict data protection regulations (India DPDP Act 2023, GDPR) and creates massive enterprise liability.
> 2. **Economic & Latency Scaling:** Cloud GPU inferences cost ₹1.70–₹8.50 ($0.02–$0.10) per scan with 1–3s latency, making real-time, in-call screening across millions of banking transactions financially impossible.
> 3. **Compression Collapse:** Standard academic detectors collapse under WhatsApp JPEG Q85 laundering and VoIP Opus re-encoding (dropping AUC below 0.65).
>
> Meanwhile, Indian citizens lose over ₹1,750+ Crores annually to UPI social engineering, digital arrest extortion calls, and spliced payment screenshots.

### Question: Core Technological Innovation (Max 200 Words)
> ONN-DETECT introduces four ML-systems breakthroughs enabling enterprise-grade forensic precision directly on edge devices:
> 1. **Base-Margin Cascade Formulation:** Injects the continuous logit of a frozen 86M Vision Transformer (INT8 SigLIP-B/16) as the `base_margin` of an XGBoost meta-learner ($\text{margin} = \text{logit}(p) + f_{\text{tree}}$). This prevents tree quantization step-ties and yields a **7× recall gain at low false alarm rates** (TPR@1%FPR: 4.0% $\rightarrow$ 28.0%; TPR@0.1%FPR: 0.5% $\rightarrow$ 25.0%).
> 2. **ViT-to-LiteRT Channels-Last Compiler Mirror:** Bypasses broken ONNX-to-TFLite ViT attention shape inference via a custom channels-last Keras mirror, achieving **$1.97 \times 10^{-6}$ FP32 numerical parity** against PyTorch and pre-computing frozen attention queries to eliminate 589K runtime parameters.
> 3. **Zero-Heap APK Memory Mapping:** Maps the 89.96 MB INT8 graph directly into Linux page cache with **0 MB Java heap overhead**, running smoothly on 4 GB budget Android devices.
> 4. **Hardware-Signed StrongBox Attestation:** Binds scan verdicts, media SHA-256, and model release hashes directly inside the physical Android StrongBox TEE Keystore, producing tamper-proof, court-admissible audit receipts.

### Question: Technology Readiness Level (TRL)
> **Select:** **System Demonstration & Prototyping (TRL 6 – TRL 7)**  
> *Justification:* Production-ready signed Android release APK (`app-release.apk`, 110.93 MB) running INT8 SigLIP-B/16, SRM/NPR CNN, and LCNN-MFM models on physical ARM64 hardware in an operational mobile environment, verified by 309 automated tests (146 Python ML + 29 script parity + 127 Kotlin unit + 2 APK elf/alignment gates).

---

## 2. Slide-by-Slide Pitch Deck Structure

### Slide 1: Title & Hook
* **Headline:** ONN-DETECT: Zero-Cloud AI Media Forensics
* **Subtitle:** On-device deepfake, voice-clone, and cyber-fraud defense in under 180 ms.
* **Metadata:** Domain: AI & Deep Tech | Stage: Pre-Seed (TRL 6–7 Validated Prototype) | IIT Kharagpur
* **Speaker Hook:**
  > *"Every day in India, citizens and businesses lose crores to AI voice-cloning kidnapping scams, digital-arrest threats, and spliced UPI payment screenshots. Existing cloud forensic tools are too slow, cost ₹2 to ₹8 per scan, and violate the DPDP Act. ONN-DETECT executes enterprise-grade forensic neural networks directly on consumer smartphones in under 180 milliseconds—100% offline, at $0 server cost."*

---

### Slide 2: The Core Problem
* **Header:** Cloud Costs, Privacy Laws, and Compression Break Legacy Detection
* **Column 1 — Live Voice & UPI Fraud:**
  * **Voice-Clone Scams:** Emergency kidnapping extortion and fake CBI/police "Digital Arrest" calls coerce instant UPI transfers.
  * **Forged Screenshots:** AI-edited payment receipts and fake PAN/Aadhaar cards defraud merchants and lenders daily.
  * **Zero In-Call Guard:** Victims lack real-time verification while speaking on active phone or WhatsApp calls.
* **Column 2 — AI Slop & KYC Spoofing:**
  * **Short-Video Flooding:** Unlabelled synthetic Reels and Shorts degrade consumer social feeds.
  * **Video-KYC Injection:** Deepfake faces and virtual cameras (OBS/ManyCam) bypass bank and fintech onboarding (RBI V-CIP bypass).
  * **Compression Collapse:** Academic CNNs fail after WhatsApp JPEG Q85 or VoIP Opus re-encoding.
* **Column 3 — Cloud Unit Economics:**
  * **Privacy Violations:** Uploading calls and biometric KYC video violates DPDP Act 2023 and GDPR.
  * **Unsustainable Cost:** Cloud GPU APIs cost $0.02–$0.10 (₹1.70–₹8.50) per scan, blocking continuous screening.
  * **Offline Blindness:** Server-dependent tools fail in low-connectivity rural areas and field investigations.

---

### Slide 3: The ONN-DETECT Solution
* **Header:** Zero-Cloud Forensics at $0 Server Cost
* **Core Pillars:**
  * **Zero-Network Sandbox:** Strips `android.permission.INTERNET` at the OS manifest level; private calls and biometrics never leave the phone.
  * **Sub-180 ms Edge Cascade:** Unifies INT8 SigLIP-B/16 (89.96 MB), 36-ch SRM/NPR CNN, LCNN-MFM speech anti-spoofing (<25 ms), and offline C2PA & Meta AudioSeal verification.
  * **Compression-Hardened Accuracy:** Reaches **0.894 Image AUC** (jumping to **0.900 under harsh JPEG Q85 laundering**) and **0.875 Speech AUC** (59.5% TPR @ 1% FPR).
  * **B2B2C Dual Deployment:** Single shared C++/Kotlin core powering both a viral consumer protection app and an enterprise Video-KYC / Insurance SDK.

---

### Slide 4: Consumer Security Suite
* **Header:** Real-Time Consumer Protection Across Calls, UPI, and Social Feeds
* **Card 1: Scam-Call & UPI Coercion Shield:**
  * *Live 4s Scoring:* LCNN-MFM (508 KiB) scores call windows in <25 ms.
  * *UPI Coercion Alert:* AccessibilityService detects simultaneous call + GPay/PhonePe/Paytm launch.
  * *Pre-Payment Block:* Pops high-priority warning before UPI PIN entry if caller exhibits synthetic speech patterns.
* **Card 2: Family Safe-Voice Vault:**
  * *15s Enrollment:* Stores encrypted family voiceprints locally in Android Keystore.
  * *Dual Biometrics:* Pairs SpeakerVerifier with LCNN anti-spoofing.
  * *Clone Immunity:* Directly eliminates the #1 family emergency extortion scam.
* **Card 3: AI Reel Filter & Forward Guard:**
  * *Feed Automation:* Uses Android 11+ Accessibility gesture APIs.
  * *Auto-Swipe & HUD:* Swipes past AI Reels or displays a live corner authenticity badge.
  * *WhatsApp/Share Guard:* 1-tap Share Sheet check on forwarded voice notes, viral videos, and QR scams.

---

### Slide 5: Enterprise B2B Monetization
* **Header:** Offline SDK for Video-KYC & Claims
* **Enterprise Features:**
  * **Video-KYC Anti-Spoofing:** Blocks face-swaps, voice clones, and virtual-camera injection (OBS/ManyCam) for RBI V-CIP and SEBI onboarding compliance.
  * **Hardware-Signed Receipts:** Signs scan verdicts, media SHA-256, and model release hashes inside Android StrongBox Keystore TEE, creating tamper-proof audit trails for insurance claim fraud.
  * **Receipt & ID Forgery Guard:** Pinpoints spliced amounts and edited fonts on UPI payment confirmation screenshots and forged identity documents.
  * **90%+ Cost Reduction:** Eliminates cloud GPU bills, saving enterprises ₹20–₹40 per onboarding verification.

---

### Slide 6: Core Technological Innovation
* **Header:** Four ML-Systems Breakthroughs Make ONN-DETECT Hard to Replicate
* **1. Base-Margin Cascade Formulation:**
  * Injects INT8 SigLIP-B/16 continuous logit as XGBoost `base_margin` ($\text{margin} = \text{logit}(p) + f_{\text{tree}}$). Prevents decision-tree ties at the top of ROC and lifts **TPR@1%FPR from 4.0% to 28.0%**.
* **2. ViT-to-LiteRT Channels-Last Compiler:**
  * Custom channels-last Keras mirror bypasses broken ONNX shape inference, hitting **$1.97 \times 10^{-6}$ FP32 parity** vs. PyTorch.
  * Static MAP Queries pre-compute frozen attention pooling, saving 589K runtime parameters.
* **3. Zero-Heap APK Memory-Mapping:**
  * Maps the 89.96 MB INT8 graph directly into OS page cache with **0 MB Java heap bloat**.
* **4. Anti-Shortcut Residual Engineering & Provenance:**
  * 36-ch SRM+NPR high-pass filters + DCPT paired loss prevent generator shortcut learning.
  * Treats cryptographic C2PA manifests and Meta AudioSeal neural watermarks as positive provenance proof.

---

### Slide 7: Measured Benchmark Validation
* **Header:** v7 Tower Architecture Lifts Sealed Image AUC to 0.894 (+17.9 pts)
* **Key Empirical Benchmarks:**
  * **0.894 Sealed Image AUC:** Tested on 3,600 images across 68 modern generators (Midjourney v6, Flux, Sora, SDXL, Ideogram).
  * **0.900 Laundered AUC:** Re-encoding test images to uniform 512px JPEG Q85 actually *raises* AUC by removing container bias.
  * **7× Low-FPR Recall Gain:** Base-margin VFM fusion elevates TPR@1%FPR to **28.0%** (vs 4.0%) and TPR@0.1%FPR to **25.0%** (vs 0.5%).
  * **0.875 Speech AUC & 59.5% TPR@1%FPR:** LCNN-MFM scores 4s audio in <25 ms.
  * **309 Automated CI Tests:** 146 Python ML + 29 script parity + 127 Kotlin unit + 2 APK gates.

---

### Slide 8: Competitive Landscape & IP Status
* **Header:** TRL 6–7 System Prototype with Structural Privacy and Cost Moats

| Capability / Metric | **ONN-DETECT (v7)** | Cloud Deepfake APIs | Academic CNNs |
|---|---|---|---|
| **Execution & Privacy** | **100% On-Device (0 Cloud)** | Cloud Upload (DPDP Risk) | Desktop GPU Only |
| **Marginal Cost / Scan** | **₹0.00 ($0 Server Cost)** | ₹1.70 – ₹8.50 ($0.02–$0.10) | N/A (Not Deployed) |
| **512px JPEG Q85 Laundering** | **0.900 AUC (Hardened)** | 0.72 – 0.81 AUC | Collapses (<0.65 AUC) |
| **Live Call & Reel Auto-Skip** | **Yes (<25 ms Audio / <180 ms)** | Impossible (1–3s Network Lag) | No Mobile Runtime |
| **Readiness & Provenance** | **TRL 6–7 \| StrongBox TEE \| IP Pipeline** | Commercial Cloud SaaS \| TLS only | TRL 2–3 Lab Papers \| None |

---

### Slide 9: Market Size & Commercial Model
* **Header:** Targeting a ₹6,800 Cr On-Device SAM and ₹85 Cr 3-Year SOM
* **Market Dimensions:**
  * **Global TAM (₹62,000 Cr / $7.4B):** Global fraud prevention, deepfake detection, and Video-KYC identity verification market.
  * **Serviceable Addressable Market (SAM — ₹6,800 Cr):** Indian banking, fintech onboarding, insurance claim inspection, and consumer mobile cybersecurity.
  * **3-Year SOM (₹85 Cr):**
    * *B2B SDK (₹52.5 Cr):* 35 fintech, NBFC, and insurance enterprise clients at ₹1.5 Cr average ACV (or ₹20–₹40 per KYC verification).
    * *B2C App (₹32.5 Cr):* 1.1M active consumer subscribers at ₹299/year.
* **Go-To-Market Execution:**
  * *B2B:* RBI V-CIP compliant SDK design partnerships with digital lenders and payment gateways.
  * *B2C:* WhatsApp family group fact-checking wedge and UPI Digital Arrest warning shield.

---

### Slide 10: Funding Ask, Runway & KGP Forge Support
* **Header:** ₹25 Lakh Seed Grant Fuels a 12-Month Runway to Enterprise Pilots
* **The Ask:** **₹25,00,000 (₹25 Lakhs)** for 12 months of runway.
* **Budget Allocation:**
  * **CapEx (35% / ₹8.75L):** GPU compute for B-Free synthetic twin dataset generation (₹5.25L) + physical Android NPU testing matrix across MediaTek/Snapdragon chipsets (₹3.50L).
  * **OpEx (65% / ₹16.25L):** Core systems engineering stipends (₹9.50L), enterprise pilot deployment ops (₹3.50L), provisional patent filings & DPDP compliance audit (₹3.25L).
* **KGP Forge & z21 Acceleration Catalyst:**
  * Direct introductions to institutional banking/fintech design partners (Razorpay, PhonePe, ICICI, HDFC).
  * Mentorship on commercial enterprise procurement and IP expansion via IIT Kharagpur SRIC.

---

### Slide 11: Closing
* **Tagline:** **Authenticity at the Edge. Zero Cloud. Zero Compromise.**
* **Team:** Founder: Shubham Kumar | IIT Kharagpur | KGP Forge
* **Repository:** `https://github.com/shubhamiitshk/Onn-Detect-`
* **Release Artifact:** `android/app/build/outputs/apk/release/app-release.apk` (110.93 MB, signed, zero network permissions).
