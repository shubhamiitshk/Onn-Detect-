# ONN-DETECT Architectural & Engineering Roadmap

**Current Release:** `onn-prod-v10` / `v10.1`
**Last Updated:** 2026-10-26
**Reference Document:** `docs/STATE_OF_THE_ART_REVIEW_2026.md`

---

## Strategic Architecture Overview

ONN-DETECT delivers zero-cloud, privacy-preserving AI media forensics directly on mobile devices (Android 15/16 64-bit ARM64, 16 KiB ELF page alignment). The system operates within an OS-level **zero-network sandbox** (`android.permission.INTERNET` and `ACCESS_NETWORK_STATE` strictly excluded) and enforces deterministic missing-evidence handling (`Float.NaN` for absent signals, serializing to `null`).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          ONN-DETECT ENGINE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  IMAGE FORENSICS                                                                │
│  - Engine A: 20-Channel SRM PatchCraft CNN (142 KiB INT8, 128×128)              │
│  - Engine C: SigLIP-B/16 Foundation Probe (Linear head shipped; TAP candidate)  │
│  - Meta-Fusion: 10-Slot Base-Margin XGBoost Meta-Classifier (20 KiB)            │
├─────────────────────────────────────────────────────────────────────────────────┤
│  AUDIO & VOICE FORENSICS                                                        │
│  - Passive Anti-Spoofing: LCNN-MFM + Versioned LFCC v3 Frontend (508 KiB INT8)  │
│  - Temporal Gating: VoicedDurationGate (≥ 300 ms) & Log-Sum-Exp Pooling         │
│  - Proactive Watermarking: AudioSeal Continuous Presence & Bonferroni Bit Sign │
├─────────────────────────────────────────────────────────────────────────────────┤
│  VIDEO & PHYSIOLOGICAL DYNAMICS                                                 │
│  - Perioral Biomechanics: Landmark Jitter Telemetry (LADDER-inspired, Slot 9)   │
│  - Physiological Signals: Bilateral Malar rPPG Phase Telemetry (CHROM-inspired) │
│  - Cross-Modal Synchrony: AVFF Lip-Sync Cross-Correlation (MAR vs. RMS, Slot 8) │
│  - Multimodal Escalation: VideoScorePolicy with Monotonic Candidate Updates    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  PROVENANCE & HARDWARE ATTESTATION                                              │
│  - Cryptographic Provenance: Offline C2PA v2.1/v2.3 Manifest Verification        │
│  - Enterprise Attestation: Android KeyMint StrongBox Hardware Signing (OID ..17)│
│  - Video-KYC Protection: Camera2 HAL Physical Sensor Audit & Dynamic Hook Guard│
│  - Biometric Security: Family Safe-Voice Vault (AES-256-GCM + AAD Binding)     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Milestone Status & Release Schedule

### Version 10.0 / 10.1 (Current Production Release — Completed)
- [x] **Milestone 1: Cryptographic & Concurrency Hardening**:
  - `StrongBoxReceiptSigner`: Thread-safe double-checked locking, `@Volatile testKeyPair`, fail-closed Keystore handling, `@JvmOverloads` constructor.
  - `VoiceVaultCrypto`: Value equality (`Arrays.equals`/`hashCode`) on `SealedBlob` for deduplication and Room persistence.
  - Release Lineage Reconciliation: Synchronized 400 calibration samples and SHA-256 digests in `release_manifest.json`.
- [x] **Milestone 2 & Milestone 2 Round 2: Video Forensics & NaN Integrity**:
  - `PerioralLandmarkAnalyzer`: Deterministic resource cleanup (`pfd.use`, `MediaMetadataRetriever.release()`, explicit bitmap recycling), MediaPipe 0.10+ reflection, 478-point mesh tracking.
  - BioLip & Bilateral Malar rPPG: Timeline gap preservation without index collapsing, contiguous run segmentation ($\ge 4$ frames) for cardiac phase coherence.
  - Zero NaN Collapse: Replaced toxic `-1f` sentinels with `Float.NaN` across `VideoForensicAnalyzer`, `SignalBreakdownJson`, `ScanResult`, and `ForensicCertificateExporter`. Guarded `VideoScorePolicy.combineMultimodalScore` against IEEE 754 NaN collapse under uncalibrated thresholds.
- [x] **Milestone 3: SOTA Literature Report & Docs Alignment**:
  - Authored comprehensive 2026 SOTA technical report (`docs/STATE_OF_THE_ART_REVIEW_2026.md`) with 8 chapters, rigorous mathematics, 2024–2026 paper citations, and itemized audit changelog.
  - Aligned all documentation (`TODO.md`, `ROADMAP.md`, `docs/ARCHITECTURE.md`, `docs/DUAL_ENGINE_CASCADE.md`, `README.md`) with the verified codebase.

---

### Version 10.2 (Target: Q1 2027 — Maintenance & Device Promotion)
- [ ] **Physical Android Reference Hardware Evaluation**:
  - Capture and attach an independent on-device evaluation report (`measurement_kind: on_device_android`) from physical Android devices (Google Pixel 8/9, Samsung Galaxy S24) to promote `validated: true` per `docs/DEPLOYMENT.md`.
  - Validate thermal throttle resistance and sustained ARM64 XNNPACK latency bounds under repeated 100-image batch stress.
- [ ] **Keras-Native AudioSeal 1D SEANet Lowering**:
  - Construct a direct Keras mirror (`ml/src/export/keras_audioseal_mirror.py`) replicating Meta's 1D SEANet architecture with 1D dilated convolutions and LeakyReLU.
  - Lower the model directly via `tf.lite.TFLiteConverter.from_keras_model` with INT8 quantization, packaging `assets/models/audioseal_detector_int8.tflite` with bit-exact agreeing-bit parity ($|\Delta \text{bits}| = 0$).
- [ ] **Dynamic Mondrian Conformal Prediction Calibration Export**:
  - Export stratified non-conformity quantiles $\hat{q}_{1-\alpha}(m)$ into `assets/constants/conformal_calibration.json` across the 4 primary missingness masks.
  - Expose prediction set size $|\mathcal{C}(X, m)|$ in the UI, enforcing automatic abstention to `INCONCLUSIVE` when $|\mathcal{C}| \neq 1$ (noting exchangeability assumption required for theoretical error bounds).

---

### Version 11.0 (Target: Q2–Q3 2027 — Next-Generation Architectural Upgrade)
- [ ] **B-Free Paired Synthetic Twin Training (Breaking Acquisition Source Bias)**:
  - Rebuild the visual training corpus using SDEdit and latent diffusion autoencoder reconstruction directly on real LAION-5B, Pexels, and DocCI images.
  - Guarantee that every real sample has a semantically identical synthetic twin, eliminating the acquisition source shortcut ($I(\text{source}; Y) = 0$) and reducing per-source real-image FPR below the strict $5\%$ release bound.
- [ ] **Acoustic Presentation Degradation Matching**:
  - Pass both authentic and synthetic audio through identical telephone/VoIP channel chains (Opus $12\text{--}32\text{ kbps}$, AMR-WB, AAC, acoustic reverberation) to neutralize the in-the-wild background noise shortcut.
- [ ] **End-to-End Spatio-Temporal 3D Video Transformer Probe**:
  - Replace frame-sampled keyframe cascades with an on-device 3D lightweight spatio-temporal video transformer (e.g., VideoMAE / CoCoDetect tier 2) operating over 16-frame clips.
  - Direct end-to-end learning of temporal motion vectors and facial articulation biomechanics.
- [ ] **Expanded Multi-Profile Biometric Voice Vault**:
  - Support multi-speaker family enrollment with encrypted local speaker clusters, incremental voiceprint updates, and StrongBox KeyMint hardware attestation receipts per enrolled speaker.

---

## Architectural Principles & Invariants

1. **Zero-Network Airgap**: Under no circumstances will network permissions (`INTERNET`, `ACCESS_NETWORK_STATE`) be added to the Android app manifest. Privacy is guaranteed by OS-level sandbox isolation.
2. **Missing-Evidence Truthfulness**: Absent evidence is strictly `Float.NaN`, never $0.0$, $0.5$, or $-1.0$. Missing values route to learned missing branches in decision trees and serialize to `null` in JSON.
3. **Fail-Closed Provenance**: Cryptographic provenance verification enforces a fail-closed conjunction ($hashVerified \land coseSignatureValid \land certChainValid$). A missing or altered certificate immediately yields `ProvenanceEvidence.Absent`.
4. **Reproducible Release Gates**: No binary asset is promoted without passing all 6 release lineage, consistency, unit, and E2E verification gates.
