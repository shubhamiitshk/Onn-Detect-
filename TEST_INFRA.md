# ONN-DETECT: E2E Test Infrastructure & Verification Specification

**Document Version:** 1.0.0  
**Target Repository:** ONN-DETECT (`onn-prod-v10` / `v10.1`)  
**Scope:** Dual-Track Opaque-Box E2E Testing across Tiers 1–4 covering all 12 features in `PROJECT.md`.  
**Execution Context:** Zero-Cloud On-Device AI Media Forensics Engine & Android SDK.  

---

## 1. Architecture & Testing Strategy Overview

The ONN-DETECT verification architecture employs a **Dual-Track Testing Model** ensuring rigorous, independent quality assurance:
- **Implementation Track**: Authors and refactors subsystem implementations, fixes defects, and runs white-box unit tests.
- **E2E Testing Track**: Designs and maintains an independent, opaque-box end-to-end test suite operating strictly through public APIs, CLI entry points, artifact contracts, and data-flow interfaces.

### Core Testing Principles
1. **Opaque-Box Independence**: Tests exercise components through their external contracts without coupling to internal private states or monkey-patching implementation details.
2. **Authoritative Expected Output Derivation**: Every test case derives expected results from mathematically proven formulations, reference oracles (NumPy, RFC 8785, RFC 9052, IEEE 754), or authoritative specifications in `PROJECT.md` and `CLAUDE.md`.
3. **Truthfulness & Missing Evidence Contract (`CLAUDE.md`)**:
   - Scores are strictly real=0.0 (authentic), synthetic=1.0 (manipulated), bounded in $[0.0, 1.0]$.
   - Missing evidence is represented as `Float.NaN` or `null`, never `-1.0` or `0.0`.
   - Ambiguous or conflicting signals strictly abstain as `INCONCLUSIVE`.
4. **Zero-Network Air-Gap Assertion**: The system operates completely offline; zero network calls or network permissions (`INTERNET`, `ACCESS_NETWORK_STATE`) are permitted.

---

## 2. Four-Tier Testing Hierarchy & Methodology

```
+------------------------------------------------------------------------------------+
| Tier 1: Feature Coverage (Category-Partition Method)                               |
|   >= 5 tests per feature across all 12 features in PROJECT.md Feature Inventory     |
|   Nominal inputs, primary behavior, success paths, contract conformance            |
+------------------------------------------------------------------------------------+
| Tier 2: Boundary & Corner Cases (Boundary Value Analysis & Stress)                 |
|   >= 5 tests per feature covering NaN, 0-byte buffers, extreme aspect ratios,      |
|   clamping limits, single-bit corruptions, malformed containers                    |
+------------------------------------------------------------------------------------+
| Tier 3: Cross-Feature Combinations (Pairwise & Multi-Engine Interactions)          |
|   Combinatorial verification: Image + Video + Audio + C2PA + B2B Security          |
|   Dual-Engine Cascade, Lip-Sync + VoiceVault, Virtual Camera + StrongBox           |
+------------------------------------------------------------------------------------+
| Tier 4: Real-World Application Scenarios (End-to-End Operational Workflows)       |
|   WhatsApp/Telegram Laundered Media, RBI V-CIP Video-KYC Live Verification,        |
|   Safe-Voice Banking Vault Authentication, Newsroom C2PA Deepfake Triage           |
+------------------------------------------------------------------------------------+
```

### 2.1 Methodology Details

- **Tier 1 (Category-Partition)**: Identifies functional categories, parameters, and nominal equivalence partitions. Validates that each feature produces expected outputs given valid inputs.
- **Tier 2 (Boundary Value Analysis)**: Tests the extremities of input domains: $\epsilon$ below minimum, exact minimum, nominal, exact maximum, $\epsilon$ above maximum, empty buffers, all-zero arrays, NaN/infinity, corrupt payloads, and memory exhaustion boundaries.
- **Tier 3 (Pairwise Combinatorial Testing)**: Evaluates interactions between subsystem pairs to detect emergent defects, race conditions, score-masking, and contradictory verdicts.
- **Tier 4 (Real-World Workloads)**: Simulates end-to-end user workflows with realistic multi-modal media files, transcoded streams, security challenges, and enterprise attestation chains.

---

## 3. Feature Coverage Mapping Matrix (12 Features)

| Feature # | Feature Name | Tier 1 Tests | Tier 2 Tests | Primary Focus |
|---|---|---|---|---|
| **F01** | Tooling & Receipt Verification Harness | 5 | 5 | CLI flags, exit codes, cert chain validation, tamper detection, missing deps |
| **F02** | StrongBoxReceiptSigner Concurrency | 5 | 5 | Thread safety, race conditions, key overwrite prevention, volatile key |
| **F03** | VoiceVaultCrypto SealedBlob Equality | 5 | 5 | Structural equality, hashCode consistency, AES-256-GCM tamper rejection |
| **F04** | Lineage Metadata Reconciliation | 5 | 5 | SHA-256 bindings, n_calibration alignment, TFL3 headers, split manifests |
| **F05** | LipSyncDetector Lifecycle & Descriptors | 5 | 5 | Resource cleanup, FileDescriptor closure, bitmap recycling, memory leaks |
| **F06** | BioLip / rPPG Temporal Continuity | 5 | 5 | Acceleration calculation, non-consecutive frame handling, cheek rPPG phase |
| **F07** | NaN Truthfulness & Contract Enforcement | 5 | 5 | Sentinel elimination (-1f -> NaN/null), score bounds [0, 1], JSON serialization |
| **F08** | 2025–2026 SOTA Literature Report | 5 | 5 | 8-chapter completeness, SOTA methods, citation validity, mathematical formulas |
| **F09** | Documentation Synchronization | 5 | 5 | Cross-file consistency (README, ROADMAP, TODO, ARCHITECTURE, DUAL_ENGINE) |
| **F10** | AudioSeal Runtime & Parity Test Suite | 5 | 5 | 16-bit sign test, exact binomial p-value, absence invariant, silence handling |
| **F11** | VideoScorePolicy Boundary & NaN Tests | 5 | 5 | Multimodal arbitration, temporal/lip anomaly boost, missing modality handling |
| **F12** | E2E Test Suite & Adversarial Hardening | 5 | 5 | Full pipeline execution, adversarial compression, zero-network validation |
| **Total** | **All 12 Features** | **60** | **60** | **120 Core Tier 1 & 2 Tests + Pairwise + Real-World** |

---

## 4. Test Suite Layout & Implementation Organization

```
tests/e2e/
├── __init__.py                     # Package marker
├── conftest.py                     # Synthetic fixtures, mock data, sample generators
├── test_tier1_features.py          # Tier 1: 60+ Feature Coverage tests (F01-F12)
├── test_tier2_boundaries.py        # Tier 2: 60+ Boundary & Corner Case tests (F01-F12)
├── test_tier3_pairwise.py          # Tier 3: 15+ Cross-Feature Pairwise Interaction tests
├── test_tier4_scenarios.py         # Tier 4: 10+ Real-World Application Workflow tests
└── run_e2e_tests.py                # Standalone test runner with rich terminal reporting
```

---

## 5. Test Runner & Execution Commands

### Primary Pytest Execution
```powershell
.\.venv\Scripts\python.exe -m pytest tests/e2e -v
```

### Standalone Self-Contained Runner
```powershell
.\.venv\Scripts\python.exe tests/e2e/run_e2e_tests.py
```

### Supplementary Subsystem Gates
- **Root Tooling & Receipt Suite**: `.\.venv\Scripts\python.exe -m unittest discover -s scripts/tests -v`
- **Release Lineage Gate**: `.\.venv\Scripts\python.exe onn_release_lineage_gate.py --repo .`
- **Cross-Platform Consistency**: `.\.venv\Scripts\python.exe scripts/check_android_consistency.py`
- **Python ML Parity Suite**: `.\.venv\Scripts\python.exe -m pytest ml/tests`
- **Android Unit Tests**: `cd android && .\gradlew.bat testDebugUnitTest`
- **APK Verification Gate**: `.\.venv\Scripts\python.exe scripts/verify_android_artifact.py --apk android/app/build/outputs/apk/release/app-release.apk`

---

## 6. Coverage Thresholds & Quality Gates

1. **Pass Rate Threshold**: 100% of tests in Tiers 1–4 must execute cleanly with zero errors.
2. **Deterministic Execution**: Tests must not rely on network access, non-deterministic random seeds without explicit initialization, or system-dependent file order.
3. **Isolation & Cleanup**: Every test must create its own isolated temporary workspaces, mock data, and cleanly destroy them upon teardown.
4. **Contract Fidelity**: No test shall bypass or weaken the `CLAUDE.md` truthfulness invariants.
