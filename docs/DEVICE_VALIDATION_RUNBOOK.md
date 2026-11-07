# v10.2 Device Validation Runbook — ONN-DETECT (`onn-prod-v10`)

**Status:** `validated: false` is the correct state until this runbook is executed on physical ARM64 hardware and a `device_evaluation.json` report is attached via `release_bundle.py`. Desktop AUC (fusion 0.9974) is *not* a claim about phone performance.

## Why this exists (research)

- **LAION-Mobile (Sep 2026, 935k real phone photos):** no published detector is deployable today. Paper AUC ≥0.99 collapses on real phone photos — e.g. UnivFD FPR 0.2% → **39.1%** on identical phone images. Modern phone ISPs (HDR fusion, denoising, motion suppression) create learned artifacts that resemble generative fingerprints. Fix: `current-device negatives + calibrated threshold + monitored FPR` per deployment.
- **Deepfake-Eval-2024 (CVPR 2026, in-the-wild 2024):** open-source AUC drops **~45–50%** (image/audio/video) vs lab benchmarks; best commercial <90% human-forensic bar. Lesson: score candidates on your own traffic, not public leaderboards.
- **C2PA 2026:** official Trust List + Conformance Program (mid-2025) is the 2026 trust layer; **Interim Trust List frozen Jan 1, 2026**. Valid credential ≠ truth; missing ≠ AI. Show signer + chain; keep ITL fallback for old assets.

Your v10 design already encodes these lessons: `Float.NaN` for missing, learned missing branches, fail-closed C2PA (`hashVerified ∧ coseSignatureValid ∧ certChainValid`), `INCONCLUSIVE` abstention, per-source FPR bound. This runbook is the last gate before `validated: true`.

## Hardware

- One Google Pixel 8/9 (Tensor G3/G4) **or** Samsung Galaxy S24 (Snapdragon 8 Gen 3), ARM64, Android 14+ (app targets SDK 36, `minSdk 26`).
- USB debugging on, `adb devices` shows `device` (not `unauthorized`).
- Battery >60%, charger connected (thermal throttle test), same Wi-Fi/USB for the whole run.

## Build + zero-network audit (must pass first)

```powershell
Push-Location android
try {
  $env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
  .\gradlew.bat testDebugUnitTest assembleRelease
  if ($LASTEXITCODE -ne 0) { throw 'build failed' }
} finally { Pop-Location }
.\.venv\Scripts\python.exe onn_release_lineage_gate.py --repo .
.\.venv\Scripts\python.exe scripts\check_android_consistency.py
.\.venv\Scripts\python.exe scripts\verify_android_artifact.py --apk android\app\build\outputs\apk\release\app-release.apk
```

Expect: lineage PASS; consistency PASS + one warning `desktop-validated only`; APK PASS (permissions, 16 KiB ELF, hashes). Anything else → stop, fix, re-run.

## Corpus — phone negatives (LAION-Mobile style)

Desktop calibration (400 sealed samples) does not cover modern ISPs. Collect **real** negatives from the target device itself, consented:

```powershell
# 1. Pull the phone gallery (or place consented photos in a folder)
.\.venv\Scripts\python.exe scripts\collect_phone_negatives.py --source .\phone_negatives_raw --out .\phone_negatives_manifest.json --consent-confirmed
```

Collector (stdlib + optional Pillow) does:
- keep JPEG/HEIC only; reject screenshots / screen photos / uniform blocks (non-photo heuristics);
- require smartphone EXIF (Make/Model) and record per-image device metadata;
- dedup by SHA-256; strip GPS from the working copy (retain EXIF device metadata in the manifest only);
- emit a manifest you review by eye before use. **Do not train on it until reviewed.**

Minimum for a credible FPR: **≥200 real photos per device**, across lighting / skin tones / ages, group-disjoint from the desktop calibration split. Add **contiguous AI positives from current generators** (post your training cutoff — diffusion video, Sora/Runway-class) with the same ISP/compression chain where possible. If phone FPR >5% per source at the desktop threshold → **do not promote**; recalibrate threshold on the phone set (Mondrian conformal) and accept lower TPR.

`scripts/generate_bfree_twins.py` exists for the v11 B-Free fix (paired real↔synthetic twins, `I(source;Y)=0`). Use it when the phone FPR bound fails, not before measuring it.

## On-device execution

```powershell
# Real run (not --dry-run). Runs 50 warm scans, cold start, meminfo, thermal drift.
.\.venv\Scripts\python.exe scripts\benchmark_device_arm64.py --package com.onndetect
# macOS/Linux: python3 scripts/benchmark_device_arm64.py

Push-Location android
try { .\gradlew.bat connectedDebugAndroidTest; if ($LASTEXITCODE -ne 0) { throw 'device tests failed' } }
finally { Pop-Location }
```

Pass criteria (all must hold; report carries them):
- `measurement_kind == on_device_android`, `on_device_measurements` non-empty, no desktop-only keys in per-model entries (`assert_device_report_is_from_a_device` in `ml/scripts/release_bundle.py` enforces this).
- Every packaged model: SHA-256 + `frontend_id` match manifest; `parity_passed == true` (Kotlin vs Python within sealed bounds).
- Fusion: `fusion_revision == release_id`, `kotlin_parity_passed`, `offline_verified`, `fusion_kotlin_sha256` matches `assets/fusion/XGBoostMetaClassifier.kt`, `source_snapshot_sha256` matches manifest.
- Zero-network: `INTERNET`/`ACCESS_NETWORK_STATE` absent (audit in benchmark + APK gate); `dumpsys meminfo` Dalvik < ~20 MB, total PSS consistent; thermal drift <25% over 50 scans; p50/p90/p99 recorded.
- Phone FPR at ship threshold recorded per device; if any device exceeds the 5% release bound → stop. Current corpus is known to violate this bound (see `ml/scripts/release_bundle.py --allow-source-fpr-violation` note) — that violation is recorded metadata, not a pass.

Writes `docs/test_artifacts/device_evaluation.json` (benchmark) as an intermediate. The full report attached to the bundle needs the fields above; `release_bundle.py` validates them strictly.

## Promotion (no hand-edits)

```powershell
.\.venv\Scripts\python.exe ml/scripts/release_bundle.py --bundle android/app/src/main/assets --device-report <phone-report.json> --require-validated
```

- `attach_device_report` refuses if `validated` is already true ("use a new release version") and refuses if `*.previous-device-*` exists. It stages a copy, validates the candidate (`require_validated=True`), renames bundle→backup, candidate→bundle, with rollback retry.
- Verify after promotion: `release_manifest.json` has `validated: true`, `validation_status: desktop_and_independent_device_validation_passed`, `device_validation.passed: true` with `report_hash` == SHA-256 of `device_evaluation.json`, and `assets/device_evaluation.json` present.
- Re-run `onn_release_lineage_gate.py`, `check_android_consistency.py`, `verify_android_artifact.py` (now without the desktop-only warning). Install only with `--install <dest> --kotlin-destination <XGBoostMetaClassifier.kt>`.

**Anti-patterns (never do):** fabricating a report from desktop timing; copying `device_evaluation.json` from another device/SoC; promoting with `--install-desktop-validated` and flipping `validated` by hand; deleting the `*.previous-device-*` backup to hide a failure.

## C2PA trust note (2026)

`assets/trust/c2pa_trust_anchors.pem` (currently 32 anchors, 40,108 B) ships offline. `scripts/fetch_c2pa_trust_anchors.sh` already points at the official `c2pa-org/conformance-public … C2PA-TRUST-LIST.pem` URL (>=30-anchor sanity, optional `C2PA_TRUST_ANCHORS_SHA256` pin). Before every release:

```bash
C2PA_TRUST_ANCHORS_SHA256=<pinned> bash scripts/fetch_c2pa_trust_anchors.sh
```

Pin the digest in CI. Keep the Interim TL values only as legacy fallback for pre-2026 assets; surface "official vs interim" in the result UI. `C2PATrustStore` already fails closed on empty list and expired/not-yet-valid certs.

## Forward Guard beta scope (until a phone passes)

Image-only share-sheet (`image/*`): processor = provenance (C2PA fail-closed) + statistical score at ship threshold. Voice/audio/video routes are **diagnostic only** (`INCONCLUSIVE` unless provenance AT1 confirms). Chat/VLM explains evidence; it never invents a verdict. Hold v11 (B-Free, 3D video probe, codec matching) until this gate passes.
