# Remaining acceptance work

Implemented code fixes and concrete check results are tracked in docs/FIX_STATUS_2026-10-07.md.

- Acquire licensed larger image and speech corpora with immutable versions and authoritative original/group identities. The pilot downloader does not fabricate missing metadata.
- Train matched higher-resolution image and corrected LFCC/LCNN weights; compare compact candidates rather than selecting one from published headline metrics.
- Evaluate sealed unseen generator/version holdouts, modern platform samples and legitimate edits; establish low false-positive operating points with uncertainty and subgroup reporting.
- Validate speech/non-speech routing and file-level audio aggregation; validate video sampled aggregation separately.
- Run codec/bitmap/frontend/fusion fixtures and performance/thermal/memory tests on representative devices. Emulator regression evidence does not replace phone validation.
- Attach release-specific independent device reports and pass the strict distribution APK gate before publishing a validated detector.
- Implement true cryptographic provenance or watermark verification only when a usable verifier and offline license/runtime are established.

Legacy model binaries are retained and explicitly unvalidated. Changing that flag or inventing test evidence is not a fix.
