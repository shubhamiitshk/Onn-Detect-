# Roadmap

The implementation status and exhaustive review are in docs/. Correctness fixes are being validated; training and device evidence remain separate deliverables.

1. Establish immutable licensed data sources and original groups. Preserve official holdouts. Prepare image train/val/stacking/fusion_val/calibration/test and speech train/val/calibration/test.
2. Run the configured baseline training pipeline with higher-resolution imagery and diverse speech. Rebuild weights for the corrected LFCC and LCNN architecture.
3. Compare residual and compact semantic image candidates using unseen generators and legitimate hard negatives. Train/distill a stronger speech teacher into an offline student and evaluate music/environmental routing separately.
4. Calibrate actual INT8 image fusion and speech operating points; establish validated window-to-file audio and video aggregation and abstention.
5. Measure Android bitmap/codec/frontend/fusion parity, seeded migrations, UI interruption and memory/latency/thermal behavior on representative phones.
6. Attach independent release-specific device reports; install the complete matching bundle and fusion source; inspect and sign the minified release.
7. Add generator/domain drift evaluation with frozen challenge sets. Keep the Android scan offline; any future model updates require an explicitly designed offline import mechanism.

The current legacy release remains Inconclusive. No universal detector guarantee or new headline accuracy is claimed.
