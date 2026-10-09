# Model card: current baseline

## Intended use

Offline media diagnostics in an Android research application. Current bundled weights cannot support a validated authenticity verdict. Results are Inconclusive; successful feature extraction may still expose labeled diagnostic signals.

## Assets and lineage

The existing SRM image and LCNN speech TFLite binaries are preserved. SHA256 and declared tensor contracts are in android/app/src/main/assets/constants/release_manifest.json. The release is legacy-bundled-unvalidated with validated=false. Hash integrity establishes bytes, not detector quality.

Image input is float32 NHWC[1,32,32,3]. Speech input is float32 NHWC[1,401,40,1]. Output is float32[1,1]. Quantized internal operations do not imply integer I/O. The runtime validates actual shapes, dtypes and finite output.

The previous training source identities, original groups, sealed splits, calibration and original mobile parity are not proven. Historical metrics are not current acceptance evidence. The source LCNN now retains frequency ordering and the shared LFCC contract; this requires newly trained matched weights.

## Output and unsupported cases

Validated releases use real=0/synthetic=1, a frozen calibrated threshold and an explicit operating point. A score is not confidence or a proof of authenticity. Failures and unavailable evidence cannot produce an authentic verdict.

Speech-window, file-level audio, speech/music routing and video aggregation are separate contracts. Existing sampled audio/video cannot produce validated file-level verdicts. Biometric enrollment, face landmarks, lip-sync, true PRNU attribution, cryptographic C2PA verification and SynthID verification are not implemented.

EXIF, metadata strings, Laplacian variance and temporal descriptors are diagnostic observations with alternative explanations. They do not override a verdict. Missing metadata does not establish AI generation.

## Limits

The image baseline is especially limited by32px training and unknown generator diversity. Recompression, screenshots, recapture, retouching, partial edits, unfamiliar camera pipelines and unseen generator updates can shift results. Speech can shift with language, speaker, channel, codec, length, noise and synthesis engine. Non-speech audio is outside a speech detector's validation.

A future release must bind model/frontend/calibration/fusion/data/source/report hashes and pass independent phone testing. The installer rejects unvalidated releases. Do not set validated=true manually or promote historical AUC into a device validation certificate.

## Improvement work

See the complete review for larger corpora, license constraints, teacher/student experiments and distinct speech/music/environmental datasets. The training pipeline now supports strict lazy loading, reproducible checkpoints, configured image resolution, independent fusion splits and staged INT8 exports. These are training infrastructure improvements, not proof of improved detector accuracy.
