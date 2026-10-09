# Benchmark evidence

No reproducible accuracy benchmark for the current mobile frontend and bundled weights is available.

The old ml/evaluation_results.json and numeric values in legacy constants are retained as historical artifacts. They used inconsistent audio polarity/frontends and inadequately established image/fusion splits. Do not quote them as current accuracy, use them to choose a deployment threshold or invert a score to improve a headline AUC.

The October6 review records the original measurements and defects. The implementation status document records code/build/regression checks. Neither is a newly trained detector benchmark.

## Required experiment record

A release report must identify source snapshot/commit, effective configs and resolved dependencies, immutable dataset revisions, original source/group IDs, disjoint splits and their manifests, checkpoint hashes, tensor/frontend contracts and the actual quantized models.

Select a model on validation only. Fit operating thresholds on calibration only. Report sealed-test counts per class and generator/speaker/source family, AUC, EER, accuracy at the frozen threshold, TPR at low fixed FPR and real-media false positives. Add uncertainty intervals and subgroup results before comparing models.

For modern platform images, reserve independently collected ChatGPT/GPT-image, Nano Banana and other unseen generator/version samples, plus camera photos and difficult legitimate edits. Evaluate screenshot, JPEG, resize, crop, blur, recapture and mixed real/synthetic edits separately. Do not reuse the sealed test for model or threshold selection.

Speech requires multilingual, speaker-disjoint, generator-disjoint and recording/channel/codec holdouts. Music and environmental sound need their own benchmark and routing. Window scores do not establish calibrated file-level probability.

## Deployment comparison

Compare PyTorch, Keras and INT8 outputs on identical labeled samples. Enforce score error, agreement at the actual operational threshold, AUC loss and low-FPR sensitivity loss. Independently check the Android decoder/frontend and compiled fusion, then measure latency, peak memory, temperature and behavior on interruption and damaged media.

Current regression tests can establish arithmetic/contracts and safe abstention. They cannot establish universal detection or new accuracy.
