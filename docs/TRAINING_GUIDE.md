# Training guide

Training is a separate workstation/runner task. Android inference remains offline. Current binaries are preserved and unvalidated; corrected source code does not retroactively retrain them.

## Environment

Use Python 3.11 and a dedicated environment. Direct dependencies are pinned in requirements.txt, requirements-export.txt, requirements-data.txt and requirements-dev.txt. Keep the resolved dependency snapshot with every experiment. Install the CUDA-compatible torch build deliberately for a GPU runner; do not force CUDA on a hosted CPU runner.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ml/requirements.txt -r ml/requirements-export.txt -r ml/requirements-data.txt -r ml/requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e ml
.\.venv\Scripts\python.exe -m unittest discover -s ml/tests -p "test_*.py" -v
.\.venv\Scripts\python.exe scripts/verify_front_end_parity.py --require-torch
.\.venv\Scripts\python.exe scripts/verify_export_path.py --full
```

The existing development machine has about 8 GB RAM and a 4 GB RTX 3050 laptop GPU; its installed torch is CPU-only. Use batch sizes that fit measured memory, zero workers initially and lazy loaders. Do not eager-cache a large corpus. The provided 256px config is a baseline experiment, not the best model established by evidence. Large teacher training is better suited to a provisioned workstation.

The optional Docker image needs the **complete Git checkout** mounted at `/workspace/repository`, including `.git`, Android sources and root tooling. An image built from `ml/` alone can display help, but the pipeline rejects it before training because release source snapshots require the complete checkout. From a Linux checkout, a CPU run can use:

```bash
docker build -t onndetect-ml ml
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e KERAS_HOME=/tmp/onn-keras \
  -v "$PWD:/workspace/repository:ro" \
  -v /absolute/verified-datasets:/workspace/data:ro \
  -v /absolute/experiment-output:/workspace/output \
  onndetect-ml \
  --visual-config configs/srm_cnn_openfake.yaml \
  --audio-config configs/lcnn_asvspoof.yaml \
  --data-root /workspace/data --device cpu --output-dir /workspace/output
```

Create the host output directory with the same user that owns the checkout. Matching the container UID/GID avoids Git's ownership rejection and keeps output files writable by that user; the Keras configuration uses a writable container temporary directory. Use a fresh output directory. The pipeline preserves existing checkpoints and staged releases; verified checkpoint reuse requires `--skip-training`, and individual continuation requires `--resume`. Set up compatible CUDA dependencies and the GPU container runtime separately for a GPU image. The Docker image was reviewed but not built during the local validation run.

## Data acquisition and identity

See the complete review for primary-source links, OpenFake/Community Forensics, ASVspoof5, MLAAD/SpeechFake/AUDETER, music/sound corpora and license/access restrictions. Non-commercial dataset licenses and generated-content/API rights are separate from this repository's code license.

The pilot downloader only names the actual supported source:
```powershell
python ml/scripts/download_all_datasets.py --dataset cifake --revision IMMUTABLE_40_CHARACTER_HF_COMMIT --output-dir D:/data/cifake-source --limit 2000
python ml/scripts/download_all_datasets.py --dataset gary-stafford-audio --revision IMMUTABLE_40_CHARACTER_HF_COMMIT --output-dir D:/data/audio-pilot
```

Replace the revision placeholder with the verified source commit. No implicit corpus download, guessed numeric labels, dataset deletion or automatic counter-based train/test split occurs. These pilots retain original source splits and do not establish verified original groups. They cannot train a release until preparation is complete. The Gary Stafford corpus is not ASVspoof.

Obtain original image/recording IDs, generator/version, subject/speaker/session and parent/derived relationships from source metadata. All crops, recompressions, edits, synthesis variants and related recordings of an original must stay in one split. Keep official benchmark holdouts sealed. Use perceptual/audio similarity to find near duplicates in addition to exact hashes; hash equality alone cannot exclude all leakage.

## Prepared layout

Each corpus has real=0/fake=1 class folders and an authoritative dataset_manifest.json.

```text
visual/
  train/{real,fake}/
  val/{real,fake}/
  stacking/{real,fake}/
  fusion_val/{real,fake}/
  calibration/{real,fake}/
  test/{real,fake}/
audio/
  train/{real,fake}/
  val/{real,fake}/
  calibration/{real,fake}/
  test/{real,fake}/
```

Files may be nested within a class folder; labels are determined by the first class folder. A group-map JSON maps every root-relative path to authoritative source_id and group_id. For example, an original camera photo and all derived versions share its group_id. A generated image paired with that original may also need the same group when the protocol would otherwise leak the source.

```powershell
python ml/scripts/prepare_dataset_manifest.py --root D:/verified-datasets/highres_visual --source VERIFIED_SOURCE_NAME --revision IMMUTABLE_REVISION --group-map D:/metadata/visual-groups.json --include-fusion
python ml/scripts/prepare_dataset_manifest.py --root D:/verified-datasets/asvspoof2019 --source OFFICIAL_ASVSPOOF_SOURCE --revision IMMUTABLE_RELEASE --group-map D:/metadata/audio-groups.json
```

This hashes and decodes media, checks missing splits/classes and rejects exact/group overlap. It verifies the supplied group identities, not their semantic correctness. Preserve grouping metadata and independently audit it. Corrupt inputs raise errors for quarantine; no black images or silence substitutes are emitted.

## Train and export

Effective YAML configs control image size, sample rate/window, batch/worker counts, AMP, optimizer, scheduler and early stopping. The image v2 policy orients EXIF, converts embedded profiles to sRGB, composites transparency on white and creates a bounded working view before square resize; resolution_norm retains original dimensions. The shared frontend ID is rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-sizeN-float32. Pillow and BitmapFactory sampling still require measured native device parity. Speech float32 LFCC/resampling must also match the mobile contract. Random audio crops are training-only; evaluation is deterministic. Checkpoints retain model, optimizer, scaler, early stopping, history, RNG and data/frontend contract for faithful resume.

```powershell
cd ml
../.venv/Scripts/python.exe scripts/train_srm_cnn.py --config configs/srm_cnn_openfake.yaml --device cuda --output-dir runs/experiment-001/checkpoints
../.venv/Scripts/python.exe scripts/train_lcnn_audio.py --config configs/lcnn_asvspoof.yaml --device cuda --output-dir runs/experiment-001/checkpoints
```

Use --resume with a verified last checkpoint and the same optimization/data config. Legacy checkpoints lacking lineage are rejected; deliberate retraining is required. Do not automatically negate below-chance predictions.

The combined pipeline preflights both corpora before expensive work:
```powershell
../.venv/Scripts/python.exe scripts/run_training_pipeline.py --visual-config configs/srm_cnn_openfake.yaml --audio-config configs/lcnn_asvspoof.yaml --data-root D:/verified-datasets --device cuda --output-dir runs/experiment-001
```

With --data-root, a relative configured data root uses its basename beneath that directory; absolute roots remain absolute. The high-resolution config's folder is highres_visual. Defaults name the actual CIFAKE pilot rather than pretending it is full GenImage.

The pipeline trains base models, exports genuine INT8 models with float I/O, verifies labeled parity, fits fusion using deployed INT8 base scores and independent stacking/fusion_val, calibrates actual deployed operating thresholds and reports sealed tests. It stages all weights, constants, fusion source and evidence as one bundle. It fails on absent calibration, mismatched frontend or missing lineage. AUC above chance is a minimum sanity gate, not a sufficient product accuracy target.

Synthetic export tests exercise layout and conversion; they do not measure detector performance. Keep the parity sample cap separate from sealed-test scope. Use low-FPR sensitivity, real-media false positives, uncertainty and unseen-generator/domain results to compare experiments.

## Larger datasets and models

Acquire licensed modern-platform samples with immutable generator/version metadata and a sealed challenge set. Include difficult legitimate camera/edit/compression cases. Add dataset mixture/source balancing and unseen-family evaluation before increasing scale.

Media loading is lazy and bounded, but JSON manifests, membership indexes and grouping metadata still use memory proportional to file count. Measure this overhead before preparing a corpus with millions of items. Sharded manifests with a disk-backed duplicate/group index and source-balanced sampling are additional scaling work; the current pipeline does not claim constant-memory metadata processing.

Compare high-resolution residual cues with a compact semantic encoder and distill a stronger image teacher when justified by measured generalization and phone budgets. The code does not yet implement a production multi-patch ensemble or select a compact encoder from an untested leaderboard.

For speech, compare the corrected LFCC/LCNN student with modern self-supervised/raw-audio teachers. Train multilingual speaker/source/generator-disjoint data and channel/codec augmentations, then distill/export a phone-compatible student. Music and environmental sounds need different data, routing and task models. These research experiments require real weights and evidence; naming a teacher does not implement it.

## Install and distribute

Desktop exports remain validated=false until independent, release-specific Android model/frontend/fusion/offline evidence is attached. Read DEPLOYMENT.md for the measured report schema and installer. Audio/video file verdicts additionally require their own routing/aggregation calibration.

```powershell
cd .. # Return from ml/ to the repository root.
python ml/scripts/release_bundle.py --bundle ml/runs/experiment-001/bundle --validate-only
python ml/scripts/release_bundle.py --bundle ml/runs/experiment-001/bundle --device-report D:/measured/device_evaluation.json --require-validated
```

Use scripts/package_for_android.sh for the explicit complete install transaction. It supplies both assets and matching compiled Kotlin fusion destinations. The previous assets are retained for recovery. Build and inspect the resulting minified APK before signing and distributing.

CI runs CPU contract tests without datasets. Manual training uses a provisioned self-hosted runner tagged onndetect-ml with external data_root. Manual artifact verification fetches a specific training run's complete bundle; it does not pretend to rerun accuracy without the datasets. Keep that runner current for the pinned GitHub actions.
