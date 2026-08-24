# Training Guide

All commands run from the repo root on Windows (PowerShell) unless noted.

## 1. Environment

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision torchaudio
.venv\Scripts\python.exe -m pip install -r ml\requirements.txt
.venv\Scripts\python.exe -m pip install -e ml
```

## 2. Datasets

Expected layout:

```
datasets/genimage/{train,val}/{real,fake}/*.{png,jpg}      # native 32x32 thumbnails
datasets/asvspoof2019/{train,val,test}/{real,fake}/*.flac
```

Helper downloaders live in `ml/scripts/download_*.py` (Hugging Face based).
**Never resize the 32x32 images up** before training or inference; bilinear
upsampling interpolates away exactly the noise statistics SRM filters need
(this caused the original chance-level AUC 0.500).

## 3. Train

```bash
cd ml
make train-visual     # scripts/train_srm_cnn.py   (SRM-CNN, native res)
make train-audio      # scripts/train_lcnn_audio.py (LCNN-MFM, ASVspoof)
make train-meta       # scripts/train_xgboost_meta.py (fusion + Kotlin transpile)
```

Useful overrides: `--device cuda --epochs N --batch-size B`. High-resolution
retraining is supported via `IMG_SIZE=256` + `DATASET_DIR` env vars, but the
shipped model assumes native 32x32 input.

## 4. Export to Android

```bash
make export           # PyTorch -> Keras -> TFLite INT8, parity-checked on real inputs,
                      # copies *.tflite into android/app/src/main/assets/models/
python scripts/generate_android_constants.py   # baselines, EER threshold, meta contract, hashes
python ../scripts/check_android_consistency.py # static cross-platform contract check
```

The export pipeline refuses to ship silently-broken models: it validates
PyTorch-vs-TFLite agreement on real calibration inputs and verifies the
transpiled Kotlin against `booster.predict()` (<1e-4).

## 5. Evaluate

```bash
make eval             # writes ml/evaluation_results.json from held-out splits
```
