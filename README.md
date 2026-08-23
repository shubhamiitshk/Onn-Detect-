# 🕵️‍♂️ ONN-DETECT

ONN-DETECT is a multimodal deepfake detection pipeline designed for high-accuracy and on-device (Android) inference. It combines a lightweight visual CNN with an audio classification network, fusing their outputs via an XGBoost meta-classifier.

## 🚀 Features

- **Visual Deepfake Detection (SRM-CNN)**: A MobileNetV3-inspired architecture prepended with fixed Spatial Rich Model (SRM) filters to detect high-frequency AI-generation artifacts. Trained on the CIFAKE dataset.
- **Audio Deepfake Detection (LCNN-MFM)**: A Light-CNN with Max-Feature-Map activations trained to distinguish authentic human speech from TTS and synthetic audio. 
- **Meta-Classifier Fusion (XGBoost)**: Fuses scores from the visual and audio models. Automatically exports decision trees into pure, zero-dependency Kotlin code for lightning-fast Android inference.
- **Optimized Training**: Features Automatic Mixed Precision (AMP), multi-core DataLoader fetching, `torchcodec` audio loading, and robust metric calculations (EER, AUC).

---

## 📊 Dataset Requirements

The pipeline automatically fetches its required datasets via HuggingFace:
1. **Visual (CIFAKE)**: 120,000 images (60k CIFAR-10 real, 60k Stable Diffusion fake).
2. **Audio**: 1,866 authentic/synthetic speech audio clips.

---

## 🛠️ Installation & Setup

We recommend using [`uv`](https://github.com/astral-sh/uv) for lightning-fast dependency management.

1. **Clone & Environment Setup:**
   ```powershell
   git clone https://github.com/your-username/Onn-Detect.git
   cd Onn-Detect
   uv venv --python 3.10
   & .venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies:**
   ```powershell
   # Install CUDA PyTorch (Adjust index URL for your OS/GPU)
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # Install ML dependencies and local package
   uv pip install -r ml/requirements.txt datasets huggingface_hub pillow tqdm soundfile torchcodec
   uv pip install -e ml/
   ```

---

## 🏃‍♂️ Training Pipeline

### 1. Download Datasets
Fetches the CIFAKE and Audio datasets and organizes them into `train/`, `val/`, and `test/` splits.
```powershell
cd ml
uv run python scripts/download_all_datasets.py
uv run python -c "import shutil, glob, os; [shutil.copy(f, '../datasets/genimage/test/real/' + os.path.basename(f)) for f in glob.glob('../datasets/genimage/val/real/*.jpg')]; [shutil.copy(f, '../datasets/genimage/test/fake/' + os.path.basename(f)) for f in glob.glob('../datasets/genimage/val/fake/*.jpg')]"
```

### 2. Train Visual Model (SRM-CNN)
Trains the image forgery detector. Expect ~0.98 AUC after 10-15 epochs.
```powershell
uv run python scripts/train_srm_cnn.py --device cuda --amp --epochs 25 --batch-size 64
```
*(To run evaluation only on the best checkpoint, pass `--eval-only`)*

### 3. Train Audio Model (LCNN-MFM)
Trains the audio forgery detector.
```powershell
uv run python scripts/train_lcnn_audio.py --device cuda --amp
```

### 4. Train XGBoost Meta-Classifier
Fuses the visual and audio outputs and generates the `XGBoostMetaClassifier.kt` file.
```powershell
uv run python scripts/train_xgboost_meta.py
```

---

## 📈 Performance Benchmarks
*Hardware: RTX A4000 / i9-14900K*

### Visual Model (SRM-CNN)
- **Test AUC**: 0.9804
- **Test Accuracy**: 91.20%
- **Speed**: ~2.5 mins per epoch (using `num_workers=4` & AMP)

### Audio Model (LCNN-MFM)
- **Test AUC**: 0.9755
- **Test EER**: 5.00%
- **Speed**: ~20 seconds per epoch

### Fusion Model (XGBoost Meta-Classifier)
- **Test AUC**: 1.0000
- **Test Accuracy**: 99.75%
