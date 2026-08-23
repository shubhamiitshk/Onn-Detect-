"""Export trained ONN-DETECT models to TFLite for on-device inference.

Fixes vs the previous version:
- Actually LOADS the trained checkpoints (previously exported random-init
  weights, making the .tflite files useless even when they existed).
- Conversion goes PyTorch -> Keras (weight transfer) -> TFLite, since
  ai-edge-torch needs torch_xla which has no Windows wheels.
- Calibration datasets use REAL data from the training distribution.
- Parity validation runs PyTorch-vs-TFLite on REAL inputs, not noise.
- Copies final INT8 models into the Android assets directory.

Android input contract (NHWC float32):
- srm_forensic_cnn_int8.tflite : (1,32,32,3) RGB in [0,1] -> (1,1) P(fake)
- lcnn_audio_int8.tflite       : (1,T,40,1) LFCC          -> (1,1) P(fake)
"""

import os
import sys
import glob
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src")))

import numpy as np
import torch
from PIL import Image

from models.srm_forensic_cnn import SRMForensicCNN
from models.lcnn_mfm_audio import LCNNMFMAudio
from export.quantize_int8 import (
    quantize_keras_model,
    visual_calibration_dataset,
    audio_specs_calibration_dataset,
)
from export.torch_to_tflite_keras import (
    build_keras_srm,
    build_keras_lcnn,
    keras_to_tflite,
    keras_parity_check,
)
from validate_export import validate_models_real

EXPORT_DIR = os.path.join(ROOT_DIR, "ml", "exported")
ASSETS_DIR = os.path.join(ROOT_DIR, "android", "app", "src", "main", "assets", "models")
CKPT_DIR = os.path.join(ROOT_DIR, "ml", "checkpoints")
GENIMAGE_VAL = os.path.join(ROOT_DIR, "datasets", "genimage", "val")
ASVSPOOF_TEST = os.path.join(ROOT_DIR, "datasets", "asvspoof2019", "test")


def list_files(folder, exts, limit):
    out = []
    for cls in ("real", "fake"):
        d = os.path.join(folder, cls)
        if not os.path.isdir(d):
            continue
        files = sorted(
            glob.glob(os.path.join(d, f"*{exts[0]}"))
            + glob.glob(os.path.join(d, f"*{exts[1]}"))
        )
        out.extend(files[: max(limit // 2, 1)])
    return out


def load_srm():
    ckpt = torch.load(os.path.join(CKPT_DIR, "srm_cnn_genimage_best.pth"),
                      map_location="cpu", weights_only=True)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model = SRMForensicCNN()
    model.load_state_dict(state)
    model.eval()
    print(f"SRM-CNN loaded (training best val AUC {float(ckpt.get('best_metric', 0)):.4f})")
    return model


def load_lcnn():
    ckpt = torch.load(os.path.join(CKPT_DIR, "lcnn_audio_asvspoof_best.pth"),
                      map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model = LCNNMFMAudio()
    model.load_state_dict(state)
    model.eval()
    return model


def make_lfcc_fn():
    from data.lfcc_extractor import LFCCExtractor
    import torchaudio

    extractor = LFCCExtractor(sample_rate=16000)

    def flac_to_lfcc(path: str) -> np.ndarray:
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        # center crop / pad to exactly 4 s like the ASVspoofDataset eval path
        target = int(4.0 * 16000)
        frames = wav.shape[-1]
        if frames > target:
            start = (frames - target) // 2
            wav = wav[:, start:start + target]
        elif frames < target:
            wav = torch.nn.functional.pad(wav, (0, target - frames))
        with torch.no_grad():
            lfcc = extractor.extract(wav)  # PT layout [1,1,T,F]
        return lfcc.numpy().astype(np.float32)

    return flac_to_lfcc


def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    print("=" * 50)
    print("ONN-DETECT TFLite Export Pipeline (trained weights)")
    print("=" * 50)

    image_paths = list_files(GENIMAGE_VAL, (".jpg", ".jpeg"), 400)
    flac_paths = list_files(ASVSPOOF_TEST, (".flac", ".wav"), 200)
    print(f"Calibration pool: {len(image_paths)} images, {len(flac_paths)} audio files")

    # ---------------- SRM Forensic CNN ----------------
    srm = load_srm()
    srm_int8 = os.path.join(EXPORT_DIR, "srm_forensic_cnn_int8.tflite")
    srm_keras = build_keras_srm(srm)
    calib_imgs = [Image.open(p).convert("RGB") for p in image_paths[:400]]
    real_imgs = [im for im in calib_imgs if im.size == (32, 32)]
    keras_parity_check(
        srm_keras, srm,
        [(np.asarray(im, dtype=np.float32) / 255.0) for im in real_imgs[:32]],
        "visual")
    quantize_keras_model(srm_keras, visual_calibration_dataset(image_paths, 200), srm_int8)
    real_imgs = [Image.open(p).convert("RGB") for p in image_paths[:100]]
    validate_models_real(srm, srm_int8, "visual", real_images=real_imgs)

    # ---------------- LCNN-MFM Audio ----------------
    lcnn = load_lcnn()
    lfcc_fn = make_lfcc_fn()
    probe = lfcc_fn(flac_paths[0])
    n_frames = int(probe.shape[-2])
    print(f"LFCC probe shape {probe.shape} -> fixed time axis T={n_frames}")

    specs = []
    for p in flac_paths[:120]:
        try:
            specs.append(lfcc_fn(p))
        except Exception:
            continue

    lcnn_int8 = os.path.join(EXPORT_DIR, "lcnn_audio_int8.tflite")
    lcnn_keras = build_keras_lcnn(lcnn, input_frames=n_frames)
    quantize_keras_model(lcnn_keras, audio_specs_calibration_dataset(specs, 80), lcnn_int8)
    validate_models_real(lcnn, lcnn_int8, "audio", real_specs_pt=specs,
                         target_frames=n_frames, num_samples=60)

    # ---------------- Ship to Android assets ----------------
    for name in ("srm_forensic_cnn_int8.tflite", "lcnn_audio_int8.tflite"):
        shutil.copy2(os.path.join(EXPORT_DIR, name), os.path.join(ASSETS_DIR, name))
        size_kb = os.path.getsize(os.path.join(ASSETS_DIR, name)) / 1024
        print(f"Shipped {name} -> android assets ({size_kb:.0f} KB)")

    print("Export pipeline complete.")


if __name__ == "__main__":
    main()
