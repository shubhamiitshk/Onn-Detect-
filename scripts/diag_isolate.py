"""Isolate: (a) are val images really 32x32? (b) does INT8 TFLite diverge from
PyTorch fp32 on IDENTICAL inputs? (c) which step causes the real-image score lift?
"""
import os, sys, random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "ml"))
import torch
from PIL import Image
from torchvision import transforms
from ml.src.models.srm_forensic_cnn import SRMForensicCNN

model = SRMForensicCNN()
ckpt = torch.load(os.path.join(ROOT, "ml", "checkpoints", "srm_cnn_genimage_best.pth"), map_location="cpu", weights_only=False)
state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in state.items()})
model.eval()

import tensorflow as tf
p_fp32 = os.path.join(ROOT, "ml", "exported", "srm_forensic_cnn_fp32.tflite")
p_int8 = os.path.join(ROOT, "android", "app", "src", "main", "assets", "models", "srm_forensic_cnn_int8.tflite")
i32 = tf.lite.Interpreter(model_path=p_int8); i32.allocate_tensors()
ifp = tf.lite.Interpreter(model_path=p_fp32); ifp.allocate_tensors()

def run(interp, x):
    i = interp.get_input_details()[0]
    interp.set_tensor(i["index"], x[None].astype(np.float32))
    interp.invoke()
    return float(interp.get_tensor(interp.get_output_details()[0]["index"]).ravel()[0])

random.seed(0)
for cls in ("real", "fake"):
    d = os.path.join(ROOT, "datasets", "genimage", "val", cls)
    fs = random.sample([f for f in os.listdir(d)], 5)
    print(f"\n--- {cls} ---")
    for f in fs:
        img = Image.open(os.path.join(d, f)).convert("RGB")
        # torch native
        t = transforms.ToTensor()(img)
        s_torch = float(model(t.unsqueeze(0)).item())
        x = (t.permute(1, 2, 0).numpy()).astype(np.float32)  # NHWC [0,1]
        s_i8 = run(i32, x)
        s_f32 = run(ifp, x)
        print(f"{f} size={img.size} torch={s_torch:.4f} tflite_fp32={s_f32:.4f} tflite_int8={s_i8:.4f}")

# also: what does PIL bilinear do when resizing 32->32 (identity?) plus pow2 decim path
img = Image.open(os.path.join(ROOT, "datasets", "genimage", "val", "real", os.listdir(os.path.join(ROOT, "datasets", "genimage", "val", "real"))[0])).convert("RGB")
t = transforms.ToTensor()(img)
x = t.permute(1, 2, 0).numpy().astype(np.float32)
r = img.resize((32, 32), Image.BILINEAR)
x2 = np.asarray(r, dtype=np.float32) / 255.0
print("\nresize-to-same-size max pixel delta:", float(np.abs(x - x2).max()))
