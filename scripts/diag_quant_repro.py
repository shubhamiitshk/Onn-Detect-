"""Reproduce the exact production quantize step and compare modes."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "ml"))
import tensorflow as tf
import torch
from PIL import Image

sys.path.append(os.path.join(ROOT, "ml", "src"))
from models.srm_forensic_cnn import SRMForensicCNN
from export.torch_to_tflite_keras import build_keras_srm
from export.quantize_int8 import visual_calibration_dataset

model = SRMForensicCNN()
ckpt = torch.load(os.path.join(ROOT, "ml", "checkpoints", "srm_cnn_genimage_best.pth"), map_location="cpu", weights_only=True)
state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
model.load_state_dict(state); model.eval()
km = build_keras_srm(model)

paths = []
for cls in ("real", "fake"):
    d = os.path.join(ROOT, "datasets", "genimage", "val", cls)
    paths += [os.path.join(d, f) for f in sorted(os.listdir(d))[:200]]

# EXACTLY what production does:
out = os.path.join(ROOT, "scripts", "_repro_fullint8.tflite")
from export.quantize_int8 import quantize_keras_model
kind = quantize_keras_model(km, visual_calibration_dataset(paths, 200), out)
print("quant kind:", kind)

itp = tf.lite.Interpreter(model_path=out); itp.allocate_tensors()
iid, oid = itp.get_input_details()[0], itp.get_output_details()[0]
print("in dtype:", iid["dtype"], "out dtype:", oid["dtype"])

xs, yt = [], []
for p in paths[:16]:
    img = Image.open(p).convert("RGB")
    xs.append(np.asarray(img, dtype=np.float32)[None] / 255.0)
    t = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0)
    with torch.no_grad():
        yt.append(float(model(t).item()))

yq = []
for x in xs:
    itp.set_tensor(iid["index"], x.astype(iid["dtype"]))
    itp.invoke()
    o = float(itp.get_tensor(oid["index"]).ravel()[0])
    if oid["dtype"] != np.float32:
        s, z = oid["quantization"]; o = (o - z) * s
    yq.append(o)

for i, (a, b) in enumerate(zip(yt, yq)):
    print(f"{i:2d} torch={a:.4f}  quant={b:.4f}")
err = np.abs(np.array(yt) - np.array(yq))
print(f"MAE={err.mean():.4f} MAX={err.max():.4f}")

# inspect intermediate: which layer saturates? dump quantization ranges
print("\n--- operator quantization params ---")
for d in itp.get_tensor_details():
    q = d.get("quantization")
    if q and (q[0] or q[1]):
        name = d["name"][:60]
        print(f"{d['index']:3d} scale={q[0]:<12g} zp={q[1]:<6} {name}")
