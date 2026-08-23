"""Where does parity break: torch -> keras -> tflite_fp32 -> tflite_int8?"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "ml"))
import tensorflow as tf
import torch
from PIL import Image
from torchvision import transforms

from ml.src.models.srm_forensic_cnn import SRMForensicCNN
from ml.src.export.torch_to_tflite_keras import build_keras_srm

model = SRMForensicCNN()
ckpt = torch.load(os.path.join(ROOT, "ml", "checkpoints", "srm_cnn_genimage_best.pth"), map_location="cpu", weights_only=False)
state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in state.items()})
model.eval()

# real val inputs, NHWC
files = []
for cls in ("real", "fake"):
    d = os.path.join(ROOT, "datasets", "genimage", "val", cls)
    files += [os.path.join(d, f) for f in sorted(os.listdir(d))[:8]]
xs = []
for p in files:
    t = transforms.ToTensor()(Image.open(p).convert("RGB"))
    xs.append(t.permute(1, 2, 0).numpy().astype(np.float32))
X = np.stack(xs)

# 1) torch scores
with torch.no_grad():
    yt = model(torch.from_numpy(X.transpose(0, 3, 1, 2))).numpy().ravel()

# 2) keras scores
km = build_keras_srm(model)
yk = km.predict(X, verbose=0).ravel()

print("\nidx  torch     keras     |err|")
for i in range(len(yt)):
    print(f"{i:3d}  {yt[i]:.5f}   {yk[i]:.5f}   {abs(yt[i]-yk[i]):.5f}")
print(f"max |torch-keras| = {np.abs(yt-yk).max():.6f}")

# 3) convert keras->tflite fp32 right here and score
conv = tf.lite.TFLiteConverter.from_keras_model(km)
tfl = conv.convert()
tmp = os.path.join(ROOT, "scripts", "_srm_check.tflite")
with open(tmp, "wb") as f:
    f.write(tfl)
itp = tf.lite.Interpreter(model_path=tmp); itp.allocate_tensors()
iid = itp.get_input_details()[0]; oid = itp.get_output_details()[0]
ytf = []
for x in X:
    itp.set_tensor(iid["index"], x[None])
    itp.invoke()
    ytf.append(float(itp.get_tensor(oid["index"]).ravel()[0]))
ytf = np.array(ytf)
print(f"max |keras-tflite_fp32| = {np.abs(yk-ytf).max():.6f}")

# 4) int8 dynamic-range quantization of same model
conv2 = tf.lite.TFLiteConverter.from_keras_model(km)
conv2.optimizations = [tf.lite.Optimize.DEFAULT]
tfl8 = conv2.convert()
tmp8 = os.path.join(ROOT, "scripts", "_srm_check_int8.tflite")
with open(tmp8, "wb") as f:
    f.write(tfl8)
itp8 = tf.lite.Interpreter(model_path=tmp8); itp8.allocate_tensors()
iid8 = itp8.get_input_details()[0]; oid8 = itp8.get_output_details()[0]
yt8 = []
for x in X:
    itp8.set_tensor(iid8["index"], x[None].astype(iid8["dtype"]))
    itp8.invoke()
    o = itp8.get_tensor(oid8["index"]).ravel()[0]
    if oid8["dtype"] != np.float32:
        s, z = oid8["quantization"]
        o = (float(o) - z) * s
    yt8.append(o)
yt8 = np.array(yt8)
print(f"int8 input dtype={iid8['dtype']}, output dtype={oid8['dtype']}")
print(f"max |fp32_tflite-int8_tflite| = {np.abs(ytf-yt8).max():.6f}")
print(f"max |torch-int8_tflite|       = {np.abs(yt-yt8).max():.6f}")
