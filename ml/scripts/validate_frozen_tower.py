"""Validate the frozen MobileCLIP2-S0 tower semantically.

If the tower is loaded correctly, CLIP zero-shot text prompts must separate
real photographs from AI-generated images well above chance. A chance-level
zero-shot AUC means the tower or the embeddings are broken, independently
of any probe.

Also compares the tower's own embeddings with the ones the release pipeline
produces, so a frontend drift would be caught.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.contracts import IMAGE_EXTS, discover_samples
from data.image_frontend import frozen_probe_image_tensor

import open_clip
from open_clip.transform import image_transform
from safetensors.torch import load_file

PROMPTS = [
    "a real photograph",
    "an AI generated image",
    "a real photo taken with a camera",
    "a computer generated picture",
    "a natural photo of a real place",
    "a synthetic rendering made by a neural network",
]


def main():
    pattern = os.path.join(
        os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
        "hub", "models--timm--MobileCLIP2-S0-OpenCLIP", "snapshots", "**", "*.safetensors",
    )
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        print("no cached checkpoint found")
        return 1

    state = load_file(candidates[0])
    model = open_clip.create_model("MobileCLIP2-S0", pretrained=None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"full model load: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    cfg = dict(model.visual.preprocess_cfg)
    transform = image_transform(
        image_size=cfg["size"], is_train=False, mean=cfg["mean"], std=cfg["std"],
        interpolation=cfg["interpolation"], resize_mode=cfg["resize_mode"],
    )
    tokenizer = open_clip.get_tokenizer("MobileCLIP2-S0")

    items = discover_samples(Path(r"D:\onn_data\openfake_visual"), "calibration", IMAGE_EXTS)
    # The manifest stores each split in label order, so a plain prefix slice
    # would contain ONE class only. Balance explicitly.
    from data.contracts import balanced_subset
    items = balanced_subset(items, 160)
    labels = np.array([label for _, label in items])
    print(f"calibration probe set: {len(items)} images, {int((labels == 1).sum())} fake / {int((labels == 0).sum())} real")

    with torch.no_grad():
        text = model.encode_text(tokenizer(PROMPTS)).float()
        text = text / text.norm(dim=-1, keepdim=True)

    # Embeddings through the tower's own transform.
    own = []
    for path, _ in items:
        image = Image.open(path).convert("RGB")
        own.append(transform(image))
    with torch.no_grad():
        features = model.encode_image(torch.stack(own)).float()
    features = features / features.norm(dim=-1, keepdim=True)
    sims = (features @ text.T).numpy()

    print("\nCLIP zero-shot AUC (real=0, ai=1):")
    for index, prompt in enumerate(PROMPTS):
        print(f"  {prompt:<45s} {roc_auc_score(labels, sims[:, index]):.4f}")

    # Embeddings through the release frontend, to catch frontend drift.
    mine = []
    for path, _ in items:
        image = Image.open(path).convert("RGB")
        mine.append(frozen_probe_image_tensor(image, 256).numpy())
    mine_t = torch.from_numpy(np.stack(mine))
    with torch.no_grad():
        features2 = model.encode_image(mine_t).float()
    features2 = features2 / features2.norm(dim=-1, keepdim=True)
    sims2 = (features2 @ text.T).numpy()
    print("\nSame, through the release frontend:")
    for index, prompt in enumerate(PROMPTS):
        print(f"  {prompt:<45s} {roc_auc_score(labels, sims2[:, index]):.4f}")
    print(f"\nmax |own - release| embedding difference: "
          f"{float(np.abs(features.numpy() - features2.numpy()).max()):.6f}")

    # A simple kNN on the release embeddings.
    stack = balanced_subset(
        discover_samples(Path(r"D:\onn_data\openfake_visual"), "stacking", IMAGE_EXTS), 200
    )
    with torch.no_grad():
        stack_t = torch.from_numpy(np.stack([
            frozen_probe_image_tensor(Image.open(p).convert("RGB"), 256).numpy() for p, _ in stack
        ]))
        stack_f = model.encode_image(stack_t).float()
    stack_f = (stack_f / stack_f.norm(dim=-1, keepdim=True)).numpy()
    stack_y = np.array([label for _, label in stack])
    similarity = features2.numpy() @ stack_f.T
    for k in (1, 10):
        index = np.argsort(-similarity, axis=1)[:, :k]
        print(f"kNN({k}) vs stacking: AUC {roc_auc_score(labels, stack_y[index].mean(axis=1)):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
