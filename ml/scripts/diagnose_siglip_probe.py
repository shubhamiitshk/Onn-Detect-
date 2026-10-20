"""Compare the two cited vision-tower probes under the repo's split discipline.

The SmolVLM/SigLIP teacher logits are already cached from the distillation
run, so this costs nothing. It reports:

  * the probe's AUC on the split it was FIT on (what a leaky evaluation
    reports), and
  * the probe's AUC on a HELD-OUT split (what the repo's protocol requires).

The gap between the two is the whole story.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.contracts import IMAGE_EXTS, discover_samples

CACHE = Path(r"D:\onn_data\runs\release_v5\checkpoints\teacher_distill_cache.npz")
DATA = Path(r"D:\onn_data\openfake_visual")


def main():
    with np.load(CACHE) as d:
        train_logits = d["train_teacher_logits"]
        val_logits = d["val_teacher_logits"]

    train_items = discover_samples(DATA, "train", IMAGE_EXTS)
    val_items = discover_samples(DATA, "val", IMAGE_EXTS)
    train_y = np.asarray([label for _, label in train_items])
    val_y = np.asarray([label for _, label in val_items])

    print("SmolVLM-500M (SigLIP-B/16) vision-tower probe")
    print("=" * 72)
    print(f"train split : n={len(train_y)}  AUC (fit split)  = {roc_auc_score(train_y, train_logits):.4f}")
    print(f"val   split : n={len(val_y)}  AUC (HELD OUT)    = {roc_auc_score(val_y, val_logits):.4f}")

    # How much of the signal is memorisation?
    saturated = np.abs(train_logits) >= 15.0
    print(f"\ntrain logits saturated at the +/-15 clip: {saturated.mean():.1%}")
    print(f"val   logits saturated at the +/-15 clip: {(np.abs(val_logits) >= 15.0).mean():.1%}")

    # A generator-blind sanity check: does the probe separate the two classes
    # WITHIN the calibration split (a leaky but informative number)?
    print("\nInterpretation:")
    print("  A high 'fit split' AUC with a held-out AUC near 0.5 means the")
    print("  probe memorised the fitting split and carries no transferable signal.")


if __name__ == "__main__":
    main()
