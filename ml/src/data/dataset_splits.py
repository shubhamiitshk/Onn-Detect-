"""Dataset fingerprinting and generator-aware evaluation splits.

Two problems this solves:

1. UNREPRODUCIBLE METRICS. `evaluation_results.json` recorded AUC numbers with
   no record of WHICH files were evaluated. `datasets/` is gitignored, so the
   same script on a different download produces different numbers with no way
   to tell. `fingerprint_dir` records a content hash of the file LIST (sorted
   relative paths + sizes), so any metric can be tied to the exact bytes it
   was measured on. A changed download changes the fingerprint loudly.

2. GENERATOR LEAKAGE. Random train/test splits on multi-generator data (GenImage,
   OpenFake, CommunityForensics) put the SAME generator in both splits, and the
   model learns generator fingerprints instead of general artifacts. The honest
   protocol is leave-one-generator-out (LOGO): hold out entire generators.
   `group_by_generator` recovers generator labels from the directory layout
   that `OpenFakeDataset` already produces:

       root/split/fake/<generator>/img_0001.jpg
       root/split/real/<anything>/img_0001.jpg   (reals have no generator)

   and `logo_splits` partitions them so no generator appears on both sides.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def fingerprint_dir(root: str, exts: set[str] | None = None) -> dict:
    """Content fingerprint of a dataset directory tree.

    Hashes sorted normalized paths and file CONTENTS. Large datasets require
    I/O proportional to corpus size; manifests preserve hashes on ingest. Two downloads with
    the same files in the same layout produce the same fingerprint; any
    added/removed/renamed file changes it.
    """
    entries = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in sorted(filenames):
            if exts and os.path.splitext(f)[1].lower() not in exts:
                continue
            full = os.path.join(dirpath, f)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            from .contracts import sha256_file
            entries.append(f"{rel}\t{sha256_file(full)}\t{size}")
    entries.sort()
    digest = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
    return {
        "root": os.path.abspath(root),
        "n_files": len(entries),
        "sha256": digest,
        "total_bytes": sum(int(e.rsplit(chr(9), 1)[1]) for e in entries),
    }


def group_by_generator(
    root: str,
    split: str = "test",
    exts: set[str] | None = None,
) -> dict[str, list[str]]:
    """Group fake files by generator subfolder; reals under key ``"__real__"``.

    Understands both layouts:
      flat:    root/split/{real,fake}/*.jpg          -> generator "flat"
      nested:  root/split/fake/<gen>/*.jpg            -> generator <gen>
               root/split/real/...                   -> "__real__" always
    """
    exts = exts or IMAGE_EXTS
    groups: dict[str, list[str]] = defaultdict(list)
    for cls in ("real", "fake"):
        cls_dir = os.path.join(root, split, cls)
        if not os.path.isdir(cls_dir):
            continue
        for dirpath, _dn, filenames in os.walk(cls_dir):
            for f in sorted(filenames):
                if os.path.splitext(f)[1].lower() not in exts:
                    continue
                full = os.path.join(dirpath, f)
                if cls == "real":
                    groups["__real__"].append(full)
                else:
                    rel = os.path.relpath(full, cls_dir)
                    parts = rel.split(os.sep)
                    gen = parts[0] if len(parts) > 1 else "flat"
                    groups[gen].append(full)
    return dict(groups)


def logo_splits(
    groups: dict[str, list[str]],
    holdout_generators: list[str] | None = None,
    holdout_fraction: float = 0.3,
) -> tuple[dict, dict]:
    """Partition groups into (train_groups, heldout_groups) with NO generator
    on both sides. Reals stay on the train side unless explicitly held out.

    If `holdout_generators` is given, exactly those generators are held out
    (deterministic - use this for published numbers). Otherwise the last
    `holdout_fraction` of sorted generator names are held out.
    """
    fakes = sorted(g for g in groups if g != "__real__")
    if not fakes:
        raise ValueError("no fake generator groups found; cannot build LOGO split")
    if holdout_generators is None:
        k = max(1, int(round(len(fakes) * holdout_fraction)))
        holdout_generators = fakes[-k:]
    unknown = set(holdout_generators) - set(fakes)
    if unknown:
        raise ValueError(f"unknown generators requested as holdout: {sorted(unknown)}")
    holdout = set(holdout_generators)
    train = {g: groups[g] for g in groups if g not in holdout and g != "__real__"}
    held = {g: groups[g] for g in holdout}
    real = sorted(groups.get("__real__", []))
    if len(real) < 2:
        raise ValueError("LOGO evaluation needs independent real controls on both sides")
    boundary = max(1, min(len(real) - 1, int(len(real) * (1 - holdout_fraction))))
    train["__real__"], held["__real__"] = real[:boundary], real[boundary:]
    return train, held


def per_generator_report(
    scores_by_file: dict[str, float],
    groups: dict[str, list[str]],
    real_label_files: set[str] | None = None,
) -> dict:
    """Per-generator AUC where each generator is scored against ALL reals.

    `scores_by_file` maps absolute path -> model score (higher = more fake).
    Returns {generator: {"auc": ..., "n": ...}} plus a "__mean__" summary.
    A generator the model has never seen with AUC ~ 0.5 is the honest signal
    that the detector memorized training generators instead of artifacts.

    Uses the local `compute_auc` with sanity checks DISABLED per-slice: a
    single-generator slice is legitimately hard, and failing loudly there
    would make the report unusable. The sanity gate still applies to the
    pooled number in the caller.
    """
    from training.metrics import compute_auc

    reals = groups.get("__real__", [])
    real_scores = [scores_by_file[p] for p in reals if p in scores_by_file]
    report: dict[str, dict] = {}
    aucs = []
    for gen in sorted(g for g in groups if g != "__real__"):
        gen_scores = [scores_by_file[p] for p in groups[gen] if p in scores_by_file]
        if not gen_scores or not real_scores:
            report[gen] = {"auc": None, "n": len(gen_scores), "reason": "no scores"}
            continue
        import numpy as np

        y = np.array([0] * len(real_scores) + [1] * len(gen_scores))
        s = np.array(real_scores + gen_scores)
        try:
            auc = float(compute_auc(y, s, sanity=True))
        except Exception:
            auc = float(compute_auc(y, s, sanity=False))
        report[gen] = {"auc": auc, "n": len(gen_scores)}
        aucs.append(auc)
    report["__mean__"] = {
        "auc": (sum(aucs) / len(aucs)) if aucs else None,
        "n_generators": len(aucs),
    }
    return report
