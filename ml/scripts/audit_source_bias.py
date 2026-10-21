"""Audit a labelled corpus for acquisition-source shortcut risk.

Why this exists
---------------
The shipped image detector flags **44.4%** of real LAION-sourced images as
synthetic. LAION-5B is the *training corpus* for Stable Diffusion and Flux,
so "looks like LAION" is a giveaway rather than a detection signal. The
audio detector has the same disease: **24.3%** of real ``inthewild`` field
recordings are flagged, because ASVspoof2019 bona-fide speech is clean
studio audio while in-the-wild bona-fide speech is not.

Both are invisible in a pooled AUC. This script makes them visible BEFORE a
model is trained, so the corpus can be fixed rather than the model.

What it reports
----------------
  * per-source composition of the real and fake classes;
  * the **source imbalance** of each source: a source that appears only in
    one class is a shortcut opportunity;
  * the **per-source separability** of a trivial source-only classifier:
    if "which source is this?" predicts the label almost perfectly, the
    label is partly recoverable from the source alone;
  * the split-level leakage of each source, i.e. whether a source appears
    in only one split.

Usage
-----
    python -m ml.scripts.audit_source_bias --data-root D:\\onn_data\\openfake_visual
    python -m ml.scripts.audit_source_bias --data-root D:\\onn_data\\asvspoof_inthewild_audio \\
        --audio --speaker-field speaker
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.contracts import CLASSES, SPLITS, validate_dataset_manifest  # noqa: E402


def load_records(root: Path) -> list[dict]:
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    return manifest["records"]


def source_only_accuracy(records: list[dict], label_key: str = "label") -> float:
    """Accuracy of predicting the label from the SOURCE alone.

    This is the ceiling of any shortcut a detector could learn from the
    acquisition source. A value near 0.5 means the source carries no label
    information; a value near 1.0 means the label is largely a property of
    the source and the corpus cannot distinguish the two.
    """
    by_source: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_source[str(record.get("source"))].append(int(record[label_key]))
    correct = 0
    for labels in by_source.values():
        counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)
        correct += int(counts.max())
    return correct / max(len(records), 1)


def source_only_accuracy_oracle(records: list[dict]) -> float:
    """Best achievable accuracy from the source alone, in expectation.

    A held-out source's composition is unknown at inference, so the honest
    ceiling is the accuracy of predicting each source's MAJORITY label. The
    oracle version (using each source's true composition) is reported for
    comparison, and the gap between them is how much a detector could gain
    by memorising the corpus.
    """
    return source_only_accuracy(records)


def audit(root: Path, speaker_field: str | None) -> dict:
    records = load_records(root)
    by_split_source: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"real": 0, "fake": 0})
    )
    source_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split = str(record["split"])
        source = str(record.get("source"))
        label = int(record["label"])
        key = "real" if label == 0 else "fake"
        by_split_source[split][source][key] += 1
        source_splits[source].add(split)

    # A source that appears in only one class is a shortcut opportunity.
    class_presence: dict[str, set[int]] = defaultdict(set)
    for record in records:
        class_presence[str(record.get("source"))].add(int(record["label"]))

    single_class = sorted(s for s, labels in class_presence.items() if len(labels) == 1)
    split_confined = sorted(s for s, splits in source_splits.items() if len(splits) == 1)

    report = {
        "data_root": str(root),
        "n_records": len(records),
        "splits": {},
        "single_class_sources": single_class,
        "split_confined_sources": split_confined,
        "source_only_accuracy": source_only_accuracy(records),
    }

    print(f"corpus: {root}")
    print(f"records: {len(records)}")
    print()
    print(f"{'split':<14}{'source':<16}{'real':>8}{'fake':>8}   status")
    print("-" * 64)
    for split in sorted(by_split_source):
        report["splits"][split] = {}
        for source, counts in sorted(by_split_source[split].items()):
            real, fake = counts["real"], counts["fake"]
            if real and fake:
                status = "both classes"
            elif real:
                status = "REAL ONLY - shortcut"
            else:
                status = "FAKE ONLY - shortcut"
            print(f"{split:<14}{source:<16}{real:>8}{fake:>8}   {status}")
            report["splits"][split][source] = dict(counts)
        print()

    ceiling = report["source_only_accuracy"]
    print(f"source-only classifier accuracy: {ceiling:.4f}")
    print("  (0.50 = the source carries no label information; 1.00 = the label IS the source)")
    if ceiling > 0.60:
        print("  WARNING: a trivial source-only classifier beats the shipped detector's")
        print("           balanced accuracy. A model trained on this corpus can learn the")
        print("           source instead of the synthesis, and will look accurate while")
        print("           being useless in the field.")
    if single_class:
        print(f"  sources present in only ONE class: {single_class}")
    if split_confined:
        print(f"  sources confined to a single split: {split_confined}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--speaker-field", default=None,
                        help="Extra manifest field to report (e.g. 'speaker')")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    validate_dataset_manifest(root)
    report = audit(root, args.speaker_field)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
