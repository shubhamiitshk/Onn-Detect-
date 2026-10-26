from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from ml.scripts.generate_bfree_twins import generate_twins_corpus


def test_generate_bfree_twins_zero_source_bias() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        in_dir = root / "raw_photos"
        out_dir = root / "bfree_out"
        (in_dir / "pexels").mkdir(parents=True)
        (in_dir / "docci").mkdir(parents=True)

        rng = np.random.RandomState(123)
        for idx, sub in enumerate(["pexels", "pexels", "docci", "docci"]):
            arr = rng.randint(16, 240, size=(96, 96, 3), dtype=np.uint8)
            Image.fromarray(arr, mode="RGB").save(in_dir / sub / f"img_{idx}.png")

        records, audit = generate_twins_corpus(
            input_dir=in_dir,
            output_dir=out_dir,
            engine="frequency_synthesizer",
            jpeg_quality=92,
            default_split="train",
        )
        assert len(records) == 8
        assert audit["bias_free"] is True
        assert abs(float(audit["source_only_accuracy"]) - 0.5) < 1e-6
        assert (out_dir / "manifest.jsonl").is_file()
        assert (out_dir / "source_bias_audit.json").is_file()
