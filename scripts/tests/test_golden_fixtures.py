"""Regression tests for the golden-fixture regeneration loop.

`scripts/gen_lfcc_golden.py` only printed its output, so the Kotlin literals in
`LFCCExtractorTest.kt` were hand-transcribed with nothing comparing the two. A
stale transcription that happened to sit inside the 2e-3 JVM tolerance would
have passed. `--json` / `--check` close that loop; these tests prove the check
actually fails on drift and on a frontend ID change.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]      # repository root
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ml" / "src"))

import gen_lfcc_golden as gen

COMMITTED = ROOT / "scripts" / "lfcc_golden.json"


def _printed_problems(document, path):
    """Re-run the check and capture its per-problem output lines."""
    import io
    import contextlib
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            gen._check_goldens(str(path), gen._golden_document(gen.reference_goldens(gen.two_tone_signal())))
    except SystemExit:
        pass
    return buffer.getvalue()


class GoldenFixtureLoopTest(unittest.TestCase):
    def test_committed_goldens_exist_and_are_self_consistent(self):
        with open(COMMITTED, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["schema_version"], 1)
        self.assertIn("lfcc-v", document["frontend_id"])
        goldens = document["goldens"]
        for key in ("frame50", "frame0", "frame99", "frame_last", "energy", "shape"):
            self.assertIn(key, goldens)

    def test_committed_goldens_match_a_fresh_computation(self):
        document = gen._golden_document(gen.reference_goldens(gen.two_tone_signal()))
        with open(COMMITTED, encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertEqual(stored["frontend_id"], document["frontend_id"])
        self.assertEqual(stored["parameters"], document["parameters"])
        for key in ("shape", "energy"):
            self.assertEqual(stored["goldens"][key], document["goldens"][key])
        for key in ("frame50", "frame0", "frame99", "frame_last"):
            for stored_value, fresh in zip(stored["goldens"][key], document["goldens"][key]):
                self.assertAlmostEqual(float(stored_value), fresh, places=6)

    def test_kotlin_literals_match_the_committed_goldens(self):
        """The Kotlin test must assert the values that are actually committed."""
        kt = (ROOT / "android/app/src/test/java/com/onndetect/LFCCExtractorTest.kt").read_text(encoding="utf-8")
        with open(COMMITTED, encoding="utf-8") as handle:
            goldens = json.load(handle)["goldens"]
        for key in ("frame50", "frame0", "frame99", "frame_last"):
            for value in goldens[key]:
                literal = repr(float(value))
                self.assertIn(
                    literal, kt,
                    f"golden {key} value {value} is not present in LFCCExtractorTest.kt",
                )

    def test_check_fails_on_value_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden.json"
            document = gen._golden_document(gen.reference_goldens(gen.two_tone_signal()))
            document["goldens"]["frame50"][0] += 0.5
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                gen._check_goldens(str(path), gen._golden_document(gen.reference_goldens(gen.two_tone_signal())))
            self.assertIn("Golden drift detected", str(caught.exception))

    def test_check_fails_on_frontend_id_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden.json"
            document = gen._golden_document(gen.reference_goldens(gen.two_tone_signal()))
            document["frontend_id"] = "lfcc-v2-some-old-contract-float32"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                gen._check_goldens(str(path), gen._golden_document(gen.reference_goldens(gen.two_tone_signal())))
            self.assertIn("frontend_id changed", str(caught.exception) + _printed_problems(document, path))

    def test_check_fails_on_parameter_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden.json"
            document = gen._golden_document(gen.reference_goldens(gen.two_tone_signal()))
            document["parameters"]["hop_length"] = 320
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(SystemExit):
                gen._check_goldens(str(path), gen._golden_document(gen.reference_goldens(gen.two_tone_signal())))

    def test_check_passes_on_an_unchanged_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden.json"
            document = gen._golden_document(gen.reference_goldens(gen.two_tone_signal()))
            path.write_text(json.dumps(document), encoding="utf-8")
            gen._check_goldens(str(path), gen._golden_document(gen.reference_goldens(gen.two_tone_signal())))

    def test_frame_indices_are_literal_not_relative(self):
        """frame_last must be the real last frame, distinct from frame99."""
        goldens = gen.reference_goldens(gen.two_tone_signal())
        self.assertNotEqual(goldens["frame_last"], goldens["frame99"])
        self.assertEqual(goldens["shape"][0], 101)
        self.assertAlmostEqual(goldens["energy"], 100.0 / 101.0, places=5)

    def test_dynamic_probe_is_covered_by_the_parity_gate(self):
        """The stationary probe alone cannot detect a precision regression."""
        import numpy as np
        from data.lfcc_reference import am_speech_like_probe, lfcc_raw, normalize_cmn

        probe = am_speech_like_probe()
        self.assertEqual(len(probe), 64000)
        self.assertTrue(np.isfinite(probe).all())
        # It must actually be dynamic, otherwise it adds no coverage.
        self.assertGreater(float(np.abs(np.diff(probe)).std()), 1e-3)
        features = normalize_cmn(lfcc_raw(probe)).T
        self.assertEqual(features.shape, (401, 40))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()