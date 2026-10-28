"""Unit tests for evaluate_embeddinggemma2_probe.py benchmark script."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ml.scripts.evaluate_embeddinggemma2_probe import (
    compute_forensic_metrics,
    discover_split_records,
    run_self_test,
    select_threshold_on_calibration,
)


class EvaluateEmbeddingGemma2ProbeTest(unittest.TestCase):

    def test_compute_forensic_metrics_accuracy(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
        sources = ["laion", "laion", "pexels", "pexels", "midjourney", "midjourney", "flux", "flux"]
        threshold = 0.5

        metrics = compute_forensic_metrics(y_true, probs, sources, threshold)

        self.assertAlmostEqual(metrics["auc"], 1.0)
        self.assertAlmostEqual(metrics["eer"], 0.0)
        self.assertAlmostEqual(metrics["accuracy_at_threshold"], 1.0)
        self.assertIn("laion", metrics["per_source_real_fpr"])
        self.assertIn("pexels", metrics["per_source_real_fpr"])
        self.assertEqual(metrics["per_source_real_fpr"]["laion"]["fpr"], 0.0)
        self.assertEqual(metrics["per_source_real_fpr"]["pexels"]["fpr"], 0.0)

    def test_select_threshold_on_calibration_obeys_max_fpr(self):
        y_cal = np.array([0] * 50 + [1] * 50, dtype=np.int32)
        # Create non-overlapping distributions
        rng = np.random.default_rng(42)
        probs_real = rng.uniform(0.01, 0.40, size=50)
        probs_fake = rng.uniform(0.60, 0.99, size=50)
        probs_cal = np.concatenate([probs_real, probs_fake])

        thr = select_threshold_on_calibration(y_cal, probs_cal, max_fpr=0.05)
        self.assertTrue(0.01 < thr < 0.99)

        # Check empirical FPR on calibration negatives at this threshold
        emp_fpr = np.mean(probs_real >= thr)
        self.assertLessEqual(emp_fpr, 0.05)

    def test_discover_split_records_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_path = root / "stacking" / "real" / "img1.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16)).save(img_path)

            manifest = {
                "schema_version": 1,
                "records": [
                    {
                        "path": "stacking/real/img1.png",
                        "label": 0,
                        "split": "stacking",
                        "source_id": "test_laion",
                        "generator": "authentic",
                    }
                ],
            }
            (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            records = discover_split_records(root, "stacking")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["label"], 0)
            self.assertEqual(records[0]["source"], "test_laion")
            self.assertEqual(records[0]["path"], img_path)

    def test_self_test_end_to_end_and_head_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "out"
            ret = run_self_test(Path(tmp_dir))
            self.assertEqual(ret, 0)

            head_file = out_dir / "embeddinggemma2_forensic_head.json"
            report_file = out_dir / "embeddinggemma2_evaluation_report.json"

            self.assertTrue(head_file.exists())
            self.assertTrue(report_file.exists())

            head_data = json.loads(head_file.read_text(encoding="utf-8"))
            self.assertEqual(head_data["schema_version"], 1)
            self.assertEqual(head_data["embedding_dim"], 768)
            self.assertEqual(len(head_data["weights"]), 768)
            self.assertTrue(isinstance(head_data["bias"], float))
            self.assertTrue(isinstance(head_data["decision_threshold"], float))
            self.assertTrue(head_data.get("diagnostic_only", False))


if __name__ == "__main__":
    unittest.main()
