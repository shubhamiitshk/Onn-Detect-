"""Adversarial and regression tests for onn_release_lineage_gate.py."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import onn_release_lineage_gate as gate


class LineageGateTests(unittest.TestCase):
    def test_current_repo_lineage_passes(self):
        errors = gate.check_lineage(ROOT)
        self.assertEqual(errors, [], f"Current repository lineage gate must pass: {errors}")

    def test_adversarial_parity_assertion_without_bounds_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create minimal assets structure mirroring repo
            assets_src = ROOT / "android" / "app" / "src" / "main" / "assets"
            assets_dst = temp_path / "android" / "app" / "src" / "main" / "assets"
            shutil.copytree(assets_src, assets_dst)

            # Adversarially wipe numerical parity error bounds while leaving passed=True
            report_path = assets_dst / "evaluation_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["models"]["srm_forensic_cnn_int8.tflite"]["parity"] = {"passed": True}
            report_path.write_text(json.dumps(report), encoding="utf-8")

            errors = gate.check_lineage(temp_path)
            self.assertTrue(
                any("parity" in e for e in errors),
                f"Gate must reject parity assertion without numerical bounds: {errors}",
            )

    def test_git_lfs_pointer_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            assets_src = ROOT / "android" / "app" / "src" / "main" / "assets"
            assets_dst = temp_path / "android" / "app" / "src" / "main" / "assets"
            shutil.copytree(assets_src, assets_dst)

            srm_file = assets_dst / "models" / "srm_forensic_cnn_int8.tflite"
            srm_file.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:1234\nsize 1234\n")

            errors = gate.check_lineage(temp_path)
            self.assertTrue(
                any("Git LFS text pointer" in e for e in errors),
                f"Gate must reject Git LFS text pointer: {errors}",
            )

    def test_cross_file_frontend_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            assets_src = ROOT / "android" / "app" / "src" / "main" / "assets"
            assets_dst = temp_path / "android" / "app" / "src" / "main" / "assets"
            shutil.copytree(assets_src, assets_dst)

            meta_res_path = assets_dst / "fusion" / "meta_results.json"
            meta_res = json.loads(meta_res_path.read_text(encoding="utf-8"))
            meta_res["calibration"]["frontend_id"] = "different-frontend-contract-v9"
            meta_res_path.write_text(json.dumps(meta_res), encoding="utf-8")

            errors = gate.check_lineage(temp_path)
            self.assertTrue(
                any("Frontend mismatch" in e for e in errors),
                f"Gate must reject cross-file frontend mismatch: {errors}",
            )

    def test_acceptance_and_evaluation_report_auc_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            assets_src = ROOT / "android" / "app" / "src" / "main" / "assets"
            assets_dst = temp_path / "android" / "app" / "src" / "main" / "assets"
            shutil.copytree(assets_src, assets_dst)

            manifest_path = assets_dst / "constants" / "release_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["acceptance"]["srm_forensic_cnn_int8.tflite"]["measured_auc"] = 0.5000
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            errors = gate.check_lineage(temp_path)
            self.assertTrue(
                any("Acceptance AUC mismatch" in e for e in errors),
                f"Gate must reject acceptance vs evaluation report AUC mismatch: {errors}",
            )


if __name__ == "__main__":
    unittest.main()
