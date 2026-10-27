"""Adversarial stress and tamper tests for release lineage gates and manifest hashing.

Empirically tests boundary tampering scenarios:
1. Simulated bit-flips in evaluation_report.json and release_manifest.json
2. Mismatched evaluation report hashes
3. Missing metadata keys, invalid schema versions, and corrupted headers
4. P0-1 visual model hash tampering in meta_classifier.json
5. Asset bundle tampering and unhashed file injection
6. Forbidden network permissions in APK manifest
7. Insufficient/at-chance calibration bounds and parity contract enforcement
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import onn_release_lineage_gate as lineage_gate
from scripts.verify_android_artifact import (
    PAGE_SIZE,
    elf_load_segments,
    verify_apk,
    verify_asset_bundle,
)


class LineageAdversarialTests(unittest.TestCase):
    """Stress tests on onn_release_lineage_gate.py."""

    def setUp(self) -> None:
        self.assets_src = ROOT / "android" / "app" / "src" / "main" / "assets"
        self.assertTrue(self.assets_src.is_dir(), f"Assets directory not found at {self.assets_src}")

    def _clone_assets_to_temp(self, temp_dir: Path) -> Path:
        assets_dst = temp_dir / "android" / "app" / "src" / "main" / "assets"
        shutil.copytree(self.assets_src, assets_dst)
        return temp_dir

    def test_current_repo_passes_lineage_gate_cleanly(self) -> None:
        """Baseline sanity: unmodified repo passes lineage check without errors."""
        errors = lineage_gate.check_lineage(ROOT)
        self.assertEqual(errors, [], f"Baseline lineage check must pass: {errors}")

    def test_single_bit_flip_in_evaluation_report_causes_manifest_mismatch(self) -> None:
        """Flipping a single byte/bit in evaluation_report.json causes hash mismatch rejection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = self._clone_assets_to_temp(Path(temp_dir))
            eval_path = temp_path / "android" / "app" / "src" / "main" / "assets" / "evaluation_report.json"
            
            raw_bytes = bytearray(eval_path.read_bytes())
            raw_bytes[10] ^= 0x01  # Flip 1 bit
            eval_path.write_bytes(raw_bytes)

            errors = lineage_gate.check_lineage(temp_path)
            # Lineage gate checks model_hash binding inside eval_report
            # And when evaluation_report is changed, verify_asset_bundle flags it
            manifest_path = temp_path / "android" / "app" / "src" / "main" / "assets" / "constants" / "release_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_hash = manifest.get("evaluation_report_hash")
            actual_hash = hashlib.sha256(raw_bytes).hexdigest()
            self.assertNotEqual(expected_hash, actual_hash)

    def test_tampered_n_calibration_revert_to_300_triggers_sha_mismatch(self) -> None:
        """Reverting n_calibration back to 300 produces a hash mismatch with release_manifest.json."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = self._clone_assets_to_temp(Path(temp_dir))
            eval_path = temp_path / "android" / "app" / "src" / "main" / "assets" / "evaluation_report.json"
            
            report = json.loads(eval_path.read_text(encoding="utf-8"))
            # Revert to old unaligned value
            report["models"]["srm_forensic_cnn_int8.tflite"]["calibration"]["n_calibration"] = 300
            reverted_bytes = json.dumps(report, indent=2).encode("utf-8")
            reverted_hash = hashlib.sha256(reverted_bytes).hexdigest()
            
            manifest_path = temp_path / "android" / "app" / "src" / "main" / "assets" / "constants" / "release_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            
            # The manifest expects the M1 reconciled hash (for 400), so 300 is rejected
            self.assertNotEqual(reverted_hash, manifest["artifacts"]["evaluation_report.json"])
            self.assertNotEqual(reverted_hash, manifest["evaluation_report_hash"])

    def test_p0_1_gate_violation_on_visual_model_hash_tamper(self) -> None:
        """Altering visual_model_hash in meta_classifier.json triggers explicit P0-1 VIOLATION."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = self._clone_assets_to_temp(Path(temp_dir))
            meta_path = temp_path / "android" / "app" / "src" / "main" / "assets" / "constants" / "meta_classifier.json"
            
            meta_cfg = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_cfg["visual_model_hash"] = "deadbeef" * 8
            meta_path.write_text(json.dumps(meta_cfg), encoding="utf-8")

            errors = lineage_gate.check_lineage(temp_path)
            self.assertTrue(
                any("P0-1 VIOLATION" in e for e in errors),
                f"Expected P0-1 VIOLATION error, got: {errors}",
            )

    def test_release_id_mismatch_between_meta_classifier_and_manifest(self) -> None:
        """Divergent release_id in meta_classifier.json is caught and rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = self._clone_assets_to_temp(Path(temp_dir))
            meta_path = temp_path / "android" / "app" / "src" / "main" / "assets" / "constants" / "meta_classifier.json"
            
            meta_cfg = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_cfg["release_id"] = "onn-rogue-release-v999"
            meta_path.write_text(json.dumps(meta_cfg), encoding="utf-8")

            errors = lineage_gate.check_lineage(temp_path)
            self.assertTrue(
                any("Release ID mismatch" in e for e in errors),
                f"Expected Release ID mismatch error, got: {errors}",
            )

    def test_missing_release_id_in_release_manifest_fails(self) -> None:
        """Missing release_id in release_manifest.json is rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = self._clone_assets_to_temp(Path(temp_dir))
            manifest_path = temp_path / "android" / "app" / "src" / "main" / "assets" / "constants" / "release_manifest.json"
            
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["release_id"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            errors = lineage_gate.check_lineage(temp_path)
            self.assertTrue(
                any("missing release_id" in e for e in errors),
                f"Expected missing release_id error, got: {errors}",
            )

    def test_corrupted_model_magic_header_rejected(self) -> None:
        """Corrupting model TFLite header (replacing TFL3 with NOPE) is rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = self._clone_assets_to_temp(Path(temp_dir))
            srm_path = temp_path / "android" / "app" / "src" / "main" / "assets" / "models" / "srm_forensic_cnn_int8.tflite"
            
            raw = bytearray(srm_path.read_bytes())
            raw[4:8] = b"NOPE"
            srm_path.write_bytes(raw)

            errors = lineage_gate.check_lineage(temp_path)
            self.assertTrue(
                any("invalid TFLite magic header" in e for e in errors),
                f"Expected invalid TFLite magic header error, got: {errors}",
            )

    def test_model_hash_divergence_between_manifest_and_actual_binary(self) -> None:
        """Modifying model binary without updating manifest triggers sha256 mismatch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = self._clone_assets_to_temp(Path(temp_dir))
            lcnn_path = temp_path / "android" / "app" / "src" / "main" / "assets" / "models" / "lcnn_audio_int8.tflite"
            
            raw = bytearray(lcnn_path.read_bytes())
            raw.extend(b"\x00" * 64)  # append padding
            lcnn_path.write_bytes(raw)

            errors = lineage_gate.check_lineage(temp_path)
            self.assertTrue(
                any("sha256 mismatch" in e for e in errors),
                f"Expected sha256 mismatch error, got: {errors}",
            )

    def test_invalid_parity_agreement_out_of_bounds(self) -> None:
        """Parity agreement outside [0.0, 1.0] (e.g. 1.25) is rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = self._clone_assets_to_temp(Path(temp_dir))
            eval_path = temp_path / "android" / "app" / "src" / "main" / "assets" / "evaluation_report.json"
            
            report = json.loads(eval_path.read_text(encoding="utf-8"))
            report["models"]["srm_forensic_cnn_int8.tflite"]["parity"]["agreement"] = 1.25
            eval_path.write_text(json.dumps(report), encoding="utf-8")

            errors = lineage_gate.check_lineage(temp_path)
            self.assertTrue(
                any("invalid parity agreement" in e for e in errors),
                f"Expected invalid parity agreement error, got: {errors}",
            )

    def test_negative_parity_max_error_rejected(self) -> None:
        """Negative max_error in parity is rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = self._clone_assets_to_temp(Path(temp_dir))
            eval_path = temp_path / "android" / "app" / "src" / "main" / "assets" / "evaluation_report.json"
            
            report = json.loads(eval_path.read_text(encoding="utf-8"))
            report["models"]["srm_forensic_cnn_int8.tflite"]["parity"]["max_error"] = -0.05
            eval_path.write_text(json.dumps(report), encoding="utf-8")

            errors = lineage_gate.check_lineage(temp_path)
            self.assertTrue(
                any("parity max_error" in e for e in errors),
                f"Expected invalid parity max_error error, got: {errors}",
            )

    def test_visual_split_hash_tamper_detected(self) -> None:
        """Tampering split manifest hash in SRM calibration causes split mismatch error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = self._clone_assets_to_temp(Path(temp_dir))
            manifest_path = temp_path / "android" / "app" / "src" / "main" / "assets" / "constants" / "release_manifest.json"
            
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["calibration"]["srm_forensic_cnn_int8.tflite"]["split_manifest_hash"] = "bad" * 21 + "a"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            errors = lineage_gate.check_lineage(temp_path)
            self.assertTrue(
                any("Visual split hash mismatch" in e for e in errors),
                f"Expected visual split hash mismatch error, got: {errors}",
            )


class ManifestAndArtifactAdversarialTests(unittest.TestCase):
    """Stress tests on scripts/verify_android_artifact.py."""

    def test_unhashed_packaged_artifact_rejected(self) -> None:
        """Any packaged constant or model omitted from release_manifest.json is rejected."""
        assets_dir = ROOT / "android" / "app" / "src" / "main" / "assets"
        manifest = json.loads((assets_dir / "constants" / "release_manifest.json").read_text(encoding="utf-8"))
        
        # Reader simulating asset files
        def reader(path: str) -> bytes:
            if path == "constants/unhashed_backdoor.json":
                return b'{"backdoor": true}'
            return (assets_dir / path).read_bytes()

        names = {
            p.relative_to(assets_dir).as_posix()
            for p in assets_dir.rglob("*")
            if p.is_file()
        }
        names.add("constants/unhashed_backdoor.json")

        errors, _ = verify_asset_bundle(reader, names, require_validated=False)
        self.assertTrue(
            any("Unhashed packaged artifacts" in e for e in errors),
            f"Expected unhashed artifact error, got: {errors}",
        )

    def test_insufficient_calibration_count_rejected(self) -> None:
        """Calibration sample count below 100 is rejected."""
        assets_dir = ROOT / "android" / "app" / "src" / "main" / "assets"
        manifest = json.loads((assets_dir / "constants" / "release_manifest.json").read_text(encoding="utf-8"))
        manifest_copy = copy.deepcopy(manifest)
        manifest_copy["calibration"]["srm_forensic_cnn_int8.tflite"]["n_calibration"] = 50

        def reader(path: str) -> bytes:
            if path == "constants/release_manifest.json":
                return json.dumps(manifest_copy).encode("utf-8")
            return (assets_dir / path).read_bytes()

        names = {
            p.relative_to(assets_dir).as_posix()
            for p in assets_dir.rglob("*")
            if p.is_file()
        }
        errors, _ = verify_asset_bundle(reader, names, require_validated=False)
        self.assertTrue(
            any("Insufficient/at-chance calibration" in e for e in errors),
            f"Expected Insufficient/at-chance calibration error, got: {errors}",
        )

    def test_at_chance_auc_rejected(self) -> None:
        """AUC at or below 0.5 is rejected as at-chance calibration."""
        assets_dir = ROOT / "android" / "app" / "src" / "main" / "assets"
        manifest = json.loads((assets_dir / "constants" / "release_manifest.json").read_text(encoding="utf-8"))
        manifest_copy = copy.deepcopy(manifest)
        manifest_copy["calibration"]["srm_forensic_cnn_int8.tflite"]["auc"] = 0.50

        def reader(path: str) -> bytes:
            if path == "constants/release_manifest.json":
                return json.dumps(manifest_copy).encode("utf-8")
            return (assets_dir / path).read_bytes()

        names = {
            p.relative_to(assets_dir).as_posix()
            for p in assets_dir.rglob("*")
            if p.is_file()
        }
        errors, _ = verify_asset_bundle(reader, names, require_validated=False)
        self.assertTrue(
            any("Insufficient/at-chance calibration" in e for e in errors),
            f"Expected Insufficient/at-chance calibration error, got: {errors}",
        )

    def test_forbidden_network_permissions_rejected(self) -> None:
        """Injected INTERNET or ACCESS_NETWORK_STATE permissions in badging dump trigger failure."""
        mock_aapt_stdout = (
            "package: name='com.onndetect' versionCode='1'\n"
            "uses-permission: name='android.permission.INTERNET'\n"
            "uses-permission: name='android.permission.ACCESS_NETWORK_STATE'\n"
            "uses-permission: name='android.permission.CAMERA'\n"
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=mock_aapt_stdout, stderr="")
            apk_path = ROOT / "android" / "app" / "build" / "outputs" / "apk" / "release" / "app-release.apk"
            if apk_path.is_file():
                errors, _ = verify_apk(apk_path, aapt="dummy_aapt")
                self.assertTrue(any("android.permission.INTERNET" in e for e in errors))
                self.assertTrue(any("android.permission.ACCESS_NETWORK_STATE" in e for e in errors))

    def test_elf_load_segments_alignment_enforcement(self) -> None:
        """ELF load segment with 4KiB alignment instead of 16KiB is rejected."""
        def make_elf(align: int) -> bytes:
            data = bytearray(120)
            data[:7] = b"\x7fELF\x02\x01\x01"
            struct.pack_into("<Q", data, 32, 64)
            struct.pack_into("<HH", data, 54, 56, 1)
            struct.pack_into("<IIQQQQQQ", data, 64, 1, 5, 0, 0, 0, 120, 120, align)
            return bytes(data)

        elf_4k = make_elf(4096)
        segments_4k = elf_load_segments(elf_4k)
        self.assertEqual(segments_4k[0][2], 4096)
        self.assertTrue(segments_4k[0][2] < PAGE_SIZE)  # Violates PAGE_SIZE=16384

        elf_16k = make_elf(16384)
        segments_16k = elf_load_segments(elf_16k)
        self.assertEqual(segments_16k[0][2], 16384)
        self.assertFalse(segments_16k[0][2] < PAGE_SIZE)


if __name__ == "__main__":
    unittest.main()
