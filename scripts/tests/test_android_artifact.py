"""Regression tests for malformed releases and distinct ELF/ZIP guarantees."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_android_artifact import elf_load_segments, verify_apk, verify_asset_bundle


def elf64(alignment=16384, offset=0, address=0):
    data = bytearray(120)
    data[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<Q", data, 32, 64)
    struct.pack_into("<HH", data, 54, 56, 1)
    struct.pack_into("<IIQQQQQQ", data, 64, 1, 5, offset, address, address, 120, 120, alignment)
    return bytes(data)


def diagnostic_assets():
    model = b"\x00\x00\x00\x00TFL3contract-test-only"
    files = {f"models/{name}": model for name in ("srm_forensic_cnn_int8.tflite", "lcnn_audio_int8.tflite")}
    models = {path.split("/")[1]: {"sha256": hashlib.sha256(content).hexdigest(),
              "input_shape": [1, 32, 32, 3], "output_shape": [1, 1],
              "dtype": "FLOAT32", "frontend_id": "test-only"} for path, content in files.items()}
    manifest = {"schema_version": 1, "release_id": "test-only", "validated": False,
                "score_convention": "real=0_fake=1", "models": models,
                "artifacts": {path: hashlib.sha256(content).hexdigest() for path, content in files.items()}}
    files["constants/release_manifest.json"] = json.dumps(manifest).encode()
    return files


class ArtifactTests(unittest.TestCase):
    def test_missing_or_malformed_manifest_fails(self):
        for files in ({}, {"constants/release_manifest.json": b"{"}, {"constants/release_manifest.json": b"[]"}):
            errors, _ = verify_asset_bundle(files.__getitem__, set(files))
            self.assertTrue(errors)

    def test_diagnostic_bundle_does_not_claim_validated_release(self):
        files = diagnostic_assets()
        errors, warnings = verify_asset_bundle(files.__getitem__, set(files))
        self.assertEqual(errors, [])
        self.assertTrue(warnings)
        release_errors, _ = verify_asset_bundle(files.__getitem__, set(files), require_validated=True)
        self.assertTrue(release_errors)

    def test_swapped_srm_without_retrained_meta_classifier_is_rejected(self):
        files = diagnostic_assets()
        manifest = json.loads(files["constants/release_manifest.json"])
        meta_bytes = json.dumps({
            "release_id": "test-only",
            "visual_model_hash": "0" * 64,
        }).encode()
        files["constants/meta_classifier.json"] = meta_bytes
        manifest["artifacts"]["constants/meta_classifier.json"] = hashlib.sha256(meta_bytes).hexdigest()
        files["constants/release_manifest.json"] = json.dumps(manifest).encode()
        errors, _ = verify_asset_bundle(files.__getitem__, set(files))
        self.assertTrue(any("visual_model_hash" in x for x in errors), errors)

    def test_changed_model_or_unhashed_constant_rejected(self):
        files = diagnostic_assets()
        files["models/lcnn_audio_int8.tflite"] += b"damage"
        files["constants/new_threshold.json"] = b"{}"
        errors, _ = verify_asset_bundle(files.__getitem__, set(files))
        self.assertTrue(any("hash mismatch" in x for x in errors))
        self.assertTrue(any("Unhashed" in x for x in errors))

    def test_unsupported_tensor_contract_rejected(self):
        files = diagnostic_assets()
        manifest = json.loads(files["constants/release_manifest.json"])
        manifest["models"]["lcnn_audio_int8.tflite"]["input_shape"] = [True, -1, 40, 1]
        manifest["models"]["lcnn_audio_int8.tflite"]["dtype"] = "INT8"
        files["constants/release_manifest.json"] = json.dumps(manifest).encode()
        errors, _ = verify_asset_bundle(files.__getitem__, set(files))
        self.assertTrue(any("shape" in x for x in errors))
        self.assertTrue(any("dtype" in x for x in errors))

    def test_setting_validated_flag_cannot_promote_legacy_assets(self):
        files = diagnostic_assets()
        manifest = json.loads(files["constants/release_manifest.json"])
        manifest['validated'] = True
        manifest['calibration'] = {'lcnn_audio_int8.tflite': 'malformed'}
        manifest['device_validation'] = {'passed': 'true'}
        files["constants/release_manifest.json"] = json.dumps(manifest).encode()
        errors, _ = verify_asset_bundle(files.__getitem__, set(files), require_validated=True)
        self.assertTrue(any('device report' in x.lower() for x in errors))
        self.assertTrue(any('Calibration/model' in x for x in errors))

    def test_path_aliases_and_non_boolean_validation_rejected(self):
        for path in ('../escape', 'models/../escape', '/escape', 'models//alias', 'models/./alias'):
            files = diagnostic_assets()
            manifest = json.loads(files["constants/release_manifest.json"])
            manifest['artifacts'][path] = 'a' * 64
            manifest['validated'] = 'false'
            files["constants/release_manifest.json"] = json.dumps(manifest).encode()
            errors, _ = verify_asset_bundle(files.__getitem__, set(files))
            self.assertTrue(any('Invalid artifact' in x for x in errors))
            self.assertTrue(any('Boolean' in x for x in errors))

    def test_boolean_schema_and_output_shape_are_not_integer_contracts(self):
        for field in ('schema', 'output'):
            files = diagnostic_assets()
            manifest = json.loads(files['constants/release_manifest.json'])
            if field == 'schema':
                manifest['schema_version'] = True
            else:
                manifest['models']['lcnn_audio_int8.tflite']['output_shape'] = [True, True]
            files['constants/release_manifest.json'] = json.dumps(manifest).encode()
            errors, _ = verify_asset_bundle(files.__getitem__, set(files))
            self.assertTrue(errors)

    def test_malformed_model_record_returns_errors_even_in_release_mode(self):
        files = diagnostic_assets()
        manifest = json.loads(files['constants/release_manifest.json'])
        manifest['validated'] = True
        manifest['models']['lcnn_audio_int8.tflite'] = 'bad contract'
        files['constants/release_manifest.json'] = json.dumps(manifest).encode()
        errors, _ = verify_asset_bundle(files.__getitem__, set(files), require_validated=True)
        self.assertTrue(any('Invalid model contract' in x for x in errors))

    def test_program_headers_inspected_and_truncation_rejected(self):
        self.assertEqual(elf_load_segments(elf64()), [(0, 0, 16384)])
        for data in (b"", b"NotELF" * 20, elf64()[:100]):
            with self.assertRaises(ValueError):
                elf_load_segments(data)

    def test_manifest_permission_and_zip_alignment_are_independent_gates(self):
        files = diagnostic_assets()
        with tempfile.TemporaryDirectory() as directory:
            apk = Path(directory) / "test.apk"
            with zipfile.ZipFile(apk, "w") as archive:
                for path, content in files.items():
                    archive.writestr("assets/" + path, content)
                archive.writestr("lib/arm64-v8a/libtest.so", elf64())
            output = "package: name='example'\nuses-permission: name='android.permission.INTERNET'\n"
            with patch("verify_android_artifact.subprocess.run") as process:
                process.return_value.returncode = 0
                process.return_value.stdout = output
                process.return_value.stderr = ""
                errors, _ = verify_apk(apk, "aapt")
            self.assertTrue(any("permission violation" in x for x in errors))
            self.assertTrue(any("ZIP 16KiB" in x for x in errors))
            self.assertFalse(any("ELF 16KiB" in x for x in errors))

    def test_adversarial_parity_without_bounds_in_asset_bundle_is_rejected(self):
        assets_dir = Path(__file__).resolve().parents[2] / "android" / "app" / "src" / "main" / "assets"
        files = {p.relative_to(assets_dir).as_posix(): p.read_bytes() for p in assets_dir.rglob("*") if p.is_file()}
        report = json.loads(files["evaluation_report.json"])
        report["models"]["srm_forensic_cnn_int8.tflite"]["parity"] = {"passed": True}
        files["evaluation_report.json"] = json.dumps(report).encode("utf-8")
        manifest = json.loads(files["constants/release_manifest.json"])
        manifest["artifacts"]["evaluation_report.json"] = hashlib.sha256(files["evaluation_report.json"]).hexdigest()
        manifest["evaluation_report_hash"] = manifest["artifacts"]["evaluation_report.json"]
        files["constants/release_manifest.json"] = json.dumps(manifest).encode("utf-8")
        errors, _ = verify_asset_bundle(files.__getitem__, set(files))
        self.assertTrue(any("parity missing numerical bounds" in e for e in errors), f"Must reject parity without bounds: {errors}")


if __name__ == "__main__":
    unittest.main()
