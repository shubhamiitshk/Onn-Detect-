"""Inspect the packaged APK, not source comments, before distribution.

Default mode permits explicitly unvalidated research models. Release mode
requires validated lineage. Both modes enforce no network permissions,
model/artifact hashes, and 16 KiB compatibility for packaged 64-bit libraries.
No TensorFlow installation or network access is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import sys
import zipfile

PAGE_SIZE = 16_384
FORBIDDEN_PERMISSIONS = {"android.permission.INTERNET", "android.permission.ACCESS_NETWORK_STATE"}
REQUIRED_MODELS = {"srm_forensic_cnn_int8.tflite", "lcnn_audio_int8.tflite"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def elf_load_segments(data: bytes) -> list[tuple[int, int, int]]:
    """Return (offset, virtual address, alignment) for LOAD program headers."""
    if len(data) < 64 or data[:4] != b"\x7fELF" or data[5] not in (1, 2):
        raise ValueError("invalid ELF header")
    endian = "<" if data[5] == 1 else ">"
    if data[4] == 2:
        phoff = struct.unpack_from(endian + "Q", data, 32)[0]
        size, count = struct.unpack_from(endian + "HH", data, 54)
        fmt = endian + "IIQQQQQQ"
    elif data[4] == 1:
        phoff = struct.unpack_from(endian + "I", data, 28)[0]
        size, count = struct.unpack_from(endian + "HH", data, 42)
        fmt = endian + "IIIIIIII"
    else:
        raise ValueError("unsupported ELF class")
    if count == 0 or count > 1024 or size < struct.calcsize(fmt) or phoff + size * count > len(data):
        raise ValueError("invalid/truncated ELF program headers")
    loads = []
    for i in range(count):
        header = struct.unpack_from(fmt, data, phoff + i * size)
        if header[0] == 1:
            offset, address = (header[2], header[3]) if data[4] == 2 else (header[1], header[2])
            loads.append((offset, address, header[-1]))
    if not loads:
        raise ValueError("ELF has no LOAD segments")
    return loads


def zip_data_offset(archive: zipfile.ZipFile, item: zipfile.ZipInfo) -> int:
    with open(archive.filename, "rb") as stream:
        stream.seek(item.header_offset)
        header = stream.read(30)
    if len(header) != 30 or header[:4] != b"PK\x03\x04":
        raise ValueError("invalid ZIP local header")
    name_size, extra_size = struct.unpack_from("<HH", header, 26)
    return item.header_offset + 30 + name_size + extra_size


def _safe_asset_path(path: str) -> bool:
    if not isinstance(path, str):
        return False
    part = PurePosixPath(path)
    return bool(path) and part.as_posix() == path and not part.is_absolute() and ".." not in part.parts and "\\" not in path and ":" not in path


def verify_asset_bundle(read, names: set[str], require_validated: bool = False) -> tuple[list[str], list[str]]:
    """Check a mapping of assets-relative paths; reusable for local assets/tests."""
    errors, warnings = [], []
    manifest_path = "constants/release_manifest.json"
    try:
        manifest = json.loads(read(manifest_path))
    except (KeyError, FileNotFoundError, ValueError, UnicodeError) as exc:
        return [f"Missing/invalid required release manifest: {exc}"], warnings
    if not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int or manifest.get("schema_version") != 1:
        return ["Unsupported release manifest schema"], warnings
    if not isinstance(manifest.get("validated"), bool) or not isinstance(manifest.get("release_id"), str) or not manifest["release_id"]:
        errors.append("Release ID and Boolean validated status are required")
    if manifest.get("score_convention") != "real=0_fake=1":
        errors.append("Unsupported or missing score convention")
    if manifest.get("validated") is not True:
        message = "Bundled models are unvalidated; app must return inconclusive assessments"
        (errors if require_validated else warnings).append(message)
    models, artifacts = manifest.get("models"), manifest.get("artifacts")
    if not isinstance(models, dict) or not REQUIRED_MODELS.issubset(models):
        errors.append("Both required model contracts must be declared")
        models = models if isinstance(models, dict) else {}
    if not isinstance(artifacts, dict) or not artifacts:
        errors.append("A nonempty complete artifact hash map is required")
        artifacts = {}
    for name, contract in models.items():
        if not _safe_asset_path(name) or "/" in name or not isinstance(contract, dict):
            errors.append(f"Invalid model contract: {name}")
            continue
        path = f"models/{name}"
        try:
            data = read(path)
            digest = hashlib.sha256(data).hexdigest()
            if len(data) < 8 or data[4:8] != b"TFL3":
                errors.append(f"Invalid TFLite header: {path}")
            if digest != contract.get("sha256"):
                errors.append(f"Model contract hash mismatch: {path}")
        except (KeyError, FileNotFoundError):
            errors.append(f"Missing model: {path}")
        shape = contract.get("input_shape")
        if not isinstance(shape, list) or len(shape) != 4 or shape[0] != 1 or any(type(n) is not int or n <= 0 for n in shape):
            errors.append(f"Invalid model input shape: {name}")
        output_shape = contract.get('output_shape')
        if output_shape != [1, 1] or any(type(n) is not int for n in output_shape) or contract.get("dtype") != "FLOAT32" or not isinstance(contract.get('frontend_id'), str) or not contract['frontend_id']:
            errors.append(f"Invalid output/dtype/frontend contract: {name}")
    for path, expected in artifacts.items():
        if not isinstance(path, str) or not _safe_asset_path(path) or not isinstance(expected, str) or not SHA256.fullmatch(expected):
            errors.append(f"Invalid artifact hash declaration: {path}")
            continue
        try:
            if hashlib.sha256(read(path)).hexdigest() != expected:
                errors.append(f"Artifact hash mismatch: {path}")
        except (KeyError, FileNotFoundError):
            errors.append(f"Missing declared artifact: {path}")
    relevant = {n for n in names if n.startswith(("models/", "constants/", "fusion/")) and not n.endswith("/")}
    uncovered = relevant - set(artifacts) - {manifest_path}
    if uncovered:
        errors.append(f"Unhashed packaged artifacts: {sorted(uncovered)}")
    if manifest.get("validated") is True:
        for field in ("source_commit", "data_manifest_hashes", "evaluation_report_hash", "calibration", "validation_status"):
            if not manifest.get(field):
                errors.append(f"Validated release lacks {field}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("source_commit", ""))):
            errors.append("Invalid validated source commit")
        data_hashes = manifest.get("data_manifest_hashes", {})
        if not isinstance(data_hashes, dict) or not data_hashes or any(not isinstance(v, str) or not SHA256.fullmatch(v) for v in data_hashes.values()):
            errors.append("Invalid validated dataset lineage")
            data_hashes = {}
        evaluation_hash = manifest.get("evaluation_report_hash")
        if not isinstance(evaluation_hash, str) or artifacts.get("evaluation_report.json") != evaluation_hash or not SHA256.fullmatch(evaluation_hash):
            errors.append("Evaluation report is not bound to the artifact manifest")
        snapshot_hash = manifest.get('source_snapshot_sha256')
        if not isinstance(snapshot_hash, str) or not SHA256.fullmatch(snapshot_hash) or artifacts.get('source_snapshot.json') != snapshot_hash:
            errors.append('Actual source snapshot is not bound to this release')
        revision = manifest.get("fusion_revision")
        if not isinstance(revision, str) or revision != manifest.get("release_id") or "fusion/XGBoostMetaClassifier.kt" not in artifacts:
            errors.append("Fusion source/release binding missing")
        else:
            try:
                source = read("fusion/XGBoostMetaClassifier.kt").decode("utf-8")
                if f"const val RELEASE_ID = {json.dumps(revision)}" not in source:
                    errors.append("Staged fusion source revision mismatch")
            except (KeyError, FileNotFoundError, UnicodeError):
                errors.append("Missing/invalid staged fusion source")
        device = manifest.get("device_validation", {})
        if not isinstance(device, dict) or device.get("passed") is not True or artifacts.get("device_evaluation.json") != device.get("report_hash") or not isinstance(device.get("report_hash"), str) or not SHA256.fullmatch(device["report_hash"]):
            errors.append("Independent device report is not bound to this release")
        for report_path in ("evaluation_report.json", "device_evaluation.json"):
            try:
                report = json.loads(read(report_path))
                if not isinstance(report, dict) or report.get("release_id") != manifest.get("release_id"):
                    errors.append(f"Report release identity mismatch: {report_path}")
                    continue
                if report_path == 'evaluation_report.json' and report.get('source_snapshot_sha256') != snapshot_hash:
                    errors.append('Evaluation source snapshot mismatch')
                for name, contract in models.items():
                    if not isinstance(contract, dict):
                        continue
                    entries = report.get("models", {})
                    entry = entries.get(name, {}) if isinstance(entries, dict) else {}
                    if not isinstance(entry, dict):
                        entry = {}
                    if report_path == "evaluation_report.json":
                        parity, cal = entry.get("parity", {}), entry.get("calibration", {})
                        if not isinstance(parity, dict) or parity.get("passed") is not True or not isinstance(cal, dict) or cal.get("model_hash") != contract.get("sha256") or cal.get('frontend_id') != contract.get('frontend_id'):
                            errors.append(f"Export report/model parity mismatch: {name}")
                    elif entry.get("sha256") != contract.get("sha256") or entry.get("frontend_id") != contract.get("frontend_id") or entry.get("parity_passed") is not True:
                        errors.append(f"Device report/model parity mismatch: {name}")
                if report_path == "device_evaluation.json" and (report.get("passed") is not True or report.get("fusion_revision") != revision or report.get("kotlin_parity_passed") is not True or report.get("offline_verified") is not True):
                    errors.append("Device report lacks passed fusion/offline evidence")
                if report_path == 'device_evaluation.json' and (report.get('source_snapshot_sha256') != snapshot_hash or report.get('fusion_kotlin_sha256') != artifacts.get('fusion/XGBoostMetaClassifier.kt')):
                    errors.append('Device evidence is not bound to actual source and fusion bytes')
            except (KeyError, FileNotFoundError, ValueError, UnicodeError):
                errors.append(f"Missing/invalid required report: {report_path}")
        calibration = manifest.get("calibration", {})
        for name, contract in models.items():
            if not isinstance(contract, dict):
                continue
            item = calibration.get(name, {}) if isinstance(calibration, dict) else {}
            if not isinstance(item, dict) or item.get("model_hash") != contract.get("sha256") or item.get("frontend_id") != contract.get("frontend_id"):
                errors.append(f"Calibration/model contract mismatch: {name}")
            if not isinstance(item, dict):
                item = {}
            threshold = item.get("threshold") if isinstance(item, dict) else None
            if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
                errors.append(f"Invalid calibration threshold: {name}")
            auc, count = item.get("auc"), item.get("n_calibration")
            if type(auc) not in (int, float) or not math.isfinite(auc) or not .5 < auc <= 1 or type(count) is not int or count < 100:
                errors.append(f"Insufficient/at-chance calibration: {name}")
            if item.get("split_manifest_hash") not in data_hashes.values():
                errors.append(f"Calibration split lineage missing: {name}")
    return errors, warnings


def find_aapt(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    for command in ("aapt2", "aapt"):
        found = shutil.which(command)
        if found:
            return found
    roots = [os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT"),
             str(Path.home() / "AppData/Local/Android/Sdk"), str(Path.home() / "Android/Sdk")]
    candidates = []
    for root in filter(None, roots):
        for version in (Path(root) / "build-tools").glob("*"):
            for name in ("aapt2.exe", "aapt2", "aapt.exe", "aapt"):
                candidate = version / name
                if candidate.is_file():
                    candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError("Android SDK aapt2/aapt is required to inspect actual permissions")
    return str(sorted(candidates, key=lambda p: tuple(int(x) for x in re.findall(r"\d+", p.parent.name)), reverse=True)[0])


def verify_apk(path: Path, aapt: str, require_validated: bool = False) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    result = subprocess.run([aapt, "dump", "badging", str(path)], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"Cannot inspect APK manifest: {result.stderr.strip()}")
    if "package: name=" not in result.stdout:
        raise ValueError("aapt did not return a parsed APK manifest")
    permissions = set(re.findall(r"uses-permission[^:]*:\s+name='([^']+)'", result.stdout))
    for permission in sorted(permissions & FORBIDDEN_PERMISSIONS):
        errors.append(f"Offline permission violation: {permission}")
    with zipfile.ZipFile(path) as apk:
        if len(apk.namelist()) != len(set(apk.namelist())):
            errors.append("Duplicate APK entries")
        asset_names = {n.removeprefix("assets/") for n in apk.namelist() if n.startswith("assets/")}
        asset_errors, asset_warnings = verify_asset_bundle(lambda name: apk.read("assets/" + name), asset_names, require_validated)
        errors.extend(asset_errors)
        warnings.extend(asset_warnings)
        for item in apk.infolist():
            parts = PurePosixPath(item.filename).parts
            if len(parts) != 3 or parts[0] != "lib" or not item.filename.endswith(".so"):
                continue
            # Android 16 KiB devices are 64-bit; 32-bit ABIs do not require it.
            if parts[1] not in {"arm64-v8a", "x86_64"}:
                continue
            with apk.open(item) as stream:
                header = stream.read(128 * 1024)
            try:
                for offset, address, alignment in elf_load_segments(header):
                    if alignment < PAGE_SIZE or alignment & (alignment - 1) or (address - offset) % PAGE_SIZE:
                        errors.append(f"ELF 16KiB alignment violation: {item.filename}")
                        break
                if item.compress_type == zipfile.ZIP_STORED and zip_data_offset(apk, item) % PAGE_SIZE:
                    errors.append(f"ZIP 16KiB alignment violation: {item.filename}")
            except ValueError as exc:
                errors.append(f"{item.filename}: {exc}")
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apk", type=Path, required=True)
    parser.add_argument("--aapt", help="Explicit SDK aapt2/aapt executable")
    parser.add_argument("--require-validated-models", action="store_true", help="Required for distributing a detector release")
    args = parser.parse_args()
    try:
        errors, warnings = verify_apk(args.apk, find_aapt(args.aapt), args.require_validated_models)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, subprocess.TimeoutExpired) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"FAIL: {error}", file=sys.stderr)
    if not errors:
        print("APK permissions, artifact contracts and 64-bit 16KiB alignment verified.")
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
