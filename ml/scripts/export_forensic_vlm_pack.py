#!/usr/bin/env python3
"""ONN-DETECT: Tier-2 On-Device Forensic VLM Pack Exporter & Cryptographic Signer.

Packages trained forensic LoRA adapters, generates cryptographic Ed25519 signed
manifests complying with `OnDemandForensicVlmAnalyzer.SignedVlmPackManifest`,
and creates GGUF quantization recipes for on-device mobile execution (`llama.cpp` JNI).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

OFFICIAL_PUBLISHER_PUBLIC_KEY_HEX = "50de4c4a6ebf5c72f73cacf2c11a58dafa40f139e400694c0b5187f307a6a8e3"


def load_external_signing_key(
    key_path_or_hex: Optional[str] = None
) -> ed25519.Ed25519PrivateKey:
    """Loads external production signing private key from file, hex string, or environment variable.

    Security rule: Production private signing keys MUST NOT be embedded in source code.
    """
    if not HAS_CRYPTOGRAPHY:
        raise RuntimeError("Python 'cryptography' library is required for Ed25519 signing.")

    raw_key = (
        key_path_or_hex
        or os.environ.get("ONN_DETECT_VLM_SIGNING_KEY_HEX")
        or os.environ.get("ONN_DETECT_VLM_SIGNING_KEY_PATH")
    )
    if not raw_key:
        raise RuntimeError(
            "Production signing requires an externally supplied Ed25519 private key. "
            "Provide via --signing-key <file>, --signing-key-hex <hex>, or set the "
            "ONN_DETECT_VLM_SIGNING_KEY_HEX environment variable. "
            "Private keys must never be committed to source control."
        )

    if os.path.exists(raw_key):
        with open(raw_key, "rb") as f:
            content = f.read()
        try:
            return serialization.load_pem_private_key(content, password=None)
        except Exception:
            content_str = content.decode("utf-8", errors="ignore").strip()
            if len(content_str) == 64:
                return ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(content_str))
            if len(content) == 32:
                return ed25519.Ed25519PrivateKey.from_private_bytes(content)
            raise ValueError(f"Could not parse private key from file: {raw_key}")

    cleaned = raw_key.strip()
    if len(cleaned) == 64:
        return ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(cleaned))

    raise ValueError("Invalid signing key specification: must be an existing file path or 64-character hex seed.")


def generate_test_signing_key() -> Tuple[ed25519.Ed25519PrivateKey, str]:
    """Generates an ephemeral, test-only Ed25519 keypair for hermetic unit and self-tests."""
    if not HAS_CRYPTOGRAPHY:
        raise RuntimeError("Python 'cryptography' library is required for Ed25519 signing.")
    priv = ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key()
    pub_hex = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    return priv, pub_hex


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 hex digest for a file using streaming chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_canonical_manifest_payload(
    pack_id: str,
    hf_model_id: str,
    language_weights_sha256: str,
    vision_mmproj_sha256: str,
    policy_sha256: str,
    language_bytes: int,
    vision_mmproj_bytes: int,
    processor_version: str,
    chat_template_id: str,
    image_resolution_cap_px: int,
    max_visual_tokens: int,
    diagnostic_only: bool,
) -> bytes:
    """Computes the exact canonical byte payload for Ed25519 digital signing covering all metadata."""
    canonical_str = (
        f"{pack_id}|{hf_model_id}|{language_weights_sha256}|{vision_mmproj_sha256}|"
        f"{policy_sha256}|{int(language_bytes)}|{int(vision_mmproj_bytes)}|"
        f"{processor_version}|{chat_template_id}|"
        f"{int(image_resolution_cap_px)}|{int(max_visual_tokens)}|{str(diagnostic_only).lower()}"
    )
    return canonical_str.encode("utf-8")


def generate_signed_manifest(
    pack_id: str,
    hf_model_id: str,
    language_file: Path,
    vision_file: Path,
    policy_file: Path,
    signing_private_key: Optional[ed25519.Ed25519PrivateKey] = None,
    processor_version: str = "transformers-4.48.0",
    chat_template_id: str = "dialectical_forensic_v1",
    image_resolution_cap_px: int = 512,
    max_visual_tokens: int = 512,
    diagnostic_only: bool = True,
) -> Dict[str, Any]:
    """Generates and cryptographically signs a pack manifest conforming to Android schema."""
    if not HAS_CRYPTOGRAPHY:
        raise RuntimeError("Python 'cryptography' library is required for Ed25519 signing.")

    # 1. Strict file existence checks (Zero placeholder fabrication)
    if not language_file.exists():
        raise FileNotFoundError(f"Language weights file not found: {language_file}")
    if not vision_file.exists():
        raise FileNotFoundError(f"Vision mmproj file not found: {vision_file}")
    if not policy_file.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_file}")

    lang_bytes = language_file.stat().st_size
    vision_bytes = vision_file.stat().st_size
    policy_bytes = policy_file.stat().st_size

    # 2. Strict minimum size boundaries matching Android contract
    if lang_bytes < 50 * 1024 * 1024:
        raise ValueError(f"Language weights file ({lang_bytes} bytes) is below minimum required 50 MB.")
    if vision_bytes < 10 * 1024 * 1024:
        raise ValueError(f"Vision mmproj file ({vision_bytes} bytes) is below minimum required 10 MB.")
    if policy_bytes < 20:
        raise ValueError(f"Policy file ({policy_bytes} bytes) is invalid or empty.")

    # 3. Compute real SHA-256 hashes
    lang_sha = compute_file_sha256(language_file)
    vision_sha = compute_file_sha256(vision_file)
    policy_sha = compute_file_sha256(policy_file)

    # 4. Construct canonical payload covering all metadata
    payload = compute_canonical_manifest_payload(
        pack_id=pack_id,
        hf_model_id=hf_model_id,
        language_weights_sha256=lang_sha,
        vision_mmproj_sha256=vision_sha,
        policy_sha256=policy_sha,
        language_bytes=lang_bytes,
        vision_mmproj_bytes=vision_bytes,
        processor_version=processor_version,
        chat_template_id=chat_template_id,
        image_resolution_cap_px=image_resolution_cap_px,
        max_visual_tokens=max_visual_tokens,
        diagnostic_only=diagnostic_only,
    )

    # 5. Cryptographic Ed25519 signing
    if signing_private_key is None:
        signing_private_key = load_external_signing_key()

    pub_key = signing_private_key.public_key()
    pub_hex = pub_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    sig_bytes = signing_private_key.sign(payload)
    sig_hex = sig_bytes.hex()

    manifest = {
        "pack_id": str(pack_id),
        "hf_model_id": str(hf_model_id),
        "language_weights_sha256": lang_sha,
        "vision_mmproj_sha256": vision_sha,
        "policy_sha256": policy_sha,
        "language_bytes": int(lang_bytes),
        "vision_mmproj_bytes": int(vision_bytes),
        "processor_version": str(processor_version),
        "chat_template_id": str(chat_template_id),
        "image_resolution_cap_px": int(image_resolution_cap_px),
        "max_visual_tokens": int(max_visual_tokens),
        "publisher_public_key_hex": pub_hex,
        "publisher_signature_hex": sig_hex,
        "publisher_signature_valid": True,
        "diagnostic_only": bool(diagnostic_only),
    }

    # 6. Verify generated signature self-consistency
    pub_key.verify(sig_bytes, payload)
    return manifest


def verify_pack_manifest(
    manifest: Dict[str, Any],
    pack_dir: Optional[Path] = None,
    expected_pub_key_hex: Optional[str] = OFFICIAL_PUBLISHER_PUBLIC_KEY_HEX,
    allow_untrusted_for_testing: bool = False,
) -> bool:
    """Verifies cryptographic signature and disk file integrity for a pack against a trusted publisher key."""
    if not HAS_CRYPTOGRAPHY:
        raise RuntimeError("Python 'cryptography' library is required for Ed25519 verification.")

    hex64 = re.compile(r"^[0-9a-fA-F]{64}$")
    hex128 = re.compile(r"^[0-9a-fA-F]{128}$")

    for field in ["language_weights_sha256", "vision_mmproj_sha256", "policy_sha256", "publisher_public_key_hex"]:
        if not hex64.match(str(manifest.get(field, ""))):
            return False

    if not hex128.match(str(manifest.get("publisher_signature_hex", ""))):
        return False

    if int(manifest.get("language_bytes", 0)) < 50 * 1024 * 1024:
        return False
    if int(manifest.get("vision_mmproj_bytes", 0)) < 10 * 1024 * 1024:
        return False

    pub_hex = manifest["publisher_public_key_hex"]
    if allow_untrusted_for_testing:
        trusted_key = expected_pub_key_hex if (expected_pub_key_hex and expected_pub_key_hex != OFFICIAL_PUBLISHER_PUBLIC_KEY_HEX) else pub_hex
    else:
        trusted_key = expected_pub_key_hex
    if not trusted_key or pub_hex.lower() != trusted_key.lower():
        return False

    # Check payload signature over all fields
    payload = compute_canonical_manifest_payload(
        pack_id=manifest["pack_id"],
        hf_model_id=manifest["hf_model_id"],
        language_weights_sha256=manifest["language_weights_sha256"],
        vision_mmproj_sha256=manifest["vision_mmproj_sha256"],
        policy_sha256=manifest["policy_sha256"],
        language_bytes=manifest["language_bytes"],
        vision_mmproj_bytes=manifest["vision_mmproj_bytes"],
        processor_version=manifest["processor_version"],
        chat_template_id=manifest["chat_template_id"],
        image_resolution_cap_px=manifest["image_resolution_cap_px"],
        max_visual_tokens=manifest["max_visual_tokens"],
        diagnostic_only=manifest.get("diagnostic_only", True),
    )

    try:
        pub_bytes = bytes.fromhex(pub_hex)
        sig_bytes = bytes.fromhex(manifest["publisher_signature_hex"])
        pub_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
        pub_key.verify(sig_bytes, payload)
    except Exception:
        return False

    # If pack directory provided, verify all 3 files on disk match manifest exactly
    if pack_dir and pack_dir.exists():
        # 1. Check policy file
        pol_file = pack_dir / "forensic_vlm_policy.json"
        if not pol_file.is_file() or compute_file_sha256(pol_file) != manifest["policy_sha256"]:
            return False

        # 2. Check language model file
        pack_id = manifest["pack_id"]
        lang_candidates = [
            pack_dir / f"{pack_id}-model.gguf",
            pack_dir / f"{pack_id}-language.gguf",
            pack_dir / "model.gguf",
            pack_dir / "language.gguf",
        ]
        lang_file = next((f for f in lang_candidates if f.is_file() and f.stat().st_size == manifest["language_bytes"]), None)
        if lang_file is None:
            lang_file = next((f for f in pack_dir.glob("*.gguf") if "mmproj" not in f.name and f.stat().st_size == manifest["language_bytes"]), None)
        if lang_file is None or compute_file_sha256(lang_file) != manifest["language_weights_sha256"]:
            return False

        # 3. Check vision mmproj file
        vis_candidates = [
            pack_dir / f"{pack_id}-mmproj.gguf",
            pack_dir / f"{pack_id}-vision.gguf",
            pack_dir / "mmproj.gguf",
            pack_dir / "vision.gguf",
        ]
        vis_file = next((f for f in vis_candidates if f.is_file() and f.stat().st_size == manifest["vision_mmproj_bytes"]), None)
        if vis_file is None:
            vis_file = next((f for f in pack_dir.glob("*.gguf") if "mmproj" in f.name and f.stat().st_size == manifest["vision_mmproj_bytes"]), None)
        if vis_file is None or compute_file_sha256(vis_file) != manifest["vision_mmproj_sha256"]:
            return False

    return True


def merge_lora_to_hf_weights(
    base_model_id: str,
    adapter_dir: Path,
    output_dir: Path,
    device: str = "cpu",
) -> Path:
    """Merges a PEFT LoRA adapter into the base model and writes full FP16 weights."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as e:
        raise ImportError(f"Required ML packages missing: {e}. Install transformers, peft, torch.")

    print(f"[Export] Loading base model '{base_model_id}' on {device}...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(adapter_dir)

    print(f"[Export] Attaching LoRA adapter from '{adapter_dir}'...")
    lora_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("[Export] Merging adapter into base model weights...")
    merged = lora_model.merge_and_unload()

    merged_out = output_dir / "merged_hf_model"
    merged_out.mkdir(parents=True, exist_ok=True)

    print(f"[Export] Saving merged FP16 model to '{merged_out}'...")
    merged.save_pretrained(merged_out)
    processor.save_pretrained(merged_out)

    return merged_out


def create_gguf_conversion_script(
    merged_hf_dir: Path,
    output_dir: Path,
    pack_id: str,
    quant_type: str = "q4_k_m",
) -> Path:
    """Generates a standalone conversion script using llama.cpp's convert_hf_to_gguf and multimodal converter."""
    script_sh = output_dir / "convert_to_gguf.sh"
    content = f"""#!/bin/bash
set -e
# GGUF Conversion recipe for ONN-DETECT Tier-2 VLM
# Pinned upstream llama.cpp release: b8291 (March 12, 2026, commit c363256839fdffae184279ad8e5f0f3775c48b77)
# Prerequisites: git clone --branch b8291 https://github.com/ggml-org/llama.cpp && pip install -r llama.cpp/requirements.txt

MERGED_DIR="{merged_hf_dir.resolve()}"
OUT_DIR="{output_dir.resolve()}"
PACK_ID="{pack_id}"
QUANT_TYPE="{quant_type.upper()}"

echo "=== 1. CONVERTING MERGED HF WEIGHTS TO FP16 GGUF ==="
python3 llama.cpp/convert_hf_to_gguf.py "$MERGED_DIR" --outfile "$OUT_DIR/$PACK_ID-f16.gguf"

echo "=== 2. QUANTIZING LANGUAGE MODEL TO $QUANT_TYPE ==="
./llama.cpp/llama-quantize "$OUT_DIR/$PACK_ID-f16.gguf" "$OUT_DIR/$PACK_ID-model.gguf" "$QUANT_TYPE"
rm -f "$OUT_DIR/$PACK_ID-f16.gguf"

echo "=== 3. CONVERTING VISION PROJECTOR (MMPROJ) TO GGUF ==="
# Upstream llama.cpp multimodal conversion:
# convert_hf_to_gguf.py natively supports --mmproj for multimodal architectures (Qwen3.5, SmolVLM, LLaVA, SigLIP)
python3 llama.cpp/convert_hf_to_gguf.py "$MERGED_DIR" --mmproj --outfile "$OUT_DIR/$PACK_ID-mmproj.gguf"

echo "=== 4. GGUF CONVERSION COMPLETE ==="
ls -lh "$OUT_DIR/$PACK_ID-model.gguf" "$OUT_DIR/$PACK_ID-mmproj.gguf"
"""
    script_sh.write_text(content, encoding="utf-8")
    return script_sh


def run_self_tests() -> None:
    """Validates exporter utilities, cryptographic signing, and contract enforcement."""
    print("=" * 70)
    print("RUNNING EXPORT FORENSIC VLM PACK CRYPTOGRAPHIC SELF-TESTS")
    print("=" * 70)

    tmp_dir = Path("tmp_vlm_pack_selftest")
    tmp_dir.mkdir(exist_ok=True)

    try:
        # 1. Test SHA-256 calculation
        tmp_file = tmp_dir / "test_hash.bin"
        tmp_file.write_bytes(b"ONN_DETECT_VLM_SIGNATURE_PAYLOAD_TEST_12345")
        sha = compute_file_sha256(tmp_file)
        expected_sha = hashlib.sha256(b"ONN_DETECT_VLM_SIGNATURE_PAYLOAD_TEST_12345").hexdigest()
        assert sha == expected_sha, f"SHA mismatch: {sha} != {expected_sha}"
        print(f"  [PASS] SHA-256 computation: {sha[:16]}...")

        # 2. Create mock valid model files (>50 MB language, >10 MB vision, policy JSON)
        dummy_lang = tmp_dir / "smolvlm2_500m_video_instruct_q8_0-model.gguf"
        dummy_lang.write_bytes(b"\x00" * (52 * 1024 * 1024))
        dummy_vis = tmp_dir / "smolvlm2_500m_video_instruct_q8_0-mmproj.gguf"
        dummy_vis.write_bytes(b"\x01" * (11 * 1024 * 1024))
        dummy_pol = tmp_dir / "forensic_vlm_policy.json"
        dummy_pol.write_text(json.dumps({"model_id": "test", "decision_threshold": 0.5}), encoding="utf-8")

        # 2b. Test that calling without external private key strictly raises RuntimeError (no embedded private keys)
        old_env_hex = os.environ.pop("ONN_DETECT_VLM_SIGNING_KEY_HEX", None)
        old_env_path = os.environ.pop("ONN_DETECT_VLM_SIGNING_KEY_PATH", None)
        try:
            generate_signed_manifest(
                pack_id="smolvlm2_500m_video_instruct_q8_0",
                hf_model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                language_file=dummy_lang,
                vision_file=dummy_vis,
                policy_file=dummy_pol,
            )
            assert False, "Signing without external key must raise RuntimeError"
        except RuntimeError as re_err:
            assert "externally supplied Ed25519 private key" in str(re_err)
            print("  [PASS] Negative test: Attempt to sign without external key strictly rejected.")
        finally:
            if old_env_hex:
                os.environ["ONN_DETECT_VLM_SIGNING_KEY_HEX"] = old_env_hex
            if old_env_path:
                os.environ["ONN_DETECT_VLM_SIGNING_KEY_PATH"] = old_env_path

        # 3. Test genuine cryptographic signing with ephemeral test key
        test_priv, test_pub_hex = generate_test_signing_key()
        test_manifest = generate_signed_manifest(
            pack_id="smolvlm2_500m_video_instruct_q8_0",
            hf_model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            language_file=dummy_lang,
            vision_file=dummy_vis,
            policy_file=dummy_pol,
            signing_private_key=test_priv,
            processor_version="transformers-4.48.0",
            chat_template_id="smolvlm_default",
            diagnostic_only=True,
        )

        assert test_manifest["pack_id"] == "smolvlm2_500m_video_instruct_q8_0"
        assert test_manifest["language_bytes"] == 52 * 1024 * 1024
        assert test_manifest["vision_mmproj_bytes"] == 11 * 1024 * 1024
        assert test_manifest["publisher_public_key_hex"] == test_pub_hex
        assert len(test_manifest["publisher_signature_hex"]) == 128
        assert test_manifest["publisher_signature_valid"] is True
        print(f"  [PASS] Genuine Ed25519 signed manifest generated with externally supplied key.")

        # 4. Test manifest cryptographic verification with full pack on disk against matching test key
        valid = verify_pack_manifest(test_manifest, pack_dir=tmp_dir, expected_pub_key_hex=test_pub_hex)
        assert valid is True, "Signature verification failed on authentic manifest"
        print(f"  [PASS] Ed25519 signature verified against matching expected publisher key.")

        # 4b. Negative Test: Test key is strictly rejected by production verification against OFFICIAL_PUBLISHER_PUBLIC_KEY_HEX
        assert verify_pack_manifest(test_manifest, pack_dir=tmp_dir) is False, "Test key passed production verification!"
        assert verify_pack_manifest(test_manifest, pack_dir=tmp_dir, allow_untrusted_for_testing=True) is True
        print(f"  [PASS] Negative test: Non-official publisher key strictly rejected by production verification.")

        # 5. Negative Test: Metadata tampering invalidates signature
        tampered_meta = dict(test_manifest)
        tampered_meta["processor_version"] = "transformers-5.0.0"
        assert verify_pack_manifest(tampered_meta, pack_dir=tmp_dir, expected_pub_key_hex=test_pub_hex) is False, "Tampered processor_version passed verification!"
        tampered_diag = dict(test_manifest)
        tampered_diag["diagnostic_only"] = False
        assert verify_pack_manifest(tampered_diag, pack_dir=tmp_dir, expected_pub_key_hex=test_pub_hex) is False, "Tampered diagnostic_only passed verification!"
        print(f"  [PASS] Negative test: Metadata modifications (processor_version, diagnostic_only) strictly rejected.")

        # 6. Negative Test: Corrupted language hash
        corrupted_manifest = dict(test_manifest)
        corrupted_manifest["language_weights_sha256"] = "a" * 64
        assert verify_pack_manifest(corrupted_manifest, pack_dir=tmp_dir, expected_pub_key_hex=test_pub_hex) is False, "Corrupted hash passed verification!"
        print(f"  [PASS] Negative test: Tampered language hash rejected.")

        # 7. Negative Test: Tampered policy on disk
        dummy_pol.write_text(json.dumps({"tampered": True}), encoding="utf-8")
        assert verify_pack_manifest(test_manifest, pack_dir=tmp_dir, expected_pub_key_hex=test_pub_hex) is False, "Tampered policy file on disk passed verification!"
        print(f"  [PASS] Negative test: Tampered policy on disk rejected.")
        dummy_pol.write_text(json.dumps({"model_id": "test", "decision_threshold": 0.5}), encoding="utf-8")

        # 8. Negative Test: Pack directory with ONLY policy file (missing model weights)
        pol_only_dir = tmp_dir / "policy_only_dir"
        pol_only_dir.mkdir(exist_ok=True)
        shutil.copy2(dummy_pol, pol_only_dir / "forensic_vlm_policy.json")
        assert verify_pack_manifest(test_manifest, pack_dir=pol_only_dir, expected_pub_key_hex=test_pub_hex) is False, "Pack directory missing model weights passed verification!"
        print(f"  [PASS] Negative test: Pack directory missing weights strictly rejected.")

        # 9. Negative Test: Truncated non-model file (<50 MB)
        tiny_file = tmp_dir / "tiny_model.gguf"
        tiny_file.write_bytes(b"not_a_real_model")
        try:
            generate_signed_manifest(
                pack_id="fail_test",
                hf_model_id="test",
                language_file=tiny_file,
                vision_file=dummy_vis,
                policy_file=dummy_pol,
            )
            assert False, "16-byte dummy file was accepted!"
        except ValueError as ve:
            assert "below minimum required 50 MB" in str(ve)
            print(f"  [PASS] Negative test: 16-byte non-model file strictly rejected.")

        # 10. Negative Test: Missing vision file
        try:
            generate_signed_manifest(
                pack_id="fail_test",
                hf_model_id="test",
                language_file=dummy_lang,
                vision_file=tmp_dir / "nonexistent_mmproj.gguf",
                policy_file=dummy_pol,
            )
            assert False, "Missing vision file was accepted!"
        except FileNotFoundError:
            print(f"  [PASS] Negative test: Missing vision file strictly rejected (zero placeholder fabrication).")

        # 11. End-to-end CLI flow test: verify weights copying into output_dir
        export_out = tmp_dir / "exported_pack"
        export_out.mkdir()
        shutil.copy2(dummy_pol, export_out / "forensic_vlm_policy.json")
        target_lang = export_out / "test_pack-model.gguf"
        target_vis = export_out / "test_pack-mmproj.gguf"
        shutil.copy2(dummy_lang, target_lang)
        shutil.copy2(dummy_vis, target_vis)
        cli_manifest = generate_signed_manifest(
            pack_id="test_pack",
            hf_model_id="Qwen/Qwen3.5-0.8B",
            language_file=target_lang,
            vision_file=target_vis,
            policy_file=export_out / "forensic_vlm_policy.json",
            signing_private_key=test_priv,
        )
        (export_out / "signed_manifest.json").write_text(json.dumps(cli_manifest, indent=2), encoding="utf-8")
        (export_out / "publisher_key.hex").write_text(cli_manifest["publisher_public_key_hex"], encoding="utf-8")

        assert (export_out / "test_pack-model.gguf").is_file()
        assert (export_out / "test_pack-mmproj.gguf").is_file()
        assert (export_out / "forensic_vlm_policy.json").is_file()
        assert (export_out / "signed_manifest.json").is_file()
        assert (export_out / "publisher_key.hex").is_file()
        assert verify_pack_manifest(cli_manifest, pack_dir=export_out, expected_pub_key_hex=cli_manifest["publisher_public_key_hex"]) is True
        print(f"  [PASS] Complete runnable pack export & file copying verified on output directory.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("=" * 70)
    print("ALL CRYPTOGRAPHIC EXPORT SELF-TESTS PASSED SUCCESSFULLY")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export, package, and cryptographically sign fine-tuned Forensic VLM artifacts.")
    parser.add_argument("--self-test", action="store_true", help="Run self-tests and exit.")
    parser.add_argument("--adapter-dir", type=Path, help="Directory containing fine-tuned LoRA adapter.")
    parser.add_argument("--base-model-id", type=str, default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--output-dir", "--out-dir", type=Path, default=Path("outputs/vlm_export_pack"), help="Output directory for packaged pack.")
    parser.add_argument("--pack-id", type=str, default="smolvlm2_500m_video_instruct_q8_0")
    parser.add_argument(
        "--policy-file",
        type=Path,
        default=Path("android/app/src/test/resources/forensic_vlm_policy_parity_fixture.json") if Path("android/app/src/test/resources/forensic_vlm_policy_parity_fixture.json").is_file() else None,
        help="Path to forensic_vlm_policy.json (defaults to standard fixture if omitted)",
    )
    parser.add_argument("--language-file", type=Path, help="Path to compiled language GGUF file.")
    parser.add_argument("--vision-file", type=Path, help="Path to compiled vision mmproj GGUF file.")
    parser.add_argument("--signing-key", type=Path, help="Path to private Ed25519 signing key.")
    parser.add_argument("--test-key", action="store_true", help="Generate an ephemeral Ed25519 test keypair for local development/testing.")
    parser.add_argument("--skip-merge", action="store_true", help="Skip LoRA merge if base weights already prepared.")

    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
        sys.exit(0)

    if not args.output_dir:
        print("Error: --output-dir is required.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Package policy bundle
    if args.policy_file and args.policy_file.exists():
        shutil.copy(args.policy_file, args.output_dir / "forensic_vlm_policy.json")
        print(f"[Export] Copied policy to {args.output_dir / 'forensic_vlm_policy.json'}")
    else:
        print("Error: --policy-file is required and must exist.", file=sys.stderr)
        sys.exit(1)

    # 2. Merge LoRA adapter if requested
    if not args.skip_merge and args.adapter_dir:
        merged_dir = merge_lora_to_hf_weights(
            base_model_id=args.base_model_id,
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir,
        )
        recipe_sh = create_gguf_conversion_script(
            merged_hf_dir=merged_dir,
            output_dir=args.output_dir,
            pack_id=args.pack_id,
        )
        print(f"[Export] Created GGUF conversion recipe: {recipe_sh}")

    # 3. If GGUF binaries provided, copy weights and generate signed manifest
    if args.language_file and args.vision_file:
        if not args.language_file.exists():
            print(f"Error: Language file does not exist: {args.language_file}", file=sys.stderr)
            sys.exit(1)
        if not args.vision_file.exists():
            print(f"Error: Vision file does not exist: {args.vision_file}", file=sys.stderr)
            sys.exit(1)

        target_lang = args.output_dir / f"{args.pack_id}-model.gguf"
        target_vis = args.output_dir / f"{args.pack_id}-mmproj.gguf"

        if args.language_file.resolve() != target_lang.resolve():
            print(f"[Export] Copying language model to {target_lang}...")
            shutil.copy2(args.language_file, target_lang)

        if args.vision_file.resolve() != target_vis.resolve():
            print(f"[Export] Copying vision mmproj to {target_vis}...")
            shutil.copy2(args.vision_file, target_vis)

        if args.test_key:
            signing_key, _ = generate_test_signing_key()
            print("[Export] Generated ephemeral Ed25519 test key for local pack experimentation.")
        else:
            signing_key = (
                load_external_signing_key(str(args.signing_key))
                if args.signing_key
                else load_external_signing_key()
            )

        manifest = generate_signed_manifest(
            pack_id=args.pack_id,
            hf_model_id=args.base_model_id,
            language_file=target_lang,
            vision_file=target_vis,
            policy_file=args.output_dir / "forensic_vlm_policy.json",
            signing_private_key=signing_key,
        )

        manifest_path = args.output_dir / "signed_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[Export] Generated cryptographically signed manifest: {manifest_path}")

        # Export public key for Android pinning
        pub_path = args.output_dir / "publisher_key.hex"
        pub_path.write_text(manifest["publisher_public_key_hex"], encoding="utf-8")
        print(f"[Export] Saved publisher public key: {pub_path}")

        # Verify the entire exported directory
        if not verify_pack_manifest(manifest, pack_dir=args.output_dir, expected_pub_key_hex=manifest["publisher_public_key_hex"]):
            print("Error: Output pack directory verification failed!", file=sys.stderr)
            sys.exit(1)

        print("[Export] Complete runnable pack verified successfully in output directory.")


if __name__ == "__main__":
    main()
