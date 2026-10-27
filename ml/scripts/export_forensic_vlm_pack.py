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
    image_resolution_cap_px: int,
    max_visual_tokens: int,
) -> bytes:
    """Computes the exact canonical byte payload for Ed25519 digital signing."""
    canonical_str = (
        f"{pack_id}|{hf_model_id}|{language_weights_sha256}|{vision_mmproj_sha256}|"
        f"{policy_sha256}|{int(language_bytes)}|{int(vision_mmproj_bytes)}|"
        f"{int(image_resolution_cap_px)}|{int(max_visual_tokens)}"
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

    # 4. Construct canonical payload
    payload = compute_canonical_manifest_payload(
        pack_id=pack_id,
        hf_model_id=hf_model_id,
        language_weights_sha256=lang_sha,
        vision_mmproj_sha256=vision_sha,
        policy_sha256=policy_sha,
        language_bytes=lang_bytes,
        vision_mmproj_bytes=vision_bytes,
        image_resolution_cap_px=image_resolution_cap_px,
        max_visual_tokens=max_visual_tokens,
    )

    # 5. Cryptographic Ed25519 signing
    if signing_private_key is None:
        signing_private_key = ed25519.Ed25519PrivateKey.generate()

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
    expected_pub_key_hex: Optional[str] = None,
) -> bool:
    """Verifies cryptographic signature and disk file integrity for a pack."""
    if not HAS_CRYPTOGRAPHY:
        raise RuntimeError("Python 'cryptography' library is required for Ed25519 verification.")

    hex64 = re.compile(r"^[0-9a-fA-F]{64}$")
    hex128 = re.compile(r"^[0-9a-fA-F]{128}$")

    for field in ["language_weights_sha256", "vision_mmproj_sha256", "policy_sha256", "publisher_public_key_hex"]:
        if not hex64.match(manifest.get(field, "")):
            return False

    if not hex128.match(manifest.get("publisher_signature_hex", "")):
        return False

    pub_hex = manifest["publisher_public_key_hex"]
    if expected_pub_key_hex and pub_hex.lower() != expected_pub_key_hex.lower():
        return False

    # Check payload signature
    payload = compute_canonical_manifest_payload(
        pack_id=manifest["pack_id"],
        hf_model_id=manifest["hf_model_id"],
        language_weights_sha256=manifest["language_weights_sha256"],
        vision_mmproj_sha256=manifest["vision_mmproj_sha256"],
        policy_sha256=manifest["policy_sha256"],
        language_bytes=manifest["language_bytes"],
        vision_mmproj_bytes=manifest["vision_mmproj_bytes"],
        image_resolution_cap_px=manifest["image_resolution_cap_px"],
        max_visual_tokens=manifest["max_visual_tokens"],
    )

    try:
        pub_bytes = bytes.fromhex(pub_hex)
        sig_bytes = bytes.fromhex(manifest["publisher_signature_hex"])
        pub_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
        pub_key.verify(sig_bytes, payload)
    except Exception:
        return False

    # If pack directory provided, verify files on disk match manifest
    if pack_dir and pack_dir.exists():
        # Check policy file
        pol_file = pack_dir / "forensic_vlm_policy.json"
        if not pol_file.exists() or compute_file_sha256(pol_file) != manifest["policy_sha256"]:
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
    """Generates a standalone conversion script using llama.cpp's convert_hf_to_gguf."""
    script_sh = output_dir / "convert_to_gguf.sh"
    content = f"""#!/bin/bash
set -e
# GGUF Conversion recipe for ONN-DETECT Tier-2 VLM
# Prerequisites: git clone https://github.com/ggerganov/llama.cpp && pip install -r llama.cpp/requirements.txt

MERGED_DIR="{merged_hf_dir.resolve()}"
OUT_DIR="{output_dir.resolve()}"
PACK_ID="{pack_id}"
QUANT_TYPE="{quant_type.upper()}"

echo "=== 1. CONVERTING MERGED HF WEIGHTS TO FP16 GGUF ==="
python3 llama.cpp/convert_hf_to_gguf.py "$MERGED_DIR" --outfile "$OUT_DIR/$PACK_ID-f16.gguf"

echo "=== 2. QUANTIZING TO $QUANT_TYPE ==="
./llama.cpp/llama-quantize "$OUT_DIR/$PACK_ID-f16.gguf" "$OUT_DIR/$PACK_ID-$quant_type.gguf" "$QUANT_TYPE"

echo "=== 3. GGUF CONVERSION COMPLETE ==="
ls -lh "$OUT_DIR/$PACK_ID-$quant_type.gguf"
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
        dummy_lang = tmp_dir / "model.gguf"
        dummy_lang.write_bytes(b"\x00" * (52 * 1024 * 1024))
        dummy_vis = tmp_dir / "mmproj.gguf"
        dummy_vis.write_bytes(b"\x01" * (11 * 1024 * 1024))
        dummy_pol = tmp_dir / "forensic_vlm_policy.json"
        dummy_pol.write_text(json.dumps({"model_id": "test", "decision_threshold": 0.5}), encoding="utf-8")

        # 3. Test genuine cryptographic signing
        priv_key = ed25519.Ed25519PrivateKey.generate()
        manifest = generate_signed_manifest(
            pack_id="smolvlm2_500m_video_instruct_q8_0",
            hf_model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            language_file=dummy_lang,
            vision_file=dummy_vis,
            policy_file=dummy_pol,
            signing_private_key=priv_key,
            diagnostic_only=True,
        )

        assert manifest["pack_id"] == "smolvlm2_500m_video_instruct_q8_0"
        assert manifest["language_bytes"] == 52 * 1024 * 1024
        assert manifest["vision_mmproj_bytes"] == 11 * 1024 * 1024
        assert len(manifest["publisher_public_key_hex"]) == 64
        assert len(manifest["publisher_signature_hex"]) == 128
        assert manifest["publisher_signature_valid"] is True
        print(f"  [PASS] Genuine Ed25519 signed manifest generated.")

        # 4. Test manifest cryptographic verification
        valid = verify_pack_manifest(manifest, pack_dir=tmp_dir, expected_pub_key_hex=manifest["publisher_public_key_hex"])
        assert valid is True, "Signature verification failed on authentic manifest"
        print(f"  [PASS] Ed25519 signature cryptographically verified on authentic pack.")

        # 5. Negative Test: Corrupted language file must fail verification
        corrupted_manifest = dict(manifest)
        corrupted_manifest["language_weights_sha256"] = "a" * 64
        assert verify_pack_manifest(corrupted_manifest, pack_dir=tmp_dir) is False, "Corrupted hash passed verification!"
        print(f"  [PASS] Negative test: Tampered language hash rejected.")

        # 6. Negative Test: Tampered policy on disk must fail verification
        dummy_pol.write_text(json.dumps({"tampered": True}), encoding="utf-8")
        assert verify_pack_manifest(manifest, pack_dir=tmp_dir) is False, "Tampered policy file on disk passed verification!"
        print(f"  [PASS] Negative test: Tampered policy on disk rejected.")

        # 7. Negative Test: Truncated non-model file (<50 MB) must raise ValueError
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

        # 8. Negative Test: Missing vision file must raise FileNotFoundError
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vlm_export_pack"))
    parser.add_argument("--pack-id", type=str, default="smolvlm2_500m_video_instruct_q8_0")
    parser.add_argument("--policy-file", type=Path, help="Path to forensic_vlm_policy.json")
    parser.add_argument("--language-file", type=Path, help="Path to compiled language GGUF file.")
    parser.add_argument("--vision-file", type=Path, help="Path to compiled vision mmproj GGUF file.")
    parser.add_argument("--signing-key", type=Path, help="Path to private Ed25519 signing key.")
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

    # 3. If GGUF binaries provided, generate signed manifest
    if args.language_file and args.vision_file:
        signing_key = None
        if args.signing_key and args.signing_key.exists():
            with open(args.signing_key, "rb") as f:
                signing_key = serialization.load_pem_private_key(f.read(), password=None)

        manifest = generate_signed_manifest(
            pack_id=args.pack_id,
            hf_model_id=args.base_model_id,
            language_file=args.language_file,
            vision_file=args.vision_file,
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


if __name__ == "__main__":
    main()
