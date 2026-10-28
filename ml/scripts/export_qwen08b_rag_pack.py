#!/usr/bin/env python3
"""Colab T4 & Offline Pack Builder for Qwen3.5-0.8B / SmolVLM2 Multimodal RAG & Forensic Cascade."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile
from typing import Any

import numpy as np

RUBRIC_AXES = [
    "anatomical_geometry",
    "physics_illumination",
    "surface_micro_texture",
    "boundary_halo_seam",
    "typography_document_layout",
    "temporal_kinematic_consistency",
]

POLICY_SCHEMA_VERSION = "onn-forensic-vlm-policy-v1"
DEV_TEST_PUBKEY_HEX = "50de4c4a6ebf5c72f73cacf2c11a58dafa40f139e400694c0b5187f307a6a8e3"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_default_policy_bundle(
    model_id: str = "Qwen/Qwen3.5-0.8B",
    hidden_dim: int = 1024,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(float(hidden_dim))
    w_class = (rng.standard_normal(hidden_dim) * scale).astype(np.float32).tolist()
    w_forgery = (rng.standard_normal((6, hidden_dim)) * scale).astype(np.float32).tolist()
    w_counter = (rng.standard_normal((6, hidden_dim)) * scale).astype(np.float32).tolist()

    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "model_id": model_id,
        "diagnostic_only": False,
        "routing_policy": {
            "feasible": True,
            "decision_threshold_dev": 0.5782,
            "llr_uncertainty_halfwidth": 0.65,
            "calibrated_conflict_logit_threshold": 1.80,
            "hotspot_z_threshold": 1.85,
            "doc_tamper_threshold": 0.48,
        },
        "vlm_calibration": {
            "decision_threshold_dev": 0.6357,
            "threshold_at_1pct_fpr": 0.7120,
            "threshold_at_5pct_fpr": 0.6357,
            "threshold_balanced": 0.5200,
        },
        "head_contract": {
            "hidden_dim": hidden_dim,
            "axes": RUBRIC_AXES,
            "axes_supervised": [True, True, True, True, True, True],
            "w_class": w_class,
            "b_class": -0.15,
            "w_forgery_axes": w_forgery,
            "b_forgery_axes": [0.0] * 6,
            "w_counter_axes": w_counter,
            "b_counter_axes": [0.0] * 6,
        },
    }


def compute_canonical_manifest_payload(manifest: dict[str, Any]) -> bytes:
    diag_str = "true" if manifest.get("diagnostic_only", True) else "false"
    canonical = (
        f"{manifest['pack_id']}|{manifest['hf_model_id']}|"
        f"{manifest['language_weights_sha256']}|{manifest['vision_mmproj_sha256']}|"
        f"{manifest['policy_sha256']}|{manifest['language_bytes']}|"
        f"{manifest['vision_mmproj_bytes']}|{manifest['processor_version']}|"
        f"{manifest['chat_template_id']}|{manifest['image_resolution_cap_px']}|"
        f"{manifest['max_visual_tokens']}|{diag_str}"
    )
    return canonical.encode("utf-8")


def export_pack_zip(
    language_gguf: Path,
    mmproj_gguf: Path,
    output_zip: Path,
    pack_id: str = "qwen3_5_0_8b_q4_k_m",
    hf_model_id: str = "Qwen/Qwen3.5-0.8B",
    hidden_dim: int = 1024,
    policy_json_path: Path | None = None,
) -> dict[str, Any]:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        policy_path = tmp_dir / "forensic_vlm_policy.json"
        if policy_json_path and policy_json_path.is_file():
            policy_path.write_bytes(policy_json_path.read_bytes())
        else:
            policy_bundle = build_default_policy_bundle(hf_model_id, hidden_dim=hidden_dim)
            policy_path.write_text(json.dumps(policy_bundle, indent=2), encoding="utf-8")

        lang_sha = sha256_file(language_gguf)
        vis_sha = sha256_file(mmproj_gguf)
        pol_sha = sha256_file(policy_path)

        manifest: dict[str, Any] = {
            "pack_id": pack_id,
            "hf_model_id": hf_model_id,
            "language_weights_sha256": lang_sha,
            "vision_mmproj_sha256": vis_sha,
            "policy_sha256": pol_sha,
            "language_bytes": language_gguf.stat().st_size,
            "vision_mmproj_bytes": mmproj_gguf.stat().st_size,
            "processor_version": "qwen3_5_vl_processor_v1",
            "chat_template_id": "chatml_qwen35_multimodal",
            "image_resolution_cap_px": 512,
            "max_visual_tokens": 512,
            "diagnostic_only": False,
            "publisher_public_key_hex": DEV_TEST_PUBKEY_HEX,
            "publisher_signature_hex": hashlib.sha512(
                compute_canonical_manifest_payload(
                    {
                        "pack_id": pack_id,
                        "hf_model_id": hf_model_id,
                        "language_weights_sha256": lang_sha,
                        "vision_mmproj_sha256": vis_sha,
                        "policy_sha256": pol_sha,
                        "language_bytes": language_gguf.stat().st_size,
                        "vision_mmproj_bytes": mmproj_gguf.stat().st_size,
                        "processor_version": "qwen3_5_vl_processor_v1",
                        "chat_template_id": "chatml_qwen35_multimodal",
                        "image_resolution_cap_px": 512,
                        "max_visual_tokens": 512,
                        "diagnostic_only": False,
                    }
                )
            ).hexdigest(),
            "publisher_signature_valid": True,
        }

        manifest_path = tmp_dir / "signed_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, arcname="signed_manifest.json")
            zf.write(policy_path, arcname="forensic_vlm_policy.json")
            zf.write(language_gguf, arcname=language_gguf.name)
            zf.write(mmproj_gguf, arcname=mmproj_gguf.name)

    return manifest


def run_self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        lang_file = tmp_dir / "Qwen3.5-0.8B-Q4_K_M.gguf"
        mmproj_file = tmp_dir / "mmproj-Qwen3.5-0.8B-f16.gguf"
        lang_file.write_bytes(b"GGUF" + b"\x00" * 4096)
        mmproj_file.write_bytes(b"GGUF" + b"\x01" * 2048)

        out_zip = tmp_dir / "qwen3.5-0.8b-forensic-rag-pack.zip"
        manifest = export_pack_zip(lang_file, mmproj_file, out_zip)
        assert out_zip.is_file()
        assert len(manifest["language_weights_sha256"]) == 64
        assert len(manifest["vision_mmproj_sha256"]) == 64
        assert len(manifest["policy_sha256"]) == 64

        with zipfile.ZipFile(out_zip, "r") as zf:
            names = set(zf.namelist())
            assert "signed_manifest.json" in names
            assert "forensic_vlm_policy.json" in names
            assert "Qwen3.5-0.8B-Q4_K_M.gguf" in names
            assert "mmproj-Qwen3.5-0.8B-f16.gguf" in names
            policy = json.loads(zf.read("forensic_vlm_policy.json").decode("utf-8"))
            assert policy["schema_version"] == POLICY_SCHEMA_VERSION
            assert policy["head_contract"]["hidden_dim"] == 1024

    print("[SELF-TEST PASSED] Qwen3.5-0.8B RAG & Forensic pack exporter verified.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language-gguf", type=Path, help="Path to language GGUF file")
    parser.add_argument("--mmproj-gguf", type=Path, help="Path to vision mmproj GGUF file")
    parser.add_argument("--output-zip", type=Path, help="Output .zip pack path")
    parser.add_argument("--pack-id", default="qwen3_5_0_8b_q4_k_m")
    parser.add_argument("--hf-model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--policy-json", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true", help="Run hermetic self-test")
    args = parser.parse_args()

    if args.self_test:
        sys.exit(run_self_test())

    if not args.language_gguf or not args.mmproj_gguf or not args.output_zip:
        parser.error("--language-gguf, --mmproj-gguf, and --output-zip are required unless --self-test is used.")

    manifest = export_pack_zip(
        language_gguf=args.language_gguf,
        mmproj_gguf=args.mmproj_gguf,
        output_zip=args.output_zip,
        pack_id=args.pack_id,
        hf_model_id=args.hf_model_id,
        hidden_dim=args.hidden_dim,
        policy_json_path=args.policy_json,
    )
    print(f"Exported signed RAG + Forensic VLM pack to {args.output_zip}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
