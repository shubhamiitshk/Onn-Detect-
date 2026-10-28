#!/usr/bin/env python3
"""Helper script to download genuine on-device VLM GGUF models & vision projectors from Hugging Face."""
import argparse
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

PRESETS = {
    "qwen3.5-0.8b": {
        "description": "Qwen3.5-0.8B text backbone + OCR vision projector (4-bit, ~550 MB)",
        "lang_repo": "ggml-org/Qwen3.5-0.8B-GGUF",
        "lang_file": "Qwen3.5-0.8B-Q4_0.gguf",
        "vis_repo": "joni3/mmproj-ocr-qwen3.5-0.8b-Q8_0",
        "vis_file": "mmproj-english-document-ocr-qwen3.5-0.8b-Q8_0.gguf",
    },
    "smolvlm-500m": {
        "description": "SmolVLM-500M multimodal model + vision projector (official ggml-org, ~490 MB)",
        "lang_repo": "ggml-org/SmolVLM-500M-Instruct-GGUF",
        "lang_file": "SmolVLM-500M-Instruct-Q8_0.gguf",
        "vis_repo": "ggml-org/SmolVLM-500M-Instruct-GGUF",
        "vis_file": "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
    },
    "smolvlm2-500m": {
        "description": "SmolVLM2-500M Video-Instruct + vision projector (official ggml-org, ~490 MB)",
        "lang_repo": "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF",
        "lang_file": "SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
        "vis_repo": "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF",
        "vis_file": "mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
    },
}

def main() -> None:
    parser = argparse.ArgumentParser(description="Download GGUF VLM models from Hugging Face.")
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="qwen3.5-0.8b",
        help="Model preset to download (default: qwen3.5-0.8b)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("downloaded_models"),
        help="Destination directory (default: ./downloaded_models)",
    )
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    dest = args.out_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Downloading VLM Preset: {args.preset}")
    print(f"Description: {preset['description']}")
    print(f"Destination: {dest}")
    print("=" * 60)

    print(f"\n[1/2] Downloading language weights ({preset['lang_file']})...")
    lang_path = hf_hub_download(
        repo_id=preset["lang_repo"],
        filename=preset["lang_file"],
        local_dir=str(dest),
    )
    print(f"      Saved: {lang_path}")

    print(f"\n[2/2] Downloading vision projector ({preset['vis_file']})...")
    vis_path = hf_hub_download(
        repo_id=preset["vis_repo"],
        filename=preset["vis_file"],
        local_dir=str(dest),
    )
    print(f"      Saved: {vis_path}")

    print("\n" + "=" * 60)
    print("SUCCESS: Both weights downloaded!")
    print(f"Language Model: {lang_path}")
    print(f"Vision mmproj:  {vis_path}")
    print("=" * 60)
    print("\nNext step: to push to your phone via ADB run:")
    print(f"  adb push \"{lang_path}\" /data/local/tmp/vlm_model.gguf")
    print(f"  adb push \"{vis_path}\" /data/local/tmp/vlm_mmproj.gguf")

if __name__ == "__main__":
    main()
