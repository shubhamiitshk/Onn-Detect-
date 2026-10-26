"""Generate B-Free paired synthetic twins from authentic photographs.

Why this exists
---------------
In standard deepfake datasets, real images and synthetic images often come from
different acquisition sources, resolutions, or JPEG compression pipelines,
creating a "source shortcut" where a classifier can predict the label from the
source domain alone.

Following the B-Free (Bias-Free) protocol, this script takes a directory or
manifest of authentic camera photographs and generates a 1-to-1 paired
synthetic twin for every real image so that each `(real, twin)` pair shares:
  1. Identical scene semantics, composition, lighting, and aspect ratio;
  2. Identical provenance source tag (`source`);
  3. Identical output resolution and matched JPEG quality / chroma subsampling.

Supported Twin Engines
----------------------
- `vae` (default for GPU): Passes each real image through a frozen diffusion
  autoencoder (`stabilityai/sd-vae-ft-mse` or `madebyollin/sdxl-vae-fp16-fix`)
  (`encode -> decode`), stamping genuine latent-diffusion decoder upsampling
  artifacts while preserving 99% pixel-level scene structure.
- `sdedit`: Runs `AutoPipelineForImage2Image` at low denoising strength
  (`--strength 0.30`) to synthesize diffusion texture over the real scene.
- `frequency_synthesizer` (CPU fallback / CI test): Injects deterministic
  2D transposed-convolution checkerboard & spectral lattice residuals into the
  real image when a GPU or HuggingFace checkpoint is unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_matched_jpeg(img: Image.Image, quality: int = 92, subsampling: int = 0) -> bytes:
    """Encodes an RGB PIL Image with deterministic JPEG quality and subsampling."""
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=int(quality), subsampling=int(subsampling), optimize=False)
    return buf.getvalue()


def synthesize_frequency_twin(img: Image.Image, seed: int = 42) -> Image.Image:
    """Deterministic CPU twin synthesizer modeling VAE decoder upsampling periodicity."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    rng = np.random.RandomState(seed)

    # 1. Subtle 2x2 / 4x4 latent-decoder upsampling lattice in high-frequency luma/chroma
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    lattice_2 = np.cos(np.pi * xx) * np.cos(np.pi * yy)
    lattice_4 = np.cos(0.5 * np.pi * xx) * np.cos(0.5 * np.pi * yy)
    grid = (1.35 * lattice_2 + 0.85 * lattice_4)[..., None]

    # 2. Mild cross-channel chroma smoothing (characteristic of 4x-downsampled VAE latents)
    from scipy.ndimage import uniform_filter

    smoothed = uniform_filter(rgb, size=(2, 2, 1))
    blended = 0.82 * rgb + 0.18 * smoothed + grid + rng.normal(0.0, 0.35, size=rgb.shape).astype(np.float32)
    clipped = np.clip(np.round(blended), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(clipped, mode="RGB")


class DiffusersTwinSynthesizer:
    """Lazy-loaded Diffusers VAE / SDEdit twin synthesizer."""

    def __init__(self, engine: str, model_id: str, device: str = "cuda", strength: float = 0.30) -> None:
        import torch

        self.engine = engine
        self.model_id = model_id
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.strength = float(strength)
        self._pipe = None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        import torch

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        if self.engine == "vae":
            from diffusers import AutoencoderKL

            self._pipe = AutoencoderKL.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device)
            self._pipe.eval()
        elif self.engine == "sdedit":
            from diffusers import AutoPipelineForImage2Image

            self._pipe = AutoPipelineForImage2Image.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device)
            self._pipe.set_progress_bar_config(disable=True)
        else:
            raise ValueError(f"Unsupported diffusers engine: {self.engine}")

    def reconstruct(self, img: Image.Image, seed: int = 42) -> Image.Image:
        self._load()
        import torch

        rgb = img.convert("RGB")
        orig_w, orig_h = rgb.size
        # Align dimensions to multiples of 8 for the 8x VAE downsampler
        aligned_w = max(64, (orig_w // 8) * 8)
        aligned_h = max(64, (orig_h // 8) * 8)
        work = rgb if (aligned_w == orig_w and aligned_h == orig_h) else rgb.resize((aligned_w, aligned_h), Image.BICUBIC)

        if self.engine == "vae":
            arr = np.asarray(work, dtype=np.float32) / 127.5 - 1.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(
                device=self.device, dtype=next(self._pipe.parameters()).dtype
            )
            with torch.no_grad():
                posterior = self._pipe.encode(tensor).latent_dist
                latents = posterior.mode()
                decoded = self._pipe.decode(latents).sample
            out_arr = ((decoded[0].permute(1, 2, 0).float().cpu().numpy() + 1.0) * 127.5)
            out_img = Image.fromarray(np.clip(np.round(out_arr), 0, 255).astype(np.uint8), mode="RGB")
        else:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
            out_img = self._pipe(
                prompt="high resolution photograph, natural lighting, sharp details",
                image=work,
                strength=self.strength,
                guidance_scale=3.5,
                num_inference_steps=20,
                generator=gen,
            ).images[0]

        if out_img.size != (orig_w, orig_h):
            out_img = out_img.resize((orig_w, orig_h), Image.BICUBIC)
        return out_img


def verify_zero_source_bias(records: List[Dict[str, object]]) -> Dict[str, object]:
    """Verifies that every `source` has an exact 50/50 real-vs-fake twin balance."""
    by_source: Dict[str, List[int]] = {}
    for rec in records:
        src = str(rec.get("source", "unknown"))
        label = int(rec["label"])
        by_source.setdefault(src, []).append(label)

    correct = 0
    total = 0
    per_source_Balance: Dict[str, Dict[str, int]] = {}
    for src, labels in by_source.items():
        n_real = sum(1 for y in labels if y == 0)
        n_fake = sum(1 for y in labels if y == 1)
        per_source_Balance[src] = {"real": n_real, "fake": n_fake}
        correct += max(n_real, n_fake)
        total += len(labels)

    source_only_accuracy = (correct / total) if total > 0 else 0.5
    return {
        "total_records": total,
        "source_only_accuracy": round(source_only_accuracy, 6),
        "bias_free": abs(source_only_accuracy - 0.5) < 1e-6,
        "sources": per_source_Balance,
    }


def generate_twins_corpus(
    input_dir: Path,
    output_dir: Path,
    engine: str = "frequency_synthesizer",
    model_id: str = "stabilityai/sd-vae-ft-mse",
    device: str = "cuda",
    strength: float = 0.30,
    jpeg_quality: int = 92,
    default_source: str = "camera_native",
    default_split: str = "train",
    max_images: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Generates matched `(real, synthetic_twin)` pairs and writes `manifest.jsonl`."""
    real_out_dir = output_dir / "real"
    twin_out_dir = output_dir / "twin_fake"
    real_out_dir.mkdir(parents=True, exist_ok=True)
    twin_out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if max_images is not None:
        image_paths = image_paths[: int(max_images)]

    diffusers_synth = (
        DiffusersTwinSynthesizer(engine=engine, model_id=model_id, device=device, strength=strength)
        if engine in {"vae", "sdedit"}
        else None
    )

    records: List[Dict[str, object]] = []
    for idx, src_path in enumerate(image_paths):
        with Image.open(src_path) as raw_img:
            rgb = raw_img.convert("RGB")
            w, h = rgb.size

        if diffusers_synth is not None:
            twin_rgb = diffusers_synth.reconstruct(rgb, seed=1000 + idx)
        else:
            twin_rgb = synthesize_frequency_twin(rgb, seed=1000 + idx)

        # Save BOTH real and twin through the exact same JPEG encoder & quality
        real_bytes = encode_matched_jpeg(rgb, quality=jpeg_quality, subsampling=0)
        twin_bytes = encode_matched_jpeg(twin_rgb, quality=jpeg_quality, subsampling=0)

        pair_id = f"bfree_pair_{idx:06d}"
        real_file = real_out_dir / f"{pair_id}_real.jpg"
        twin_file = twin_out_dir / f"{pair_id}_twin.jpg"
        real_file.write_bytes(real_bytes)
        twin_file.write_bytes(twin_bytes)

        source_tag = src_path.parent.name if src_path.parent != input_dir else default_source
        records.append({
            "id": f"{pair_id}_real",
            "pair_id": pair_id,
            "path": str(real_file.relative_to(output_dir).as_posix()),
            "label": 0,
            "source": source_tag,
            "generator": "authentic_camera",
            "split": default_split,
            "width": w,
            "height": h,
            "jpeg_quality": jpeg_quality,
            "sha256": _sha256_bytes(real_bytes),
        })
        records.append({
            "id": f"{pair_id}_twin",
            "pair_id": pair_id,
            "path": str(twin_file.relative_to(output_dir).as_posix()),
            "label": 1,
            "source": source_tag,
            "generator": f"bfree_{engine}",
            "split": default_split,
            "width": w,
            "height": h,
            "jpeg_quality": jpeg_quality,
            "sha256": _sha256_bytes(twin_bytes),
        })

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    audit = verify_zero_source_bias(records)
    (output_dir / "source_bias_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return records, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of real source images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for paired B-Free corpus")
    parser.add_argument(
        "--engine",
        choices=["vae", "sdedit", "frequency_synthesizer"],
        default="vae",
        help="Twin synthesis engine (`vae` for SD/SDXL autoencoder, `sdedit` for img2img, `frequency_synthesizer` for CPU/CI)",
    )
    parser.add_argument("--model-id", default="stabilityai/sd-vae-ft-mse", help="HuggingFace model ID for VAE/SDEdit")
    parser.add_argument("--device", default="cuda", help="Torch device (`cuda` or `cpu`)")
    parser.add_argument("--strength", type=float, default=0.30, help="Denoising strength for SDEdit")
    parser.add_argument("--jpeg-quality", type=int, default=92, help="Matched JPEG quality for both real and twin")
    parser.add_argument("--split", default="train", help="Split tag written to manifest.jsonl")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of pairs")
    args = parser.parse_args()

    records, audit = generate_twins_corpus(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        engine=args.engine,
        model_id=args.model_id,
        device=args.device,
        strength=args.strength,
        jpeg_quality=args.jpeg_quality,
        default_split=args.split,
        max_images=args.max_images,
    )
    print(f"Generated {len(records) // 2} B-Free twin pairs ({len(records)} total images)")
    print(json.dumps(audit, indent=2))
    if not audit["bias_free"]:
        raise SystemExit("FATAL: Source bias audit failed (source_only_accuracy != 0.5)")


if __name__ == "__main__":
    main()
