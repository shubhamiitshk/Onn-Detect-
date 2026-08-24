#!/usr/bin/env python3
"""
Unified Advanced Dataset Downloader for ONN-DETECT
Streams and splits large-scale visual and audio deepfake datasets:
- WaveFake / In-The-Wild Audio
- GenImage / Diffusion Artifacts Visual
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset, Audio
from tqdm import tqdm
from PIL import Image

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = SCRIPT_DIR.parent.parent

def setup_dirs(base_path: Path):
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            (base_path / split / label).mkdir(parents=True, exist_ok=True)

def download_audio_inthewild(max_samples: int = 30000):
    print("=" * 70)
    print("DOWNLOADING AUDIO DATASET (WaveFake / In-The-Wild)")
    print("=" * 70)
    
    target_dir = ROOT_DIR / "datasets" / "inthewild_audio"
    setup_dirs(target_dir)
    
    # Try In-the-wild audio or fallback to wavefake
    hf_dataset_name = "garystafford/deepfake-audio-detection"
    
    print(f"Connecting to HuggingFace dataset: {hf_dataset_name}...")
    try:
        dataset = load_dataset(
            hf_dataset_name,
            split="train",
            streaming=True
        ).cast_column("audio", Audio(decode=False))
        
        real_count = 0
        fake_count = 0
        target_per_class = max_samples // 2
        
        for item in tqdm(dataset, desc="Streaming audio files"):
            if real_count >= target_per_class and fake_count >= target_per_class:
                break
                
            label = item.get('label', 0)
            is_real = (label == 1)
            
            count = real_count if is_real else fake_count
            if count < int(target_per_class * 0.8):
                split = 'train'
            elif count < int(target_per_class * 0.9):
                split = 'val'
            else:
                split = 'test'
                
            label_name = 'real' if is_real else 'fake'
            out_path = target_dir / split / label_name / f"audio_{count:06d}.wav"
            
            raw_bytes = item['audio']['bytes']
            with open(out_path, 'wb') as f:
                f.write(raw_bytes)
                
            if is_real:
                real_count += 1
            else:
                fake_count += 1
                
        print(f"[OK] Audio dataset ready at: {target_dir}")
        print(f"   Real: {real_count} | Fake: {fake_count}")
        return True
    except Exception as e:
        print(f"[ERROR] Error downloading audio dataset: {e}")
        return False

def download_visual_genimage(max_samples: int = 50000):
    print("=" * 70)
    print("DOWNLOADING VISUAL DATASET (GenImage / Diffusion Artifacts)")
    print("=" * 70)
    
    target_dir = ROOT_DIR / "datasets" / "highres_visual"
    setup_dirs(target_dir)
    
    hf_dataset_name = "ngscamdev/deepfake-and-real-images"
    
    print(f"Connecting to HuggingFace dataset: {hf_dataset_name}...")
    try:
        dataset = load_dataset(
            hf_dataset_name,
            split="train",
            streaming=True
        )
        
        real_count = 0
        fake_count = 0
        target_per_class = max_samples // 2
        
        for item in tqdm(dataset, desc="Streaming visual files"):
            if real_count >= target_per_class and fake_count >= target_per_class:
                break
                
            img = item['image']
            label = item['label']
            is_real = (label == 1)
            
            count = real_count if is_real else fake_count
            if count < int(target_per_class * 0.8):
                split = 'train'
            elif count < int(target_per_class * 0.9):
                split = 'val'
            else:
                split = 'test'
                
            label_name = 'real' if is_real else 'fake'
            out_path = target_dir / split / label_name / f"img_{count:06d}.jpg"
            
            if img.width >= 256 and img.height >= 256:
                img.save(out_path, quality=95)
                if is_real:
                    real_count += 1
                else:
                    fake_count += 1
                    
        print(f"[OK] Visual dataset ready at: {target_dir}")
        print(f"   Real: {real_count} | Fake: {fake_count}")
        return True
    except Exception as e:
        print(f"[ERROR] Error downloading visual dataset: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Advanced Datasets for ONN-DETECT")
    parser.add_argument("--mode", choices=["all", "visual", "audio"], default="all")
    parser.add_argument("--visual-samples", type=int, default=50000)
    parser.add_argument("--audio-samples", type=int, default=10000)
    args = parser.parse_args()
    
    if args.mode in ["all", "visual"]:
        download_visual_genimage(max_samples=args.visual_samples)
        
    if args.mode in ["all", "audio"]:
        download_audio_inthewild(max_samples=args.audio_samples)

if __name__ == "__main__":
    main()
