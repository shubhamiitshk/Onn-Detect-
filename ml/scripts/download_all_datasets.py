#!/usr/bin/env python3
"""
Complete dataset downloader for ONN-DETECT
Downloads CIFAKE (visual) and Deepfake Audio datasets from HuggingFace
"""

import os
import shutil
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import soundfile as sf

def setup_directories():
    """Create clean directory structure"""
    print("🔧 Setting up directory structure...")
    
    base_path = Path("../datasets")
    
    # Clean old data
    for folder in ["asvspoof2019"]:
        target = base_path / folder
        if target.exists():
            print(f"  Removing old {folder}...")
            shutil.rmtree(target)
    
    # Create new structure
    dirs = [
        "genimage/train/real",
        "genimage/train/fake",
        "genimage/val/real",
        "genimage/val/fake",
        "genimage/test/real",
        "genimage/test/fake",
        "asvspoof2019/train/real",
        "asvspoof2019/train/fake",
        "asvspoof2019/val/real",
        "asvspoof2019/val/fake",
        "asvspoof2019/test/real",
        "asvspoof2019/test/fake",
    ]
    
    for dir_path in dirs:
        (base_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created\n")

def download_cifake():
    """Download CIFAKE dataset - 120K images (60K real + 60K AI-generated)"""
    print("=" * 70)
    print("📥 VISUAL DATASET: CIFAKE")
    print("   Source: dragonintelligence/CIFAKE-image-dataset")
    print("   Real: CIFAR-10 photographs")
    print("   Fake: Stable Diffusion AI-generated images")
    print("=" * 70)
    
    base_path = Path("../datasets/genimage")
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset(
            "dragonintelligence/CIFAKE-image-dataset",
            trust_remote_code=True
        )
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"   Splits available: {list(dataset.keys())}")
        
        # Process each split
        for split_name in ['train', 'test']:
            if split_name not in dataset:
                print(f"⚠️  Split '{split_name}' not found, skipping...")
                continue
                
            split_data = dataset[split_name]
            print(f"\n📊 Processing {split_name} split: {len(split_data)} images")
            
            # Map to our directory structure
            target_split = split_name if split_name != 'test' else 'val'
            
            for idx, item in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
                # Determine if real or fake
                label = item.get('label', 0)  # 0 = real, 1 = fake
                label_name = 'real' if label == 0 else 'fake'
                
                # Get image
                img = item['image']
                if isinstance(img, str):
                    # If it's a file path
                    img = Image.open(img).convert('RGB')
                elif not isinstance(img, Image.Image):
                    # Convert to PIL Image if needed
                    img = Image.fromarray(img).convert('RGB')
                else:
                    img = img.convert('RGB')
                
                # Save image
                output_path = base_path / target_split / label_name / f"img_{idx:06d}.jpg"
                img.save(output_path, quality=95)
            
            print(f"✅ Saved {len(split_data)} images to {target_split}/")
        
        # Print statistics
        print("\n📊 CIFAKE Dataset Statistics:")
        for split in ['train', 'val', 'test']:
            split_path = base_path / split
            if split_path.exists():
                real_count = len(list((split_path / 'real').glob('*.jpg')))
                fake_count = len(list((split_path / 'fake').glob('*.jpg')))
                print(f"   {split:5s}: {real_count:6d} real + {fake_count:6d} fake = {real_count + fake_count:6d} total")
        
        print("\n✅ CIFAKE dataset ready!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading CIFAKE: {e}")
        print("   Trying alternative approach...")
        return False

def download_audio_dataset():
    """Download deepfake audio dataset"""
    print("=" * 70)
    print("🎵 AUDIO DATASET: Deepfake Audio Detection")
    print("   Source: garystafford/deepfake-audio-detection")
    print("   Real: 933 authentic human recordings")
    print("   Fake: 933 synthetic/TTS-generated samples")
    print("=" * 70)
    
    base_path = Path("../datasets/asvspoof2019")
    
    try:
        # Load dataset without decoding audio to bypass torchcodec Windows DLL issues
        from datasets import Audio
        dataset = load_dataset(
            "garystafford/deepfake-audio-detection",
            trust_remote_code=True
        ).cast_column('audio', Audio(decode=False))
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"   Splits available: {list(dataset.keys())}")
        
        # Process each split
        split_mapping = {
            'train': 'train',
            'validation': 'val',
            'test': 'test'
        }
        
        for split_name in ['train', 'validation', 'test']:
            if split_name not in dataset:
                print(f"⚠️  Split '{split_name}' not found, skipping...")
                continue
            
            split_data = dataset[split_name]
            target_split = split_mapping.get(split_name, split_name)
            
            print(f"\n📊 Processing {split_name} split: {len(split_data)} audio files")
            
            real_count = 0
            fake_count = 0
            
            for idx, item in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
                try:
                    # Determine label
                    label = item.get('label', item.get('is_fake', 0))
                    label_name = 'real' if label == 0 else 'fake'
                    
                    # Get audio bytes
                    if 'audio' in item and isinstance(item['audio'], dict) and 'bytes' in item['audio']:
                        audio_bytes = item['audio']['bytes']
                    else:
                        continue
                    
                    # Save audio file
                    output_path = base_path / target_split / label_name / f"audio_{idx:06d}.flac"
                    with open(output_path, 'wb') as f:
                        f.write(audio_bytes)
                    
                    if label == 0:
                        real_count += 1
                    else:
                        fake_count += 1
                        
                except Exception as e:
                    print(f"⚠️  Error processing audio {idx}: {e}")
                    continue
            
            print(f"✅ Saved {real_count} real + {fake_count} fake to {target_split}/")
        
        # Print statistics
        print("\n📊 Audio Dataset Statistics:")
        for split in ['train', 'val', 'test']:
            split_path = base_path / split
            if split_path.exists():
                real_count = len(list((split_path / 'real').glob('*.wav')))
                fake_count = len(list((split_path / 'fake').glob('*.wav')))
                print(f"   {split:5s}: {real_count:4d} real + {fake_count:4d} fake = {real_count + fake_count:4d} total")
        
        print("\n✅ Audio dataset ready!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading audio dataset: {e}")
        return False

def verify_datasets():
    """Verify all datasets are properly downloaded"""
    print("=" * 70)
    print("🔍 VERIFYING DATASETS")
    print("=" * 70)
    
    base_path = Path("../datasets")
    all_good = True
    
    # Check visual dataset
    visual_path = base_path / "genimage"
    if visual_path.exists():
        print("\n📊 Visual Dataset (CIFAKE):")
        for split in ['train', 'val', 'test']:
            split_path = visual_path / split
            if split_path.exists():
                real = len(list((split_path / 'real').glob('*.jpg')))
                fake = len(list((split_path / 'fake').glob('*.jpg')))
                total = real + fake
                status = "✅" if total > 0 else "❌"
                print(f"   {status} {split:5s}: {real:6d} real + {fake:6d} fake")
                if total == 0:
                    all_good = False
            else:
                print(f"   ❌ {split:5s}: Directory not found")
                all_good = False
    else:
        print("❌ Visual dataset directory not found")
        all_good = False
    
    # Check audio dataset
    audio_path = base_path / "asvspoof2019"
    if audio_path.exists():
        print("\n🎵 Audio Dataset:")
        for split in ['train', 'val', 'test']:
            split_path = audio_path / split
            if split_path.exists():
                real = len(list((split_path / 'real').glob('*.wav')))
                fake = len(list((split_path / 'fake').glob('*.wav')))
                total = real + fake
                status = "✅" if total > 0 else "⚠️"
                print(f"   {status} {split:5s}: {real:4d} real + {fake:4d} fake")
                if total == 0 and split in ['train', 'val']:
                    all_good = False
            else:
                print(f"   ⚠️  {split:5s}: Directory not found")
    else:
        print("⚠️  Audio dataset directory not found")
    
    print("\n" + "=" * 70)
    if all_good:
        print("✅ ALL DATASETS VERIFIED SUCCESSFULLY!")
        print("   You can now run: uv run python scripts/train_srm_cnn.py --device cuda")
    else:
        print("⚠️  Some datasets have issues, but you can still proceed")
        print("   Visual training will work if CIFAKE downloaded successfully")
    print("=" * 70)
    
    return all_good

def main():
    print("\n" + "=" * 70)
    print("🚀 ONN-DETECT DATASET DOWNLOADER")
    print("=" * 70)
    print()
    
    # Setup
    setup_directories()
    
    # Download visual dataset (Already downloaded!)
    visual_success = True
    
    # Download audio dataset
    audio_success = download_audio_dataset()
    
    # Verify
    verify_datasets()
    
    print("\n📋 NEXT STEPS:")
    if visual_success:
        print("   1. Train visual model:")
        print("      uv run python scripts/train_srm_cnn.py --device cuda")
        print()
    if audio_success:
        print("   2. Train audio model:")
        print("      uv run python scripts/train_lcnn_audio.py --device cuda")
        print()
    
    if not visual_success and not audio_success:
        print("   ❌ Dataset download failed. Please check your internet connection")
        print("   and try again, or manually download from:")
        print("   - Visual: https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset")
        print("   - Audio: https://huggingface.co/datasets/garystafford/deepfake-audio-detection")

if __name__ == "__main__":
    main()
