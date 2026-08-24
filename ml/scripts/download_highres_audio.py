import os
import io
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Force paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = SCRIPT_DIR.parent.parent
DATASET_DIR = ROOT_DIR / "datasets" / "asvspoof2019"

def setup():
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            (DATASET_DIR / split / label).mkdir(parents=True, exist_ok=True)

def download_audio():
    print("=" * 70)
    print("HIGH-RES AUDIO DATASET: Deepfake Audio")
    print("=" * 70)
    
    try:
        print("Fetching high-res audio dataset...")
        import datasets
        dataset = load_dataset(
            "garystafford/deepfake-audio-detection", 
            split="train",
            streaming=True
        )
        # Fix WinError 127 by bypassing soundfile decoding entirely
        dataset = dataset.cast_column("audio", datasets.features.Audio(decode=False))
        
        real_count = 0
        fake_count = 0
        target = 25000  
        
        print("Streaming audio files to disk... (This will take a while!)")
        
        for count, item in enumerate(tqdm(dataset)):
            if real_count >= target and fake_count >= target:
                break
                
            label = item['label']
            is_real = (label == 1)
            
            c = real_count if is_real else fake_count
            if c < int(target * 0.8):
                split = 'train'
            elif c < int(target * 0.9):
                split = 'val'
            else:
                split = 'test'
                
            label_name = 'real' if is_real else 'fake'
            out_path = DATASET_DIR / split / label_name / f"audio_{c:06d}.wav"
            
            raw_bytes = item['audio']['bytes']
            
            with open(out_path, 'wb') as f:
                f.write(raw_bytes)
            
            if is_real:
                real_count += 1
            else:
                fake_count += 1
                
        print("Audio Dataset downloaded successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    setup()
    download_audio()
