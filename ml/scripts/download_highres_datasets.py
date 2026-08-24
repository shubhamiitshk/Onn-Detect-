import os
import shutil
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# Force paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = SCRIPT_DIR.parent.parent
DATASET_DIR = ROOT_DIR / "datasets" / "highres_visual"

def setup():
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            (DATASET_DIR / split / label).mkdir(parents=True, exist_ok=True)

def download_highres_visual():
    print("=" * 70)
    print("HIGH-RES VISUAL DATASET: Deepfake Data")
    print("   Source: bitmind/diffusion-artifacts")
    print("   Resolution: 512x512 to 1024x1024")
    print("=" * 70)
    
    try:
        # Downloading a massive dataset requires streaming or specific splits
        # We will use a smaller slice of a high-res dataset for demonstration, 
        # but in production, you would stream this.
        print("Fetching high-res dataset index...")
        dataset = load_dataset(
            "ngscamdev/deepfake-and-real-images", 
            split="train",
            streaming=True
        )
        
        # We will save 100,000 images (50k real, 50k fake)
        # 80/10/10 split
        
        real_count = 0
        fake_count = 0
        target = 50000
        
        print("Streaming high-res images to disk... (This will take a while!)")
        
        for item in tqdm(dataset, total=target*2):
            if real_count >= target and fake_count >= target:
                break
                
            img = item['image']
            # For this dataset, 'Fake' is 0 and 'Real' is 1
            label = item['label'] 
            is_real = (label == 1)
            
            # Determine split (80/10/10)
            count = real_count if is_real else fake_count
            if count < int(target * 0.8):
                split = 'train'
            elif count < int(target * 0.9):
                split = 'val'
            else:
                split = 'test'
                
            label_name = 'real' if is_real else 'fake'
            out_path = DATASET_DIR / split / label_name / f"img_{count:06d}.jpg"
            
            # Save only if it's high res (>256)
            if img.width >= 256 and img.height >= 256:
                img.save(out_path, quality=95)
                if is_real:
                    real_count += 1
                else:
                    fake_count += 1
                    
        print("High-Res Dataset downloaded successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    setup()
    download_highres_visual()
