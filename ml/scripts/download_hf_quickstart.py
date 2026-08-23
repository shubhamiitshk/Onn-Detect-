import os
import shutil
from datasets import load_dataset
from tqdm import tqdm

def clean_old_data():
    print("🧹 Cleaning broken downloads...")
    for folder in ['../datasets/genimage', '../datasets/asvspoof2019']:
        if os.path.exists(folder):
            shutil.rmtree(folder)

def download_visual():
    print("\n📥 Downloading REAL images (Tiny ImageNet)...")
    os.makedirs('../datasets/genimage/train/real', exist_ok=True)
    os.makedirs('../datasets/genimage/train/fake', exist_ok=True)
    os.makedirs('../datasets/genimage/val/real', exist_ok=True)
    os.makedirs('../datasets/genimage/val/fake', exist_ok=True)
    os.makedirs('../datasets/genimage/test/real', exist_ok=True)
    os.makedirs('../datasets/genimage/test/fake', exist_ok=True)

    # Real: High-quality natural photographs
    real_ds = load_dataset("zh-plus/tiny-imagenet", split="train")
    for i, item in enumerate(tqdm(real_ds.select(range(1000)), desc="Saving Real (Train)")):
        item['image'].convert('RGB').save(f'../datasets/genimage/train/real/img_{i}.jpg')
    for i, item in enumerate(tqdm(real_ds.select(range(1000, 1200)), desc="Saving Real (Val)")):
        item['image'].convert('RGB').save(f'../datasets/genimage/val/real/img_{i}.jpg')
    for i, item in enumerate(tqdm(real_ds.select(range(1200, 1400)), desc="Saving Real (Test)")):
        item['image'].convert('RGB').save(f'../datasets/genimage/test/real/img_{i}.jpg')

    # Fake: AI-generated images (Stable Diffusion / DALL-E style)
    print("\n📥 Downloading FAKE images (AI Generated)...")
    fake_ds = load_dataset("poloclub/diffusiondb", "2m_first_5k", split="train") # AI generated art
    for i, item in enumerate(tqdm(fake_ds.select(range(1000)), desc="Saving Fake (Train)")):
        item['image'].convert('RGB').save(f'../datasets/genimage/train/fake/img_{i}.jpg')
    for i, item in enumerate(tqdm(fake_ds.select(range(1000, 1200)), desc="Saving Fake (Val)")):
        item['image'].convert('RGB').save(f'../datasets/genimage/val/fake/img_{i}.jpg')
    for i, item in enumerate(tqdm(fake_ds.select(range(1200, 1400)), desc="Saving Fake (Test)")):
        item['image'].convert('RGB').save(f'../datasets/genimage/test/fake/img_{i}.jpg')

    print("✅ Visual Dataset Ready: 1000 Real + 1000 Fake per split!")

if __name__ == "__main__":
    clean_old_data()
    download_visual()
    print("\n🎉 SUCCESS! You can now run train_srm_cnn.py")
