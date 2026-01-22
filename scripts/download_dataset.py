"""
Download and prepare bear detection dataset
"""

from pathlib import Path
from PIL import Image, ImageDraw
import random
import os

class BearDatasetDownloader:
    def __init__(self, data_dir='./dataset'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def create_sample_dataset(self):
        """Create sample dataset for quick testing"""
        print("\n[*] Creating sample dataset...")
        
        for split in ['train', 'val', 'test']:
            img_dir = self.data_dir / 'images' / split
            lbl_dir = self.data_dir / 'labels' / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            
            count = 100 if split == 'train' else 20
            
            for i in range(count):
                # Create image
                img = Image.new('RGB', (640, 480), 
                               color=(random.randint(50, 200), 
                                     random.randint(50, 150), 
                                     random.randint(30, 100)))
                draw = ImageDraw.Draw(img)
                
                # Draw shapes
                for _ in range(random.randint(1, 3)):
                    x = random.randint(50, 550)
                    y = random.randint(50, 400)
                    w = random.randint(50, 150)
                    h = random.randint(50, 150)
                    draw.ellipse([x, y, x+w, y+h], fill=(139, 69, 19))
                
                img_path = img_dir / f'bear_{i:05d}.jpg'
                img.save(img_path)
                
                # Create label
                lbl_path = lbl_dir / f'bear_{i:05d}.txt'
                with open(lbl_path, 'w') as f:
                    f.write('0 0.5 0.5 0.4 0.4\n')
        
        print("[+] Sample dataset created")
        print("    train: 100 images")
        print("    val:   20 images")
        print("    test:  20 images")
    
    def download_inaturalist_bears(self, num_images=500):
        """Download bears from iNaturalist"""
        print(f"\n[*] Note: iNaturalist download is slow")
        print(f"    Consider extracting Bear.v1i.yolov8.zip instead")
        print(f"    Or run: python quickstart.py --sample")
        return False
    
    def verify_dataset(self):
        """Verify dataset structure"""
        print("\n[+] Dataset structure:")
        
        total = 0
        for split in ['train', 'val', 'test']:
            img_path = self.data_dir / 'images' / split
            lbl_path = self.data_dir / 'labels' / split
            
            if img_path.exists() and lbl_path.exists():
                img_count = len(list(img_path.glob('*')))
                lbl_count = len(list(lbl_path.glob('*')))
                total += img_count
                print(f"    {split}: {img_count} images, {lbl_count} labels")
        
        print(f"    Total: {total} images")
        return total > 0

if __name__ == "__main__":
    downloader = BearDatasetDownloader('./dataset')
    downloader.create_sample_dataset()
    downloader.verify_dataset()
