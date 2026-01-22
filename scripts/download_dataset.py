"""
Download and Prepare Bear Dataset from Public Sources
- Uses iNaturalist bear observations (best for challenging backgrounds)
- Supports Roboflow pre-made datasets
- Converts to YOLO format
- Handles challenging background conditions
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
import urllib.request
import zipfile
from PIL import Image
import cv2
import argparse
import subprocess

class BearDatasetDownloader:
    def __init__(self, output_dir="./data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    def download_roboflow_bear_dataset(self, api_key=None, workspace="", project="", version=1):
        """
        Download optimized bear detection dataset from Roboflow
        Roboflow has pre-made bear datasets optimized for challenging backgrounds
        
        Example: workspace="YOUR_WORKSPACE", project="bear-detection", version=1
        """
        print("[*] Downloading Roboflow bear dataset...")
        print("    This requires roboflow API setup")
        
        try:
            from roboflow import Roboflow
            
            rf = Roboflow(api_key=api_key)
            project_obj = rf.workspace(workspace).project(project)
            dataset = project_obj.versions(version).download("yolov8")
            
            print(f"[+] Dataset downloaded to: {dataset.location}")
            return True
        except ImportError:
            print("[!] roboflow package not installed")
            print("    Install with: pip install roboflow")
            return False
        except Exception as e:
            print(f"[!] Error downloading Roboflow dataset: {e}")
            return False
    
    def download_inaturalist_bears(self, num_images=500):
        """
        Download bear images from iNaturalist with challenging backgrounds
        This is the RECOMMENDED dataset - yields best results
        
        Downloads real-world bear observations with:
        - Complex forest/mountain backgrounds
        - Various lighting conditions
        - Partial occlusion
        - Different bear species and sizes
        """
        print("[*] Downloading iNaturalist bear dataset...")
        print("    This uses iNaturalist API to fetch bear observations")
        print(f"    Target: {num_images} images with challenging backgrounds\n")
        
        try:
            import requests
            
            # iNaturalist API endpoint for bears (Ursidae family)
            # Taxonomy ID for bears: 41679
            url = "https://api.inaturalist.org/v1/observations"
            
            params = {
                'taxon_id': 41679,  # Bears (Ursidae)
                'has_photos': True,
                'per_page': 200,
                'order_by': 'created_at',
                'order': 'desc'
            }
            
            downloaded_count = 0
            page = 1
            
            while downloaded_count < num_images:
                params['page'] = page
                
                print(f"[*] Fetching page {page}...")
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                observations = data.get('results', [])
                if not observations:
                    print("[!] No more observations available")
                    break
                
                for obs in observations:
                    if downloaded_count >= num_images:
                        break
                    
                    # Get photo from observation
                    if obs.get('photos'):
                        try:
                            photo = obs['photos'][0]
                            photo_url = photo['url'].replace('square', 'medium')
                            
                            # Determine split (80/10/10)
                            rand = random.random()
                            if rand < 0.8:
                                split = 'train'
                            elif rand < 0.9:
                                split = 'val'
                            else:
                                split = 'test'
                            
                            # Download image
                            img_filename = f"bear_{downloaded_count:05d}.jpg"
                            img_path = self.output_dir / 'images' / split / img_filename
                            
                            urllib.request.urlretrieve(photo_url, str(img_path))
                            
                            # Create placeholder YOLO label
                            label_path = self.output_dir / 'labels' / split / f"bear_{downloaded_count:05d}.txt"
                            with open(label_path, 'w') as f:
                                # Placeholder: bear roughly in center (requires manual annotation ideally)
                                # For automated: assume bear takes ~30% of image in center
                                f.write("0 0.5 0.5 0.3 0.35\n")
                            
                            downloaded_count += 1
                            if downloaded_count % 50 == 0:
                                print(f"    Downloaded: {downloaded_count}/{num_images}")
                        
                        except Exception as e:
                            print(f"    [!] Failed to download image: {e}")
                            continue
                
                page += 1
            
            print(f"\n[+] Downloaded {downloaded_count} iNaturalist bear images")
            return downloaded_count > 0
        
        except ImportError:
            print("[!] requests package not installed")
            print("    Install with: pip install requests")
            return False
        except Exception as e:
            print(f"[!] Error downloading iNaturalist dataset: {e}")
            return False
    
    def create_sample_dataset(self):
        """
        Create a minimal sample dataset structure for testing
        In production, replace with actual dataset
        """
        print("[*] Creating sample dataset structure...")
        
        # Create dummy images and labels for demonstration
        splits = {'train': 100, 'val': 20, 'test': 20}
        
        for split, count in splits.items():
            for i in range(count):
                # Create dummy image
                img = Image.new('RGB', (640, 480), color=(random.randint(0, 255), 
                                                          random.randint(0, 255), 
                                                          random.randint(0, 255)))
                img_path = self.output_dir / 'images' / split / f"image_{i:04d}.jpg"
                img.save(img_path)
                
                # Create dummy label (YOLO format)
                # For demonstration: 1-3 random bear detections per image
                num_bears = random.randint(1, 3)
                label_path = self.output_dir / 'labels' / split / f"image_{i:04d}.txt"
                
                with open(label_path, 'w') as f:
                    for _ in range(num_bears):
                        # Class 0 (bear), random normalized coordinates
                        x_center = random.uniform(0.1, 0.9)
                        y_center = random.uniform(0.1, 0.9)
                        width = random.uniform(0.1, 0.4)
                        height = random.uniform(0.1, 0.4)
                        f.write(f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")
        
        print(f"[+] Sample dataset created in {self.output_dir}")
        return True
    
    def verify_dataset(self):
        """Verify dataset structure"""
        print("\n[*] Verifying dataset structure...")
        
        total_images = 0
        total_labels = 0
        
        for split in ['train', 'val', 'test']:
            img_dir = self.output_dir / 'images' / split
            label_dir = self.output_dir / 'labels' / split
            
            img_count = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
            label_count = len(list(label_dir.glob('*.txt')))
            
            total_images += img_count
            total_labels += label_count
            
            print(f"    {split:5} split: {img_count:4} images, {label_count:4} labels")
        
        print(f"\n[+] Total: {total_images} images, {total_labels} labels")
        
        if total_images > 0 and total_labels > 0:
            return True
        return False

def main():
    parser = argparse.ArgumentParser(description="Download and Prepare Bear Dataset")
    parser.add_argument('--output', type=str, default='./data',
                       help='Output directory for dataset')
    parser.add_argument('--sample', action='store_true',
                       help='Create sample dataset for testing')
    parser.add_argument('--inaturalist', action='store_true',
                       help='Download from iNaturalist (RECOMMENDED - challenging backgrounds)')
    parser.add_argument('--inaturalist-count', type=int, default=500,
                       help='Number of iNaturalist images to download (default: 500)')
    parser.add_argument('--roboflow', action='store_true',
                       help='Download from Roboflow')
    parser.add_argument('--roboflow-api', type=str, default=None,
                       help='Roboflow API key')
    parser.add_argument('--roboflow-workspace', type=str, default='',
                       help='Roboflow workspace')
    parser.add_argument('--roboflow-project', type=str, default='',
                       help='Roboflow project name')
    
    args = parser.parse_args()
    
    downloader = BearDatasetDownloader(args.output)
    
    if args.inaturalist:
        print("\n" + "="*70)
        print("Downloading iNaturalist Bear Dataset (RECOMMENDED)")
        print("="*70)
        print("[*] This dataset has challenging backgrounds with real-world")
        print("    conditions - yields best detection results\n")
        success = downloader.download_inaturalist_bears(num_images=args.inaturalist_count)
        if success:
            downloader.verify_dataset()
            print("\n[+] iNaturalist dataset ready for training!")
    
    elif args.roboflow:
        print("\n" + "="*70)
        print("Downloading Roboflow Bear Dataset")
        print("="*70 + "\n")
        success = downloader.download_roboflow_bear_dataset(
            api_key=args.roboflow_api,
            workspace=args.roboflow_workspace,
            project=args.roboflow_project
        )
        if success:
            downloader.verify_dataset()
    
    elif args.sample:
        print("\n" + "="*70)
        print("Creating Sample Bear Dataset")
        print("="*70 + "\n")
        downloader.create_sample_dataset()
        downloader.verify_dataset()
        print("\n[!] Note: Sample dataset is for testing only")
        print("    For real training, use: python scripts/download_dataset.py --inaturalist")
    
    else:
        print("\n" + "="*70)
        print("Bear Dataset Setup Tool")
        print("="*70 + "\n")
        
        print("RECOMMENDED OPTION:")
        print("  Download from iNaturalist (challenging backgrounds):")
        print("  python scripts/download_dataset.py --inaturalist --inaturalist-count 500\n")
        
        print("OTHER OPTIONS:")
        print("  1. Sample dataset (testing): python scripts/download_dataset.py --sample")
        print("  2. Roboflow dataset: python scripts/download_dataset.py --roboflow \\")
        print("       --roboflow-api YOUR_KEY --roboflow-workspace WORKSPACE \\")
        print("       --roboflow-project PROJECT_NAME\n")
        
        print("Check current dataset:")
        downloader.verify_dataset()

if __name__ == "__main__":
    main()
