#!/usr/bin/env python3
"""
Quick Start Training Script for Bear Detection
Minimal setup needed - just run this to start training
Uses Roboflow dataset with challenging backgrounds for best results
"""

import sys
from pathlib import Path
import argparse
import time

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from train_yolo import BearYOLOTrainer
from download_dataset import BearDatasetDownloader

def main():
    parser = argparse.ArgumentParser(description="Quick Start: Train Bear Detection Model")
    parser.add_argument('--setup-only', action='store_true',
                       help='Only setup dataset, do not train')
    parser.add_argument('--sample', action='store_true',
                       help='Use sample dataset for quick testing')
    parser.add_argument('--inaturalist', action='store_true',
                       help='Use iNaturalist dataset (download, slow)')
    parser.add_argument('--dataset-count', type=int, default=500,
                       help='Number of iNaturalist images to download (default: 500)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device (0) or CPU (using -1)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("YOLO Bear Detection - Quick Start")
    print("="*70 + "\n")
    
    # Check for extracted Roboflow dataset first
    print("[1/3] Setting up dataset...")
    has_extracted_data = (Path('./train').exists() and Path('./train/images').exists()) or \
                         (Path('./data/images/train').exists())
    
    print(f"      Checking for existing data...", end=" ", flush=True)
    if has_extracted_data:
        print("✓ Found!")
    else:
        print("✗")
    
    downloader = BearDatasetDownloader('./data')
    
    if has_extracted_data:
        print("      [+] Found extracted dataset (train/valid/test folders)")
        # Organize extracted data into data/ folder structure
        if Path('./train').exists():
            print("      Organizing Roboflow dataset structure...")
            import shutil
            data_dir = Path('./data')
            
            # Move train/valid/test to data/images and data/labels
            for split_src, split_dst in [('./train', 'train'), ('./valid', 'val'), ('./test', 'test')]:
                src_path = Path(split_src)
                if src_path.exists():
                    # Images
                    img_src = src_path / 'images'
                    img_dst = data_dir / 'images' / split_dst
                    if img_src.exists():
                        img_dst.parent.mkdir(parents=True, exist_ok=True)
                        if img_dst.exists():
                            shutil.rmtree(img_dst)
                        shutil.copytree(img_src, img_dst)
                        print(f"      [+] {split_dst} images: {len(list(img_dst.glob('*')))} files")
                    
                    # Labels
                    lbl_src = src_path / 'labels'
                    lbl_dst = data_dir / 'labels' / split_dst
                    if lbl_src.exists():
                        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
                        if lbl_dst.exists():
                            shutil.rmtree(lbl_dst)
                        shutil.copytree(lbl_src, lbl_dst)
                        print(f"      [+] {split_dst} labels: {len(list(lbl_dst.glob('*')))} files")
    elif args.sample:
        print("      Using sample dataset for quick testing...")
        downloader.create_sample_dataset()
    elif args.inaturalist:
        print("      Downloading iNaturalist bear dataset...")
        print("      (Real-world challenging backgrounds - best results)\n")
        success = downloader.download_inaturalist_bears(num_images=args.dataset_count)
        if not success:
            print("\n[!] iNaturalist download failed, falling back to sample dataset...")
            downloader.create_sample_dataset()
    else:
        print("      Using existing dataset...")
        if not downloader.verify_dataset():
            print("\n[!] No dataset found, creating sample for testing...")
            downloader.create_sample_dataset()
    
    downloader.verify_dataset()
    print()  # New line
    
    if args.setup_only:
        print("[+] Dataset setup complete!")
        print("    To train: python quickstart.py")
        return
    
    # Train model
    print("[2/3] Preparing training environment...")
    print("      Loading configuration...", end=" ", flush=True)
    trainer = BearYOLOTrainer(config_path='configs/training_config.yaml')
    print("✓\n")
    
    print("[3/3] Starting training...")
    print("      Monitor progress below:\n")
    
    # Add small delay to show all messages before training starts
    time.sleep(0.5)
    
    try:
        trainer.train(dataset_yaml='configs/bear_dataset.yaml')
        print("\n" + "="*70)
        print("[SUCCESS] Training complete!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Evaluate model:")
        print("     python -m scripts.evaluate --model results/bear_detection_yolov8x/weights/best.pt")
        print("\n  2. Run inference:")
        print("     python -m scripts.inference --model results/bear_detection_yolov8x/weights/best.pt --image <path>")
        
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user")
    except Exception as e:
        print(f"\n[!] Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
