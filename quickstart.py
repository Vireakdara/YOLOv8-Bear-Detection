#!/usr/bin/env python3
"""
Quick Start Training Script for Bear Detection
Minimal setup needed - just run this to start training
Uses iNaturalist dataset with challenging backgrounds for best results
"""

import sys
from pathlib import Path
import argparse

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
    parser.add_argument('--inaturalist', action='store_true', default=True,
                       help='Use iNaturalist dataset (default, challenging backgrounds)')
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
    
    # Setup dataset
    print("[1/3] Setting up dataset...")
    downloader = BearDatasetDownloader('./data')
    
    if args.sample:
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
    
    if args.setup_only:
        print("\n[+] Dataset setup complete!")
        print("    To train: python quickstart.py")
        return
    
    # Train model
    print("\n[2/3] Preparing training...")
    trainer = BearYOLOTrainer(config_path='configs/training_config.yaml')
    
    print("\n[3/3] Starting training...")
    print("      This may take a while depending on GPU and dataset size...")
    
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
