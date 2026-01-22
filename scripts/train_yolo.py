"""
Train YOLOv8 Model for Bear Detection
- Uses transfer learning with pretrained weights
- Optimized for challenging background conditions
- Includes augmentation strategies
- Early stopping and checkpoint management
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import json

class BearYOLOTrainer:
    def __init__(self, config_path="configs/training_config.yaml"):
        """Initialize trainer with configuration"""
        self.config_path = Path(config_path)
        self.load_config()
        self.setup_device()
        
    def load_config(self):
        """Load training configuration from YAML"""
        if not self.config_path.exists():
            print(f"[!] Config file not found: {self.config_path}")
            sys.exit(1)
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("[+] Configuration loaded:")
        for key, value in self.config.items():
            if key not in ['names', 'hsv_h', 'hsv_s', 'hsv_v']:
                print(f"    {key}: {value}")
    
    def setup_device(self):
        """Setup GPU/CPU device"""
        if torch.cuda.is_available():
            self.device = 0
            print(f"[+] GPU available: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA capability: {torch.cuda.get_device_capability(0)}")
        else:
            self.device = 'cpu'
            print("[!] GPU not available, using CPU (training will be slow)")
    
    def train(self, dataset_yaml="configs/bear_dataset.yaml"):
        """
        Train YOLO model on bear dataset
        
        Args:
            dataset_yaml: Path to YOLO dataset configuration
        """
        print("\n" + "="*60)
        print("YOLO Bear Detection Training")
        print("="*60 + "\n")
        
        # Verify dataset exists
        if not Path(dataset_yaml).exists():
            print(f"[!] Dataset config not found: {dataset_yaml}")
            print("    Make sure to prepare dataset first: python scripts/prepare_dataset.py")
            sys.exit(1)
        
        # Load or create model
        model_name = self.config['model_name']
        print(f"[*] Loading model: {model_name}")
        
        # YOLOv8 models: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
        # n=nano, s=small, m=medium, l=large, x=extra-large
        model = YOLO(model_name)
        
        # Prepare training arguments
        train_args = {
            'data': dataset_yaml,
            'epochs': self.config['epochs'],
            'imgsz': self.config['img_size'],
            'batch': self.config['batch_size'],
            'patience': self.config['patience'],
            'device': self.device,
            'workers': self.config['workers'],
            'project': self.config['project'],
            'name': self.config['name'],
            'save': self.config['save'],
            'exist_ok': self.config['exist_ok'],
            'verbose': self.config['verbose'],
            
            # Augmentation
            'augment': self.config['augment'],
            'mosaic': self.config['mosaic'],
            'mixup': self.config['mixup'],
            'copy_paste': self.config['copy_paste'],
            'hsv_h': self.config['hsv_h'],
            'hsv_s': self.config['hsv_s'],
            'hsv_v': self.config['hsv_v'],
            'degrees': self.config['degrees'],
            'translate': self.config['translate'],
            'scale': self.config['scale'],
            'flipud': self.config['flipud'],
            'fliplr': self.config['fliplr'],
            
            # Optimizer
            'optimizer': self.config['optimizer'],
            'lr0': self.config['lr0'],
            'lrf': self.config['lrf'],
            'momentum': self.config['momentum'],
            'weight_decay': self.config['weight_decay'],
            
            # Loss and detection
            'cls': self.config['cls'],
            'dfl': self.config['dfl'],
            'iou': self.config['iou_threshold'],
            'conf': self.config['conf_threshold'],
            
            # Save
            'save_period': self.config['save_period'],
            'resume': self.config['resume'],
        }
        
        print("\n[*] Starting training...")
        print(f"    Model: {model_name}")
        print(f"    Epochs: {self.config['epochs']}")
        print(f"    Batch Size: {self.config['batch_size']}")
        print(f"    Image Size: {self.config['img_size']}")
        print(f"    Device: {self.device}")
        
        # Train the model
        results = model.train(**train_args)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"[+] Results saved to: {self.config['project']}/{self.config['name']}")
        print(f"[+] Best model: {self.config['project']}/{self.config['name']}/weights/best.pt")
        print(f"[+] Last model: {self.config['project']}/{self.config['name']}/weights/last.pt")
        
        return results
    
    def validate(self, model_path, dataset_yaml="configs/bear_dataset.yaml"):
        """Validate trained model"""
        print("\n[*] Validating model...")
        model = YOLO(model_path)
        results = model.val(data=dataset_yaml, device=self.device)
        return results
    
    def export_model(self, model_path, export_format='onnx'):
        """
        Export model to different formats
        Formats: 'torchscript', 'onnx', 'openvino', 'engine', 'coreml', 'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs', 'paddle'
        """
        print(f"\n[*] Exporting model to {export_format}...")
        model = YOLO(model_path)
        model.export(format=export_format)
        print(f"[+] Model exported successfully")

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Bear Detection")
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--dataset', type=str, default='configs/bear_dataset.yaml',
                       help='Path to dataset configuration file')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation only')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model for validation')
    parser.add_argument('--export', type=str, default=None,
                       help='Export model to specified format')
    
    args = parser.parse_args()
    
    trainer = BearYOLOTrainer(config_path=args.config)
    
    if args.validate:
        if not args.model:
            print("[!] --model required for validation")
            sys.exit(1)
        trainer.validate(args.model, args.dataset)
    elif args.export:
        if not args.model:
            print("[!] --model required for export")
            sys.exit(1)
        trainer.export_model(args.model, args.export)
    else:
        trainer.train(dataset_yaml=args.dataset)

if __name__ == "__main__":
    main()
