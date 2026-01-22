"""
Train YOLOv8 Model for Bear Detection
"""

import os
import sys
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO
from datetime import datetime
import time

class BearYOLOTrainer:
    def __init__(self, config_path="configs/training_config.yaml"):
        """Initialize trainer with configuration"""
        self.config_path = Path(config_path)
        self.load_config()
        self.setup_device()
        
    def load_config(self):
        """Load training configuration from YAML"""
        print("[*] Loading configuration...", end=" ", flush=True)
        if not self.config_path.exists():
            print(f"\n[!] Config file not found: {self.config_path}")
            sys.exit(1)
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        print("✓")
        
        print("[+] Configuration:")
        for key, value in self.config.items():
            if key not in ['names', 'hsv_h', 'hsv_s', 'hsv_v']:
                print(f"    • {key}: {value}")
    
    def setup_device(self):
        """Setup GPU/CPU device"""
        print("[*] Setting up device...", end=" ", flush=True)
        if torch.cuda.is_available():
            self.device = 0
            device_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            print("✓")
            print(f"[+] GPU: {device_name}")
            print(f"    Compute Capability: {capability[0]}.{capability[1]}")
            print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = 'cpu'
            print("⚠")
            print("[!] GPU not available, using CPU (training will be very slow)")
    
    def train(self, dataset_yaml="configs/data.yaml"):
        """Train YOLO model on bear dataset"""
        print("\n" + "="*70)
        print("YOLO Bear Detection Training")
        print("="*70 + "\n")
        
        print("[*] Verifying dataset...", end=" ", flush=True)
        if not Path(dataset_yaml).exists():
            print("✗")
            print(f"[!] Dataset config not found: {dataset_yaml}")
            sys.exit(1)
        print("✓")
        
        model_name = self.config['model_name']
        print(f"[*] Loading model: {model_name}...", end=" ", flush=True)
        model = YOLO(model_name)
        print("✓")
        
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
            'verbose': False,
            
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
        
        print("\n" + "="*70)
        print("[*] TRAINING CONFIGURATION")
        print("="*70)
        print(f"  Model:        {model_name}")
        print(f"  Epochs:       {self.config['epochs']}")
        print(f"  Batch Size:   {self.config['batch_size']}")
        print(f"  Image Size:   {self.config['img_size']}")
        print(f"  Device:       {'GPU (CUDA)' if self.device == 0 else 'CPU'}")
        print(f"  Started:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
        print("[*] Starting model training...")
        print("    Training in progress:\n")
        
        start_time = time.time()
        try:
            results = model.train(**train_args)
        except KeyboardInterrupt:
            print("\n\n[!] Training interrupted by user")
            elapsed_time = time.time() - start_time
            print(f"[*] Training time: {elapsed_time/60:.1f} minutes")
            return None
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("[SUCCESS] Training Complete!")
        print("="*70)
        print(f"[+] Total training time: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
        print(f"[+] Results location:  {self.config['project']}/{self.config['name']}")
        print(f"[+] Best model:        weights/best.pt")
        print(f"[+] Last model:        weights/last.pt")
        print(f"[+] Completed:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return results

if __name__ == "__main__":
    trainer = BearYOLOTrainer()
    trainer.train()
