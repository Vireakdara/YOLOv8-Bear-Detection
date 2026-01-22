"""
Evaluate and Visualize YOLO Bear Detection Model
- Generates detailed metrics
- Creates visualizations (confusion matrix, precision-recall curves)
- Produces detection statistics
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import argparse
from datetime import datetime

class BearYOLOEvaluator:
    def __init__(self, model_path, device=0):
        """Initialize evaluator"""
        if not Path(model_path).exists():
            print(f"[!] Model not found: {model_path}")
            sys.exit(1)
        
        self.model = YOLO(model_path)
        self.device = device
        print(f"[+] Model loaded: {model_path}")
    
    def evaluate(self, dataset_yaml, output_dir="results/evaluation"):
        """
        Evaluate model on validation set
        """
        print("\n" + "="*60)
        print("Model Evaluation")
        print("="*60 + "\n")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[*] Evaluating on dataset: {dataset_yaml}")
        
        # Run validation
        results = self.model.val(data=dataset_yaml, device=self.device, verbose=True)
        
        print("\n[+] Evaluation Metrics:")
        print(f"    mAP50: {results.box.map50:.4f}")
        print(f"    mAP50-95: {results.box.map:.4f}")
        print(f"    Precision: {results.box.mp:.4f}")
        print(f"    Recall: {results.box.mr:.4f}")
        
        # Save metrics
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'Precision': float(results.box.mp),
            'Recall': float(results.box.mr),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n[+] Metrics saved to: {output_path / 'metrics.json'}")
        
        return results
    
    def predict_single(self, image_path, conf=0.25, output_dir="results/predictions"):
        """
        Run inference on single image
        """
        if not Path(image_path).exists():
            print(f"[!] Image not found: {image_path}")
            return None
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[*] Running inference on: {image_path}")
        
        # Inference
        results = self.model.predict(
            source=image_path,
            conf=conf,
            device=self.device,
            save=False,
            verbose=False
        )
        
        if len(results) == 0:
            print("[!] No results returned")
            return None
        
        result = results[0]
        
        # Draw predictions
        annotated_img = result.plot()
        
        # Save annotated image
        output_img_path = output_path / f"detected_{Path(image_path).name}"
        cv2.imwrite(str(output_img_path), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        
        print(f"[+] Detections found: {len(result.boxes)}")
        
        if len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                conf_val = box.conf.item()
                print(f"    Bear {i+1}: confidence={conf_val:.4f}")
            print(f"[+] Annotated image saved: {output_img_path}")
        else:
            print("[!] No bears detected in image")
        
        return result
    
    def predict_batch(self, image_dir, conf=0.25, output_dir="results/predictions"):
        """
        Run inference on batch of images
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"[!] Directory not found: {image_dir}")
            return None
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[*] Running batch inference on: {image_dir}")
        
        # Get all image files
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        print(f"[*] Found {len(image_files)} images")
        
        if len(image_files) == 0:
            print("[!] No images found in directory")
            return None
        
        results = self.model.predict(
            source=str(image_dir),
            conf=conf,
            device=self.device,
            save=False,
            verbose=False
        )
        
        print(f"[+] Batch inference complete. Processed {len(results)} images")
        
        # Save results with annotations
        for result, img_file in zip(results, image_files[:len(results)]):
            annotated_img = result.plot()
            output_img_path = output_path / f"detected_{img_file.name}"
            cv2.imwrite(str(output_img_path), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
            print(f"  ✓ {img_file.name}: {len(result.boxes)} detections")
        
        print(f"[+] Results saved to: {output_path}")
        
        return results
    
    def create_visualizations(self, results_dir="results/bear_detection_yolov8x"):
        """
        Create visualization plots from training results
        """
        results_path = Path(results_dir)
        
        if not results_path.exists():
            print(f"[!] Results directory not found: {results_dir}")
            return
        
        print(f"\n[*] Creating visualizations from: {results_dir}")
        
        # Look for results files
        results_json = results_path / 'results.json'
        if results_json.exists():
            with open(results_json, 'r') as f:
                results_data = json.load(f)
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            metrics = list(results_data.keys())
            epochs = len(results_data[metrics[0]])
            
            # Plot metrics
            for i, metric in enumerate(['box_loss', 'cls_loss', 'dfl_loss']):
                if metric in results_data:
                    ax = axes[i // 2, i % 2]
                    ax.plot(results_data[metric])
                    ax.set_title(metric)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(Path(results_dir) / 'training_curves.png')
            print(f"[+] Training curves saved: {Path(results_dir) / 'training_curves.png'}")
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Bear Detection Model")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (best.pt)')
    parser.add_argument('--dataset', type=str, default='configs/bear_dataset.yaml',
                       help='Path to dataset configuration')
    parser.add_argument('--image', type=str, default=None,
                       help='Single image for inference')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Directory of images for batch inference')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--device', type=int, default=0,
                       help='Device ID (0 for GPU, cpu for CPU)')
    
    args = parser.parse_args()
    
    evaluator = BearYOLOEvaluator(args.model, device=args.device)
    
    # Evaluate on dataset
    evaluator.evaluate(args.dataset)
    
    # Run inference if image specified
    if args.image:
        evaluator.predict_single(args.image, conf=args.conf)
    
    # Batch inference if directory specified
    if args.image_dir:
        evaluator.predict_batch(args.image_dir, conf=args.conf)

if __name__ == "__main__":
    main()
