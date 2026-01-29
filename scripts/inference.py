"""
Real-time and Batch Inference for Bear Detection
- Single image inference with visualization
- Video inference with real-time detection
- Batch processing
- Confidence filtering and result logging
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import json
from datetime import datetime

class BearDetectionInference:
    def __init__(self, model_path, device=0, conf_threshold=0.25):
        """Initialize inference engine"""
        if not Path(model_path).exists():
            print(f"[!] Model not found: {model_path}")
            sys.exit(1)
        
        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        print(f"[+] Model loaded: {model_path}")
        print(f"[+] Confidence threshold: {conf_threshold}")
    
    def infer_image(self, image_path, output_path=None):
        """
        Detect bears in a single image
        """
        if not Path(image_path).exists():
            print(f"[!] Image not found: {image_path}")
            return None
        
        print(f"\n[*] Processing image: {image_path}")
        
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            device=self.device,
            save=False,
            verbose=False
        )
        
        result = results[0]
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw detections
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf.item()
            
            # Draw bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw confidence label
            label = f"Bear: {conf:.2f}"
            cv2.putText(img, label, (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class': 'bear'
            })
        
        print(f"[+] Detections: {len(detections)}")
        for i, det in enumerate(detections):
            print(f"    Bear {i+1}: confidence={det['confidence']:.4f}")
        
        # Save result
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), img)
            print(f"[+] Annotated image saved: {output_path}")
        
        return detections
    
    def infer_video(self, video_path, output_path=None, fps=30):
        """
        Detect bears in video frames
        """
        if not Path(video_path).exists():
            print(f"[!] Video not found: {video_path}")
            return None
        
        print(f"\n[*] Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"    Total frames: {total_frames}")
        print(f"    Resolution: {width}x{height}")
        
        # Setup video writer if output specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        else:
            out = None
        
        frame_count = 0
        detections_log = []
        
        print("[*] Processing frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"    Processing frame {frame_count}/{total_frames}")
            
            # Inference
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                device=self.device,
                save=False,
                verbose=False
            )
            
            result = results[0]
            annotated_frame = result.plot()
            
            # Log detections
            frame_detections = {
                'frame': frame_count,
                'detections': len(result.boxes),
                'boxes': []
            }
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                frame_detections['boxes'].append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf)
                })
            
            if len(result.boxes) > 0:
                detections_log.append(frame_detections)
            
            # Write frame
            if out:
                out.write(annotated_frame)
        
        cap.release()
        if out:
            out.release()
            print(f"[+] Output video saved: {output_path}")
        
        print(f"[+] Video processing complete")
        print(f"    Total frames: {frame_count}")
        print(f"    Frames with detections: {len(detections_log)}")
        
        return detections_log
    
    def infer_webcam(self, duration=30):
        """
        Real-time detection from webcam
        """
        print("[*] Starting webcam inference (press 'q' to quit)...")
        
        cap = cv2.namedWindow('Bear Detection', cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(0)
        
        start_time = datetime.now()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Inference
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                device=self.device,
                save=False,
                verbose=False
            )
            
            result = results[0]
            annotated_frame = result.plot()
            
            # Display
            cv2.imshow('Bear Detection', annotated_frame)
            
            frame_count += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if elapsed > duration:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"[+] Webcam session complete")
        print(f"    Total frames: {frame_count}")
        print(f"    Duration: {elapsed:.1f}s")

def main():
    parser = argparse.ArgumentParser(description="Bear Detection Inference")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--image', type=str, default=None,
                       help='Input image path')
    parser.add_argument('--video', type=str, default=None,
                       help='Input video path')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam for inference')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--device', type=int, default=0,
                       help='Device ID (0 for GPU)')
    
    args = parser.parse_args()
    
    inference = BearDetectionInference(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf
    )
    
    if args.image:
        inference.infer_image(args.image, output_path=args.output)
    elif args.video:
        inference.infer_video(args.video, output_path=args.output)
    elif args.webcam:
        inference.infer_webcam()
    else:
        print("[!] Please specify --image, --video, or --webcam")

if __name__ == "__main__":
    main()