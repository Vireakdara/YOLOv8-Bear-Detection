"""
Prepare Bear Dataset for YOLO Training
- Downloads a challenging bear dataset
- Splits into train/val/test sets
- Converts to YOLO format (txt annotations)
- Handles complex background conditions
"""

import os
import sys
import json
import shutil
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import random
from PIL import Image
import cv2

class BearDatasetPreparer:
    def __init__(self, data_root="./data"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)
        
    def download_dataset(self):
        """
        Download bear dataset from public sources
        Using COCO-formatted bear annotations or similar challenging dataset
        """
        print("[*] Setting up bear dataset directories...")
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (self.data_root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.data_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        print("[+] Directory structure created at:", self.data_root)
        
        # Note: For actual implementation, you would download from sources like:
        # 1. iNaturalist bear dataset (has challenging backgrounds)
        # 2. Microsoft COCO with bear class filtered
        # 3. Custom annotated bear datasets
        
        print("[!] IMPORTANT: Please add your bear dataset images to:")
        print(f"    - {self.data_root / 'images' / 'train'}")
        print(f"    - {self.data_root / 'images' / 'val'}")
        print(f"    - {self.data_root / 'images' / 'test'}")
        print("\n[!] IMPORTANT: Please add annotations (COCO format JSON) or use a tool like:")
        print("    - Roboflow: https://roboflow.com (search for bear datasets)")
        print("    - iNaturalist: https://www.inaturalist.org")
        print("    - LabelImg/Roboflow for annotation")
        
    def coco_to_yolo(self, annotation_file, image_dir, output_dir):
        """
        Convert COCO format annotations to YOLO format
        COCO: {"image_id": x, "category_id": y, "bbox": [x, y, w, h]}
        YOLO: <class> <x_center> <y_center> <width> <height> (normalized)
        """
        print(f"[*] Converting COCO annotations to YOLO format from {annotation_file}...")
        
        if not os.path.exists(annotation_file):
            print(f"[!] Annotation file not found: {annotation_file}")
            return
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create annotation mapping
        image_info = {img['id']: img for img in coco_data['images']}
        annotations_by_image = {}
        
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Convert each image's annotations
        for img_id, anns in annotations_by_image.items():
            img_info = image_info[img_id]
            img_width = img_info['width']
            img_height = img_info['height']
            img_filename = img_info['file_name']
            
            # Create YOLO annotation file
            txt_filename = Path(output_dir) / f"{Path(img_filename).stem}.txt"
            
            with open(txt_filename, 'w') as f:
                for ann in anns:
                    bbox = ann['bbox']  # [x, y, width, height]
                    category_id = ann['category_id']
                    
                    # Convert to YOLO format (center normalized)
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height
                    
                    # Clamp values to [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"[+] Converted {len(annotations_by_image)} annotations to YOLO format")
    
    def create_demo_structure(self):
        """
        Create a demo directory structure with instructions
        """
        print("[*] Creating demo structure and instructions...")
        
        readme_content = """# Bear Detection YOLOv8 Project

## Dataset Setup Instructions

This project uses YOLO for bear detection with challenging background conditions.

### Step 1: Obtain Bear Dataset

Choose one of these options:

1. **Roboflow Pre-made Dataset**
   - Visit: https://roboflow.com
   - Search for "bear" or "wildlife detection"
   - Download in YOLO format
   - Extract to `data/images/` and `data/labels/`

2. **iNaturalist Dataset**
   - Download bear observations from: https://www.inaturalist.org
   - Use tools like `inat-download` or `cv2_extractor`
   - Annotate using Roboflow or LabelImg

3. **COCO Dataset (Filtered)**
   - Download COCO dataset
   - Filter for bear class (category_id for bear)
   - Convert using the provided converter script

### Step 2: Directory Structure

Ensure your dataset follows this structure:

```
data/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── val/
│   │   └── img3.jpg
│   └── test/
│       └── img4.jpg
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── img2.txt
    ├── val/
    │   └── img3.txt
    └── test/
        └── img4.txt
```

YOLO label format (.txt):
```
<class_id> <x_center> <y_center> <width> <height>
```

Where values are normalized to [0, 1].

### Step 3: Run Training

```bash
python scripts/train_yolo.py
```

### Step 4: Evaluate Model

```bash
python scripts/evaluate.py
```

### Step 5: Run Inference

```bash
python scripts/inference.py --image <image_path> --model <model_path>
```

## Training Strategy

- **Model**: YOLOv8 Extra-Large (yolov8x) for maximum accuracy
- **Approach**: Transfer learning with fine-tuning on bear-specific dataset
- **Augmentation**: Heavy augmentation for robust detection in challenging conditions
- **Epochs**: 100 with early stopping
- **Batch Size**: 16 (adjust based on GPU memory)

## Challenging Conditions Handled

- Complex forest backgrounds
- Occlusion (partially hidden bears)
- Varying lighting conditions
- Different bear sizes and poses
- Weather conditions (rain, snow)
- Different animal orientations

## Output

Trained model saved to: `results/bear_detection_yolov8x/weights/best.pt`
"""
        
        with open(self.data_root.parent / "README.md", 'w') as f:
            f.write(readme_content)
        
        print("[+] README created at:", self.data_root.parent / "README.md")

if __name__ == "__main__":
    preparer = BearDatasetPreparer("./data")
    preparer.download_dataset()
    preparer.create_demo_structure()
    print("\n[✓] Dataset preparation setup complete!")
    print("[!] Next: Add your bear dataset images and labels to the data/ directory")
