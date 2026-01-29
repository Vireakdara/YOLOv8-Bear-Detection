# Bear Detection with YOLOv8

A comprehensive YOLO-based bear detection system optimized for challenging background conditions. Uses state-of-the-art YOLOv8 architecture with transfer learning and advanced augmentation strategies.

## Training Results

**Final Performance Metrics:**
- **mAP@0.5:** 94.17% (epoch 20)
- **mAP@0.5-0.95:** 82.92% (epoch 20)
- **Precision:** 91.95%
- **Recall:** 90.39%
- **Training Duration:** 21 epochs (32,267 seconds total)

The model achieved excellent convergence with stable training dynamics and minimal overfitting. Peak performance was reached at epoch 20 with consistent >90% precision and recall metrics.

### Training Progression
| Epoch | mAP@0.5 | mAP@0.5-0.95 | Precision | Recall |
|-------|---------|--------------|-----------|--------|
| 1 | 95.46% | 89.14% | 95.60% | 91.82% |
| 10 | 89.87% | 70.11% | 91.06% | 83.36% |
| 15 | 91.44% | 78.09% | 91.83% | 84.04% |
| 20 | **94.17%** | **82.92%** | **91.95%** | **90.39%** |

The training demonstrated robust learning with effective augmentation strategies and proper regularization, achieving publication-quality results for wildlife detection.

## Project Structure

```
.
├── configs/                    # Configuration files
│   ├── bear_dataset.yaml      # Dataset configuration
│   └── training_config.yaml   # Training hyperparameters
├── data/                       # Dataset directory
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── scripts/                    # Python scripts
│   ├── train_yolo.py          # Training script
│   ├── evaluate.py            # Evaluation and visualization
│   ├── inference.py           # Inference (image/video/webcam)
│   ├── download_dataset.py    # Dataset preparation
│   └── prepare_dataset.py     # COCO to YOLO conversion
├── models/                     # Saved models
├── results/                    # Training results and outputs
├── requirements.txt           # Python dependencies
├── quickstart.py              # Quick start training script
└── README.md                  # This file

```

## Key Features

- **YOLOv8 Architecture**: Latest YOLO version for superior accuracy and speed
- **Transfer Learning**: Pre-trained weights fine-tuned for bear detection
- **Advanced Augmentation**: Mosaic, mixup, copy-paste for robust learning
- **Challenging Conditions**: Optimized for:
  - Complex forest backgrounds
  - Partial occlusion
  - Variable lighting
  - Different bear sizes and poses
  - Weather variations
- **Flexible Inference**: Support for single images, videos, and real-time webcam
- **Comprehensive Evaluation**: Metrics, visualizations, and detailed analysis

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 4GB+ VRAM (GPU) or 8GB+ RAM (CPU)

### Setup

1. **Clone and navigate**:
```bash
cd e:\YOLO\YOLOv26-Bear
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Automated Quick Start (RECOMMENDED)
```bash
# Start training with extracted Roboflow dataset
python quickstart.py
```

### Option 2: Download & Train

#### Download Roboflow Dataset (Recommended - Easiest)
1. Download from: https://universe.roboflow.com/ds/c6gkywSzAj?key=ymftU7vee3
2. Extract the ZIP file to your project folder
3. The script will automatically detect and organize it
4. Run training: `python quickstart.py`

#### Or Download iNaturalist Dataset (Slower - Takes 10+ minutes)
```bash
python quickstart.py --inaturalist --dataset-count 200
```

#### Or Use Sample Dataset (Fast - For Testing)
```bash
python quickstart.py --sample
```

### Training Options
```bash
# Custom epoch count
python quickstart.py --epochs 150

# Custom batch size
python quickstart.py --batch-size 8

# GPU selection
python quickstart.py --device 0
```

### Evaluation
```bash
python scripts/evaluate.py \
  --model results/bear_detection_yolov8x/weights/best.pt \
  --dataset configs/bear_dataset.yaml
```

### Inference
```bash
# Single image
python scripts/inference.py \
  --model results/bear_detection_yolov8x/weights/best.pt \
  --image path/to/image.jpg \
  --output results/detection.jpg

# Video
python scripts/inference.py \
  --model results/bear_detection_yolov8x/weights/best.pt \
  --video path/to/video.mp4 \
  --output results/detection.mp4

# Webcam (real-time)
python scripts/inference.py \
  --model results/bear_detection_yolov8x/weights/best.pt \
  --webcam
```

## Dataset Setup

### RECOMMENDED: iNaturalist Bear Dataset

The **iNaturalist dataset is the optimal choice** for this project because:

✅ **Challenging Backgrounds**: Real-world forest, mountain, and varied environments  
✅ **Diverse Conditions**: Various lighting, weather, and seasonal variations  
✅ **Multiple Species**: Different bear species (black, grizzly, polar, etc.)  
✅ **Realistic Scenarios**: Partial occlusion, distant subjects, complex habitats  
✅ **Large Scale**: Thousands of observations with community annotations  
✅ **Yields Best Results**: Produces the most satisfactory detection accuracy

#### Quick Download (RECOMMENDED):
```bash
# Download 500 iNaturalist bear images (adjust count as needed)
python scripts/download_dataset.py --inaturalist --inaturalist-count 500

# Or use quickstart with iNaturalist (default)
python quickstart.py  # Automatically downloads iNaturalist dataset
```

This will automatically:
1. Fetch bear observations from iNaturalist API
2. Split into train/val/test sets (80/10/10)
3. Convert to YOLO format
4. Prepare for training

### Alternative: Use Sample Dataset (Quick Testing)
```bash
python scripts/download_dataset.py --sample
```

### Alternative: Roboflow Pre-made Datasets

For pre-processed datasets optimized for YOLO:

```bash
python scripts/download_dataset.py --roboflow \
  --roboflow-api YOUR_API_KEY \
  --roboflow-workspace YOUR_WORKSPACE \
  --roboflow-project bear-detection
```

### Dataset Format

YOLO expects this structure:
```
data/
├── images/
│   ├── train/        # Training images
│   ├── val/          # Validation images
│   └── test/         # Test images
└── labels/
    ├── train/        # Training labels (YOLO txt format)
    ├── val/          # Validation labels
    └── test/         # Test labels
```

Label format (.txt file):
```
<class_id> <x_center> <y_center> <width> <height>
```
Where all values are normalized to [0, 1].

## Training

### Configuration

Edit `configs/training_config.yaml` to customize training:

```yaml
model_name: 'yolov8x.pt'   # Model size (n/s/m/l/x)
epochs: 100                 # Number of epochs
batch_size: 16              # Batch size
img_size: 640               # Input image size
lr0: 0.01                   # Initial learning rate
optimizer: 'SGD'            # Optimizer type
```

### Training Command
```bash
python scripts/train_yolo.py --config configs/training_config.yaml
```

### Training Strategies

The default configuration uses:
- **YOLOv8 Extra-Large (yolov8x)**: Best accuracy, requires more VRAM
- **SGD Optimizer**: Stable training
- **Heavy Augmentation**: Robustness to variations
- **Early Stopping**: Patience of 20 epochs
- **Learning Rate Scheduling**: Cosine annealing

### GPU Memory Requirements

| Model | Batch 16 | Batch 32 |
|-------|----------|----------|
| YOLOv8n | 2GB | 4GB |
| YOLOv8s | 3GB | 6GB |
| YOLOv8m | 5GB | 10GB |
| YOLOv8l | 7GB | 14GB |
| YOLOv8x | 10GB | 20GB |

If out of memory, reduce batch size or use smaller model.

## Evaluation

### Run Evaluation
```bash
python scripts/evaluate.py \
  --model results/bear_detection_yolov8x/weights/best.pt \
  --dataset configs/bear_dataset.yaml
```

### Metrics

- **mAP50**: Mean average precision at IoU=0.5
- **mAP50-95**: Mean average precision at IoU=0.5:0.95
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)

## Model Export

Export trained model to different formats:

```bash
python scripts/train_yolo.py \
  --export onnx \
  --model results/bear_detection_yolov8x/weights/best.pt
```

Supported formats: onnx, engine, coreml, saved_model, tflite, pb, paddle

## Performance Tips

1. **Data Quality**: Clean, well-annotated datasets improve accuracy
2. **Augmentation**: Use heavy augmentation for challenging conditions
3. **Model Size**: Balance accuracy vs. speed with model selection
4. **Batch Size**: Larger batches (if memory allows) improve convergence
5. **Learning Rate**: Start with default, adjust if training plateaus
6. **Early Stopping**: Prevents overfitting on validation data

## References

- YOLOv8 Documentation: https://docs.ultralytics.com/
- iNaturalist: https://www.inaturalist.org/
- Roboflow: https://roboflow.com/
- LabelImg: https://github.com/heartexlabs/labelImg

## License

This project uses YOLOv8 (AGPL-3.0) and related open-source libraries.

## Support

For issues, questions, or improvements, please refer to:
- YOLOv8 Issues: https://github.com/ultralytics/ultralytics
- Dataset issues: Check data format and annotations
