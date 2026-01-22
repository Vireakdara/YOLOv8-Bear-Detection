# Dataset Structure

This project uses a **consolidated dataset folder** for all training data.

## Folder Layout

```
dataset/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
└── labels/
    ├── train/          # Training labels (YOLO format .txt)
    ├── val/            # Validation labels
    └── test/           # Test labels
```

## Getting Started

### Option 1: Use Roboflow Dataset (Recommended)

1. Download the bear dataset from Roboflow:
   https://universe.roboflow.com/ds/c6gkywSzAj?key=ymftU7vee3

2. Extract the downloaded ZIP file to get `train/`, `valid/`, and `test/` folders

3. Run the setup script to organize into `dataset/`:
   ```bash
   python quickstart.py --setup-only
   ```

### Option 2: Create Sample Dataset

For quick testing without downloading:
```bash
python quickstart.py --sample
```

This creates synthetic bear-like images in the `dataset/` folder.

## Directory Structure After Setup

```
YOLOv26-Bear/
├── dataset/                 ← All images and labels consolidated here
│   ├── images/
│   │   ├── train/          (600+ images)
│   │   ├── val/            (100+ images)
│   │   └── test/           (100+ images)
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── configs/
│   ├── bear_dataset.yaml   (points to ./dataset)
│   └── training_config.yaml
├── scripts/
├── results/
└── quickstart.py
```

## Configuration

The dataset path is defined in `configs/bear_dataset.yaml`:

```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test
```

This single folder approach makes it:
- ✓ Easier to manage
- ✓ Clearer to understand the structure
- ✓ Simpler to exclude from git
- ✓ Portable across different machines

## Starting Training

```bash
python quickstart.py --epochs 50
```

The script will:
1. Auto-detect extracted Roboflow data if present
2. Organize it into `dataset/` folder
3. Start training on the consolidated dataset

## Notes

- The `dataset/` folder is excluded from git (see `.gitignore`)
- Dataset is not included in the repository
- Each machine running training needs its own dataset
- Use `python quickstart.py --setup-only` to just organize data without training
