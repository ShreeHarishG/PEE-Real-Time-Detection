# PPE Dataset Structure & Labelling Guide

This document outlines the dataset structure and labelling conventions required to train the YOLOv8 EdgeVision PPE models (`ppe_v3_hn_best.pt`).

## Directory Structure
The dataset must follow the standard YOLO formatting structure:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Classes
The current model supports the following unified classes (ID mapping):
```yaml
names:
  0: person
  1: helmet
  2: no_helmet
  3: vest
  4: boots
  5: harness
```

## Labelling Guidelines
When annotating images (e.g. using CVAT or Roboflow), adhere strictly to these rules:

1. **Persons**: Draw tight bounding boxes around all visible workers (`class 0`).
2. **Helmets**: Draw a box around the hard-hat (`class 1`). If a person is clearly visible but missing a helmet, label the top of their head as `no_helmet` (`class 2`) as a negative example to penalize the model.
3. **Vests**: Draw a box around the high-visibility vest (`class 3`).
4. **Boots**: Draw a box around EACH safety boot (`class 4`).
5. **Harness**: Draw a box encompassing the full body harness straps (`class 5`).

### Difficult Classes (Hooks & Lanyards)
The PRD specifies the need to detect `hooks` and `lanyards` in the future.
- **Hooks**: Due to their small size, images must be captured at high resolution (1080p+). Annotators should zoom in fully to label the hook.
- **Lanyards**: Lanyards often blend into the harness or background. Use polygon annotations if bounding boxes include too much background noise.
- **Hard Negatives**: Include images of loose ropes, extension cords, and yellow machinery. Explicitly DO NOT label these, or label them as a background/negative class so the model learns to differentiate them.

## Data Augmentation
To improve small object detection (boots, hooks), apply the following augmentations before training:
- **Mosaic**: Combines 4 images, forcing the model to learn smaller contexts.
- **Random Crop**: Crops a section of the image containing the objects.
- **Color Jitter**: Adjust brightness/contrast to simulate harsh sunlight and low-light shadows.
