# Training Reproducibility Guide

## Dataset Structure
The dataset follows standard YOLO format:
```
datasets/
  construction-ppe/
    images/
      train/
      val/
    labels/
      train/
      val/
```

### Classes
```yaml
names:
  0: person
  1: helmet
  2: no_helmet
  3: vest
  4: boots
  5: harness
```
*(Note: Person is ignored by the PPE model as we use a dedicated YOLOv8n model for the person crop/bounding box).*

## V3-HN Architecture (Current Production Model)
The EdgeVision `V3-HN` model is fine-tuned from `yolov8n.pt` at `imgsz=512`. 

### The Hard-Negative Strategy
The primary innovation of `V3-HN` over `V2` is the hard-negative mining strategy. `V2` suffered from 154 False Positives (helmets) and 10 False Positives (vests) in real-world validation caused by yellow machinery and reflective tape on background assets.

To fix this, 73 "hard negative" frames (containing the visually confusing background elements but ZERO actual workers or PPE) were added to the training set with completely empty label files. The model was trained for 10 epochs. This successfully forced the YOLO architecture to unlearn the background noise, achieving exactly 0 FPs in real-world testing.

## Retraining (Not Recommended)
The V3-HN model is considered FROZEN. Our autonomous ML lab (Exp01 through Exp07) conclusively proved that attempting to blindly class-balance vests or train for longer epochs (20+) causes the model to "forget" the hard-negative constraints and the False Positives return. 

If you must retrain, use the exact parameters in `scripts/exp1_train.py` on the exact dataset blend.

```bash
python scripts/exp1_train.py
```
