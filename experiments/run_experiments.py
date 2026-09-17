import cv2
import yaml
import time
import os
import numpy as np
from ultralytics import YOLO

os.makedirs("fp_crops/helmet", exist_ok=True)
os.makedirs("fp_crops/vest", exist_ok=True)

print("Loading models...")
person_model = YOLO("yolov8n.pt")
person_model.to('cuda')
ppe_model = YOLO("edgevision_v2/models/ppe_best.pt")
ppe_model.to('cuda')

INPUT_VIDEO = "../docs/test.mp4"

# Part A & B: Tracking experiments
configs = [
    {"track_buffer": 30, "match_thresh": 0.8},
    {"track_buffer": 60, "match_thresh": 0.8},
    {"track_buffer": 90, "match_thresh": 0.8},
    {"track_buffer": 120, "match_thresh": 0.8},
    {"track_buffer": 90, "match_thresh": 0.9},
]

print("=== PART A: TRACKING EXPERIMENTS ===")
best_config = None
min_ids = float('inf')

for cfg in configs:
    cfg_path = f"custom_track_{cfg['track_buffer']}_{cfg['match_thresh']}.yaml"
    with open(cfg_path, 'w') as f:
        yaml.dump({
            "tracker_type": "bytetrack",
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.25,
            "track_buffer": cfg['track_buffer'],
            "match_thresh": cfg['match_thresh'],
            "fuse_score": True
        }, f)
        
    cap = cv2.VideoCapture(INPUT_VIDEO)
    unique_ids = set()
    frames = 0
    t0 = time.time()
    
    # Store history to see occlusions for the last run
    track_history = {} 
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames += 1
        
        p_res = person_model.track(frame, persist=True, classes=[0], tracker=cfg_path, conf=0.25, verbose=False, imgsz=512)[0]
        if p_res.boxes.id is not None:
            ids = p_res.boxes.id.int().cpu().tolist()
            unique_ids.update(ids)
            for pid in ids:
                if pid not in track_history:
                    track_history[pid] = []
                track_history[pid].append(frames)
                
    cap.release()
    fps = frames / (time.time() - t0)
    print(f"Config buf={cfg['track_buffer']} match={cfg['match_thresh']} | Unique IDs: {len(unique_ids)} | FPS: {fps:.2f}")
    if len(unique_ids) < min_ids:
        min_ids = len(unique_ids)
        best_config = cfg

print("\n=== PART B: OCCLUSION ANALYSIS (buf=120) ===")
# Let's see if we can find tracks that had gaps
for pid, history in track_history.items():
    if len(history) > 1:
        gaps = [history[i] - history[i-1] for i in range(1, len(history))]
        max_gap = max(gaps) if gaps else 0
        if max_gap > 5:
            print(f"Track {pid} was occluded for {max_gap} frames (disappeared at {history[gaps.index(max_gap)]}, reappeared at {history[gaps.index(max_gap)+1]}).")

print("\n=== PART C: PPE FALSE-POSITIVE ANALYSIS ===")
cap = cv2.VideoCapture(INPUT_VIDEO)
frame_idx = 0
fp_helmet = []
fp_vest = []

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1
    
    ppe_res = ppe_model(frame, conf=0.25, verbose=False)[0]
    boxes = ppe_res.boxes.xyxy.cpu().numpy()
    classes = ppe_res.boxes.cls.cpu().numpy()
    confs = ppe_res.boxes.conf.cpu().numpy()
    
    for box, cls, conf in zip(boxes, classes, confs):
        x1, y1, x2, y2 = map(int, box)
        cname = "helmet" if cls == 1 else "no_helmet" if cls == 2 else "vest" if cls == 3 else "other"
        if cname in ["helmet", "vest"]:
            crop = frame[max(0,y1):y2, max(0,x1):x2]
            if crop.size > 0:
                fp_path = f"fp_crops/{cname}/{frame_idx}_{conf:.2f}.jpg"
                cv2.imwrite(fp_path, crop)
                if cname == "helmet":
                    fp_helmet.append(conf)
                else:
                    fp_vest.append(conf)

cap.release()
print(f"Helmet FPs: {len(fp_helmet)} (Conf range: {min(fp_helmet) if fp_helmet else 0:.2f} - {max(fp_helmet) if fp_helmet else 0:.2f})")
print(f"Vest FPs: {len(fp_vest)} (Conf range: {min(fp_vest) if fp_vest else 0:.2f} - {max(fp_vest) if fp_vest else 0:.2f})")

print("\n=== PART D: ASSOCIATION ANALYSIS ===")
def iou_association(p_box, ppe_box):
    # intersection over PPE area
    px1, py1, px2, py2 = p_box
    hx1, hy1, hx2, hy2 = ppe_box
    ix1, iy1 = max(px1, hx1), max(py1, hy1)
    ix2, iy2 = min(px2, hx2), min(py2, hy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ppe_area = max(0, hx2 - hx1) * max(0, hy2 - hy1)
    return inter / ppe_area if ppe_area > 0 else 0

def spatial_association(p_box, ppe_box, ppe_cls):
    px1, py1, px2, py2 = p_box
    hx1, hy1, hx2, hy2 = ppe_box
    
    # Center of PPE
    hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
    
    # Must be inside person box horizontally
    if not (px1 <= hcx <= px2):
        return False
        
    ph = py2 - py1
    if ppe_cls == 1: # helmet
        # top 30% of person box
        return (py1 <= hcy <= py1 + 0.3 * ph)
    elif ppe_cls == 3: # vest
        # middle 60% of person box (from 20% to 80%)
        return (py1 + 0.2 * ph <= hcy <= py1 + 0.8 * ph)
    return True

print("Spatial association functions ready to test in logic.")
