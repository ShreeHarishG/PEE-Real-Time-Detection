import os
import glob
import cv2
import time
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_ROOT = "."
V2_MODEL_PATH = "edgevision_v2/models/ppe_best.pt"
HN_DIR = "edgevision_v2/datasets/merged_v21/images/train"
VAL_DATA_YAML = "edgevision_v2/datasets/merged/data.yaml"
TEST_VIDEO = "../docs/test.mp4"
UNIFIED_CLASSES = ["person", "helmet", "no_helmet", "vest", "boots", "harness"]
ZONE_RULES = {"construction": {"required": ["helmet", "vest"]}}

# ==============================================================================
# PIPELINE DEFINITIONS
# ==============================================================================
class TemporalValidator:
    def __init__(self, fps=25):
        self.history = defaultdict(lambda: deque(maxlen=10))
        self.frames = defaultdict(int)
        self.confirmed = set()

    def update(self, tid, has_viol):
        self.frames[tid] += 1
        self.history[tid].append(has_viol)
        if sum(self.history[tid]) >= 8 and self.frames[tid] >= 50 and tid not in self.confirmed and has_viol:
            self.confirmed.add(tid)
            return True
        return False

def box_center(box):
    return ((box[0] + box[2])/2, (box[1] + box[3])/2)

def iou_area(p_box, ppe_box):
    px1, py1, px2, py2 = p_box
    hx1, hy1, hx2, hy2 = ppe_box
    ix1, iy1 = max(px1, hx1), max(py1, hy1)
    ix2, iy2 = min(px2, hx2), min(py2, hy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    return (iw * ih) / max(1, (hx2 - hx1) * (hy2 - hy1))

def assoc_nearest_center(person_boxes, person_ids, ppe_boxes, ppe_classes):
    assignments = defaultdict(list)
    for p_box, p_class in zip(ppe_boxes, ppe_classes):
        cx, cy = box_center(p_box)
        best_id, best_dist = None, float("inf")
        for pbox, pid in zip(person_boxes, person_ids):
            x1, y1, x2, y2 = pbox
            if x1 - 15 <= cx <= x2 + 15 and y1 - 15 <= cy <= y2 + 15:
                pcx, pcy = box_center(pbox)
                dist = (pcx - cx)**2 + (pcy - cy)**2
                if dist < best_dist:
                    best_dist, best_id = dist, pid
        if best_id is not None:
            assignments[best_id].append(p_class)
    return assignments

def assoc_spatial_ioa(person_boxes, person_ids, ppe_boxes, ppe_classes):
    assignments = defaultdict(list)
    for p_box, p_class in zip(ppe_boxes, ppe_classes):
        hcx, hcy = box_center(p_box)
        best_id, best_ioa = None, -1
        
        for pbox, pid in zip(person_boxes, person_ids):
            px1, py1, px2, py2 = pbox
            ph = py2 - py1
            
            # Spatial constraints
            if not (px1 <= hcx <= px2): continue
            if p_class == "helmet" and not (py1 <= hcy <= py1 + 0.3 * ph): continue
            if p_class == "vest" and not (py1 + 0.2 * ph <= hcy <= py1 + 0.8 * ph): continue
            
            ioa = iou_area(pbox, p_box)
            if ioa > best_ioa:
                best_ioa, best_id = ioa, pid
                
        if best_id is not None:
            assignments[best_id].append(p_class)
    return assignments

def run_pipeline(name, helmet_conf, vest_conf, assoc_fn, person_model, ppe_model):
    print(f"\n--- Running Video Pipeline [{name}] ---")
    cap = cv2.VideoCapture(TEST_VIDEO)
    validator = TemporalValidator()
    
    frames, violations, h_det, v_det = 0, 0, 0, 0
    t0 = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames += 1
        
        p_res = person_model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", conf=0.25, verbose=False, imgsz=512)[0]
        p_boxes = p_res.boxes.xyxy.cpu().numpy().tolist() if p_res.boxes.id is not None else []
        p_ids = p_res.boxes.id.int().cpu().tolist() if p_res.boxes.id is not None else []
        
        ppe_res = ppe_model(frame, conf=0.25, verbose=False)[0] # base conf, we filter below
        ppe_boxes_filtered, ppe_classes_filtered = [], []
        
        for box, cls, conf in zip(ppe_res.boxes.xyxy.cpu().numpy(), ppe_res.boxes.cls.cpu().numpy(), ppe_res.boxes.conf.cpu().numpy()):
            cname = UNIFIED_CLASSES[int(cls)]
            if cname == "helmet" and conf < helmet_conf: continue
            if cname == "vest" and conf < vest_conf: continue
            
            ppe_boxes_filtered.append(box)
            ppe_classes_filtered.append(cname)
            if cname == "helmet": h_det += 1
            if cname == "vest": v_det += 1
            
        assignments = assoc_fn(p_boxes, p_ids, ppe_boxes_filtered, ppe_classes_filtered)
        
        for pbox, pid in zip(p_boxes, p_ids):
            worn = set(assignments.get(pid, []))
            missing = [item for item in ZONE_RULES["construction"]["required"] if item not in worn]
            if validator.update(pid, len(missing) > 0):
                violations += 1
                
    cap.release()
    fps = frames / (time.time() - t0)
    print(f"Results for {name}:")
    print(f"Helmet Detections: {h_det} | Vest Detections: {v_det}")
    print(f"Confirmed Violations: {violations}")
    print(f"FPS: {fps:.2f}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # Ensure output directory
    os.makedirs("outputs", exist_ok=True)

    print("Loading models...")
    ppe_model = YOLO(V2_MODEL_PATH)
    ppe_model.to('cuda:0')
    person_model = YOLO("yolov8n.pt")
    person_model.to('cuda:0')

    print("\n=== RUNNING FALSE POSITIVE SWEEP ON HARD NEGATIVES ===")
    # Fix: only select the 73 extracted hard negative frames, not the entire training set
    hn_files = glob.glob(f"{HN_DIR}/hn_test_vid_frame_*.jpg")
    print(f"Found {len(hn_files)} hard-negative images.")

    helmet_confs = []
    vest_confs = []

    for f in hn_files:
        res = ppe_model(f, verbose=False, imgsz=512)[0]
        boxes = res.boxes
        for cls, conf in zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
            if cls == 1: # helmet
                helmet_confs.append(conf)
            elif cls == 3: # vest
                vest_confs.append(conf)

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    fp_results = []

    for t in thresholds:
        h_fp = sum(1 for c in helmet_confs if c >= t)
        v_fp = sum(1 for c in vest_confs if c >= t)
        fp_results.append({"threshold": t, "helmet_fp": h_fp, "vest_fp": v_fp})

    df_fp = pd.DataFrame(fp_results)
    df_fp.to_csv("outputs/fp_sweep_results.csv", index=False)
    print("Saved False Positive sweep to outputs/fp_sweep_results.csv")
    print(df_fp.to_string(index=False))

    print("\n=== EVALUATING GENUINE PPE RECALL ON VALIDATION SET ===")
    val_thresholds = [0.25, 0.50, 0.75, 0.85]
    val_results = []

    for t in val_thresholds:
        print(f"Running validation with conf={t}...")
        metrics = ppe_model.val(data=VAL_DATA_YAML, split='val', imgsz=512, batch=16, conf=t, device='cuda:0', verbose=False)
        
        names = metrics.names
        for i, c in enumerate(metrics.ap_class_index):
            cls_name = names[c]
            if cls_name in ['helmet', 'vest']:
                val_results.append({
                    "threshold": t,
                    "class": cls_name,
                    "precision": round(metrics.box.p[i], 4),
                    "recall": round(metrics.box.r[i], 4),
                    "mAP50": round(metrics.box.ap50[i], 4)
                })

    df_val = pd.DataFrame(val_results)
    df_val.to_csv("outputs/genuine_recall_results.csv", index=False)
    print("Saved Genuine Validation sweep to outputs/genuine_recall_results.csv")
    print(df_val.to_string(index=False))

    HELMET_CALIBRATED_CONF = 0.65
    VEST_CALIBRATED_CONF = 0.45

    # BEFORE: Original 0.25 conf thresholds and Nearest Center association
    run_pipeline("BEFORE (0.25 Conf, Nearest Center)", 0.25, 0.25, assoc_nearest_center, person_model, ppe_model)

    # AFTER: Calibrated conf thresholds and Spatial IoA association
    run_pipeline(f"AFTER ({HELMET_CALIBRATED_CONF} Helmet, {VEST_CALIBRATED_CONF} Vest, Spatial IoA)", 
                 HELMET_CALIBRATED_CONF, VEST_CALIBRATED_CONF, assoc_spatial_ioa, person_model, ppe_model)

    print("\n--- Calibration Complete ---")
    print("Examine outputs/fp_sweep_results.csv and outputs/genuine_recall_results.csv to fine-tune thresholds further if needed.")
