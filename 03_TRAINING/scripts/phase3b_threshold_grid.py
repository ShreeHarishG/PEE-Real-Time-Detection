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

def run_pipeline(helmet_conf, vest_conf, person_model, ppe_model):
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
        
        ppe_res = ppe_model(frame, conf=0.25, verbose=False)[0]
        ppe_boxes_filtered, ppe_classes_filtered = [], []
        
        for box, cls, conf in zip(ppe_res.boxes.xyxy.cpu().numpy(), ppe_res.boxes.cls.cpu().numpy(), ppe_res.boxes.conf.cpu().numpy()):
            cname = UNIFIED_CLASSES[int(cls)]
            if cname == "helmet" and conf < helmet_conf: continue
            if cname == "vest" and conf < vest_conf: continue
            
            ppe_boxes_filtered.append(box)
            ppe_classes_filtered.append(cname)
            if cname == "helmet": h_det += 1
            if cname == "vest": v_det += 1
            
        assignments = assoc_spatial_ioa(p_boxes, p_ids, ppe_boxes_filtered, ppe_classes_filtered)
        
        for pbox, pid in zip(p_boxes, p_ids):
            worn = set(assignments.get(pid, []))
            missing = [item for item in ZONE_RULES["construction"]["required"] if item not in worn]
            if validator.update(pid, len(missing) > 0):
                violations += 1
                
    cap.release()
    fps = frames / (time.time() - t0)
    return h_det, v_det, violations, fps


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    os.makedirs("outputs", exist_ok=True)

    print("Loading models...")
    ppe_model = YOLO(V2_MODEL_PATH)
    ppe_model.to('cuda:0')
    person_model = YOLO("yolov8n.pt")
    person_model.to('cuda:0')

    helmet_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    vest_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    print("\n[1/3] Caching validation metrics for unique thresholds...")
    helmet_metrics = {}
    for ht in helmet_thresholds:
        print(f"Evaluating validation set for helmet conf={ht}...")
        metrics = ppe_model.val(data=VAL_DATA_YAML, split='val', imgsz=512, batch=16, conf=ht, device='cuda:0', verbose=False)
        names_inv = {v: k for k, v in metrics.names.items()}
        cls_idx = list(metrics.ap_class_index).index(names_inv['helmet'])
        helmet_metrics[ht] = {'p': round(metrics.box.p[cls_idx], 4), 'r': round(metrics.box.r[cls_idx], 4)}

    vest_metrics = {}
    for vt in vest_thresholds:
        print(f"Evaluating validation set for vest conf={vt}...")
        metrics = ppe_model.val(data=VAL_DATA_YAML, split='val', imgsz=512, batch=16, conf=vt, device='cuda:0', verbose=False)
        names_inv = {v: k for k, v in metrics.names.items()}
        cls_idx = list(metrics.ap_class_index).index(names_inv['vest'])
        vest_metrics[vt] = {'p': round(metrics.box.p[cls_idx], 4), 'r': round(metrics.box.r[cls_idx], 4)}

    print("\n[2/3] Extracting Hard Negative confidences...")
    hn_files = glob.glob(f"{HN_DIR}/hn_test_vid_frame_*.jpg")
    hn_helmet_confs = []
    hn_vest_confs = []
    for f in hn_files:
        res = ppe_model(f, verbose=False, imgsz=512)[0]
        boxes = res.boxes
        for cls, conf in zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
            if cls == 1: hn_helmet_confs.append(conf)
            elif cls == 3: hn_vest_confs.append(conf)

    print("\n[3/3] Running Grid Pipeline (30 combinations)...")
    results = []
    total_runs = len(helmet_thresholds) * len(vest_thresholds)
    current_run = 0

    for ht in helmet_thresholds:
        for vt in vest_thresholds:
            current_run += 1
            print(f"Testing Grid {current_run}/{total_runs}: Helmet={ht:.2f}, Vest={vt:.2f}...")
            
            # Hard negatives
            hn_h = sum(1 for c in hn_helmet_confs if c >= ht)
            hn_v = sum(1 for c in hn_vest_confs if c >= vt)
            
            # Validation stats
            val_hp, val_hr = helmet_metrics[ht]['p'], helmet_metrics[ht]['r']
            val_vp, val_vr = vest_metrics[vt]['p'], vest_metrics[vt]['r']
            
            # Video performance
            h_det, v_det, viol, fps = run_pipeline(ht, vt, person_model, ppe_model)
            
            results.append({
                "Helmet Conf": ht,
                "Vest Conf": vt,
                "Helmet Recall": val_hr,
                "Helmet Precision": val_hp,
                "Vest Recall": val_vr,
                "Vest Precision": val_vp,
                "Helmet HN FPs": hn_h,
                "Vest HN FPs": hn_v,
                "Video Helmet Det": h_det,
                "Video Vest Det": v_det,
                "Confirmed Violations": viol,
                "FPS": round(fps, 2)
            })

    df = pd.DataFrame(results)
    df.to_csv("outputs/phase3b_threshold_grid.csv", index=False)
    print("\nPhase 3B Complete! Results saved to outputs/phase3b_threshold_grid.csv")
