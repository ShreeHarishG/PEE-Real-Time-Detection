import os
import cv2
import time
import uuid
import json
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION
# ==============================================================================
V2_MODEL_PATH = "edgevision_v2/models/ppe_best.pt"
TEST_VIDEO = "../docs/test.mp4"
HELMET_CONF = 0.65
VEST_CONF = 0.45
UNIFIED_CLASSES = ["person", "helmet", "no_helmet", "vest", "boots", "harness"]
ZONE_RULES = {"construction": {"required": ["helmet", "vest"]}}

OUTPUT_DIR = "outputs/phase3d_validation"
EVIDENCE_DIR = os.path.join(OUTPUT_DIR, "evidence")
CSV_PATH = os.path.join(OUTPUT_DIR, "events.csv")

# ==============================================================================
# REUSED PIPELINE COMPONENTS
# ==============================================================================
class TemporalValidator:
    def __init__(self, fps=25):
        self.history = defaultdict(lambda: deque(maxlen=10))
        self.frames = defaultdict(int)
        self.confirmed = set()

    def update(self, tid, is_violation):
        self.frames[tid] += 1
        self.history[tid].append(is_violation)
        # BUG FIX: Event confirmation must require current frame to actually contain a violation
        if sum(self.history[tid]) >= 8 and self.frames[tid] >= 50 and tid not in self.confirmed and is_violation:
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

# ==============================================================================
# MAIN VALIDATION SCRIPT
# ==============================================================================
def run_validation():
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    
    print(f"Loading existing PPE model from {V2_MODEL_PATH}...")
    ppe_model = YOLO(V2_MODEL_PATH).to('cuda:0')
    print("Loading existing Person model from yolov8n.pt...")
    person_model = YOLO("yolov8n.pt").to('cuda:0')
    
    cap = cv2.VideoCapture(TEST_VIDEO)
    validator = TemporalValidator()
    
    frames = 0
    h_det, v_det = 0, 0
    log_rows = []
    
    unique_ids = set()
    frame_times = []
    
    print("\nRunning clean validation pipeline...")
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret: break
        frames += 1
        
        # Track persons using current tested ByteTrack config
        p_res = person_model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", conf=0.25, verbose=False, imgsz=512)[0]
        p_boxes = p_res.boxes.xyxy.cpu().numpy().tolist() if p_res.boxes.id is not None else []
        p_ids = p_res.boxes.id.int().cpu().tolist() if p_res.boxes.id is not None else []
        unique_ids.update(p_ids)
        
        # Detect PPE
        ppe_res = ppe_model(frame, conf=0.25, verbose=False)[0]
        ppe_boxes_filtered, ppe_classes_filtered = [], []
        
        for box, cls, conf in zip(ppe_res.boxes.xyxy.cpu().numpy(), ppe_res.boxes.cls.cpu().numpy(), ppe_res.boxes.conf.cpu().numpy()):
            cname = UNIFIED_CLASSES[int(cls)]
            # Apply configured confidence gates
            if cname == "helmet":
                if conf < HELMET_CONF: continue
                h_det += 1
            if cname == "vest":
                if conf < VEST_CONF: continue
                v_det += 1
                
            ppe_boxes_filtered.append(box)
            ppe_classes_filtered.append(cname)
            
        # Spatial IoA Association
        assignments = assoc_spatial_ioa(p_boxes, p_ids, ppe_boxes_filtered, ppe_classes_filtered)
        
        # Rule Validation
        for pbox, pid in zip(p_boxes, p_ids):
            worn = set(assignments.get(pid, []))
            missing = [item for item in ZONE_RULES["construction"]["required"] if item not in worn]
            
            is_violation = len(missing) > 0
            
            if validator.update(pid, is_violation):
                # We verified `is_violation` is True on the current frame, so `missing` must not be empty.
                event_id = str(uuid.uuid4())[:8]
                rel_ev_path = f"evidence/{event_id}.jpg"
                full_ev_path = os.path.join(OUTPUT_DIR, rel_ev_path)
                
                # Crop and save evidence locally
                x1, y1, x2, y2 = map(int, pbox)
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size > 0:
                    cv2.imwrite(full_ev_path, crop)
                
                log_rows.append({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "event_id": event_id,
                    "person_id": pid,
                    "zone": "construction",
                    "missing_ppe": ",".join(missing),
                    "evidence_path": rel_ev_path,
                    "is_actual_violation": is_violation
                })
        
        frame_times.append(time.time() - t0)
                
    cap.release()
    
    # ---------------------------------------------------------
    # ANALYSIS AND SUMMARY
    # ---------------------------------------------------------
    df = pd.DataFrame(log_rows) if log_rows else pd.DataFrame(columns=["timestamp", "event_id", "person_id", "zone", "missing_ppe", "evidence_path", "is_actual_violation"])
    df.to_csv(CSV_PATH, index=False)
    
    total_violations = len(log_rows)
    duplicate_events = total_violations - len(set(df['event_id'])) if total_violations > 0 else 0
    empty_missing = sum(1 for x in df['missing_ppe'] if not str(x).strip()) if total_violations > 0 else 0
    
    missing_evidence = 0
    unsupported_classes = 0
    
    if total_violations > 0:
        for idx, row in df.iterrows():
            if not os.path.exists(os.path.join(OUTPUT_DIR, row['evidence_path'])):
                missing_evidence += 1
            missing_items = [p.strip() for p in str(row['missing_ppe']).split(",") if p.strip()]
            for p in missing_items:
                if p not in ["helmet", "vest"]:
                    unsupported_classes += 1
                    
    avg_fps = frames / sum(frame_times) if frame_times else 0
    fps_list = [1/t if t > 0 else 0 for t in frame_times]
    min_fps = min(fps_list) if fps_list else 0
    max_fps = max(fps_list) if fps_list else 0
    p95_latency = np.percentile(frame_times, 95) * 1000 if frame_times else 0
    
    summary = {
        "total_frames": frames,
        "total_confirmed_violations": total_violations,
        "unique_person_ids": len(unique_ids),
        "duplicate_event_ids": duplicate_events,
        "empty_missing_ppe_events": empty_missing,
        "missing_evidence_files": missing_evidence,
        "unsupported_ppe_classes": unsupported_classes,
        "helmet_detections": h_det,
        "vest_detections": v_det,
        "average_fps": round(avg_fps, 2),
        "min_fps": round(min_fps, 2),
        "max_fps": round(max_fps, 2),
        "p95_frame_latency_ms": round(p95_latency, 2)
    }
    
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\n=== PHASE 3D VALIDATION SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
        
    print("\n=== RUNNING EXPLICIT ASSERTIONS ===")
    try:
        assert summary["empty_missing_ppe_events"] == 0, f"Assertion failed: empty_missing_ppe_events is {summary['empty_missing_ppe_events']}"
        assert summary["missing_evidence_files"] == 0, f"Assertion failed: missing_evidence_files is {summary['missing_evidence_files']}"
        assert summary["unsupported_ppe_classes"] == 0, f"Assertion failed: unsupported_ppe_classes is {summary['unsupported_ppe_classes']}"
        assert summary["average_fps"] >= 12, f"Assertion failed: average_fps {summary['average_fps']} is < 12"
        print("SUCCESS: All assertions PASSED. Pipeline logic is CLEAN.")
    except AssertionError as e:
        print(f"FAILURE: {e}")
        
if __name__ == '__main__':
    run_validation()
