import os
import glob
import cv2
import time
import uuid
import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION
# ==============================================================================
V2_MODEL_PATH = "edgevision_v2/models/ppe_v3_hn_best.pt"
INPUT_DIR = "../docs/positive_test"
OUTPUT_DIR = "outputs/phase3e_positive_validation"
EVIDENCE_DIR = os.path.join(OUTPUT_DIR, "evidence")

HELMET_CONF = 0.65
VEST_CONF = 0.45
UNIFIED_CLASSES = ["person", "helmet", "no_helmet", "vest", "boots", "harness"]
ZONE_RULES = {"construction": {"required": ["helmet", "vest"]}}

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
def main():
    if not os.path.exists(INPUT_DIR) or not os.listdir(INPUT_DIR):
        print(f"\nERROR: The positive test directory '{INPUT_DIR}' does not exist or is empty.")
        print("Please place genuine positive PPE footage (e.g., .mp4, .jpg) in this directory.")
        print("Do NOT silently substitute the previous negative test video. We must validate genuine PPE wearers.")
        return
        
    media_files = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.jpg", "*.jpeg", "*.png"]:
        media_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        
    if not media_files:
        print(f"No valid media files found in {INPUT_DIR}.")
        return

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    
    print("Loading models...")
    ppe_model = YOLO(V2_MODEL_PATH).to('cuda:0')
    person_model = YOLO("yolov8n.pt").to('cuda:0')
    
    # Process the first valid media file found
    input_media = media_files[0]
    is_video = input_media.lower().endswith(('.mp4', '.avi', '.mov'))
    
    cap = cv2.VideoCapture(input_media) if is_video else None
    
    writer = None
    if is_video:
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(os.path.join(OUTPUT_DIR, "annotated_output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    validator = TemporalValidator(fps=25)
    
    frames_processed = 0
    frame_times = []
    log_rows = []
    events_log = []
    person_stats = defaultdict(lambda: {"frames": 0, "helmet": 0, "vest": 0, "violations": 0})
    
    print(f"\nProcessing {input_media}...")
    
    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret: break
        else:
            if frames_processed >= len(media_files): break
            frame = cv2.imread(media_files[frames_processed])
            if frame is None:
                frames_processed += 1
                continue
                
        frames_processed += 1
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        p_res = person_model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", conf=0.25, verbose=False, imgsz=512)[0]
        p_boxes = p_res.boxes.xyxy.cpu().numpy().tolist() if p_res.boxes.id is not None else []
        p_ids = p_res.boxes.id.int().cpu().tolist() if p_res.boxes.id is not None else []
        
        ppe_res = ppe_model(frame, conf=0.25, verbose=False)[0]
        ppe_boxes_filtered, ppe_classes_filtered, ppe_confs_filtered = [], [], []
        
        for box, cls, conf in zip(ppe_res.boxes.xyxy.cpu().numpy(), ppe_res.boxes.cls.cpu().numpy(), ppe_res.boxes.conf.cpu().numpy()):
            cname = UNIFIED_CLASSES[int(cls)]
            if cname == "helmet" and conf < HELMET_CONF: continue
            if cname == "vest" and conf < VEST_CONF: continue
            
            ppe_boxes_filtered.append(box)
            ppe_classes_filtered.append(cname)
            ppe_confs_filtered.append(conf)
            
        assignments = assoc_spatial_ioa(p_boxes, p_ids, ppe_boxes_filtered, ppe_classes_filtered)
        
        for pbox, pid in zip(p_boxes, p_ids):
            worn = set(assignments.get(pid, []))
            has_helmet = "helmet" in worn
            has_vest = "vest" in worn
            
            missing = [item for item in ZONE_RULES["construction"]["required"] if item not in worn]
            is_violation = len(missing) > 0
            
            person_stats[pid]["frames"] += 1
            if has_helmet: person_stats[pid]["helmet"] += 1
            if has_vest: person_stats[pid]["vest"] += 1
            if is_violation: person_stats[pid]["violations"] += 1
            
            log_rows.append({
                "person_id": pid,
                "helmet_detected": has_helmet,
                "vest_detected": has_vest,
                "missing_ppe": ",".join(missing),
                "violation": is_violation,
                "confidence": float(max(ppe_confs_filtered)) if ppe_confs_filtered else 0.5,
                "frame_number": frames_processed,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            })
            
            if validator.update(pid, is_violation):
                event_id = str(uuid.uuid4())[:8]
                rel_ev_path = f"evidence/{event_id}.jpg"
                full_ev_path = os.path.join(OUTPUT_DIR, rel_ev_path)
                
                x1, y1, x2, y2 = map(int, pbox)
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size > 0:
                    cv2.imwrite(full_ev_path, crop)
                
                events_log.append({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "event_id": event_id,
                    "person_id": pid,
                    "missing_ppe": ",".join(missing),
                    "evidence_path": rel_ev_path
                })
            
            x1, y1, x2, y2 = map(int, pbox)
            color = (0, 0, 255) if is_violation else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{pid} H:{int(has_helmet)} V:{int(has_vest)}", (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        torch.cuda.synchronize()
        frame_times.append(time.perf_counter() - t0)
        
        if writer:
            writer.write(frame)
        elif not is_video:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"annotated_{os.path.basename(media_files[frames_processed-1])}"), frame)

    if cap: cap.release()
    if writer: writer.release()
    
    # ---------------------------------------------------------
    # SUMMARIZATION
    # ---------------------------------------------------------
    pd.DataFrame(events_log).to_csv(os.path.join(OUTPUT_DIR, "events.csv"), index=False)
    
    person_summary = []
    people_with_helmet = 0
    people_with_vest = 0
    people_with_both = 0
    
    # Observational Compliant Logic (heuristic: person seen with PPE heavily)
    compliant_people = [] 
    
    for pid, stats in person_stats.items():
        frames = stats["frames"]
        h_rate = stats["helmet"] / frames if frames else 0
        v_rate = stats["vest"] / frames if frames else 0
        v_viol = stats["violations"] / frames if frames else 0
        
        person_summary.append({
            "person_id": pid,
            "frames_seen": frames,
            "helmet_frames": stats["helmet"],
            "vest_frames": stats["vest"],
            "violation_frames": stats["violations"],
            "helmet_detection_rate": round(h_rate, 4),
            "vest_detection_rate": round(v_rate, 4),
            "violation_rate": round(v_viol, 4)
        })
        
        if stats["helmet"] > 0: people_with_helmet += 1
        if stats["vest"] > 0: people_with_vest += 1
        if stats["helmet"] > 0 and stats["vest"] > 0: people_with_both += 1
        
        if h_rate >= 0.5 and v_rate >= 0.5:
            compliant_people.append(pid)
            
    pd.DataFrame(person_summary).to_csv(os.path.join(OUTPUT_DIR, "person_summary.csv"), index=False)
    
    avg_fps = frames_processed / sum(frame_times) if frame_times else 0
    warm_times = frame_times[5:] if len(frame_times) > 10 else frame_times
    warm_fps = len(warm_times) / sum(warm_times) if warm_times else 0
    
    fps_list = [1/t for t in frame_times if t > 0]
    min_fps = min(fps_list) if fps_list else 0
    max_fps = max(fps_list) if fps_list else 0
    p95_latency = np.percentile(frame_times, 95) * 1000 if frame_times else 0
    
    false_violation_candidates = sum(1 for evt in events_log if evt["person_id"] in compliant_people)
    total_helmet_dets = sum(s["helmet"] for s in person_stats.values())
    total_vest_dets = sum(s["vest"] for s in person_stats.values())
    
    h_rate_comp, v_rate_comp, fv_rate_comp = 0, 0, 0
    if compliant_people:
        h_sum = sum(person_stats[p]["helmet"] for p in compliant_people)
        v_sum = sum(person_stats[p]["vest"] for p in compliant_people)
        f_sum = sum(person_stats[p]["frames"] for p in compliant_people)
        viol_sum = sum(person_stats[p]["violations"] for p in compliant_people)
        
        if f_sum > 0:
            h_rate_comp = h_sum / f_sum
            v_rate_comp = v_sum / f_sum
            fv_rate_comp = viol_sum / f_sum
            
    summary = {
        "total_frames": frames_processed,
        "total_person_ids": len(person_stats),
        "total_helmet_detections": total_helmet_dets,
        "total_vest_detections": total_vest_dets,
        "total_confirmed_violations": len(events_log),
        "average_fps": round(avg_fps, 2),
        "min_fps": round(min_fps, 2),
        "max_fps": round(max_fps, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "observational_metrics": {
            "helmet_detection_rate_for_compliant_people": round(h_rate_comp, 4),
            "vest_detection_rate_for_compliant_people": round(v_rate_comp, 4),
            "false_violation_rate_for_compliant_people": round(fv_rate_comp, 4)
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\n=== PHASE 3E POSITIVE PPE VALIDATION ===")
    print(f"Total frames: {frames_processed}")
    print(f"Unique people: {len(person_stats)}")
    print(f"People with helmet: {people_with_helmet}")
    print(f"People with vest: {people_with_vest}")
    print(f"People with both: {people_with_both}")
    print(f"Confirmed violations: {len(events_log)}")
    print(f"False violation candidates: {false_violation_candidates}")
    print(f"\nAverage FPS: {summary['average_fps']:.2f}")
    print(f"Warm FPS: {warm_fps:.2f}")
    print(f"Minimum FPS: {summary['min_fps']:.2f}")
    print(f"P95 latency: {summary['p95_latency_ms']:.2f} ms")
    print(f"\n[Observational] Helmet detection rate: {summary['observational_metrics']['helmet_detection_rate_for_compliant_people']:.4f}")
    print(f"[Observational] Vest detection rate: {summary['observational_metrics']['vest_detection_rate_for_compliant_people']:.4f}")
    
    print("\n=== VALIDATION STATUS ===")
    if len(compliant_people) == 0 and people_with_both == 0:
        print("Insufficient positive PPE coverage for a complete end-to-end validation.")
        print("POSITIVE PPE VALIDATION: REVIEW REQUIRED")
    else:
        unsupported = sum(1 for e in events_log if any(p not in ["helmet", "vest"] for p in str(e["missing_ppe"]).split(",")))
        
        if false_violation_candidates == 0 and avg_fps >= 12 and unsupported == 0:
            print("POSITIVE PPE VALIDATION: PASS")
        else:
            print("POSITIVE PPE VALIDATION: REVIEW REQUIRED")

if __name__ == '__main__':
    main()
