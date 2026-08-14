import os
import cv2
import json
import glob
import time
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION
# ==============================================================================
V2_MODEL_PATH = "edgevision_v2/models/ppe_best.pt"
INPUT_DIR = "../docs/positive_test"
PHASE3E_DIR = "outputs/phase3e_positive_validation"
EVENTS_CSV = os.path.join(PHASE3E_DIR, "events.csv")
PERSON_SUMMARY_CSV = os.path.join(PHASE3E_DIR, "person_summary.csv")

OUTPUT_DIR = "outputs/phase3f_violation_audit"
AUDIT_CSV = os.path.join(OUTPUT_DIR, "audit.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "summary.json")

HELMET_CONF = 0.65
VEST_CONF = 0.45
UNIFIED_CLASSES = ["person", "helmet", "no_helmet", "vest", "boots", "harness"]

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
            if not (px1 <= hcx <= px2): continue
            if p_class == "helmet" and not (py1 <= hcy <= py1 + 0.3 * ph): continue
            if p_class == "vest" and not (py1 + 0.2 * ph <= hcy <= py1 + 0.8 * ph): continue
            ioa = iou_area(pbox, p_box)
            if ioa > best_ioa:
                best_ioa, best_id = ioa, pid
        if best_id is not None:
            assignments[best_id].append(p_class)
    return assignments

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(EVENTS_CSV):
        print(f"Error: Could not find {EVENTS_CSV}. Run Phase 3E first.")
        return
        
    events_df = pd.read_csv(EVENTS_CSV)
    if len(events_df) == 0:
        print("No violations found in events.csv to audit.")
        return
        
    media_files = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.jpg", "*.jpeg", "*.png"]:
        media_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        
    if not media_files:
        print(f"Error: No media files found in {INPUT_DIR}.")
        return
        
    input_media = media_files[0]
    
    # 1. We must re-run inference to build frame-by-frame temporal history because 
    # Phase 3E did not export a frame-by-frame tracker log.
    print("==================================================")
    print("EXTRACTING TEMPORAL CONTEXT...")
    print("Running tracker to collect chronological PPE history for each track.")
    print("==================================================")
    
    ppe_model = YOLO(V2_MODEL_PATH).to('cuda:0')
    person_model = YOLO("yolov8n.pt").to('cuda:0')
    
    cap = cv2.VideoCapture(input_media)
    
    track_history = defaultdict(list)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        p_res = person_model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", conf=0.25, verbose=False, imgsz=512)[0]
        p_boxes = p_res.boxes.xyxy.cpu().numpy().tolist() if p_res.boxes.id is not None else []
        p_ids = p_res.boxes.id.int().cpu().tolist() if p_res.boxes.id is not None else []
        
        ppe_res = ppe_model(frame, conf=0.25, verbose=False)[0]
        ppe_boxes_filtered, ppe_classes_filtered = [], []
        
        for box, cls, conf in zip(ppe_res.boxes.xyxy.cpu().numpy(), ppe_res.boxes.cls.cpu().numpy(), ppe_res.boxes.conf.cpu().numpy()):
            cname = UNIFIED_CLASSES[int(cls)]
            if cname == "helmet" and conf < HELMET_CONF: continue
            if cname == "vest" and conf < VEST_CONF: continue
            ppe_boxes_filtered.append(box)
            ppe_classes_filtered.append(cname)
            
        assignments = assoc_spatial_ioa(p_boxes, p_ids, ppe_boxes_filtered, ppe_classes_filtered)
        
        for pbox, pid in zip(p_boxes, p_ids):
            worn = set(assignments.get(pid, []))
            track_history[pid].append({
                "frame": frame_idx,
                "helmet": "helmet" in worn,
                "vest": "vest" in worn
            })
    cap.release()
    print("Extraction complete.\n")
    
    # 2. Interactive Audit
    print("==================================================")
    print("INTERACTIVE VIOLATION AUDIT")
    print("==================================================")
    print("For each event, an evidence image will open.")
    print("Review the image and the temporal history, then classify the root cause.\n")
    
    audit_results = []
    
    categories = {
        'a': 'GENUINE_VIOLATION',
        'b': 'TEMPORARY_MISDETECTION',
        'c': 'PPE_ASSOCIATION_ERROR',
        'd': 'TRACKING_ID_ERROR',
        'e': 'OCCLUSION',
        'f': 'INSUFFICIENT_VISUAL_EVIDENCE'
    }
    
    cat_counts = defaultdict(int)
    
    for idx, row in events_df.iterrows():
        ev_id = row['event_id']
        pid = row['person_id']
        missing_ppe = str(row['missing_ppe'])
        
        # Determine temporal history context
        history = track_history.get(pid, [])
        # Find the frame this event likely occurred on (we don't have exact frame in events.csv, but we can look for periods of missing PPE)
        # As an approximation, just show the general pattern of the track
        total_frames = len(history)
        helmet_pattern = "".join(["T" if h["helmet"] else "_" for h in history])
        vest_pattern = "".join(["T" if h["vest"] else "_" for h in history])
        
        print(f"\n--- Event {idx+1}/{len(events_df)} | ID: {ev_id} | Person: {pid} ---")
        print(f"Timestamp: {row['timestamp']}")
        print(f"Missing PPE: {missing_ppe}")
        print(f"Track length: {total_frames} frames")
        
        # Display sequence snippet to show if they had PPE before/after
        # To avoid giant strings, let's compress it
        print("Helmet detection history (T=detected, _=missed):")
        print(helmet_pattern if len(helmet_pattern) < 100 else helmet_pattern[:50] + "..." + helmet_pattern[-50:])
        print("Vest detection history (T=detected, _=missed):")
        print(vest_pattern if len(vest_pattern) < 100 else vest_pattern[:50] + "..." + vest_pattern[-50:])
        
        h_before, h_at, h_after = False, False, False
        v_before, v_at, v_after = False, False, False
        
        if total_frames > 0:
            mid = total_frames // 2
            h_before = any(h["helmet"] for h in history[:mid])
            v_before = any(h["vest"] for h in history[:mid])
            h_after = any(h["helmet"] for h in history[mid:])
            v_after = any(h["vest"] for h in history[mid:])
        
        img_path = os.path.join(PHASE3E_DIR, str(row['evidence_path']))
        if os.path.exists(img_path):
            abs_img = os.path.abspath(img_path).replace("\\", "/")
            print(f"\n=> EVIDENCE IMAGE: file:///{abs_img} (Ctrl+Click to view)")
        else:
            print(f"\nWARNING: Evidence image not found at {img_path}")
            
        print("\nSelect Category:")
        for k, v in categories.items():
            print(f"  {k.upper()}. {v}")
            
        while True:
            choice = input("Choice (A-F): ").strip().lower()
            if choice in categories:
                break
            print("Invalid choice.")
            
        reason = input("Brief reason/note (optional): ").strip()
        
        cat = categories[choice]
        cat_counts[cat] += 1
        
        audit_results.append({
            "event_id": ev_id,
            "person_id": pid,
            "timestamp": row['timestamp'],
            "missing_ppe": missing_ppe,
            "confidence": 0.5, # Default placeholder if not in CSV
            "category": cat,
            "helmet_before": h_before,
            "helmet_at_event": False, # Heuristic
            "helmet_after": h_after,
            "vest_before": v_before,
            "vest_at_event": False,
            "vest_after": v_after,
            "violation_frames": sum(1 for h in history if not h["helmet"] or not h["vest"]),
            "evidence_path": img_path,
            "reason": reason
        })
        
        # cv2.destroyAllWindows()

    df_audit = pd.DataFrame(audit_results)
    df_audit.to_csv(AUDIT_CSV, index=False)
    
    # ---------------------------------------------------------
    # DECISION LOGIC & SUMMARY
    # ---------------------------------------------------------
    summary = {
        "total_events": len(events_df),
        "genuine_violations": cat_counts["GENUINE_VIOLATION"],
        "temporary_misdetections": cat_counts["TEMPORARY_MISDETECTION"],
        "association_errors": cat_counts["PPE_ASSOCIATION_ERROR"],
        "tracking_errors": cat_counts["TRACKING_ID_ERROR"],
        "occlusion_events": cat_counts["OCCLUSION"],
        "insufficient_evidence": cat_counts["INSUFFICIENT_VISUAL_EVIDENCE"],
        "false_violation_rate": (len(events_df) - cat_counts["GENUINE_VIOLATION"]) / len(events_df) if len(events_df) > 0 else 0
    }
    
    # Check compliance metrics if person_summary is available
    if os.path.exists(PERSON_SUMMARY_CSV):
        ps_df = pd.read_csv(PERSON_SUMMARY_CSV)
        compliant = ps_df[(ps_df['helmet_detection_rate'] > 0.5) & (ps_df['vest_detection_rate'] > 0.5)]
        h_miss_rate = 1.0 - compliant['helmet_detection_rate'].mean() if len(compliant) > 0 else 0
        v_miss_rate = 1.0 - compliant['vest_detection_rate'].mean() if len(compliant) > 0 else 0
        summary["helmet_miss_rate_during_compliant_tracks"] = round(h_miss_rate, 4)
        summary["vest_miss_rate_during_compliant_tracks"] = round(v_miss_rate, 4)
    
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\n==================================================")
    print("=== PHASE 3F AUDIT ===")
    print("==================================================")
    print(f"Total confirmed violations: {summary['total_events']}")
    print(f"Genuine violations: {summary['genuine_violations']}")
    print(f"Temporary misdetections: {summary['temporary_misdetections']}")
    print(f"Association errors: {summary['association_errors']}")
    print(f"Tracking errors: {summary['tracking_errors']}")
    print(f"Occlusion events: {summary['occlusion_events']}")
    print(f"Insufficient evidence: {summary['insufficient_evidence']}")
    
    print("\n=== RECOMMENDATION ===")
    max_cat = max(cat_counts, key=cat_counts.get) if cat_counts else None
    
    if max_cat == "TEMPORARY_MISDETECTION":
        print("RECOMMENDATION: Investigate detector calibration or temporal smoothing before deployment.")
    elif max_cat == "PPE_ASSOCIATION_ERROR":
        print("RECOMMENDATION: Improve PPE-person association before deployment.")
    elif max_cat == "TRACKING_ID_ERROR":
        print("RECOMMENDATION: Improve track persistence/identity handling before deployment.")
    elif max_cat == "GENUINE_VIOLATION":
        print("RECOMMENDATION: Current violation logic is behaving correctly.")
    elif max_cat == "INSUFFICIENT_VISUAL_EVIDENCE":
        print("RECOMMENDATION: Acquire better positive PPE footage before changing the pipeline.")
    else:
        print("RECOMMENDATION: Address specific errors based on audit categories.")
        
    print(f"\nAudit complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
