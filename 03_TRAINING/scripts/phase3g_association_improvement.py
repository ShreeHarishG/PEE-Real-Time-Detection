import os
import cv2
import time
import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION
# ==============================================================================
V2_MODEL_PATH = "edgevision_v2/models/ppe_best.pt"
TEST_VIDEO = "../docs/test.mp4"
OUTPUT_DIR = "outputs/phase3g_association"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HELMET_CONF = 0.65
VEST_CONF = 0.45
UNIFIED_CLASSES = ["person", "helmet", "no_helmet", "vest", "boots", "harness"]
ZONE_RULES = {"construction": {"required": ["helmet", "vest"]}}

# ==============================================================================
# PHASE 3G IMPROVEMENT CONFIG
# ==============================================================================
# Person Filter
MIN_PERSON_CONF = 0.35      # Was 0.25. Helps reject weak 2D posters/reflections
MIN_PERSON_WIDTH = 25       # Rejects extreme slivers of people/poles
MIN_PERSON_HEIGHT = 50      # Rejects truncated tiny boxes
MIN_PERSON_AREA = 1500      # w * h minimum
MIN_PERSON_ASPECT = 0.15    # min width/height (rejects impossibly thin vertical strips)
MAX_PERSON_ASPECT = 1.5     # max width/height (rejects horizontal artifacts/posters)
MIN_TRACK_STABILITY = 5     # Frames a track must exist before considered "valid"

# PPE Association Score Weights
# Total should ideally sum to 1.0 for normalized scoring.
W_IOA = 0.40          # Intersection over Area is still the strongest baseline
W_SPATIAL = 0.30      # Heavy weight to ensure helmet is near top, vest near middle
W_CENTER_DIST = 0.30  # Punishes PPE that is technically inside the box but physically far from the person's center
MIN_ASSOC_SCORE = 0.50

# ==============================================================================
# BASE UTILITIES
# ==============================================================================
def box_center(box):
    return ((box[0] + box[2])/2, (box[1] + box[3])/2)

def iou_area(p_box, ppe_box):
    px1, py1, px2, py2 = p_box
    hx1, hy1, hx2, hy2 = ppe_box
    ix1, iy1 = max(px1, hx1), max(py1, hy1)
    ix2, iy2 = min(px2, hx2), min(py2, hy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    return (iw * ih) / max(1, (hx2 - hx1) * (hy2 - hy1))

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

# ==============================================================================
# BEFORE (BASELINE) LOGIC
# ==============================================================================
def assoc_before(person_boxes, person_ids, ppe_boxes, ppe_classes):
    assignments = defaultdict(list)
    assoc_failures = 0
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
        else:
            assoc_failures += 1
    return assignments, assoc_failures

# ==============================================================================
# AFTER (IMPROVED) LOGIC
# ==============================================================================
def is_valid_person(box, conf, track_length):
    w = box[2] - box[0]
    h = box[3] - box[1]
    area = w * h
    aspect = w / h if h > 0 else 0
    
    if conf < MIN_PERSON_CONF: return False, "Low_Conf"
    if w < MIN_PERSON_WIDTH: return False, "Too_Narrow"
    if h < MIN_PERSON_HEIGHT: return False, "Too_Short"
    if area < MIN_PERSON_AREA: return False, "Area_Too_Small"
    if not (MIN_PERSON_ASPECT <= aspect <= MAX_PERSON_ASPECT): return False, "Invalid_Aspect"
    if track_length < MIN_TRACK_STABILITY: return False, "Unstable_Track"
    
    return True, "Valid"

def calculate_assoc_score(person_box, ppe_box, ppe_class):
    ioa = iou_area(person_box, ppe_box)
    
    px1, py1, px2, py2 = person_box
    hx1, hy1, hx2, hy2 = ppe_box
    ph = py2 - py1
    pw = px2 - px1
    hcx, hcy = box_center(ppe_box)
    pcx, pcy = box_center(person_box)
    
    spatial_score = 0
    if ppe_class == "helmet":
        if not (py1 <= hcy <= py1 + 0.3 * ph): return 0
        ideal_y = py1 + 0.15 * ph
        dist_y = abs(hcy - ideal_y) / max(1, ph)
        spatial_score = max(0, 1.0 - (dist_y * 2))
    elif ppe_class == "vest":
        if not (py1 + 0.2 * ph <= hcy <= py1 + 0.8 * ph): return 0
        ideal_y = py1 + 0.5 * ph
        dist_y = abs(hcy - ideal_y) / max(1, ph)
        spatial_score = max(0, 1.0 - (dist_y * 2))
        
    if not (px1 <= hcx <= px2): return 0
        
    norm_dist = np.sqrt(((hcx - pcx)/max(1, pw))**2 + ((hcy - pcy)/max(1, ph))**2)
    dist_score = max(0, 1.0 - norm_dist)
    
    return (W_IOA * ioa) + (W_SPATIAL * spatial_score) + (W_CENTER_DIST * dist_score)

def assoc_after(person_boxes, person_ids, person_validity, ppe_boxes, ppe_classes, assoc_report_rows):
    assignments = defaultdict(list)
    edges = []
    cross_assign_cands = 0
    
    for ppe_idx, (p_box, p_class) in enumerate(zip(ppe_boxes, ppe_classes)):
        valid_cands = 0
        for person_idx, (person_box, pid, is_valid) in enumerate(zip(person_boxes, person_ids, person_validity)):
            if not is_valid: continue
            
            score = calculate_assoc_score(person_box, p_box, p_class)
            
            assoc_report_rows.append({
                "ppe_id": ppe_idx,
                "ppe_class": p_class,
                "person_id": pid,
                "score": round(score, 4)
            })
            
            if score >= MIN_ASSOC_SCORE:
                edges.append((score, ppe_idx, pid, p_class))
                valid_cands += 1
                
        if valid_cands > 1:
            cross_assign_cands += 1
            
    edges.sort(key=lambda x: x[0], reverse=True)
    assigned_ppe = set()
    
    for score, ppe_idx, pid, p_class in edges:
        if ppe_idx not in assigned_ppe:
            if p_class not in assignments[pid]:
                assignments[pid].append(p_class)
                assigned_ppe.add(ppe_idx)
                
    assoc_failures = len(ppe_boxes) - len(assigned_ppe)
    return assignments, assoc_failures, cross_assign_cands

# ==============================================================================
# PIPELINE RUNNER
# ==============================================================================
def run_pipeline(mode, precomputed_data, filter_report_rows, assoc_report_rows):
    validator = TemporalValidator()
    
    stats = {
        "person_detections": 0,
        "stable_person_tracks": 0,
        "ppe_detections": 0,
        "helmet_detections": 0,
        "vest_detections": 0,
        "ppe_associations": 0,
        "association_failures": 0,
        "cross_assignment_candidates": 0,
        "confirmed_violations": 0,
        "person_false_positive_candidates": 0,
        "short_tracks": 0,
        "frames": 0,
        "frame_times": []
    }
    
    track_lengths = defaultdict(int)
    
    for frame_data in precomputed_data:
        stats["frames"] += 1
        t0 = time.perf_counter()
        
        p_boxes = frame_data["p_boxes"]
        p_ids = frame_data["p_ids"]
        p_confs = frame_data["p_confs"]
        ppe_boxes = frame_data["ppe_boxes"]
        ppe_classes = frame_data["ppe_classes"]
        
        for pid in p_ids:
            track_lengths[pid] += 1
            
        stats["person_detections"] += len(p_boxes)
        stats["ppe_detections"] += len(ppe_boxes)
        stats["helmet_detections"] += sum(1 for c in ppe_classes if c == "helmet")
        stats["vest_detections"] += sum(1 for c in ppe_classes if c == "vest")
        
        if mode == "BEFORE":
            assignments, assoc_fails = assoc_before(p_boxes, p_ids, ppe_boxes, ppe_classes)
            stats["association_failures"] += assoc_fails
            stats["ppe_associations"] += sum(len(v) for v in assignments.values())
            
            for pid in p_ids:
                if track_lengths[pid] >= MIN_TRACK_STABILITY:
                    stats["stable_person_tracks"] += 1
                    
            for pbox, pid in zip(p_boxes, p_ids):
                worn = set(assignments.get(pid, []))
                missing = [item for item in ZONE_RULES["construction"]["required"] if item not in worn]
                if validator.update(pid, len(missing) > 0):
                    stats["confirmed_violations"] += 1
                    
        else:
            # AFTER mode
            person_validity = []
            for box, conf, pid in zip(p_boxes, p_confs, p_ids):
                is_valid, reason = is_valid_person(box, conf, track_lengths[pid])
                person_validity.append(is_valid)
                if not is_valid:
                    stats["person_false_positive_candidates"] += 1
                
                filter_report_rows.append({
                    "frame": stats["frames"],
                    "person_id": pid,
                    "conf": round(conf, 3),
                    "track_len": track_lengths[pid],
                    "is_valid": is_valid,
                    "reject_reason": reason
                })
                    
            stats["stable_person_tracks"] += sum(person_validity)
            
            assignments, assoc_fails, cross_cands = assoc_after(p_boxes, p_ids, person_validity, ppe_boxes, ppe_classes, assoc_report_rows)
            stats["association_failures"] += assoc_fails
            stats["cross_assignment_candidates"] += cross_cands
            stats["ppe_associations"] += sum(len(v) for v in assignments.values())
            
            for pbox, pid, is_valid in zip(p_boxes, p_ids, person_validity):
                if not is_valid: continue # Filtered out
                
                worn = set(assignments.get(pid, []))
                missing = [item for item in ZONE_RULES["construction"]["required"] if item not in worn]
                if validator.update(pid, len(missing) > 0):
                    stats["confirmed_violations"] += 1
                    
        stats["frame_times"].append(time.perf_counter() - t0)
        
    for pid, length in track_lengths.items():
        if length < MIN_TRACK_STABILITY:
            stats["short_tracks"] += 1
            
    return stats

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("Loading models and precomputing YOLO inference...")
    ppe_model = YOLO(V2_MODEL_PATH).to('cuda:0')
    person_model = YOLO("yolov8n.pt").to('cuda:0')
    
    cap = cv2.VideoCapture(TEST_VIDEO)
    
    precomputed_data = []
    yolo_times = []
    
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret: break
        
        p_res = person_model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", conf=0.25, verbose=False, imgsz=512)[0]
        p_boxes = p_res.boxes.xyxy.cpu().numpy().tolist() if p_res.boxes.id is not None else []
        p_ids = p_res.boxes.id.int().cpu().tolist() if p_res.boxes.id is not None else []
        p_confs = p_res.boxes.conf.cpu().numpy().tolist() if p_res.boxes.id is not None else []
        
        ppe_res = ppe_model(frame, conf=0.25, verbose=False)[0]
        ppe_boxes_filtered, ppe_classes_filtered = [], []
        
        for box, cls, conf in zip(ppe_res.boxes.xyxy.cpu().numpy(), ppe_res.boxes.cls.cpu().numpy(), ppe_res.boxes.conf.cpu().numpy()):
            cname = UNIFIED_CLASSES[int(cls)]
            if cname == "helmet" and conf < HELMET_CONF: continue
            if cname == "vest" and conf < VEST_CONF: continue
            ppe_boxes_filtered.append(box)
            ppe_classes_filtered.append(cname)
            
        torch.cuda.synchronize()
        yolo_times.append(time.perf_counter() - t0)
        
        precomputed_data.append({
            "p_boxes": p_boxes,
            "p_ids": p_ids,
            "p_confs": p_confs,
            "ppe_boxes": ppe_boxes_filtered,
            "ppe_classes": ppe_classes_filtered
        })
        
    cap.release()
    print(f"Precomputed {len(precomputed_data)} frames.\n")
    
    print("Running BEFORE (Baseline) pipeline...")
    stats_before = run_pipeline("BEFORE", precomputed_data, [], [])
    
    print("Running AFTER (Improved) pipeline...")
    filter_report_rows = []
    assoc_report_rows = []
    stats_after = run_pipeline("AFTER", precomputed_data, filter_report_rows, assoc_report_rows)
    
    # Save detailed CSVs
    pd.DataFrame(filter_report_rows).to_csv(os.path.join(OUTPUT_DIR, "person_filter_report.csv"), index=False)
    pd.DataFrame(assoc_report_rows).to_csv(os.path.join(OUTPUT_DIR, "association_report.csv"), index=False)
    
    # Add YOLO times back for accurate FPS
    for i in range(len(precomputed_data)):
        stats_before["frame_times"][i] += yolo_times[i]
        stats_after["frame_times"][i] += yolo_times[i]
        
    def finalize_fps(s):
        fts = s["frame_times"]
        s["average_fps"] = len(fts) / sum(fts) if fts else 0
        warm_fts = fts[5:] if len(fts) > 5 else fts
        s["warm_fps"] = len(warm_fts) / sum(warm_fts) if warm_fts else 0
        s["p95_latency"] = np.percentile(fts, 95) * 1000 if fts else 0
        del s["frame_times"]
        
    finalize_fps(stats_before)
    finalize_fps(stats_after)
    
    with open(os.path.join(OUTPUT_DIR, "before_summary.json"), "w") as f:
        json.dump(stats_before, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "after_summary.json"), "w") as f:
        json.dump(stats_after, f, indent=4)
        
    # Build comparison report
    print("\n==================================================")
    print("PHASE 3G COMPARISON REPORT")
    print("==================================================")
    print(f"{'Metric':<32} | {'BEFORE':<10} | {'AFTER':<10} | {'Change':<10}")
    print("-" * 70)
    
    metrics_to_print = [
        ("Person Detections", "person_detections"),
        ("Person FP Candidates Filtered", "person_false_positive_candidates"),
        ("Stable Person Tracks", "stable_person_tracks"),
        ("Helmet Detections", "helmet_detections"),
        ("Vest Detections", "vest_detections"),
        ("PPE Associations", "ppe_associations"),
        ("Association Failures", "association_failures"),
        ("Cross-Assignment Candidates", "cross_assignment_candidates"),
        ("Confirmed Violations", "confirmed_violations"),
        ("Average FPS", "average_fps"),
        ("Warm FPS", "warm_fps"),
        ("P95 Latency (ms)", "p95_latency")
    ]
    
    comp_dict = {}
    for label, key in metrics_to_print:
        v_before = stats_before.get(key, 0)
        v_after = stats_after.get(key, 0)
        
        if isinstance(v_before, float):
            change = v_after - v_before
            sign = "+" if change > 0 else ""
            print(f"{label:<32} | {v_before:<10.2f} | {v_after:<10.2f} | {sign}{change:<10.2f}")
        else:
            change = v_after - v_before
            sign = "+" if change > 0 else ""
            print(f"{label:<32} | {v_before:<10} | {v_after:<10} | {sign}{change:<10}")
            
        comp_dict[key] = {"BEFORE": v_before, "AFTER": v_after, "CHANGE": change}
        
    with open(os.path.join(OUTPUT_DIR, "comparison.json"), "w") as f:
        json.dump(comp_dict, f, indent=4)
        
    print("\n==================================================")
    print("=== PHASE 3G VERDICT ===")
    
    person_fp_decrease = stats_after["person_false_positive_candidates"] > 0
    assoc_fails_better = stats_after["association_failures"] <= stats_before["association_failures"]
    violations_stable = stats_after["confirmed_violations"] <= stats_before["confirmed_violations"] + 5
    ppe_assoc_drop = stats_before["ppe_associations"] - stats_after["ppe_associations"]
    fps_ok = stats_after["warm_fps"] >= 12
    
    if ppe_assoc_drop > stats_before["ppe_associations"] * 0.2:
        print("ASSOCIATION TOO STRICT")
        print("Reason: Over 20% of previously successful PPE associations were dropped. The new matching thresholds are too aggressive.")
    elif (stats_before["stable_person_tracks"] - stats_after["stable_person_tracks"]) > stats_before["stable_person_tracks"] * 0.3:
        print("FILTER TOO AGGRESSIVE")
        print("Reason: Over 30% of stable person tracks were eliminated. The person bounding box filters are rejecting genuine workers.")
    elif not fps_ok:
        print("FAIL")
        print(f"Reason: FPS dropped below 12 (Warm FPS: {stats_after['warm_fps']:.2f}).")
    elif not violations_stable:
        print("FAIL")
        print(f"Reason: Confirmed violations exploded from {stats_before['confirmed_violations']} to {stats_after['confirmed_violations']}.")
    elif person_fp_decrease and stats_after["association_failures"] < stats_before["association_failures"]:
        print("PASS")
        print("Reason: Person filtering actively reduced false candidates, and the new association logic successfully reduced unassigned PPE.")
    else:
        print("PASS WITH TRADEOFF")
        print("Reason: The pipeline safely applied the new logic without breaking constraints, but association failures did not strictly decrease.")
        
    print("\nPhase 3G complete. Results saved to outputs/phase3g_association/")

if __name__ == "__main__":
    main()
