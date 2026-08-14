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
OUTPUT_DIR = "outputs/phase3i_association_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HELMET_CONF = 0.65
VEST_CONF = 0.45
UNIFIED_CLASSES = ["person", "helmet", "no_helmet", "vest", "boots", "harness"]
ZONE_RULES = {"construction": {"required": ["helmet", "vest"]}}

# Frozen Person Filter Config
MIN_PERSON_CONF = 0.35
MIN_PERSON_WIDTH = 25
MIN_PERSON_HEIGHT = 50
MIN_PERSON_AREA = 1500
MIN_PERSON_ASPECT = 0.15
MAX_PERSON_ASPECT = 1.5
MIN_TRACK_STABILITY = 5

W_IOA = 0.40
W_SPATIAL = 0.30
W_CENTER_DIST = 0.30

# Variables
SCORE_BASELINE = 0.50
SCORE_PROPOSED = 0.40

# ==============================================================================
# UTILS
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
# LOGIC
# ==============================================================================
def is_valid_person(box, conf, track_length):
    w = box[2] - box[0]
    h = box[3] - box[1]
    area = w * h
    aspect = w / h if h > 0 else 0
    if conf < MIN_PERSON_CONF: return False
    if w < MIN_PERSON_WIDTH: return False
    if h < MIN_PERSON_HEIGHT: return False
    if area < MIN_PERSON_AREA: return False
    if not (MIN_PERSON_ASPECT <= aspect <= MAX_PERSON_ASPECT): return False
    if track_length < MIN_TRACK_STABILITY: return False
    return True

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

def assoc_before_trace(person_boxes, person_ids, ppe_boxes, ppe_classes):
    ppe_to_person = {}
    for ppe_idx, (p_box, p_class) in enumerate(zip(ppe_boxes, ppe_classes)):
        hcx, hcy = box_center(p_box)
        best_id = None
        best_ioa = -1
        for pbox, pid in zip(person_boxes, person_ids):
            px1, py1, px2, py2 = pbox
            ph = py2 - py1
            if not (px1 <= hcx <= px2): continue
            if p_class == "helmet" and not (py1 <= hcy <= py1 + 0.3 * ph): continue
            if p_class == "vest" and not (py1 + 0.2 * ph <= hcy <= py1 + 0.8 * ph): continue
            
            ioa = iou_area(pbox, p_box)
            if ioa > best_ioa:
                best_ioa = ioa
                best_id = pid
        if best_id is not None:
            ppe_to_person[ppe_idx] = best_id
    return ppe_to_person

def assoc_after(person_boxes, person_ids, person_validity, ppe_boxes, ppe_classes, min_score):
    edges = []
    cross_assign_cands = 0
    
    for ppe_idx, (p_box, p_class) in enumerate(zip(ppe_boxes, ppe_classes)):
        valid_cands = 0
        for person_idx, (person_box, pid, is_valid) in enumerate(zip(person_boxes, person_ids, person_validity)):
            if not is_valid: continue
            score = calculate_assoc_score(person_box, p_box, p_class)
            if score >= min_score:
                edges.append((score, ppe_idx, pid, p_class))
                valid_cands += 1
        if valid_cands > 1:
            cross_assign_cands += 1
            
    edges.sort(key=lambda x: x[0], reverse=True)
    assigned_ppe = set()
    assignments = defaultdict(list)
    ppe_to_person = {}
    
    for score, ppe_idx, pid, p_class in edges:
        if ppe_idx not in assigned_ppe:
            if p_class not in assignments[pid]:
                assignments[pid].append(p_class)
                assigned_ppe.add(ppe_idx)
                ppe_to_person[ppe_idx] = pid
                
    assoc_failures = len(ppe_boxes) - len(assigned_ppe)
    return assignments, assoc_failures, cross_assign_cands, ppe_to_person

def run_pipeline(min_score, precomputed_data):
    validator = TemporalValidator()
    stats = {
        "person_detections": 0,
        "person_fp_candidates_filtered": 0,
        "stable_person_tracks": 0,
        "helmet_detections": 0,
        "vest_detections": 0,
        "ppe_associations": 0,
        "association_failures": 0,
        "cross_assignment_candidates": 0,
        "confirmed_violations": 0,
        "duplicate_events": 0,
        "empty_missing_ppe_events": 0,
        "frame_times": []
    }
    track_lengths = defaultdict(int)
    
    for frame_data in precomputed_data:
        t0 = time.perf_counter()
        
        p_boxes = frame_data["p_boxes"]
        p_ids = frame_data["p_ids"]
        p_confs = frame_data["p_confs"]
        ppe_boxes = frame_data["ppe_boxes"]
        ppe_classes = frame_data["ppe_classes"]
        
        for pid in p_ids:
            track_lengths[pid] += 1
            
        stats["person_detections"] += len(p_boxes)
        stats["helmet_detections"] += sum(1 for c in ppe_classes if c == "helmet")
        stats["vest_detections"] += sum(1 for c in ppe_classes if c == "vest")
        
        person_validity = []
        for box, conf, pid in zip(p_boxes, p_confs, p_ids):
            is_valid = is_valid_person(box, conf, track_lengths[pid])
            person_validity.append(is_valid)
            if not is_valid:
                stats["person_fp_candidates_filtered"] += 1
                
        stats["stable_person_tracks"] += sum(person_validity)
        
        assignments, assoc_fails, cross_cands, _ = assoc_after(p_boxes, p_ids, person_validity, ppe_boxes, ppe_classes, min_score)
        stats["association_failures"] += assoc_fails
        stats["cross_assignment_candidates"] += cross_cands
        stats["ppe_associations"] += sum(len(v) for v in assignments.values())
        
        for pbox, pid, is_valid in zip(p_boxes, p_ids, person_validity):
            if not is_valid: continue
            worn = set(assignments.get(pid, []))
            missing = [item for item in ZONE_RULES["construction"]["required"] if item not in worn]
            if validator.update(pid, len(missing) > 0):
                stats["confirmed_violations"] += 1
                if len(missing) == 0:
                    stats["empty_missing_ppe_events"] += 1
                
        stats["frame_times"].append(time.perf_counter() - t0)
        
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
    track_lengths = defaultdict(int)
    
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret: break
        
        p_res = person_model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", conf=0.25, verbose=False, imgsz=512)[0]
        p_boxes = p_res.boxes.xyxy.cpu().numpy().tolist() if p_res.boxes.id is not None else []
        p_ids = p_res.boxes.id.int().cpu().tolist() if p_res.boxes.id is not None else []
        p_confs = p_res.boxes.conf.cpu().numpy().tolist() if p_res.boxes.id is not None else []
        
        ppe_res = ppe_model(frame, conf=0.25, verbose=False)[0]
        ppe_boxes_filtered, ppe_classes_filtered, ppe_confs_filtered = [], [], []
        
        for box, cls, conf in zip(ppe_res.boxes.xyxy.cpu().numpy(), ppe_res.boxes.cls.cpu().numpy(), ppe_res.boxes.conf.cpu().numpy()):
            cname = UNIFIED_CLASSES[int(cls)]
            if cname == "helmet" and conf < HELMET_CONF: continue
            if cname == "vest" and conf < VEST_CONF: continue
            ppe_boxes_filtered.append(box)
            ppe_classes_filtered.append(cname)
            ppe_confs_filtered.append(conf)
            
        torch.cuda.synchronize()
        yolo_times.append(time.perf_counter() - t0)
        
        for pid in p_ids:
            track_lengths[pid] += 1
            
        person_validity = []
        for box, conf, pid in zip(p_boxes, p_confs, p_ids):
            person_validity.append(is_valid_person(box, conf, track_lengths[pid]))
            
        precomputed_data.append({
            "p_boxes": p_boxes,
            "p_ids": p_ids,
            "p_confs": p_confs,
            "person_validity": person_validity,
            "ppe_boxes": ppe_boxes_filtered,
            "ppe_classes": ppe_classes_filtered,
            "ppe_confs": ppe_confs_filtered
        })
        
    cap.release()
    print(f"Precomputed {len(precomputed_data)} frames.\n")
    
    # Run pipelines
    print(f"Running BASELINE pipeline (MIN_ASSOC_SCORE = {SCORE_BASELINE})...")
    stats_base = run_pipeline(SCORE_BASELINE, precomputed_data)
    
    print(f"Running PROPOSED pipeline (MIN_ASSOC_SCORE = {SCORE_PROPOSED})...")
    stats_prop = run_pipeline(SCORE_PROPOSED, precomputed_data)
    
    # FPS Calculation
    for i in range(len(precomputed_data)):
        stats_base["frame_times"][i] += yolo_times[i]
        stats_prop["frame_times"][i] += yolo_times[i]
        
    def finalize_fps(s):
        fts = s["frame_times"]
        s["average_fps"] = len(fts) / sum(fts) if fts else 0
        warm_fts = fts[5:] if len(fts) > 5 else fts
        s["warm_fps"] = len(warm_fts) / sum(warm_fts) if warm_fts else 0
        s["min_fps"] = 1.0 / max(fts) if fts else 0
        s["p95_latency"] = np.percentile(fts, 95) * 1000 if fts else 0
        del s["frame_times"]
        
    finalize_fps(stats_base)
    finalize_fps(stats_prop)
    
    # Audit previously unassociated PPE
    audit_stats = {
        "prev_unassoc": 0,
        "recovered_total": 0,
        "recovered_valid": 0,
        "recovered_false": 0,
        "recovered_fake_dep": 0,
        "still_unassociated": 0
    }
    
    for frame_data in precomputed_data:
        p_boxes = frame_data["p_boxes"]
        p_ids = frame_data["p_ids"]
        person_validity = frame_data["person_validity"]
        ppe_boxes = frame_data["ppe_boxes"]
        ppe_classes = frame_data["ppe_classes"]
        ppe_confs = frame_data["ppe_confs"]
        
        ppe_before = assoc_before_trace(p_boxes, p_ids, ppe_boxes, ppe_classes)
        _, _, _, ppe_050 = assoc_after(p_boxes, p_ids, person_validity, ppe_boxes, ppe_classes, SCORE_BASELINE)
        _, _, _, ppe_040 = assoc_after(p_boxes, p_ids, person_validity, ppe_boxes, ppe_classes, SCORE_PROPOSED)
        
        for ppe_idx in range(len(ppe_boxes)):
            p_b = ppe_before.get(ppe_idx)
            p_050 = ppe_050.get(ppe_idx)
            p_040 = ppe_040.get(ppe_idx)
            
            if p_b is not None and p_050 is None:
                audit_stats["prev_unassoc"] += 1
                
                pid_idx = p_ids.index(p_b) if p_b in p_ids else -1
                was_filtered = not person_validity[pid_idx] if pid_idx != -1 else True
                
                if was_filtered:
                    classification = "FAKE_PERSON_DEPENDENT"
                elif ppe_confs[ppe_idx] < 0.50:
                    classification = "LIKELY_FALSE_PPE"
                else:
                    classification = "LIKELY_VALID_ASSOCIATION"
                    
                if p_040 is not None:
                    audit_stats["recovered_total"] += 1
                    if classification == "LIKELY_VALID_ASSOCIATION": audit_stats["recovered_valid"] += 1
                    elif classification == "LIKELY_FALSE_PPE": audit_stats["recovered_false"] += 1
                    elif classification == "FAKE_PERSON_DEPENDENT": audit_stats["recovered_fake_dep"] += 1
                else:
                    audit_stats["still_unassociated"] += 1

    # Print Report
    metrics_to_print = [
        ("Person Detections", "person_detections"),
        ("Person FP Candidates Filtered", "person_fp_candidates_filtered"),
        ("Stable Person Tracks", "stable_person_tracks"),
        ("Helmet Detections", "helmet_detections"),
        ("Vest Detections", "vest_detections"),
        ("PPE Associations", "ppe_associations"),
        ("Association Failures", "association_failures"),
        ("Cross-Assignment Candidates", "cross_assignment_candidates"),
        ("Confirmed Violations", "confirmed_violations"),
        ("Duplicate Events", "duplicate_events"),
        ("Empty Missing-PPE Events", "empty_missing_ppe_events"),
        ("Average FPS", "average_fps"),
        ("Warm FPS", "warm_fps"),
        ("Minimum FPS", "min_fps"),
        ("P95 Latency (ms)", "p95_latency")
    ]
    
    comp_data = []
    for label, key in metrics_to_print:
        v_base = stats_base[key]
        v_prop = stats_prop[key]
        comp_data.append({"Metric": label, "BASELINE": v_base, "PROPOSED": v_prop, "Change": v_prop - v_base})
        
    df_comp = pd.DataFrame(comp_data)
    df_comp.to_csv(os.path.join(OUTPUT_DIR, "comparison.csv"), index=False)
    
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump({"baseline": stats_base, "proposed": stats_prop, "audit": audit_stats}, f, indent=4)
        
    print("\n=== PHASE 3I ASSOCIATION VALIDATION ===")
    print(f"BASELINE: MIN_ASSOC_SCORE = {SCORE_BASELINE}")
    print(f"PROPOSED: MIN_ASSOC_SCORE = {SCORE_PROPOSED}")
    print("-" * 75)
    print(f"{'Metric':<32} | {'BASELINE':<10} | {'PROPOSED':<10} | {'Change':<10}")
    print("-" * 75)
    for row in comp_data:
        v_b, v_p, chg = row["BASELINE"], row["PROPOSED"], row["Change"]
        if isinstance(v_b, float):
            print(f"{row['Metric']:<32} | {v_b:<10.2f} | {v_p:<10.2f} | {chg:<10.2f}")
        else:
            sign = "+" if chg > 0 else ""
            print(f"{row['Metric']:<32} | {v_b:<10} | {v_p:<10} | {sign}{chg:<10}")
            
    new_cross = stats_prop["cross_assignment_candidates"] - stats_base["cross_assignment_candidates"]
            
    print("\nPreviously unassociated PPE: {}".format(audit_stats["prev_unassoc"]))
    print("Recovered PPE: {}".format(audit_stats["recovered_total"]))
    print("Recovered likely-valid PPE: {}".format(audit_stats["recovered_valid"]))
    print("Recovered likely-false PPE: {}".format(audit_stats["recovered_false"]))
    print("Recovered fake-person-dependent PPE: {}".format(audit_stats["recovered_fake_dep"]))
    print("Still unassociated: {}".format(audit_stats["still_unassociated"]))
    print("New cross-assignments: {}".format(new_cross))
    print("Confirmed violations: {}".format(stats_prop["confirmed_violations"]))
    
    print(f"\nBaseline warm FPS: {stats_base['warm_fps']:.2f}")
    print(f"Proposed warm FPS: {stats_prop['warm_fps']:.2f}")
    print(f"Baseline P95 latency: {stats_base['p95_latency']:.2f} ms")
    print(f"Proposed P95 latency: {stats_prop['p95_latency']:.2f} ms")
    
    print("\n=== FINAL DECISION ===")
    
    recovered_valid_good = audit_stats["recovered_valid"] > 5
    recovered_fake_bad = audit_stats["recovered_fake_dep"] > 10
    cross_bad = new_cross > 20
    fps_ok = stats_prop["warm_fps"] >= 12
    violations_stable = stats_prop["confirmed_violations"] <= stats_base["confirmed_violations"] + 2
    
    if recovered_fake_bad or cross_bad or not violations_stable:
        print("KEEP 0.50")
        print("- Lowering the score caused too many dangerous fake associations or cross-assignment collisions.")
        print("- Fake-person dependent associations increased unsafely.")
    elif recovered_valid_good and fps_ok:
        print("ACCEPT 0.40")
        print("- Successfully recovered valid PPE without materially increasing fake-person associations.")
        print("- Cross-assignment candidates did not explode.")
        print("- Confirmed violations remained stable.")
        print("- Pipeline performance safely exceeded the 12 FPS minimum.")
    else:
        print("TEST INTERMEDIATE VALUE")
        print("- Valid PPE recovery was marginal or tradeoffs were slightly uncomfortable.")
        print("- Recommend testing 0.45 before finalizing the parameter.")
        print("- Please adjust SCORE_PROPOSED to 0.45 and re-run.")

if __name__ == "__main__":
    main()
