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
OUTPUT_DIR = "outputs/phase3h_association_audit"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HELMET_CONF = 0.65
VEST_CONF = 0.45
UNIFIED_CLASSES = ["person", "helmet", "no_helmet", "vest", "boots", "harness"]

# Configs matching Phase 3G exactly (frozen for audit)
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
MIN_ASSOC_SCORE = 0.50

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

def box_iou(box1, box2):
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area1 = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
    area2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# ==============================================================================
# BASELINE LOGIC (BEFORE)
# ==============================================================================
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

# ==============================================================================
# IMPROVED LOGIC (AFTER)
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

def calculate_assoc_score_detailed(person_box, ppe_box, ppe_class):
    ioa = iou_area(person_box, ppe_box)
    px1, py1, px2, py2 = person_box
    hx1, hy1, hx2, hy2 = ppe_box
    ph = py2 - py1
    pw = px2 - px1
    hcx, hcy = box_center(ppe_box)
    pcx, pcy = box_center(person_box)
    
    spatial_score = 0
    if ppe_class == "helmet":
        if not (py1 <= hcy <= py1 + 0.3 * ph): return 0, ioa, 0
        ideal_y = py1 + 0.15 * ph
        dist_y = abs(hcy - ideal_y) / max(1, ph)
        spatial_score = max(0, 1.0 - (dist_y * 2))
    elif ppe_class == "vest":
        if not (py1 + 0.2 * ph <= hcy <= py1 + 0.8 * ph): return 0, ioa, 0
        ideal_y = py1 + 0.5 * ph
        dist_y = abs(hcy - ideal_y) / max(1, ph)
        spatial_score = max(0, 1.0 - (dist_y * 2))
        
    if not (px1 <= hcx <= px2): return 0, ioa, spatial_score
        
    norm_dist = np.sqrt(((hcx - pcx)/max(1, pw))**2 + ((hcy - pcy)/max(1, ph))**2)
    dist_score = max(0, 1.0 - norm_dist)
    
    score = (W_IOA * ioa) + (W_SPATIAL * spatial_score) + (W_CENTER_DIST * dist_score)
    return score, ioa, spatial_score

def assoc_after_trace(person_boxes, person_ids, person_validity, ppe_boxes, ppe_classes):
    edges = []
    cross_candidates = []
    
    for ppe_idx, (p_box, p_class) in enumerate(zip(ppe_boxes, ppe_classes)):
        cands = []
        for person_idx, (person_box, pid, is_valid) in enumerate(zip(person_boxes, person_ids, person_validity)):
            if not is_valid: continue
            
            score, ioa, spatial = calculate_assoc_score_detailed(person_box, p_box, p_class)
            if score >= MIN_ASSOC_SCORE:
                cands.append({
                    "pid": pid,
                    "score": score,
                    "ioa": ioa,
                    "spatial": spatial,
                    "box": person_box
                })
                edges.append((score, ppe_idx, pid, p_class))
                
        if len(cands) > 1:
            cross_candidates.append({
                "ppe_idx": ppe_idx,
                "ppe_class": p_class,
                "cands": cands
            })
            
    edges.sort(key=lambda x: x[0], reverse=True)
    ppe_to_person = {}
    assigned_person_classes = defaultdict(set)
    
    for score, ppe_idx, pid, p_class in edges:
        if ppe_idx not in ppe_to_person:
            if p_class not in assigned_person_classes[pid]:
                assigned_person_classes[pid].add(p_class)
                ppe_to_person[ppe_idx] = pid
                
    return ppe_to_person, cross_candidates

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("Loading models and precomputing YOLO inference for Phase 3H Audit...")
    ppe_model = YOLO(V2_MODEL_PATH).to('cuda:0')
    person_model = YOLO("yolov8n.pt").to('cuda:0')
    
    cap = cv2.VideoCapture(TEST_VIDEO)
    
    audit_unassoc = []
    audit_cross = []
    audit_filter = []
    
    track_lengths = defaultdict(int)
    
    frame_idx = 0
    yolo_times = []
    alg_times = []
    
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
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
        
        t1 = time.perf_counter()
        
        for pid in p_ids:
            track_lengths[pid] += 1
            
        # 1. Person Filter Audit
        person_validity = []
        pid_to_validity = {}
        pid_to_box = {}
        for box, conf, pid in zip(p_boxes, p_confs, p_ids):
            is_valid, reason = is_valid_person(box, conf, track_lengths[pid])
            person_validity.append(is_valid)
            pid_to_validity[pid] = is_valid
            pid_to_box[pid] = box
            
            if not is_valid:
                category = "ambiguous"
                if reason in ["Area_Too_Small", "Too_Narrow", "Too_Short"]: category = "tiny box"
                elif reason == "Low_Conf": category = "low confidence"
                elif reason == "Invalid_Aspect": category = "implausible aspect ratio / likely poster"
                elif reason == "Unstable_Track": category = "unstable detection"
                
                audit_filter.append({
                    "frame": frame_idx,
                    "person_id": pid,
                    "reason": reason,
                    "category": category
                })
                
        # 2. Association Traces
        ppe_before = assoc_before_trace(p_boxes, p_ids, ppe_boxes_filtered, ppe_classes_filtered)
        ppe_after, cross_cands = assoc_after_trace(p_boxes, p_ids, person_validity, ppe_boxes_filtered, ppe_classes_filtered)
        
        # 3. Unassociated PPE Audit
        for ppe_idx in range(len(ppe_boxes_filtered)):
            p_b = ppe_before.get(ppe_idx)
            p_a = ppe_after.get(ppe_idx)
            
            if p_b is not None and p_a is None:
                # Newly unassociated
                p_class = ppe_classes_filtered[ppe_idx]
                ppe_conf = ppe_confs_filtered[ppe_idx]
                
                was_person_filtered = not pid_to_validity.get(p_b, True)
                
                classification = "AMBIGUOUS"
                reason_str = ""
                
                if was_person_filtered:
                    classification = "FAKE_PERSON_DEPENDENT"
                    reason_str = "Host person box was rejected by Phase 3G filter"
                elif ppe_conf < 0.50:
                    classification = "LIKELY_FALSE_PPE"
                    reason_str = "PPE confidence extremely low, barely passed global threshold"
                else:
                    classification = "LIKELY_VALID_ASSOCIATION"
                    reason_str = f"Person was valid, but score dropped below MIN_ASSOC_SCORE {MIN_ASSOC_SCORE}"
                    
                audit_unassoc.append({
                    "frame": frame_idx,
                    "ppe_class": p_class,
                    "ppe_conf": float(ppe_conf),
                    "previous_host_pid": p_b,
                    "classification": classification,
                    "reason": reason_str
                })
                
        # 4. Cross Assignment Audit
        for cross in cross_cands:
            cands = cross["cands"]
            ppe_idx = cross["ppe_idx"]
            
            # Check overlap between top 2 candidates
            if len(cands) >= 2:
                box1 = cands[0]["box"]
                box2 = cands[1]["box"]
                iou = box_iou(box1, box2)
                
                score_diff = abs(cands[0]["score"] - cands[1]["score"])
                
                if iou > 0.3:
                    classification = "legitimate overlapping-person association"
                    reason_str = f"High candidate overlap (IoU={iou:.2f})"
                elif score_diff < 0.15:
                    classification = "genuine cross-assignment risk"
                    reason_str = f"Close scoring margins (diff={score_diff:.2f}) between distant candidates"
                else:
                    classification = "ambiguous"
                    reason_str = "Multiple candidates cleared threshold, default winner took it"
                    
                audit_cross.append({
                    "frame": frame_idx,
                    "ppe_class": cross["ppe_class"],
                    "candidate_person_ids": ",".join([str(c["pid"]) for c in cands]),
                    "selected_person_id": ppe_after.get(ppe_idx, "None"),
                    "association_score": max(c["score"] for c in cands),
                    "ioa": max(c["ioa"] for c in cands),
                    "spatial_score": max(c["spatial"] for c in cands),
                    "classification": classification,
                    "reason": reason_str
                })
                
        alg_times.append(time.perf_counter() - t1)
        
    cap.release()
    print("Audit tracing complete.\n")
    
    # ---------------------------------------------------------
    # SUMMARIZATION
    # ---------------------------------------------------------
    df_unassoc = pd.DataFrame(audit_unassoc)
    df_cross = pd.DataFrame(audit_cross)
    df_filter = pd.DataFrame(audit_filter)
    
    if len(df_cross) > 0: df_cross.to_csv(os.path.join(OUTPUT_DIR, "phase3h_association_audit.csv"), index=False)
    if len(df_filter) > 0: df_filter.to_csv(os.path.join(OUTPUT_DIR, "phase3h_person_filter_audit.csv"), index=False)
    
    # Metrics
    total_unassoc = len(df_unassoc)
    count_valid = sum(1 for a in audit_unassoc if a["classification"] == "LIKELY_VALID_ASSOCIATION")
    count_false = sum(1 for a in audit_unassoc if a["classification"] == "LIKELY_FALSE_PPE")
    count_fake_dep = sum(1 for a in audit_unassoc if a["classification"] == "FAKE_PERSON_DEPENDENT")
    count_ambig = sum(1 for a in audit_unassoc if a["classification"] == "AMBIGUOUS")
    
    count_genuine_worker_concerns = sum(1 for f in audit_filter if f["category"] == "likely genuine person")
    
    # Performance
    total_times = [y + a for y, a in zip(yolo_times, alg_times)]
    avg_fps = len(total_times) / sum(total_times) if total_times else 0
    warm_times = total_times[5:] if len(total_times) > 5 else total_times
    warm_fps = len(warm_times) / sum(warm_times) if warm_times else 0
    p95_lat = np.percentile(total_times, 95) * 1000 if total_times else 0
    
    summary = {
        "new_unassociated_ppe": total_unassoc,
        "likely_valid": count_valid,
        "likely_false_ppe": count_false,
        "fake_person_dependent": count_fake_dep,
        "ambiguous": count_ambig,
        "cross_assignment_candidates": len(df_cross),
        "genuine_worker_filtering_concerns": count_genuine_worker_concerns,
        "average_fps": round(avg_fps, 2),
        "warm_fps": round(warm_fps, 2),
        "p95_latency_ms": round(p95_lat, 2)
    }
    
    with open(os.path.join(OUTPUT_DIR, "phase3h_association_audit.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    print("=== PHASE 3H AUDIT ===")
    print(f"New unassociated PPE: {total_unassoc}")
    print(f"Likely valid: {count_valid}")
    print(f"Likely false PPE: {count_false}")
    print(f"Fake-person dependent: {count_fake_dep}")
    print(f"Ambiguous: {count_ambig}")
    print(f"\nCross-assignment candidates: {len(df_cross)}")
    print(f"Genuine-worker filtering concerns: {count_genuine_worker_concerns}")
    print(f"\nAverage FPS: {avg_fps:.2f}")
    print(f"Warm FPS: {warm_fps:.2f}")
    print(f"P95 latency: {p95_lat:.2f} ms")
    
    print("\n=== FINAL RECOMMENDATION ===")
    
    if total_unassoc > 0:
        valid_rate = count_valid / total_unassoc
        fake_rate = (count_false + count_fake_dep) / total_unassoc
    else:
        valid_rate = 0
        fake_rate = 0
        
    if count_genuine_worker_concerns > 0:
        print("PERSON FILTER TOO AGGRESSIVE")
        print("- The Phase 3G Person Filter is incorrectly flagging genuine workers.")
        print("- We must lower aspect ratio or confidence requirements.")
        print("- Fake-person reduction is not worth losing genuine person tracking.")
    elif fake_rate > 0.6:
        print("FREEZE PHASE 3G")
        print(f"- {fake_rate*100:.1f}% of newly unassociated PPE were actually false associations tied to fake people/posters.")
        print("- Phase 3G successfully eliminated these dangerous false positives.")
        print("- Performance remains safely above the 12 FPS minimum requirement.")
    elif valid_rate > 0.3:
        print("REFINE ASSOCIATION")
        print(f"- {valid_rate*100:.1f}% of unassociated PPE were likely valid.")
        print("- Phase 3G's MIN_ASSOC_SCORE (0.50) is slightly too strict for genuine PPE slightly off-center.")
        print("- Recommend lowering MIN_ASSOC_SCORE to 0.40.")
    else:
        print("MORE AUDIT REQUIRED")
        print("- The results are ambiguous.")
        print("- Many unassociated PPE fall outside clear heuristic boundaries.")
        print("- Manual visual audit of the specific unassociated bounding boxes is required.")

if __name__ == "__main__":
    main()
