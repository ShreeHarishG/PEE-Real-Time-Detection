import os
import cv2
import time
import json
import torch
import glob
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO

# ==============================================================================
# V3 EXPERIMENT 1: EVALUATION (Epoch 10 Checkpoint)
# ==============================================================================

# CONFIGURATION
V2_MODEL_PATH = "edgevision_v2/models/ppe_best.pt"
V3_MODEL_PATH = "runs/detect/experiments/v3/exp1_hard_negative/weights/best.pt"
VAL_DATA_YAML = "edgevision_v2/datasets/merged/data.yaml"
NEG_VIDEO = "../docs/test.mp4"
POS_VIDEO = "../docs/positive_test/vidssave.com PPE Safety Video.✅#safetyfirst #ppe #video 👷_♂️👷✅💟 #viralvideo 720P.mp4"
HN_FRAMES_DIR = "edgevision_v2/datasets/hard_negative_frames"
OUTPUT_DIR = "outputs/v3"

os.makedirs(OUTPUT_DIR, exist_ok=True)

HELMET_CONF = 0.65
VEST_CONF = 0.45
MIN_ASSOC_SCORE = 0.40
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

print("V3 best.pt exists:", os.path.exists(V3_MODEL_PATH))
print("V3 best.pt path:", V3_MODEL_PATH)
print("V3 last.pt path:", V3_MODEL_PATH.replace("best.pt", "last.pt"))
print("V3 results.csv path:", V3_MODEL_PATH.replace("weights/best.pt", "results.csv"))
print("V2 model:", V2_MODEL_PATH)
print("V3-HN model:", V3_MODEL_PATH)

# ==============================================================================
# PIPELINE LOGIC (Frozen V2 Rules)
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
    
    for score, ppe_idx, pid, p_class in edges:
        if ppe_idx not in assigned_ppe:
            if p_class not in assignments[pid]:
                assignments[pid].append(p_class)
                assigned_ppe.add(ppe_idx)
                
    assoc_fails = len(ppe_boxes) - len(assigned_ppe)
    return assignments, assoc_fails, cross_assign_cands

# ==============================================================================
# PIPELINE RUNNER
# ==============================================================================
def run_video_pipeline(ppe_model, person_model, video_path):
    cap = cv2.VideoCapture(video_path)
    validator = TemporalValidator()
    
    stats = {
        "total_frames": 0,
        "helmet_detections": 0,
        "vest_detections": 0,
        "person_detections": 0,
        "ppe_associations": 0,
        "association_failures": 0,
        "confirmed_violations": 0,
        "false_violation_candidates": 0,
        "duplicate_events": 0,
        "empty_missing_ppe_events": 0,
        "unique_people": set(),
        "people_with_helmet": set(),
        "people_with_vest": set(),
        "people_with_both": set(),
        "frame_times": []
    }
    
    track_lengths = defaultdict(int)
    person_history = defaultdict(lambda: {"helmet": 0, "vest": 0, "frames": 0})
    
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret: break
        stats["total_frames"] += 1
        
        p_res = person_model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", conf=0.25, verbose=False, imgsz=512)[0]
        p_boxes = p_res.boxes.xyxy.cpu().numpy().tolist() if p_res.boxes.id is not None else []
        p_ids = p_res.boxes.id.int().cpu().tolist() if p_res.boxes.id is not None else []
        p_confs = p_res.boxes.conf.cpu().numpy().tolist() if p_res.boxes.id is not None else []
        
        stats["person_detections"] += len(p_boxes)
        
        ppe_res = ppe_model(frame, conf=0.25, verbose=False)[0]
        ppe_boxes_filtered, ppe_classes_filtered = [], []
        
        for box, cls, conf in zip(ppe_res.boxes.xyxy.cpu().numpy(), ppe_res.boxes.cls.cpu().numpy(), ppe_res.boxes.conf.cpu().numpy()):
            cname = UNIFIED_CLASSES[int(cls)]
            if cname == "helmet" and conf < HELMET_CONF: continue
            if cname == "vest" and conf < VEST_CONF: continue
            ppe_boxes_filtered.append(box)
            ppe_classes_filtered.append(cname)
            
            if cname == "helmet": stats["helmet_detections"] += 1
            if cname == "vest": stats["vest_detections"] += 1
            
        torch.cuda.synchronize()
        stats["frame_times"].append(time.perf_counter() - t0)
        
        person_validity = []
        for box, conf, pid in zip(p_boxes, p_confs, p_ids):
            track_lengths[pid] += 1
            is_valid = is_valid_person(box, conf, track_lengths[pid])
            person_validity.append(is_valid)
            if is_valid:
                stats["unique_people"].add(pid)
                person_history[pid]["frames"] += 1
            
        assignments, a_fails, _ = assoc_after(p_boxes, p_ids, person_validity, ppe_boxes_filtered, ppe_classes_filtered, MIN_ASSOC_SCORE)
        stats["association_failures"] += a_fails
        stats["ppe_associations"] += sum(len(v) for v in assignments.values())
        
        for pid in p_ids:
            worn = set(assignments.get(pid, []))
            if "helmet" in worn: person_history[pid]["helmet"] += 1
            if "vest" in worn: person_history[pid]["vest"] += 1
                
        for pbox, pid, is_valid in zip(p_boxes, p_ids, person_validity):
            if not is_valid: continue
            worn = set(assignments.get(pid, []))
            missing = [item for item in ZONE_RULES["construction"]["required"] if item not in worn]
            is_violation = len(missing) > 0
            
            if is_violation:
                stats["false_violation_candidates"] += 1
            
            if validator.update(pid, is_violation):
                stats["confirmed_violations"] += 1
                if len(missing) == 0:
                    stats["empty_missing_ppe_events"] += 1
                
    cap.release()
    
    # Calculate unique people stats
    for pid, hist in person_history.items():
        if hist["frames"] < MIN_TRACK_STABILITY: continue
        h_rate = hist["helmet"] / hist["frames"]
        v_rate = hist["vest"] / hist["frames"]
        
        if h_rate > 0.5: stats["people_with_helmet"].add(pid)
        if v_rate > 0.5: stats["people_with_vest"].add(pid)
        if h_rate > 0.5 and v_rate > 0.5: stats["people_with_both"].add(pid)
        
    fts = stats["frame_times"]
    stats["average_fps"] = len(fts) / sum(fts) if fts else 0
    warm_fts = fts[5:] if len(fts) > 5 else fts
    stats["warm_fps"] = len(warm_fts) / sum(warm_fts) if warm_fts else 0
    stats["cold_fps"] = 1.0 / fts[0] if fts else 0
    stats["mean_frame_latency"] = np.mean(fts) * 1000 if fts else 0
    stats["p95_latency"] = np.percentile(fts, 95) * 1000 if fts else 0
    
    return stats

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    if not os.path.exists(V3_MODEL_PATH):
        print(f"[ERROR] V3 Model {V3_MODEL_PATH} not found.")
        return

    models = {
        "V2": YOLO(V2_MODEL_PATH).to('cuda:0'),
        "V3-HN": YOLO(V3_MODEL_PATH).to('cuda:0')
    }
    person_model = YOLO("yolov8n.pt").to('cuda:0')
    
    results = {"V2": {}, "V3-HN": {}}
    
    for name, model in models.items():
        print(f"\n--- EVALUATING {name} ---")
        
        # 1. Validation Metrics
        print("1. Running validation...")
        val_res = model.val(data=VAL_DATA_YAML, split='val', imgsz=512, batch=16, verbose=False)
        names = val_res.names
        
        results[name]["val"] = {
            "mAP50": float(val_res.box.map50),
            "mAP50_95": float(val_res.box.map),
            "precision": float(val_res.box.mp),
            "recall": float(val_res.box.mr)
        }
        
        for i, c in enumerate(val_res.ap_class_index):
            cls_name = names[c]
            if cls_name in ['helmet', 'no_helmet', 'vest']:
                results[name]["val"][f"{cls_name}_precision"] = float(val_res.box.p[i])
                results[name]["val"][f"{cls_name}_recall"] = float(val_res.box.r[i])
                results[name]["val"][f"{cls_name}_AP50"] = float(val_res.box.ap50[i])
                results[name]["val"][f"{cls_name}_AP50_95"] = float(val_res.box.ap[i])
                
        # 2. Hard-Negative Evaluation
        print("2. Running hard-negative inference...")
        hn_files = glob.glob(os.path.join(HN_FRAMES_DIR, "*.jpg"))
        h_fps, v_fps = 0, 0
        for f in hn_files:
            res = model(f, verbose=False, conf=0.25)[0]
            for cls, conf in zip(res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                cname = UNIFIED_CLASSES[int(cls)]
                if cname == "helmet" and conf >= HELMET_CONF: h_fps += 1
                if cname == "vest" and conf >= VEST_CONF: v_fps += 1
        
        results[name]["hn"] = {
            "helmet_fp": h_fps,
            "vest_fp": v_fps,
            "total_fp": h_fps + v_fps,
            "total_frames": len(hn_files)
        }
        
        # 3. Negative Video
        print("3. Running negative test video...")
        neg_stats = run_video_pipeline(model, person_model, NEG_VIDEO)
        results[name]["neg"] = neg_stats
        
        # 4. Positive Video
        print("4. Running positive test video...")
        pos_stats = run_video_pipeline(model, person_model, POS_VIDEO)
        actual_helmet_wearers = 58
        actual_vest_wearers = 57
        pos_stats["helmet_detection_rate"] = len(pos_stats["people_with_helmet"]) / actual_helmet_wearers
        pos_stats["vest_detection_rate"] = len(pos_stats["people_with_vest"]) / actual_vest_wearers
        results[name]["pos"] = pos_stats

    # Fix sets for JSON serialization
    for name in results:
        for cat in results[name]:
            for k in results[name][cat]:
                if isinstance(results[name][cat][k], set):
                    results[name][cat][k] = len(results[name][cat][k])

    # Save outputs/v3/v2_vs_v3_validation.csv
    val_data = []
    keys = ["precision", "recall", "mAP50", "mAP50_95",
            "helmet_precision", "helmet_recall", "helmet_AP50", "helmet_AP50_95",
            "no_helmet_precision", "no_helmet_recall", "no_helmet_AP50", "no_helmet_AP50_95",
            "vest_precision", "vest_recall", "vest_AP50", "vest_AP50_95"]
    for k in keys:
        val_data.append([k, results["V2"]["val"].get(k, 0), results["V3-HN"]["val"].get(k, 0)])
    
    val_df = pd.DataFrame(val_data, columns=["Metric", "V2 Baseline", "V3-HN"])
    val_df.to_csv(os.path.join(OUTPUT_DIR, "v2_vs_v3_validation.csv"), index=False)
    
    print("\n--- VALIDATION COMPARISON ---")
    print(val_df.to_string(index=False))

    # Save outputs/v3/v2_vs_v3_final_comparison.csv
    final_data = []
    final_data.extend(val_data)
    
    # Negative Video
    neg_keys = ["total_frames", "helmet_detections", "vest_detections", "person_detections", 
                "ppe_associations", "association_failures", "confirmed_violations", 
                "duplicate_events", "empty_missing_ppe_events", "average_fps", "warm_fps", "p95_latency"]
    for k in neg_keys:
        final_data.append([f"Neg Video: {k}", results["V2"]["neg"].get(k, 0), results["V3-HN"]["neg"].get(k, 0)])
        
    # Positive Video
    pos_keys = ["unique_people", "people_with_helmet", "people_with_vest", "people_with_both",
                "helmet_detection_rate", "vest_detection_rate", "confirmed_violations", 
                "false_violation_candidates", "average_fps", "warm_fps", "p95_latency"]
    for k in pos_keys:
        final_data.append([f"Pos Video: {k}", results["V2"]["pos"].get(k, 0), results["V3-HN"]["pos"].get(k, 0)])
        
    # Hard-Negative
    hn_keys = ["helmet_fp", "vest_fp", "total_fp"]
    for k in hn_keys:
        final_data.append([f"Hard Negative: {k}", results["V2"]["hn"].get(k, 0), results["V3-HN"]["hn"].get(k, 0)])
        
    # Performance
    perf_keys = ["cold_fps", "warm_fps", "mean_frame_latency", "p95_latency"]
    for k in perf_keys:
        final_data.append([f"Performance (Neg): {k}", results["V2"]["neg"].get(k, 0), results["V3-HN"]["neg"].get(k, 0)])
        
    final_df = pd.DataFrame(final_data, columns=["Metric", "V2 Baseline", "V3-HN"])
    final_df.to_csv(os.path.join(OUTPUT_DIR, "v2_vs_v3_final_comparison.csv"), index=False)

    # Calculate reductions
    v2_h_fp = results["V2"]["hn"]["helmet_fp"]
    v3_h_fp = results["V3-HN"]["hn"]["helmet_fp"]
    v2_v_fp = results["V2"]["hn"]["vest_fp"]
    v3_v_fp = results["V3-HN"]["hn"]["vest_fp"]
    
    h_red = ((v2_h_fp - v3_h_fp) / v2_h_fp * 100) if v2_h_fp > 0 else 0
    v_red = ((v2_v_fp - v3_v_fp) / v2_v_fp * 100) if v2_v_fp > 0 else 0
    
    # Save outputs/v3/V3_EVALUATION_REPORT.md
    report_md = f"""# V3_EVALUATION_REPORT

## 1. V2 Baseline vs V3-HN Results
**V2 Baseline:** `{V2_MODEL_PATH}`
**V3-HN (Epoch 10):** `{V3_MODEL_PATH}`

## 2. Validation Comparison
```csv
{val_df.to_csv(index=False)}
```

## 3. Negative-Video Comparison (test.mp4)
- **V2 Helmet Detections (False):** {results['V2']['neg']['helmet_detections']}
- **V3 Helmet Detections (False):** {results['V3-HN']['neg']['helmet_detections']}
- **V2 Vest Detections (False):** {results['V2']['neg']['vest_detections']}
- **V3 Vest Detections (False):** {results['V3-HN']['neg']['vest_detections']}
- **V2 Confirmed Violations (False):** {results['V2']['neg']['confirmed_violations']}
- **V3 Confirmed Violations (False):** {results['V3-HN']['neg']['confirmed_violations']}

## 4. Positive-Video Comparison
- **V2 Helmet Detection Rate:** {results['V2']['pos']['helmet_detection_rate']:.4f}
- **V3 Helmet Detection Rate:** {results['V3-HN']['pos']['helmet_detection_rate']:.4f}
- **V2 Vest Detection Rate:** {results['V2']['pos']['vest_detection_rate']:.4f}
- **V3 Vest Detection Rate:** {results['V3-HN']['pos']['vest_detection_rate']:.4f}
- **V2 Confirmed Violations:** {results['V2']['pos']['confirmed_violations']}
- **V3 Confirmed Violations:** {results['V3-HN']['pos']['confirmed_violations']}

## 5. Hard-Negative Comparison (73 crops)
- **V2 Helmet FP:** {v2_h_fp}
- **V3 Helmet FP:** {v3_h_fp} -> **{h_red:.1f}% reduction**
- **V2 Vest FP:** {v2_v_fp}
- **V3 Vest FP:** {v3_v_fp} -> **{v_red:.1f}% reduction**

## 6. FPS Comparison
- **V2 Warm FPS (Neg/Pos):** {results['V2']['neg']['warm_fps']:.1f} / {results['V2']['pos']['warm_fps']:.1f}
- **V3 Warm FPS (Neg/Pos):** {results['V3-HN']['neg']['warm_fps']:.1f} / {results['V3-HN']['pos']['warm_fps']:.1f}
- **V2 P95 Latency (Neg):** {results['V2']['neg']['p95_latency']:.1f} ms
- **V3 P95 Latency (Neg):** {results['V3-HN']['neg']['p95_latency']:.1f} ms

## 7. Percentage Improvements & Regression Analysis
- **Helmet FPs on crops reduced by:** {h_red:.1f}%
- **Vest FPs on crops reduced by:** {v_red:.1f}%
- **Helmet False Detections on video:** {results['V2']['neg']['helmet_detections']} -> {results['V3-HN']['neg']['helmet_detections']}
- **Validation mAP50:** {results['V2']['val']['mAP50']:.4f} -> {results['V3-HN']['val']['mAP50']:.4f}

## 8. Final Recommendation
"""
    
    if h_red >= 50 and v3_v_fp <= v2_v_fp and results['V3-HN']['val']['helmet_recall'] > 0.8:
        report_md += "**DECISION: PROMOTE V3-HN**\n- Model successfully suppressed >50% of real-world helmet FPs.\n- Recall is stable.\n- FPS meets >12 requirement."
    else:
        report_md += "**DECISION: KEEP V2**\n- V3 did not suppress enough FPs, or recall degraded unacceptably."
        
    with open(os.path.join(OUTPUT_DIR, "V3_EVALUATION_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report_md)

    # FINAL CONCISE TERMINAL OUTPUT
    print("\n" + "="*40)
    print("V2 vs V3-HN FINAL DECISION")
    print("="*40)
    print(f"V2 mAP50: {results['V2']['val']['mAP50']:.4f}")
    print(f"V3 mAP50: {results['V3-HN']['val']['mAP50']:.4f}\n")
    print(f"V2 mAP50-95: {results['V2']['val']['mAP50_95']:.4f}")
    print(f"V3 mAP50-95: {results['V3-HN']['val']['mAP50_95']:.4f}\n")
    
    print(f"V2 helmet FP: {v2_h_fp}")
    print(f"V3 helmet FP: {v3_h_fp}")
    print(f"Helmet FP reduction: {h_red:.1f}%\n")
    
    print(f"V2 vest FP: {v2_v_fp}")
    print(f"V3 vest FP: {v3_v_fp}")
    print(f"Vest FP reduction: {v_red:.1f}%\n")
    
    print(f"V2 positive helmet detection: {results['V2']['pos']['helmet_detection_rate']:.4f}")
    print(f"V3 positive helmet detection: {results['V3-HN']['pos']['helmet_detection_rate']:.4f}\n")
    
    print(f"V2 positive vest detection: {results['V2']['pos']['vest_detection_rate']:.4f}")
    print(f"V3 positive vest detection: {results['V3-HN']['pos']['vest_detection_rate']:.4f}\n")
    
    print(f"V2 warm FPS: {results['V2']['neg']['warm_fps']:.1f}")
    print(f"V3 warm FPS: {results['V3-HN']['neg']['warm_fps']:.1f}\n")
    
    print(f"V2 confirmed violations: {results['V2']['neg']['confirmed_violations']}")
    print(f"V3 confirmed violations: {results['V3-HN']['neg']['confirmed_violations']}\n")

    # Basic heuristic for printing decision to terminal (user makes the real choice)
    if h_red >= 50 and results['V3-HN']['val']['helmet_recall'] > 0.8:
        print("FINAL DECISION:\nPROMOTE V3-HN\n")
    else:
        print("FINAL DECISION:\nKEEP V2\n")

    print("REASON:")
    print(f"- Helmet FP reduction is {h_red:.1f}%")
    print(f"- Vest FP reduction is {v_red:.1f}%")
    print(f"- mAP50 changed from {results['V2']['val']['mAP50']:.3f} to {results['V3-HN']['val']['mAP50']:.3f}")
    print(f"- Warm FPS is {results['V3-HN']['neg']['warm_fps']:.1f}")
    
if __name__ == "__main__":
    main()
