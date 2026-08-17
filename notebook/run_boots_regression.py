import os
import sys
import time
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

def validate_model(model_path, data_yaml):
    print(f"Validating {model_path}...")
    model = YOLO(model_path)
    # Using imgsz=512 as per production config
    results = model.val(data=data_yaml, imgsz=512, split='val', verbose=False)
    
    metrics_dict = {}
    
    # Safely extract metrics if available
    try:
        class_indices = results.ap_class_index
        names = model.names
        
        for i, cls_idx in enumerate(class_indices):
            cls_name = names[cls_idx]
            metrics_dict[cls_name] = {
                'precision': results.box.p[i],
                'recall': results.box.r[i],
                'ap50': results.box.ap50[i],
                'map50_95': results.box.ap[i]
            }
    except Exception as e:
        print(f"Warning: Could not parse YOLO validation metrics for {model_path}: {e}")
        
    return metrics_dict

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def process_video(ppe_model_path, person_model_path, video_path, max_frames=300):
    print(f"Processing video with {ppe_model_path}...")
    if not os.path.exists(video_path):
        print(f"Video {video_path} not found. Skipping video simulation.")
        return {'fps': 0, 'latency': 0, 'helmet_fp': 0, 'vest_fp': 0, 'boots_fp': 0, 'violations': 0, 'assoc_fails': 0}
        
    ppe_model = YOLO(ppe_model_path)
    person_model = YOLO(person_model_path)
    
    cap = cv2.VideoCapture(video_path)
    frames_processed = 0
    start_time = time.time()
    latencies = []
    
    stats = {
        'helmet_fp': 0,
        'vest_fp': 0,
        'boots_fp': 0,
        'violations': 0,
        'assoc_fails': 0
    }
    
    unified_classes = ["person", "helmet", "no_helmet", "vest", "boots", "harness"]
    # Check if boots is in model names, otherwise use default
    has_boots = 4 in ppe_model.names and ppe_model.names[4] == 'boots'
    
    while cap.isOpened() and frames_processed < max_frames:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect person
        p_res = person_model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", conf=0.25, verbose=False, imgsz=512)[0]
        person_boxes = p_res.boxes.xyxy.cpu().numpy().tolist() if p_res.boxes.id is not None else []
        person_ids = p_res.boxes.id.int().cpu().tolist() if p_res.boxes.id is not None else []
        
        # Detect PPE
        ppe_res = ppe_model(frame, conf=0.25, verbose=False)[0]
        ppe_boxes = ppe_res.boxes.xyxy.cpu().numpy().tolist()
        
        # Safely map classes based on model's names dict
        ppe_classes = []
        for c in ppe_res.boxes.cls.cpu().numpy():
            c_int = int(c)
            cls_name = ppe_model.names.get(c_int, f"class_{c_int}")
            ppe_classes.append(cls_name)

        # Basic association
        assignments = {}
        for pid in person_ids:
            assignments[pid] = []
            
        for p_box, p_class in zip(ppe_boxes, ppe_classes):
            if p_class == 'person': continue
            cx, cy = box_center(p_box)
            best_id, best_dist = None, float("inf")
            for person_box, pid in zip(person_boxes, person_ids):
                x1, y1, x2, y2 = person_box
                if (x1 - 15) <= cx <= (x2 + 15) and (y1 - 15) <= cy <= (y2 + 15):
                    pcx, pcy = box_center(person_box)
                    dist = (pcx - cx)**2 + (pcy - cy)**2
                    if dist < best_dist:
                        best_dist, best_id = dist, pid
            if best_id is not None:
                assignments[best_id].append(p_class)
            else:
                stats['assoc_fails'] += 1
                if p_class == 'helmet': stats['helmet_fp'] += 1
                if p_class == 'vest': stats['vest_fp'] += 1
                if p_class == 'boots': stats['boots_fp'] += 1

        # Check violations (mock construction zone: requires helmet + vest)
        # For boots model experiment, check if boots are missing too just for stats
        for pid, worn in assignments.items():
            required = ['helmet', 'vest']
            missing = [item for item in required if item not in worn]
            if missing:
                stats['violations'] += 1
                
        latencies.append((time.time() - frame_start) * 1000)
        frames_processed += 1
        
    elapsed = time.time() - start_time
    cap.release()
    
    stats['fps'] = frames_processed / elapsed if elapsed > 0 else 0
    stats['latency'] = np.percentile(latencies, 95) if latencies else 0
    return stats

def main():
    print("Starting Boot Extension Regression Test...")
    v3_hn_path = os.path.join("models", "ppe_v3_hn_best.pt")
    v3_boots_path = os.path.join("models", "experiments", "v3_boots", "best.pt")
    person_path = os.path.join("models", "yolov8n.pt")
    data_yaml = os.path.join("datasets", "ppe_extension_boots", "data.yaml")
    
    # 1. Validation Regression
    metrics_hn = validate_model(v3_hn_path, data_yaml)
    metrics_boots = validate_model(v3_boots_path, data_yaml)
    
    # 2. Video Processing
    # Use docs/test.mp4 as the test video (covers both positive/negative simulation)
    video_path = os.path.join("..", "docs", "test.mp4")
    stats_hn = process_video(v3_hn_path, person_path, video_path)
    stats_boots = process_video(v3_boots_path, person_path, video_path)
    
    # 3. Decision Logic
    boots_ap50 = metrics_boots.get('boots', {}).get('ap50', 0)
    helmet_ap50_drop = metrics_hn.get('helmet', {}).get('ap50', 0) - metrics_boots.get('helmet', {}).get('ap50', 0)
    vest_ap50_drop = metrics_hn.get('vest', {}).get('ap50', 0) - metrics_boots.get('vest', {}).get('ap50', 0)
    
    decision = "EXPERIMENTAL"
    if stats_boots['fps'] < 12 or helmet_ap50_drop > 0.05 or vest_ap50_drop > 0.05:
        decision = "REJECTED"
    elif boots_ap50 > 0.70 and stats_boots['fps'] >= 12 and stats_boots['helmet_fp'] <= stats_hn['helmet_fp'] + 5:
        decision = "CANDIDATE_FOR_INTEGRATION"
        
    # 4. Save Artifacts
    out_dir = os.path.join("outputs", "ppe_extension")
    os.makedirs(out_dir, exist_ok=True)
    
    # CSVs
    rows = []
    for cls_name in ['helmet', 'vest', 'no_helmet', 'boots']:
        rows.append({
            'class': cls_name,
            'v3_hn_ap50': metrics_hn.get(cls_name, {}).get('ap50', 0),
            'v3_boots_ap50': metrics_boots.get(cls_name, {}).get('ap50', 0)
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "boots_regression.csv"), index=False)
    
    pd.DataFrame([
        {'model': 'V3-HN', **stats_hn},
        {'model': 'V3-Boots', **stats_boots}
    ]).to_csv(os.path.join(out_dir, "boots_video_comparison.csv"), index=False)
    
    with open(os.path.join(out_dir, "BOOTS_REGRESSION_REPORT.md"), 'w') as f:
        f.write("# BOOTS REGRESSION REPORT\n")
        f.write("Generated by run_boots_regression.py\n\n")
        f.write(f"## Final Decision: {decision}\n")
    
    # 5. Exact Terminal Output
    print("\n" + "="*50)
    print("BOOT EXTENSION REGRESSION COMPLETE")
    print("="*50)
    print("\nV3-HN:")
    print(f"FPS: {stats_hn['fps']:.2f}")
    print(f"Helmet FP: {stats_hn['helmet_fp']}")
    print(f"Vest FP: {stats_hn['vest_fp']}")
    print(f"Violations: {stats_hn['violations']}\n")
    
    print("V3-BOOTS:")
    print(f"FPS: {stats_boots['fps']:.2f}")
    print(f"Helmet FP: {stats_boots['helmet_fp']}")
    print(f"Vest FP: {stats_boots['vest_fp']}")
    print(f"Boots FP: {stats_boots['boots_fp']}")
    print(f"Violations: {stats_boots['violations']}\n")
    
    print(f"Boots detection: {boots_ap50:.3f} AP50")
    print(f"Helmet regression: {-helmet_ap50_drop:.3f} AP50 difference")
    print(f"Vest regression: {-vest_ap50_drop:.3f} AP50 difference")
    print(f"Association regression: {stats_boots['assoc_fails'] - stats_hn['assoc_fails']} additional fails\n")
    
    print("FINAL DECISION:")
    print(f"{decision}\n")
    print("PRODUCTION V3-HN:")
    print("UNTOUCHED\n")
    print("HARNESS:")
    print("UNSUPPORTED\n")
    print("STOP.")
    print("="*50)

if __name__ == "__main__":
    main()
