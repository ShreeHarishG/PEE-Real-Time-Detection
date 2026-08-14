import os
import glob
import json
import csv
import numpy as np
from collections import defaultdict, Counter

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATASET_ROOT = "edgevision_v2/datasets/merged"
OUTPUT_DIR = "outputs/project_audit"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = {0: "person", 1: "helmet", 2: "no_helmet", 3: "vest", 4: "boots", 5: "harness"}
SPLITS = ["train", "val", "test"]

# ==============================================================================
# PARSE ALL LABELS
# ==============================================================================
def parse_labels(label_dir):
    """Parse all YOLO label .txt files in a directory."""
    records = []
    zero_label_files = []
    
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    for lf in label_files:
        basename = os.path.splitext(os.path.basename(lf))[0]
        with open(lf, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        if len(lines) == 0:
            zero_label_files.append(basename)
            continue
            
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            records.append({
                "file": basename,
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, f"unknown_{cls_id}"),
                "cx": cx, "cy": cy, "w": w, "h": h,
                "area": w * h  # normalized area (fraction of image)
            })
    
    return records, zero_label_files, len(label_files)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("DATASET AUDIT: edgevision_v2/datasets/merged/")
    print("=" * 60)
    
    all_records = {}
    all_zero_labels = {}
    all_file_counts = {}
    
    for split in SPLITS:
        label_dir = os.path.join(DATASET_ROOT, "labels", split)
        image_dir = os.path.join(DATASET_ROOT, "images", split)
        
        if not os.path.exists(label_dir):
            print(f"\nWARNING: {label_dir} does not exist!")
            continue
            
        records, zero_labels, n_files = parse_labels(label_dir)
        all_records[split] = records
        all_zero_labels[split] = zero_labels
        
        # Count images
        n_images = len(glob.glob(os.path.join(image_dir, "*.*")))
        all_file_counts[split] = {"images": n_images, "labels": n_files}
        
        print(f"\n--- {split.upper()} ---")
        print(f"Images: {n_images}")
        print(f"Label files: {n_files}")
        print(f"Zero-label files: {len(zero_labels)}")
        print(f"Total bounding boxes: {len(records)}")
        
        # Per-class counts
        class_counts = Counter(r["class_name"] for r in records)
        print(f"\nPer-class bounding box counts:")
        for cls_name in sorted(class_counts.keys()):
            print(f"  {cls_name:<12}: {class_counts[cls_name]}")
            
        # Images per class
        images_per_class = defaultdict(set)
        for r in records:
            images_per_class[r["class_name"]].add(r["file"])
        print(f"\nImages containing each class:")
        for cls_name in sorted(images_per_class.keys()):
            print(f"  {cls_name:<12}: {len(images_per_class[cls_name])}")
            
        # Bbox area statistics
        if records:
            areas = [r["area"] for r in records]
            print(f"\nBounding box area statistics (normalized):")
            print(f"  Mean area:   {np.mean(areas):.6f}")
            print(f"  Median area: {np.median(areas):.6f}")
            print(f"  Min area:    {np.min(areas):.6f}")
            print(f"  Max area:    {np.max(areas):.6f}")
            print(f"  Std area:    {np.std(areas):.6f}")
            
            # Small object analysis
            very_small = sum(1 for a in areas if a < 0.001)  # < 0.1% of image
            small = sum(1 for a in areas if a < 0.005)        # < 0.5% of image
            medium = sum(1 for a in areas if 0.005 <= a < 0.05)
            large = sum(1 for a in areas if a >= 0.05)
            
            print(f"\nBounding box size distribution:")
            print(f"  Very small (< 0.1%): {very_small} ({100*very_small/len(areas):.1f}%)")
            print(f"  Small (< 0.5%):      {small} ({100*small/len(areas):.1f}%)")
            print(f"  Medium (0.5-5%):     {medium} ({100*medium/len(areas):.1f}%)")
            print(f"  Large (>= 5%):       {large} ({100*large/len(areas):.1f}%)")
            
            # Per-class area stats
            print(f"\nAverage bounding box area by class:")
            for cls_name in sorted(class_counts.keys()):
                cls_areas = [r["area"] for r in records if r["class_name"] == cls_name]
                very_small_cls = sum(1 for a in cls_areas if a < 0.001)
                print(f"  {cls_name:<12}: mean={np.mean(cls_areas):.6f}, median={np.median(cls_areas):.6f}, very_small={very_small_cls}/{len(cls_areas)}")
            
        # Crowded images (>= 10 boxes)
        boxes_per_image = Counter(r["file"] for r in records)
        crowded = sum(1 for v in boxes_per_image.values() if v >= 10)
        max_boxes = max(boxes_per_image.values()) if boxes_per_image else 0
        print(f"\nCrowded images (>= 10 boxes): {crowded}")
        print(f"Maximum boxes in single image: {max_boxes}")
    
    # ==============================================================
    # TRAIN/VAL/TEST LEAKAGE CHECK
    # ==============================================================
    print("\n" + "=" * 60)
    print("TRAIN/VAL/TEST LEAKAGE CHECK")
    print("=" * 60)
    
    train_files = set(r["file"] for r in all_records.get("train", []))
    val_files = set(r["file"] for r in all_records.get("val", []))
    test_files = set(r["file"] for r in all_records.get("test", []))
    
    tv_overlap = train_files & val_files
    tt_overlap = train_files & test_files
    vt_overlap = val_files & test_files
    
    print(f"Train-Val overlap: {len(tv_overlap)} files")
    print(f"Train-Test overlap: {len(tt_overlap)} files")
    print(f"Val-Test overlap: {len(vt_overlap)} files")
    
    if tv_overlap:
        print(f"\nWARNING: Train-Val leakage detected! First 10: {list(tv_overlap)[:10]}")
    if tt_overlap:
        print(f"\nWARNING: Train-Test leakage detected! First 10: {list(tt_overlap)[:10]}")
    if vt_overlap:
        print(f"\nWARNING: Val-Test leakage detected! First 10: {list(vt_overlap)[:10]}")
    
    if not (tv_overlap or tt_overlap or vt_overlap):
        print("NO LEAKAGE DETECTED. Splits are clean.")
    
    # ==============================================================
    # CLASS IMBALANCE ANALYSIS
    # ==============================================================
    print("\n" + "=" * 60)
    print("CLASS IMBALANCE ANALYSIS (TRAIN)")
    print("=" * 60)
    
    train_records = all_records.get("train", [])
    if train_records:
        train_class_counts = Counter(r["class_name"] for r in train_records)
        max_count = max(train_class_counts.values())
        min_count = min(train_class_counts.values())
        
        print(f"\nClass distribution in training set:")
        for cls_name in sorted(train_class_counts.keys()):
            count = train_class_counts[cls_name]
            ratio = count / max_count
            bar = "█" * int(ratio * 40)
            print(f"  {cls_name:<12}: {count:>6} {bar} ({ratio:.2f}x)")
        
        print(f"\nImbalance ratio (max/min): {max_count/min_count:.1f}x")
    
    # ==============================================================
    # UNSUPPORTED CLASSES CHECK
    # ==============================================================
    print("\n" + "=" * 60)
    print("UNSUPPORTED CLASSES CHECK")
    print("=" * 60)
    
    all_classes_found = set()
    for split, records in all_records.items():
        for r in records:
            all_classes_found.add(r["class_id"])
    
    supported = {1, 3}  # helmet, vest (used by V2 zone rules)
    used_by_pipeline = {0, 1, 2, 3}  # person, helmet, no_helmet, vest
    unused = all_classes_found - used_by_pipeline
    
    print(f"Classes found in dataset: {sorted(all_classes_found)}")
    print(f"Classes used by V2 pipeline: {sorted(used_by_pipeline)}")
    print(f"Unused classes in dataset: {sorted(unused)}")
    for cls_id in sorted(unused):
        total = sum(1 for split_records in all_records.values() for r in split_records if r["class_id"] == cls_id)
        print(f"  Class {cls_id} ({CLASS_NAMES.get(cls_id, 'unknown')}): {total} total boxes")
    
    # ==============================================================
    # SAVE SUMMARY
    # ==============================================================
    summary = {
        "dataset_path": os.path.abspath(DATASET_ROOT),
        "splits": {}
    }
    
    for split in SPLITS:
        records = all_records.get(split, [])
        class_counts = dict(Counter(r["class_name"] for r in records))
        areas = [r["area"] for r in records]
        
        summary["splits"][split] = {
            "images": all_file_counts.get(split, {}).get("images", 0),
            "label_files": all_file_counts.get(split, {}).get("labels", 0),
            "zero_label_files": len(all_zero_labels.get(split, [])),
            "total_boxes": len(records),
            "class_counts": class_counts,
            "area_stats": {
                "mean": float(np.mean(areas)) if areas else 0,
                "median": float(np.median(areas)) if areas else 0,
                "min": float(np.min(areas)) if areas else 0,
                "max": float(np.max(areas)) if areas else 0
            }
        }
    
    summary["leakage"] = {
        "train_val": len(tv_overlap),
        "train_test": len(tt_overlap),
        "val_test": len(vt_overlap)
    }
    
    with open(os.path.join(OUTPUT_DIR, "dataset_audit.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nDataset audit saved to {OUTPUT_DIR}/dataset_audit.json")
    print("\nDATASET AUDIT COMPLETE.")

if __name__ == "__main__":
    main()
