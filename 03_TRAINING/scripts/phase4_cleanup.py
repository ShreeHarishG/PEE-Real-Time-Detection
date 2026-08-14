import os
import shutil
import json
import glob
import csv

# ==============================================================================
# PHASE 4 PROJECT CLEANUP + V3 STRUCTURE SETUP
# ==============================================================================
# This script:
# 1. Extracts hard-negative images from merged_v21 before deletion
# 2. Removes broken/duplicate directories
# 3. Archives obsolete files
# 4. Creates V3 experiment directory structure
# 5. Creates V2 baseline reference JSON
# 6. Creates project audit CSV outputs
#
# SAFETY: Does NOT delete V2 weights, datasets, outputs, or scripts.
# ==============================================================================

NOTEBOOK_ROOT = "."  # Run from notebook/ directory
OUTPUT_DIR = "outputs/project_audit"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_delete_dir(path, description):
    if os.path.exists(path):
        print(f"  DELETING: {path} ({description})")
        shutil.rmtree(path)
    else:
        print(f"  SKIP: {path} (does not exist)")

def safe_delete_file(path, description):
    if os.path.exists(path):
        print(f"  DELETING: {path} ({description})")
        os.remove(path)
    else:
        print(f"  SKIP: {path} (does not exist)")

def safe_move(src, dst, description):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"  MOVING: {src} -> {dst} ({description})")
        shutil.move(src, dst)
    else:
        print(f"  SKIP: {src} (does not exist)")

# ==============================================================================
# STEP 1: EXTRACT HARD-NEGATIVE IMAGES FROM MERGED_V21
# ==============================================================================
print("=" * 60)
print("STEP 1: EXTRACT HARD-NEGATIVES FROM MERGED_V21")
print("=" * 60)

hn_dest = "edgevision_v2/datasets/hard_negative_frames"
os.makedirs(hn_dest, exist_ok=True)

merged_v21_train = "edgevision_v2/datasets/merged_v21/images/train"
if os.path.exists(merged_v21_train):
    hn_files = glob.glob(os.path.join(merged_v21_train, "hn_*.jpg"))
    print(f"Found {len(hn_files)} hard-negative frames to extract.")
    for f in hn_files:
        dst = os.path.join(hn_dest, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)
    print(f"Extracted {len(hn_files)} files to {hn_dest}")
else:
    print("merged_v21/images/train not found, skipping extraction.")

# ==============================================================================
# STEP 2: DELETE BROKEN/EMPTY DIRECTORIES
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 2: DELETE BROKEN/EMPTY DIRECTORIES")
print("=" * 60)

# Broken merged_v21 duplicate (4.29 GB)
safe_delete_dir("edgevision_v2/datasets/merged_v21", "Broken duplicate dataset (4.29 GB)")

# Empty directories
safe_delete_dir("edgevision_v2/datasets/raw", "Empty placeholder")
safe_delete_dir("edgevision_v2/runs", "Empty V2 runs dir")
safe_delete_dir("edgevision_v1/datasets", "Empty V1 datasets dir")
safe_delete_dir("edgevision_v1/runs", "Empty V1 runs dir")

# Empty checkpoint caches
safe_delete_dir("../.ipynb_checkpoints", "Root-level empty Jupyter cache")
safe_delete_dir("../docs/.ipynb_checkpoints", "Docs empty Jupyter cache")
safe_delete_dir(".ipynb_checkpoints", "Notebook-level Jupyter cache")

# Empty root outputs
if os.path.exists("../outputs") and not os.listdir("../outputs"):
    safe_delete_dir("../outputs", "Empty root outputs")

# ==============================================================================
# STEP 3: DELETE OLD YOLO VALIDATION CACHES
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 3: DELETE OLD YOLO VALIDATION CACHES")
print("=" * 60)

val_dirs = glob.glob("runs/detect/val*")
total_cleaned = 0
for vd in val_dirs:
    if os.path.isdir(vd):
        safe_delete_dir(vd, "Old validation cache")
        total_cleaned += 1
print(f"Cleaned {total_cleaned} old validation directories.")

# ==============================================================================
# STEP 4: ARCHIVE OBSOLETE FILES
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 4: ARCHIVE OBSOLETE FILES")
print("=" * 60)

archive_dir = "archive"
os.makedirs(archive_dir, exist_ok=True)
os.makedirs(os.path.join(archive_dir, "v1"), exist_ok=True)
os.makedirs(os.path.join(archive_dir, "loose_scripts"), exist_ok=True)
os.makedirs(os.path.join(archive_dir, "tracker_configs"), exist_ok=True)
os.makedirs(os.path.join(archive_dir, "models"), exist_ok=True)

# Loose diagnostic/test scripts
loose_scripts = [
    "diagnostic.py", "diagnostic2.py",
    "extract_frames.py", "prepare_train_eval.py",
    "patch_notebook.py", "patch_notebook_step8.py",
    "test_bench_after_track.py", "test_cpu.py",
    "test_pipeline_notrack.py", "test_pipeline_perf.py",
    "test_tracker.py",
    "run_final_bench.py", "run_validation.py",
    "calibration_sweep.py"
]

for script in loose_scripts:
    safe_move(script, os.path.join(archive_dir, "loose_scripts", script), "Superseded by Phase 3 scripts")

# Old tracker configs
for yaml_file in glob.glob("custom_track_*.yaml"):
    safe_move(yaml_file, os.path.join(archive_dir, "tracker_configs", os.path.basename(yaml_file)), "Phase 2 experiment")

safe_move("edgevision_v2/botsort_reid.yaml", os.path.join(archive_dir, "tracker_configs", "botsort_reid.yaml"), "ReID rejected")

# Unused model weights
safe_move("yolo26n.pt", os.path.join(archive_dir, "models", "yolo26n.pt"), "Unused model")
safe_move("yolov8s.pt", os.path.join(archive_dir, "models", "yolov8s.pt"), "Unused model (may retrieve for V3 experiments)")

# V1 notebook (keep one copy in docs/notebook/)
safe_move("EdgeVision_V1_Full_Pipeline.ipynb", os.path.join(archive_dir, "v1", "EdgeVision_V1_Full_Pipeline.ipynb"), "V1 superseded by V2")

# V1 outputs
if os.path.exists("edgevision_v1/outputs"):
    safe_move("edgevision_v1/outputs", os.path.join(archive_dir, "v1", "outputs"), "V1 outputs")

# V2 training run artifacts (keep for reproducibility)
if os.path.exists("runs/detect/edgevision_v2"):
    safe_move("runs/detect/edgevision_v2", os.path.join(archive_dir, "v2_training_run"), "V2 training artifacts")

# Clean up empty edgevision_v1 if fully emptied
if os.path.exists("edgevision_v1") and not os.listdir("edgevision_v1"):
    safe_delete_dir("edgevision_v1", "Empty V1 directory after archiving")

# Clean up runs/detect if empty
if os.path.exists("runs/detect") and not os.listdir("runs/detect"):
    safe_delete_dir("runs/detect", "Empty detect dir after cleanup")
if os.path.exists("runs") and not os.listdir("runs"):
    safe_delete_dir("runs", "Empty runs dir after cleanup")

# ==============================================================================
# STEP 5: CREATE V3 EXPERIMENT STRUCTURE
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 5: CREATE V3 EXPERIMENT STRUCTURE")
print("=" * 60)

v3_dirs = [
    "experiments/v3",
    "outputs/v3",
    "edgevision_v2/datasets/hard_negative_frames",
]
for d in v3_dirs:
    os.makedirs(d, exist_ok=True)
    print(f"  Created: {d}")

# ==============================================================================
# STEP 6: CREATE V2 BASELINE REFERENCE
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 6: CREATE V2 BASELINE REFERENCE")
print("=" * 60)

v2_baseline = {
    "model": "YOLOv8n",
    "weights": "edgevision_v2/models/ppe_best.pt",
    "dataset": "edgevision_v2/datasets/merged/",
    "data_yaml": "edgevision_v2/datasets/merged/data.yaml",
    "imgsz": 512,
    "epochs": 30,
    "precision": "FP32",
    "gpu": "RTX 4050",
    "person_detector": "yolov8n.pt",
    "tracker": "bytetrack.yaml",
    "reid": False,
    "metrics": {
        "mAP50": 0.844,
        "mAP50_95": 0.501
    },
    "thresholds": {
        "HELMET_CONF": 0.65,
        "VEST_CONF": 0.45,
        "MIN_PERSON_CONF": 0.35,
        "MIN_ASSOC_SCORE": 0.40
    },
    "person_filter": {
        "MIN_PERSON_WIDTH": 25,
        "MIN_PERSON_HEIGHT": 50,
        "MIN_PERSON_AREA": 1500,
        "MIN_PERSON_ASPECT": 0.15,
        "MAX_PERSON_ASPECT": 1.5,
        "MIN_TRACK_STABILITY": 5
    },
    "association_weights": {
        "W_IOA": 0.40,
        "W_SPATIAL": 0.30,
        "W_CENTER_DIST": 0.30
    },
    "temporal_validator": {
        "window_size": 10,
        "violation_threshold": 8,
        "min_frames": 50,
        "requires_current_violation": True
    },
    "zone_rules": {
        "construction": {"required": ["helmet", "vest"]}
    },
    "real_world_performance": {
        "phase3d_negative_test": {
            "frames": 221,
            "confirmed_violations": 15,
            "avg_fps": 26.56,
            "p95_latency_ms": 41.15
        },
        "phase3e_positive_test": {
            "frames": 400,
            "unique_people": 63,
            "people_with_both": 55,
            "confirmed_violations": 13,
            "helmet_detection_rate": 0.8987,
            "vest_detection_rate": 0.8263,
            "avg_fps": 26.13,
            "warm_fps": 30.30
        },
        "phase3i_final": {
            "ppe_associations": 1036,
            "association_failures": 59,
            "cross_assignment_candidates": 14,
            "confirmed_violations": 14,
            "warm_fps": 22.35,
            "p95_latency_ms": 55.64
        }
    }
}

with open("experiments/v3/v2_baseline.json", "w") as f:
    json.dump(v2_baseline, f, indent=4)
print("  Created: experiments/v3/v2_baseline.json")

# Create model comparison CSV template
with open("outputs/v3/model_comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "experiment", "model", "dataset", "imgsz", "epochs",
        "precision", "recall", "mAP50", "mAP50_95",
        "helmet_precision", "helmet_recall", "vest_precision", "vest_recall",
        "helmet_false_positives", "vest_false_positives",
        "avg_fps", "warm_fps", "p95_latency", "VRAM_GB",
        "model_path", "status"
    ])
    # V2 baseline row
    writer.writerow([
        "exp0_v2_baseline", "YOLOv8n", "merged", 512, 30,
        "", "", 0.844, 0.501,
        "", "", "", "",
        1476, 137,
        26.56, 22.35, 55.64, "",
        "edgevision_v2/models/ppe_best.pt", "V2_BASELINE"
    ])
print("  Created: outputs/v3/model_comparison.csv")

# ==============================================================================
# STEP 7: GENERATE PROJECT AUDIT CSVs
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 7: GENERATE PROJECT AUDIT CSVs")
print("=" * 60)

# Model inventory
with open(os.path.join(OUTPUT_DIR, "model_inventory.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "size_MB", "classification", "notes"])
    writer.writerow(["edgevision_v2/models/ppe_best.pt", 6.2, "V2_BASELINE", "Trained YOLOv8n PPE detector"])
    writer.writerow(["yolov8n.pt", 6.5, "V2_BASELINE", "Pretrained person detector"])
    writer.writerow(["archive/models/yolov8s.pt", 22.6, "ARCHIVED", "Unused larger model"])
    writer.writerow(["archive/models/yolo26n.pt", 5.5, "ARCHIVED", "Unused YOLO26 model"])
print("  Created: model_inventory.csv")

# Dataset inventory
with open(os.path.join(OUTPUT_DIR, "dataset_inventory.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "size_GB", "images", "classification", "notes"])
    writer.writerow(["edgevision_v2/datasets/merged/", 4.29, 22354, "V2_BASELINE", "Training dataset"])
    writer.writerow(["datasets/HardHat-Vest/", 4.27, 22141, "V2_BASELINE", "Raw source dataset"])
    writer.writerow(["datasets/construction-ppe/", 0.17, 1416, "V2_BASELINE", "Raw source dataset"])
    writer.writerow(["edgevision_v2/datasets/hard_negative_frames/", 0, 73, "V3_EXPERIMENT", "Extracted from merged_v21"])
    writer.writerow(["harness-1/", 0.01, 233, "ARCHIVED", "Incompatible class mapping"])
    writer.writerow(["fp_crops/", 0, 1540, "V2_BASELINE", "False positive crops for analysis"])
print("  Created: dataset_inventory.csv")

print("\n" + "=" * 60)
print("PHASE 4 CLEANUP COMPLETE")
print("=" * 60)
print("Next step: Run phase4_dataset_audit.py for detailed label analysis.")
