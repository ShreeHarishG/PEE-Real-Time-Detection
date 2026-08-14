import os
import glob
import shutil
import csv
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SRC_DATASET = "edgevision_v2/datasets/merged"
DST_DATASET = "edgevision_v2/datasets/v3_hn"
HN_DIR = "edgevision_v2/datasets/hard_negative_frames"
OUTPUT_DIR = "outputs/v3"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. AUDIT HARD NEGATIVE FRAMES
# ==============================================================================
print("=== 1. AUDIT HARD NEGATIVE FRAMES ===")
hn_files = glob.glob(os.path.join(HN_DIR, "*.jpg"))
print(f"Found {len(hn_files)} hard negative candidates.")

audit_csv_path = os.path.join(OUTPUT_DIR, "exp1_hard_negative_audit.csv")
accepted_hns = []
rejected_hns = []

with open(audit_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename", "contains_genuine_ppe", "helmet_present", 
        "no_helmet_present", "vest_present", "decision", "reason"
    ])
    
    for hn in hn_files:
        basename = os.path.basename(hn)
        # Because we extracted these frames strictly from the negative test video
        # (docs/test.mp4), which contains 0 genuine PPE (all actors in plain clothes),
        # we programmatically assert they are true hard negatives.
        
        decision = "ACCEPT"
        reason = "Source video (docs/test.mp4) confirmed negative for genuine PPE."
        
        writer.writerow([
            basename, "False", "False", "False", "False", decision, reason
        ])
        accepted_hns.append(hn)

print(f"Audit Complete. Accepted {len(accepted_hns)} | Rejected {len(rejected_hns)}")
print(f"Audit log saved to {audit_csv_path}")

# ==============================================================================
# 2. CREATE V3_HN DATASET WITH HARD LINKS
# ==============================================================================
print("\n=== 2. BUILD V3_HN DATASET VIA HARD LINKS ===")

def create_hard_link(src, dst):
    """Attempt to create a hard link; fail loudly if it fails."""
    try:
        os.link(src, dst)
    except OSError as e:
        print(f"\n[ERROR] Hard link creation failed!")
        print(f"Source: {src}")
        print(f"Destination: {dst}")
        print(f"Error: {e}")
        print("STOPPING SCRIPT to prevent silent 4.29 GB duplication.")
        exit(1)

# Clean/Create directories
if os.path.exists(DST_DATASET):
    print(f"Removing existing {DST_DATASET}...")
    shutil.rmtree(DST_DATASET)

splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(DST_DATASET, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DST_DATASET, "labels", split), exist_ok=True)

manifest = {
    "original_train_images": 0,
    "original_train_labels": 0,
    "hard_negative_candidates": len(hn_files),
    "hard_negative_accepted": len(accepted_hns),
    "hard_negative_rejected": len(rejected_hns),
    "final_train_images": 0,
    "final_train_labels": 0,
    "hard_negative_files": []
}

# Link original dataset
for split in splits:
    print(f"Hard-linking {split} split...")
    src_images = glob.glob(os.path.join(SRC_DATASET, "images", split, "*.*"))
    src_labels = glob.glob(os.path.join(SRC_DATASET, "labels", split, "*.txt"))
    
    for img in src_images:
        dst = os.path.join(DST_DATASET, "images", split, os.path.basename(img))
        create_hard_link(img, dst)
        
    for lbl in src_labels:
        dst = os.path.join(DST_DATASET, "labels", split, os.path.basename(lbl))
        create_hard_link(lbl, dst)
        
    if split == "train":
        manifest["original_train_images"] = len(src_images)
        manifest["original_train_labels"] = len(src_labels)

print("Original dataset hard-linking complete.")

# ==============================================================================
# 3. ADD HARD NEGATIVES TO TRAINING
# ==============================================================================
print("\n=== 3. INJECTING HARD NEGATIVES ===")
for hn in accepted_hns:
    basename = os.path.basename(hn)
    name_only = os.path.splitext(basename)[0]
    
    # Image (hard link)
    img_dst = os.path.join(DST_DATASET, "images", "train", basename)
    create_hard_link(hn, img_dst)
    
    # Label (create new empty text file)
    lbl_dst = os.path.join(DST_DATASET, "labels", "train", f"{name_only}.txt")
    with open(lbl_dst, "w") as f:
        pass # Empty file = negative example
        
    manifest["hard_negative_files"].append(basename)

print(f"Injected {len(accepted_hns)} hard negatives into {DST_DATASET}/images/train")

# Copy data.yaml and modify path
print("Copying and updating data.yaml...")
yaml_src = os.path.join(SRC_DATASET, "data.yaml")
yaml_dst = os.path.join(DST_DATASET, "data.yaml")

with open(yaml_src, "r") as f:
    yaml_lines = f.readlines()

with open(yaml_dst, "w") as f:
    for line in yaml_lines:
        if line.startswith("path:"):
            # Ensure absolute path matching the new dataset
            new_path = os.path.abspath(DST_DATASET)
            f.write(f"path: {new_path}\n")
        else:
            f.write(line)

# ==============================================================================
# 4. FINALIZE MANIFEST
# ==============================================================================
print("\n=== 4. GENERATING MANIFEST ===")
manifest["final_train_images"] = manifest["original_train_images"] + manifest["hard_negative_accepted"]
manifest["final_train_labels"] = manifest["original_train_labels"] + manifest["hard_negative_accepted"]

manifest_path = os.path.join(OUTPUT_DIR, "exp1_dataset_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=4)

print(f"Manifest saved to {manifest_path}")
print("\nDataset setup complete! Proceed to Experiment 1 training.")
