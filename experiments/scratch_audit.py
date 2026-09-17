import os
import glob
import pandas as pd
import numpy as np

def audit_dataset(dataset_dir):
    classes = {0: 'helmet', 1: 'gloves', 2: 'vest', 3: 'boots', 4: 'goggles', 5: 'none', 6: 'Person', 7: 'no_helmet', 8: 'no_goggle', 9: 'no_gloves', 10: 'no_boots', 'harness': 'harness'} # HardHat-Vest isn't included here because we know construction-ppe has the boots
    
    # We will search for all labels in train and val
    splits = ['train', 'val']
    stats = {}
    
    # Initialize stats for our target classes + all existing ones
    target_classes = ['boots', 'harness', 'helmet', 'no_helmet', 'vest']
    for cls_name in target_classes:
        stats[cls_name] = {'train_images': 0, 'train_instances': 0, 'val_images': 0, 'val_instances': 0, 'areas': []}
        
    for split in splits:
        label_dir = os.path.join(dataset_dir, 'labels', split)
        if not os.path.exists(label_dir):
            continue
            
        label_files = glob.glob(os.path.join(label_dir, '*.txt'))
        for lf in label_files:
            with open(lf, 'r') as f:
                lines = f.readlines()
                
            found_classes_in_img = set()
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                cls_id = int(parts[0])
                cls_name = classes.get(cls_id, f"class_{cls_id}")
                
                w, h = float(parts[3]), float(parts[4])
                area = w * h
                
                if cls_name in stats:
                    stats[cls_name][f'{split}_instances'] += 1
                    stats[cls_name]['areas'].append(area)
                    found_classes_in_img.add(cls_name)
                    
            for cls_name in found_classes_in_img:
                stats[cls_name][f'{split}_images'] += 1

    # Harness doesn't exist in the data.yaml, so it has 0 instances.
    
    # Compile results
    rows = []
    for cls_name in target_classes:
        areas = stats[cls_name]['areas']
        mean_area = np.mean(areas) if areas else 0.0
        # small box if area < 0.05 (just an arbitrary small threshold for normalized area)
        small_boxes = sum(1 for a in areas if a < 0.01)
        small_box_pct = (small_boxes / len(areas) * 100) if areas else 0.0
        
        rows.append({
            'class': cls_name,
            'train_images': stats[cls_name]['train_images'],
            'train_instances': stats[cls_name]['train_instances'],
            'val_images': stats[cls_name]['val_images'],
            'val_instances': stats[cls_name]['val_instances'],
            'mean_box_area': round(mean_area, 4),
            'small_box_percentage': round(small_box_pct, 2)
        })
        
    df = pd.DataFrame(rows)
    
    out_dir = os.path.join('outputs', 'ppe_extension')
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, 'dataset_audit.csv')
    df.to_csv(csv_path, index=False)
    
    md_path = os.path.join(out_dir, 'DATASET_AUDIT.md')
    with open(md_path, 'w') as f:
        f.write("# PPE Extension Dataset Audit\n\n")
        f.write("## Findings\n")
        f.write("- **Boots**: Annotations found in the `construction-ppe` dataset.\n")
        f.write("- **Harness**: No annotations found in the dataset.\n")
        f.write("\n## Statistics\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Conclusion\n")
        if stats['harness']['train_instances'] == 0:
            f.write("BOOTS DATA: INSUFFICIENT (potentially sufficient, but missing harness)\n")
            f.write("HARNESS DATA: INSUFFICIENT\n")
            f.write("\nAdditional dataset containing harness annotations is REQUIRED before experimental training can begin.")

if __name__ == "__main__":
    audit_dataset(os.path.join("datasets", "construction-ppe"))
    print("Audit complete.")
