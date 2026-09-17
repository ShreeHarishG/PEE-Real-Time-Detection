import os
import shutil

boots_dir = r"w:\3 projects\Building\Tfrenzy\notebook\datasets\ppe_extension_boots"
harness_dir = r"w:\3 projects\Building\Tfrenzy\notebook\datasets\ppe_extension_harness"
combined_dir = r"w:\3 projects\Building\Tfrenzy\notebook\datasets\ppe_v5_combined"

print("Creating combined dataset directory...")
os.makedirs(combined_dir, exist_ok=True)
os.makedirs(os.path.join(combined_dir, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(combined_dir, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(combined_dir, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(combined_dir, "labels", "val"), exist_ok=True)

def copy_data(src_images, src_labels, split, prefix):
    if not os.path.exists(src_images) or not os.path.exists(src_labels):
        return
        
    for filename in os.listdir(src_images):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Copy image
            src_img_path = os.path.join(src_images, filename)
            dst_img_path = os.path.join(combined_dir, "images", split, f"{prefix}_{filename}")
            shutil.copy2(src_img_path, dst_img_path)
            
            # Copy corresponding label
            label_filename = filename.rsplit('.', 1)[0] + '.txt'
            src_lbl_path = os.path.join(src_labels, label_filename)
            dst_lbl_path = os.path.join(combined_dir, "labels", split, f"{prefix}_{label_filename}")
            
            if os.path.exists(src_lbl_path):
                shutil.copy2(src_lbl_path, dst_lbl_path)

print("Copying Boots dataset...")
copy_data(os.path.join(boots_dir, "images", "train"), os.path.join(boots_dir, "labels", "train"), "train", "b")
copy_data(os.path.join(boots_dir, "images", "val"), os.path.join(boots_dir, "labels", "val"), "val", "b")

print("Copying Harness dataset...")
# Note: Harness dataset uses 'valid' instead of 'val', and train/images instead of images/train
copy_data(os.path.join(harness_dir, "train", "images"), os.path.join(harness_dir, "train", "labels"), "train", "h")
copy_data(os.path.join(harness_dir, "valid", "images"), os.path.join(harness_dir, "valid", "labels"), "val", "h")

print("Writing combined data.yaml...")
yaml_content = """names:
  0: person
  1: helmet
  2: no_helmet
  3: vest
  4: boots
  5: harness
path: W:\\3 projects\\Building\\Tfrenzy\\notebook\\datasets\\ppe_v5_combined
train: images/train
val: images/val
"""
with open(os.path.join(combined_dir, "data.yaml"), "w") as f:
    f.write(yaml_content)

print("Datasets successfully merged into ppe_v5_combined!")
