import os

dataset_path = r"w:\3 projects\Building\Tfrenzy\notebook\datasets\ppe_extension_harness"
splits = ["train", "valid", "test"]

print("Fixing dataset labels...")
for split in splits:
    labels_dir = os.path.join(dataset_path, split, "labels")
    if not os.path.exists(labels_dir):
        continue
        
    for filename in os.listdir(labels_dir):
        if not filename.endswith(".txt"):
            continue
            
        filepath = os.path.join(labels_dir, filename)
        with open(filepath, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            class_id = int(parts[0])
            # The Roboflow dataset has 0: human, 1: safety-harness
            # Our Unified classes are: 0: person, 1: helmet, 2: no_helmet, 3: vest, 4: boots, 5: harness
            
            if class_id == 0:
                new_class_id = 0 # human -> person
            elif class_id == 1:
                new_class_id = 5 # safety-harness -> harness
            else:
                new_class_id = class_id
                
            new_line = f"{new_class_id} " + " ".join(parts[1:]) + "\n"
            new_lines.append(new_line)
            
        with open(filepath, "w") as f:
            f.writelines(new_lines)

print("Updating data.yaml...")
yaml_content = """names:
  0: person
  1: helmet
  2: no_helmet
  3: vest
  4: boots
  5: harness
path: W:\\3 projects\\Building\\Tfrenzy\\notebook\\datasets\\ppe_extension_harness
train: train/images
val: valid/images
test: test/images
"""

with open(os.path.join(dataset_path, "data.yaml"), "w") as f:
    f.write(yaml_content)

print("Dataset is now SAFE and ready for training!")
