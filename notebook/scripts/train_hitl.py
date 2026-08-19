import os
import sys
import shutil
import uuid
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ultralytics import YOLO

# Add parent dir to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from app import models

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://edgevision:edgevision_password@localhost:5432/edgevision")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "hitl_dataset")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8n-ppe-v5-harness.pt") # Base model

# Fallback to YOLO base if PPE model not found
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolov8n.pt"

# Unified classes
CLASSES = {
    0: "person",
    1: "helmet",
    2: "no_helmet",
    3: "vest",
    4: "boots",
    5: "harness"
}
CLASS_NAME_TO_ID = {v: k for k, v in CLASSES.items()}

def create_yolo_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "train", "labels"), exist_ok=True)

def export_hitl_data():
    print(f"Connecting to {DATABASE_URL}...")
    db = SessionLocal()
    
    # Get all acknowledged violations with feedback
    events = db.query(models.ViolationEvent).filter(
        models.ViolationEvent.feedback_correct.isnot(None)
    ).all()
    
    if not events:
        print("No HITL feedback found in the database. Cannot train.")
        db.close()
        return False
        
    print(f"Found {len(events)} HITL events. Extracting and pseudo-labeling...")
    
    create_yolo_dir(DATASET_DIR)
    
    # Load model for pseudo-labeling
    print(f"Loading {MODEL_PATH} for pseudo-labeling...")
    model = YOLO(MODEL_PATH)
    
    processed = 0
    for event in events:
        if not event.evidence_image_path:
            continue
            
        img_path = os.path.join(PROJECT_ROOT, event.evidence_image_path)
        if not os.path.exists(img_path):
            continue
            
        # Run inference to get pseudo-labels
        results = model(img_path, verbose=False)[0]
        boxes = results.boxes.xywhn.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        
        # We will build a new set of labels based on feedback
        final_labels = []
        
        # 1. Always add the person box (the crop is the person, so we use a central box)
        # We use x=0.5, y=0.5, w=0.9, h=0.9 to represent the person in the crop
        final_labels.append(f"0 0.5 0.5 0.9 0.9\n")
        
        # Helper to check if model detected a class
        def get_model_box(target_class_id):
            for i, cls_id in enumerate(classes):
                if int(cls_id) == target_class_id:
                    return boxes[i]
            return None
            
        # Helper to apply HITL logic
        def apply_feedback(item_name, feedback_val, pos_class_id, neg_class_id, heuristic_box):
            if feedback_val is True:
                # User says item IS present
                box = get_model_box(pos_class_id)
                if box is not None:
                    final_labels.append(f"{pos_class_id} {' '.join(map(str, box))}\n")
                else:
                    # Model missed it. Inject heuristic box!
                    final_labels.append(f"{pos_class_id} {' '.join(map(str, heuristic_box))}\n")
            elif feedback_val is False:
                # User says item is NOT present (use negative class if available)
                if neg_class_id is not None:
                    box = get_model_box(neg_class_id)
                    if box is not None:
                        final_labels.append(f"{neg_class_id} {' '.join(map(str, box))}\n")
                    else:
                        # Model didn't detect absence, but user says it's missing.
                        # We use the heuristic box for the negative class!
                        final_labels.append(f"{neg_class_id} {' '.join(map(str, heuristic_box))}\n")

        # Apply for each PPE type using sensible heuristic boxes for missing coordinates
        # Format: (item, feedback, pos_id, neg_id, [x_center, y_center, width, height])
        apply_feedback("helmet", event.feedback_helmet, 1, 2, [0.5, 0.15, 0.3, 0.25])
        apply_feedback("vest", event.feedback_vest, 3, None, [0.5, 0.45, 0.7, 0.5])
        apply_feedback("boots", event.feedback_boots, 4, None, [0.5, 0.9, 0.5, 0.2])
        apply_feedback("harness", event.feedback_harness, 5, None, [0.5, 0.5, 0.6, 0.6])
        
        # Copy image to dataset
        file_id = f"hitl_{event.id}_{uuid.uuid4().hex[:6]}"
        new_img_path = os.path.join(DATASET_DIR, "train", "images", f"{file_id}.jpg")
        shutil.copy2(img_path, new_img_path)
        
        # Write labels
        label_path = os.path.join(DATASET_DIR, "train", "labels", f"{file_id}.txt")
        with open(label_path, "w") as f:
            f.writelines(final_labels)
            
        processed += 1
        
    db.close()
    
    if processed == 0:
        print("No valid image files found for the HITL events.")
        return False
        
    print(f"Successfully generated YOLO dataset with {processed} pseudo-labeled HITL images.")
    
    # Create data.yaml
    yaml_content = {
        "path": DATASET_DIR,
        "train": "train/images",
        "val": "train/images", # Use train for val just to satisfy YOLO requirements for fine-tuning
        "names": CLASSES
    }
    with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
        yaml.dump(yaml_content, f)
        
    return True

def train_hitl():
    print("\n--- Starting YOLOv8 Training ---")
    model = YOLO(MODEL_PATH)
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    
    # Train using the local GPU
    print(f"Training on dataset: {yaml_path}")
    results = model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        device=0, # Use local GPU as requested
        batch=16,
        name="yolov8n-ppe-hitl-v6"
    )
    
    print("\n✅ Training Complete!")
    print("New model saved at: runs/detect/yolov8n-ppe-hitl-v6/weights/best.pt")

if __name__ == "__main__":
    success = export_hitl_data()
    if success:
        train_hitl()
