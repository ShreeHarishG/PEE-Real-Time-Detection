import os
import json
import time
import torch
import ultralytics
from ultralytics import YOLO

# ==============================================================================
# V3 EXPERIMENT 1: HARD-NEGATIVE TRAINING
# ==============================================================================
# Goal: Train a model (V3-HN) with the identical configuration as V2, but on the 
# new v3_hn dataset which includes 73 empty-label hard-negative frames.
# ==============================================================================

# Configuration matches V2 Baseline strictly
MODEL_ARCH = "yolov8n.pt"  # Pretrained weights
DATA_YAML = "edgevision_v2/datasets/v3_hn/data.yaml"
IMGSZ = 512
EPOCHS = 30
PROJECT = "experiments/v3"
NAME = "exp1_hard_negative"

os.makedirs(os.path.join(PROJECT, NAME), exist_ok=True)

def main():
    print("=" * 60)
    print(f"STARTING V3 EXPERIMENT 1: HARD-NEGATIVE TRAINING")
    print("=" * 60)
    print(f"Architecture: {MODEL_ARCH}")
    print(f"Dataset:      {DATA_YAML}")
    print(f"Resolution:   {IMGSZ}")
    print(f"Epochs:       {EPOCHS}")
    print("=" * 60)

    # 1. Initialize model
    model = YOLO(MODEL_ARCH)
    
    # 2. Record hyperparameters and environment
    start_time = time.time()
    env_info = {
        "model_architecture": MODEL_ARCH,
        "dataset_yaml": DATA_YAML,
        "imgsz": IMGSZ,
        "epochs": EPOCHS,
        "batch_size": 16, # Default
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "pytorch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    
    with open(os.path.join(PROJECT, NAME, "training_config.json"), "w") as f:
        json.dump(env_info, f, indent=4)
        
    print("\nTraining Configuration Saved. Starting Ultralytics Training...")
    
    # 3. Train
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        project=PROJECT,
        name=NAME,
        batch=16,
        seed=42, # Fixed seed for reproducibility
        exist_ok=True,
        verbose=True
    )
    
    # 4. Finalize
    end_time = time.time()
    duration = end_time - start_time
    
    env_info["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    env_info["training_duration_seconds"] = round(duration, 2)
    env_info["final_model_path"] = os.path.join(PROJECT, NAME, "weights", "best.pt")
    
    with open(os.path.join(PROJECT, NAME, "training_config.json"), "w") as f:
        json.dump(env_info, f, indent=4)
        
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE IN {duration/60:.2f} MINUTES")
    print(f"Best model saved to: {env_info['final_model_path']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
