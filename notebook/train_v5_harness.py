from ultralytics import YOLO
import os

def main():
    # Load the best V4 model (so it remembers helmet, vest, and boots)
    print("Loading V4-Boots base model...")
    model_path = r"runs\detect\runs\detect\ppe_v4_boots\weights\best.pt"
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. Falling back to yolov8n.pt")
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO(model_path) 
    
    # Path to your new combined dataset YAML
    data_yaml = r"datasets\ppe_v5_combined\data.yaml"
    
    print(f"Starting training on {data_yaml}...")
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=20,              # As requested!
        imgsz=512,              
        batch=16,               
        project="runs/detect",
        name="ppe_v5_harness",
        save=True,
        device=0,           
        patience=20             
    )
    
    # Evaluate model performance on the validation set
    print("Running validation...")
    metrics = model.val()
    print("--------------------------------------------------")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print("--------------------------------------------------")
    print("Training complete! The best model is saved in runs/detect/ppe_v5_harness/weights/best.pt")

if __name__ == "__main__":
    if not os.path.exists("datasets"):
        print("Please run this script from the 'notebook' directory!")
        exit(1)
        
    main()
