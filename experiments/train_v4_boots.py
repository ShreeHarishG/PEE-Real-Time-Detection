from ultralytics import YOLO
import os

def main():
    # Start from a clean YOLOv8 nano model (you can also use yolov8s.pt or yolov8m.pt for higher accuracy)
    print("Loading YOLOv8n base model...")
    model = YOLO("yolov8n.pt") 
    
    # Path to your boots dataset YAML
    data_yaml = r"datasets\ppe_extension_boots\data.yaml"
    
    print(f"Starting training on {data_yaml}...")
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=20,              # Reduced for faster CPU training
        imgsz=512,              # Same image size as previous models for consistency
        batch=16,               # Adjust based on your GPU VRAM
        project="runs/detect",
        name="ppe_v4_boots",
        save=True,
        device="cpu",             # Fallback to CPU since no CUDA device was found
        patience=20             # Early stopping if no improvement after 20 epochs
    )
    
    # Evaluate model performance on the validation set
    print("Running validation...")
    metrics = model.val()
    print("--------------------------------------------------")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print("--------------------------------------------------")
    print("Training complete! The best model is saved in runs/detect/ppe_v4_boots/weights/best.pt")

if __name__ == "__main__":
    # Ensure we are in the notebook directory when running this
    if not os.path.exists("datasets"):
        print("Please run this script from the 'notebook' directory!")
        exit(1)
        
    main()
