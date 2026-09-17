import os
import argparse
from ultralytics import YOLO

def export_to_onnx(model_path, output_dir, imgsz=512):
    print(f"Exporting model {model_path} to ONNX format...")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return False
        
    try:
        model = YOLO(model_path)
        # Exporting in half-precision for Jetson TensorRT compatibility
        success = model.export(format='onnx', imgsz=imgsz, half=True, dynamic=False, simplify=True)
        print(f"Export successful. Saved to {success}")
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export EdgeVision model to ONNX")
    parser.add_argument("--model", type=str, default="models/ppe_v3_hn_best.pt", help="Path to YOLOv8 model")
    parser.add_argument("--imgsz", type=int, default=512, help="Input resolution")
    args = parser.parse_args()
    
    export_to_onnx(args.model, "models/", args.imgsz)
