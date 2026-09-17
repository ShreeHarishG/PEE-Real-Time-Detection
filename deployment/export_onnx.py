import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX for TensorRT deployment")
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLOv8 PyTorch weights (.pt)")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size for inference")
    parser.add_argument("--half", action="store_true", help="Export in FP16 precision")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found at {args.weights}")
        return

    print(f"Loading YOLOv8 model from {args.weights}...")
    model = YOLO(args.weights)
    
    print(f"Exporting to ONNX format (imgsz={args.imgsz}, half={args.half})...")
    # Export the model
    # Note: Simplify=True is recommended for TensorRT, but requires onnxsim package
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        simplify=True,
        opset=12,
        dynamic=False
    )
    
    print(f"Successfully exported ONNX model to: {onnx_path}")
    print("\nNext step:")
    print(f"Copy the {onnx_path} file to your Jetson device to build the TensorRT engine.")

if __name__ == "__main__":
    main()
