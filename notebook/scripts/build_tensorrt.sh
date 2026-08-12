#!/bin/bash
# Builds TensorRT engine from ONNX model on Jetson hardware

MODEL_NAME="ppe_v3_hn_best"
ONNX_PATH="../models/${MODEL_NAME}.onnx"
ENGINE_PATH="../models/${MODEL_NAME}.engine"
WORKSPACE=4096

if [ ! -f "$ONNX_PATH" ]; then
    echo "Error: ONNX file not found at $ONNX_PATH"
    echo "Please run python export_onnx.py first."
    exit 1
fi

echo "Building TensorRT FP16 Engine for $MODEL_NAME..."
# Note: trtexec is usually located at /usr/src/tensorrt/bin/trtexec on Jetson devices
/usr/src/tensorrt/bin/trtexec \
    --onnx=$ONNX_PATH \
    --saveEngine=$ENGINE_PATH \
    --fp16 \
    --workspace=$WORKSPACE \
    --verbose

echo "Engine build complete: $ENGINE_PATH"
echo "NOTE: Physical Jetson testing is PENDING HARDWARE."
