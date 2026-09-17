#!/bin/bash
# Benchmarks the TensorRT engine on Jetson hardware

ENGINE_PATH="../models/ppe_v3_hn_best.engine"

if [ ! -f "$ENGINE_PATH" ]; then
    echo "Error: Engine file not found at $ENGINE_PATH"
    exit 1
fi

echo "Benchmarking TensorRT Engine..."
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=$ENGINE_PATH \
    --avgRuns=100 \
    --verbose

echo "Ensure tegrastats is running to monitor temperature and power modes."
