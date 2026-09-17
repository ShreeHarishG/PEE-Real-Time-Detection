# Troubleshooting Guide

## Common Issues & Fixes

### 1. `ModuleNotFoundError: No module named 'ultralytics'`
**Cause:** The Python virtual environment is not activated, or dependencies were not installed.
**Fix:** Ensure you run `ppe-env\Scripts\activate` (Windows) or `source ppe-env/bin/activate` (Linux) before running the pipeline or dashboard. Run `pip install -r requirements.txt`.

### 2. CUDA Out of Memory (OOM)
**Cause:** The GPU does not have enough VRAM to run both YOLOv8n (Person) and V3-HN (PPE) simultaneously alongside the tracking algorithms.
**Fix:** 
- Reduce the batch size if processing multiple streams.
- Ensure no other heavy applications (like training scripts) are consuming GPU memory.
- If using DeepStream/TensorRT on Jetson, ensure FP16 quantization is enabled to reduce memory footprint.

### 3. Dashboard Video Not Found
**Cause:** The dashboard attempts to read `outputs/results/annotated_output_functional.mp4`, but the file hasn't been generated yet.
**Fix:** Run `python src/pipeline.py` (or `.\scripts\run_pipeline.ps1`) to process the test video first.

### 4. Poor FPS (< 12 FPS)
**Cause:** Running the inference pipeline on CPU instead of GPU, or running an unoptimized PyTorch model on a Jetson edge device.
**Fix:**
- Verify CUDA is being used by checking the console logs for `device: cuda:0`.
- On Jetson, PyTorch `.pt` models will run very slowly. You MUST export the model to ONNX and generate a TensorRT `.engine` file as outlined in `JETSON_DEPLOYMENT.md`.

### 5. No Violations Detected
**Cause:** The zone rules or thresholds might be misconfigured.
**Fix:** 
- Ensure `ACTIVE_ZONE` in `src/pipeline.py` matches a key in `ZONE_RULES` (default is "construction").
- Check that the `CONF_THRESHOLD` is not set too high (default is `0.25`).
- Confirm the `INPUT_VIDEO` actually contains instances of workers without required PPE.
