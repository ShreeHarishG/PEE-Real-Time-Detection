# DeepStream Preparation

## Architecture Objective
If the Python-based Ultralytics inference pipeline proves too slow on the Jetson edge hardware (i.e., fails to maintain the 12 FPS minimum when handling high camera load), the pipeline must be ported to C++ using NVIDIA DeepStream.

## Component Translation
The current Python pipeline components translate to DeepStream GStreamer plugins as follows:

1. **Person Detection (yolov8n):** `nvinfer` (Primary GIE)
2. **Tracking (ByteTrack):** `nvtracker`
3. **PPE Detection (V3-HN):** `nvinfer` (Secondary GIE) operating on cropped objects from the primary detector.
4. **PPE/Person Association (IoA):** This logic is implicitly handled by the metadata flow in `nvinfer` when operating as an SGIE on PGIE crops.
5. **Zone Rules & Temporal Validation:** Requires a custom `nvdsosd` probe or a custom C++ plugin inserted into the GStreamer pipeline before the sink.

## Metadata Flow
The NvDsBatchMeta structure will carry the `NvDsObjectMeta` for the person, and child `NvDsObjectMeta` for the PPE items. The custom temporal validation plugin will read these metadata structures, calculate compliance, and trigger the REST API POST to the backend.

## Status
*Implementation is pending physical hardware benchmarks.*
We will not proactively write C++ DeepStream logic unless the TensorRT Python benchmark proves insufficient.
