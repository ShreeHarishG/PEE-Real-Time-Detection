# EdgeVision Performance Benchmarks

*Note: This report reflects the benchmark of the EdgeVision V3 pipeline (`ppe_v3_hn_best.pt`) on the target edge hardware.*

## Target Hardware
- **Device**: NVIDIA Jetson Orin Nano (8GB)
- **Power Mode**: MAXN (15W)
- **Deployment**: TensorRT FP16 (DeepStream 6.3)

---

## Model Accuracy Evaluation

| Metric | Score | Purpose |
|--------|-------|---------|
| **mAP50** | 0.892 | Overall object-detection accuracy across all classes. |
| **mAP50-95** | 0.674 | Strict overall accuracy metric. |
| **Violation Precision** | 0.941 | Ratio of true violations over all generated violation alerts. |

### Recall per PPE Class
Measures the percentage of actual PPE successfully detected.
- `person`: 0.98
- `helmet`: 0.95
- `vest`: 0.96
- `boots`: 0.82 *(Requires further small-object tuning)*
- `harness`: 0.79 *(Requires more dataset augmentation)*

---

## Throughput & Latency

### FPS (Real-time performance)
- **Target**: 20 FPS
- **Actual (Single 1080p Stream)**: 28 FPS
- **Actual (Dual 1080p Streams)**: 14 FPS per stream

### Inference Latency (P95)
- **Target**: < 50ms
- **Actual**: 36ms per frame (TensorRT FP16)

---

## Resource Utilisation

- **GPU Usage**: 65% - 85% sustained
- **CPU Usage**: 30% (Data ingestion & rule engine)
- **Memory**: 3.2 GB / 8.0 GB
- **Temperature**: Sustained at 62°C (Active cooling on)
- **False Alerts per Hour**: 0.4 (Successfully mitigated using Temporal Validation window: 8/10 frames)
