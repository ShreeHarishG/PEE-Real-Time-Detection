# EdgeVision Internal Report Data

A. Exact verified project title: EdgeVision PPE Compliance and Work-at-Height Safety Platform
B. Problem statement: Manual industrial safety monitoring is prone to human error and delayed detection, leading to missed compliance violations for PPE and restricted zones.
C. Objectives: Real-time worker tracking, PPE detection, zone awareness, temporal validation to avoid alert fatigue, and database logging with a Web Dashboard.
D. System requirements: See Objectives + 12 FPS minimum deployment performance.
E. AI/ML approach: YOLOv8n for Person Tracking via ByteTrack, YOLOv8s V3-HN for PPE Detection, Spatial Association, Temporal Validator.
F. Dataset information: Construction-PPE with hard-negative mining.
G. Dataset statistics:
- Train: 17,452 images, 148,487 boxes (43905 helmet, 6326 vest, 98256 no_helmet)
- Val: 2,438 images, 21,032 boxes (13576 no_helmet, 6586 helmet, 870 vest)
- Test: 2,464 images, 20,193 boxes (6749 helmet, 935 vest, 12509 no_helmet)
H. Classes: person, helmet, vest, no_helmet (boots and harness unsupported).
I. Model versions: V2, V3-HN (Production), V3-BOOTS (Experimental).
J. V2 baseline:
- mAP50: 84.45%
- mAP50-95: 50.12%
- Helmet Precision: 91.16%, Recall: 84.56%, AP50: 90.15%
- Vest Precision: 78.34%, Recall: 76.67%, AP50: 82.37%
- Warm FPS: 24.3
- False Positives: 154 (Helmet), 10 (Vest)
K. V3 production model: V3-HN
- mAP50: 84.20%
- mAP50-95: 48.77%
- Helmet Precision: 91.07%, Recall: 82.33%, AP50: 88.65%
- Vest Precision: 78.86%, Recall: 73.76%, AP50: 79.77%
- Warm FPS: 16.2
- P95 Latency: 134.63ms
- False Positives: 0 (Helmet), 0 (Vest)
- Confirmed Violations: 14
L. V3 hard-negative model information: Background images containing visually similar objects added as empty labels to suppress false positives.
M. Training configuration: YOLOv8s, imgsz=512, FP16 precision on Dev Workstation.
N. Evaluation metrics: V2 vs V3-HN metrics extracted from `final_model_comparison.csv`.
O. Accuracy metrics: See J and K.
P. Precision: See J and K.
Q. Recall: See J and K.
R. mAP50: See J and K.
S. mAP50-95: See J and K.
T. Confusion matrix information: Verified false positive elimination.
U. Training curves: N/A, using final validation metrics.
V. FPS: 16.2 (V3-HN), 24.3 (V2).
W. latency: P95 134.63ms (V3-HN).
X. memory: Not explicitly benchmarked, Jetson pending.
Y. hardware: RTX GPU Workstation (Development), Jetson Orin Nano / NX (Target - Pending).
Z. database architecture: PostgreSQL schema with cameras, zones, violation_events, inference_metrics tables.
AA. API architecture: FastAPI REST endpoints for zones, cameras, upload, jobs.
AB. frontend architecture: Next.js Web Dashboard.
AC. inference pipeline: RTSP -> YOLOv8n (ByteTrack) -> YOLOv8s (PPE) -> Spatial Association -> Temporal Rules.
AD. person-to-PPE association: Bounding box overlap / spatial containment.
AE. rule engine: Zone-specific PPE requirements.
AF. temporal validation: Hysteresis/persistence tracking to ensure violations are sustained before firing alerts.
AG. violation generation: DB insertion with evidence links.
AH. evidence capture: Image snapshots saved to `outputs/evidence/`.
AI. deployment architecture: ONNX export, TensorRT FP16 engine, DeepStream integration.
AJ. ONNX/TensorRT pipeline: `export_onnx.py` script provided, execution on target device.
AK. Jetson status: Pending Target Hardware Validation.
AL. testing results: Pytest 100% passing, NPM Build passing.
AM. Docker/PostgreSQL verification: PASS.
AN. API verification: PASS.
AO. known limitations: Jetson pending, Boots/Harness unsupported due to data constraints, slight recall regression for false-positive trade-off.
AP. future improvements: Jetson testing, multi-camera scaling, advanced tracking, harness dataset collection.
AQ. final QA status: READY.
