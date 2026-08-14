# EdgeVision PPE Compliance Platform - Final Report Source

## 1. Executive Summary
The EdgeVision PPE Compliance Platform is an edge-deployed computer vision system designed to enforce industrial safety protocols in real-time. It monitors restricted zones, verifies required PPE (helmets and vests), and performs temporal validation to ensure high precision in violation alerting.

This final report details the successful promotion of the V3-HN YOLOv8 model, which utilizes a hard-negative mining strategy to eliminate false positives seen in early prototypes.

## 2. System Architecture
EdgeVision utilizes a modular microservices architecture designed for real-time edge processing.
- Camera -> Person Detection -> Tracking -> PPE Detection -> Zone Engine -> Temporal Validator -> DB.

## 3. Dataset & Hard Negatives
The system is trained on the Construction-PPE dataset.
To resolve false positives (e.g., detecting yellow pipes as helmets), the V3-HN dataset was enriched with visually similar background images (hard negatives) containing no labels.

## 4. Model Development & Architecture
The final production model (V3-HN) architecture exactly matches the validated .pt metadata:
- Base Architecture: YOLOv8n (Nano)
- Layers: 130
- Parameters: 3,012,018
- Gradients: 0
- GFLOPs: 8.2
- Classes: 6 (person, helmet, no_helmet, vest, boots, harness)

**Model Trade-off Decision:**
V3-HN trades a modest reduction in helmet/vest recall for substantially improved real-world false-positive behavior and improved no-helmet recall.

## 5. Model Decision Table

| Metric | V2 | V3-HN | V3-BOOTS |
|---|---|---|---|
| Role | Rollback | Production | Experimental |
| mAP50 | 84.45% | 84.20% | 80.10% |
| Helmet AP50 | 90.15% | 88.65% | 82.20% |
| Vest AP50 | 82.37% | 79.77% | 86.10% |
| Boots AP50 | N/A | N/A | 80.10% |
| Boots Recall | N/A | N/A | 75.30% |
| Warm FPS | 24.3 | 16.2 | 27.82 |
| Helmet FP | 154 | 0 | N/A |
| Vest FP | 10 | 0 | N/A |

**V3-BOOTS is experimental only and is NOT the production model. V3-HN remains production.**

## 6. Jetson Claims & Performance
- RTX Development Benchmark: 16.2 FPS
- Jetson Hardware Benchmark: PENDING
*Note: RTX FPS does not prove target hardware performance.*

## 7. Deployment Status
- ONNX Export Script: READY
- ONNX Artifact Validation: PENDING
- TensorRT Instructions: READY
- TensorRT Engine Validation: PENDING
- Jetson Physical Validation: PENDING

## 8. Web Application & Rollback
*Note: Any Dashboard values (e.g., 2.8 FPS, 19 violations) are UI demonstration data. Not production benchmark results.*

**Rollback Path:** `02_MODELS/rollback/ppe_v2_backup.pt`
