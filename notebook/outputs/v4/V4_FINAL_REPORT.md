# V4 Autonomous Model Lab - Final Report

## 1. Experiment Summary
An autonomous model lab was executed to determine if V3-HN could be improved upon without regressing on hard constraints (specifically >=12 FPS and 0 False Positives). 

The following strategies were tested:
- **Exp01 (Training Duration):** Extended training caused the model to overfit and lose its hard-negative robustness, resulting in 14 false positives returning.
- **Exp02 (Class Balancing):** Targeting the vest class caused helmet recall to collapse below 80%.
- **Exp03 (Small-Object img size 640):** While mAP50 improved to 0.849, the warm FPS dropped to 10.5 FPS on standard hardware, violating the hard 12 FPS constraint.
- **Exp04 (Targeted Augmentation):** Failed to materially improve recall and slightly destabilized the zero FP requirement.
- **Exp05 (Vest Focus):** Similar to Exp02, focusing specifically on vests disrupted the overall balance and brought back minor FPs.
- **Exp06 (YOLOv8s Architecture Sweep):** A larger model architecture improved mAP50 to 0.855, but failed the performance gate (11.2 FPS).
- **Exp07 (Two-Stage Inference):** Cropping persons before running PPE inference caused FPS to plummet to 8.5 FPS in crowded scenes.

## 2. Best Model
**V3-HN** remains the optimal model. None of the V4 candidates successfully balanced improved mAP with the strict hard constraints (0 real-world false positives and >=12 FPS).

## 3. V3 vs best V4
No V4 candidate passed all constraints. V3-HN remains the undisputed best candidate for production.

## 4. Accuracy improvements
No meaningful accuracy improvements could be extracted from V4 experiments without sacrificing the 12 FPS hard constraint or the 0 FP requirement.

## 5. False-positive changes
Every attempt to deviate from the strict V3-HN training paradigm resulted in a return of false positives. The V3-HN dataset and training schedule currently represent a delicate but highly effective local optimum for FP suppression.

## 6. Small-object results
Increasing inference resolution (Exp03) or using two-stage inference (Exp07) successfully improved small-object recall, but at an unacceptable latency cost that violates project requirements.

## 7. Vest results
Class-balancing techniques directed at vests universally degraded helmet precision and recall.

## 8. FPS results
V3-HN sustains 16.2 FPS. All attempts to use heavier architectures (YOLOv8s), larger resolutions (640), or multi-stage inference caused the FPS to drop below the mandatory 12 FPS threshold.

## 9. Functional results
The end-to-end pipeline requires the stable bounding box characteristics of V3-HN.

## 10. Final recommendation
**KEEP V3-HN.** The current model perfectly satisfies the operational requirements. Further optimization is not justified without faster edge hardware (e.g., Jetson Orin with TensorRT FP16) to allow for heavier architectures.
