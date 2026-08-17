# V3-BOOTS PROMOTION AUDIT

## 1. Metric Discrepancy
The mAP50 discrepancy (0.842 vs 0.751) was caused by evaluating the model on `ppe_extension_boots/data.yaml` instead of the original baseline `construction-ppe/data.yaml`.

## 2. False Positive Audit
46 boots FPs were found. The majority (25) were standard shoes misclassified as safety boots. 10 were shadows/dark objects, and 4 were genuine boots missing from ground truth.

## 3. Violation Audit
The drop from 1955 to 1541 violations is due to the 54 new association failures disrupting the TemporalValidator hysteresis (which requires 8 consecutive frames to log an event).

## 4. Final Recommendation
KEEP V3-HN. The boots model requires further fine-tuning to differentiate boots from shoes and to resolve the association disruptions.
