# EdgeVision Model Documentation

## Production Model: V3-HN

**File:** `models/ppe_v3_hn_best.pt`  
**Architecture:** YOLOv8n (Nano)  
**Input Resolution:** 512×512  
**Parameters:** 3,012,018  
**GFLOPs:** 8.2  
**Training Epochs:** 10  
**Status:** FROZEN

---

## Validated Metrics (Desktop GPU, Warm)

| Metric | Value |
| :--- | :--- |
| mAP50 | 84.20% (0.841955) |
| mAP50-95 | 48.77% (0.487671) |
| Helmet AP50 | 88.6% |
| No-Helmet AP50 | 80.8% |
| Vest AP50 | 82.4% |
| Helmet Recall | 82.33% (0.823261) |
| Vest Recall | 73.76% (0.737585) |
| Helmet Precision | 91.07% |
| Vest Precision | 79.79% |

---

## Real-World Negative-Video Testing

| Test | Result |
| :--- | :--- |
| Helmet False Positives | **0** |
| Vest False Positives | **0** |
| Confirmed Violations (positive video) | 14 |
| Duplicate Events | 0 |
| Empty Events | 0 |
| Association Failures | 0 |

---

## Real-World Positive-Video Testing

| Metric | Value |
| :--- | :--- |
| Helmet detection rate | 63.79% (0.6379) |
| Vest detection rate | 66.67% (0.6667) |

---

## Inference Performance

| Metric | Value |
| :--- | :--- |
| Warm FPS (RTX 4050) | 16.2 |
| P95 Latency | 134.63 ms |
| Minimum FPS Requirement | 12 FPS |
| Status | PASS |

---

## Hard-Negative Training Strategy
V3-HN's key innovation over V2 is the **hard-negative mining strategy**.

V2 produced:
- 154 false positive helmets (caused by yellow machinery, reflective tape)
- 10 false positive vests

To correct this, 73 hard-negative frames (no workers, no PPE) were added to the training set with empty label files. The model was fine-tuned for 10 epochs. This drove real-world false positives to exactly zero while retaining detection accuracy.

---

## Supported PPE Classes (TRAINED + VALIDATED)

| Class | Status |
| :--- | :--- |
| helmet | ✅ TRAINED + VALIDATED |
| no_helmet | ✅ TRAINED + VALIDATED |
| vest | ✅ TRAINED + VALIDATED |

---

## Unsupported PPE Classes (NOT TRAINED)

| Class | Status |
| :--- | :--- |
| boots | ❌ NOT TRAINED — do not claim detection |
| harness | ❌ NOT TRAINED — do not claim detection |
| lanyard | ❌ NOT TRAINED — do not claim detection |
| hook | ❌ NOT TRAINED — do not claim detection |
| anchor_point | ❌ NOT TRAINED — do not claim detection |

---

## Rollback
**V2 backup:** `models/ppe_v2_backup.pt`

To activate rollback, change `config/model_versions.yaml`:
```yaml
production:
  version: V2
```
Then restart the ML pipeline.
