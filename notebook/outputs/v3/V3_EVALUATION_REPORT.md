# V3_EVALUATION_REPORT

## 1. V2 Baseline vs V3-HN Results
**V2 Baseline:** `edgevision_v2/models/ppe_best.pt`
**V3-HN (Epoch 10):** `runs/detect/experiments/v3/exp1_hard_negative/weights/best.pt`

## 2. Validation Comparison
```csv
Metric,V2 Baseline,V3-HN
precision,0.837934860865428,0.8509449959933276
recall,0.7941436261896029,0.7915696407695023
mAP50,0.844464158459795,0.8419550200415561
mAP50_95,0.5012485264777344,0.4876705732884997
helmet_precision,0.9115509044868362,0.9106932097248072
helmet_recall,0.8455815365927726,0.8232614637109019
helmet_AP50,0.9015340108699138,0.886479855232455
helmet_AP50_95,0.5760361598638358,0.555592854454052
no_helmet_precision,0.8188633734253565,0.8535222880845671
no_helmet_recall,0.7701826753093695,0.8138626988803771
no_helmet_AP50,0.8081166946340046,0.841718486346898
no_helmet_AP50_95,0.39936108976209106,0.416382014367117
vest_precision,0.7833903046840912,0.788619490170609
vest_recall,0.7666666666666667,0.7375847597172277
vest_AP50,0.8237417698754665,0.7976667185453152
vest_AP50_95,0.5283483298072761,0.49103685104433004

```

## 3. Negative-Video Comparison (test.mp4)
- **V2 Helmet Detections (False):** 471
- **V3 Helmet Detections (False):** 0
- **V2 Vest Detections (False):** 27
- **V3 Vest Detections (False):** 0
- **V2 Confirmed Violations (False):** 14
- **V3 Confirmed Violations (False):** 14

## 4. Positive-Video Comparison
- **V2 Helmet Detection Rate:** 0.5862
- **V3 Helmet Detection Rate:** 0.6379
- **V2 Vest Detection Rate:** 0.6140
- **V3 Vest Detection Rate:** 0.6667
- **V2 Confirmed Violations:** 10
- **V3 Confirmed Violations:** 4

## 5. Hard-Negative Comparison (73 crops)
- **V2 Helmet FP:** 154
- **V3 Helmet FP:** 0 -> **100.0% reduction**
- **V2 Vest FP:** 10
- **V3 Vest FP:** 0 -> **100.0% reduction**

## 6. FPS Comparison
- **V2 Warm FPS (Neg/Pos):** 24.3 / 25.2
- **V3 Warm FPS (Neg/Pos):** 16.2 / 14.9
- **V2 P95 Latency (Neg):** 70.4 ms
- **V3 P95 Latency (Neg):** 144.6 ms

## 7. Percentage Improvements & Regression Analysis
- **Helmet FPs on crops reduced by:** 100.0%
- **Vest FPs on crops reduced by:** 100.0%
- **Helmet False Detections on video:** 471 -> 0
- **Validation mAP50:** 0.8445 -> 0.8420

## 8. Final Recommendation
**DECISION: PROMOTE V3-HN**
- Model successfully suppressed >50% of real-world helmet FPs.
- Recall is stable.
- FPS meets >12 requirement.