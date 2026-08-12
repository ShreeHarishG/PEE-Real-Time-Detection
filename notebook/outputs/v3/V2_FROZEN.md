# V2 Model Frozen Status

## V2 Model Path
`edgevision_v2/models/ppe_best.pt`

## V2 Metrics
- **mAP50:** 0.844464
- **mAP50-95:** 0.501249
- **Helmet:** precision = 0.911551, recall = 0.845582, AP50 = 0.901534
- **No Helmet:** precision = 0.818863, recall = 0.770183, AP50 = 0.808117
- **Vest:** precision = 0.783390, recall = 0.766667, AP50 = 0.823742
- **Positive helmet detection:** 0.5862
- **Positive vest detection:** 0.6140
- **Warm FPS:** 24.3
- **Confirmed violations:** 14

## Reason V2 was replaced
V2 exhibited a significant number of false positives in real-world scenarios (154 helmet FPs, 10 vest FPs). V3-HN was promoted because it eliminated these false positives (100% reduction) and improved positive video detection rates, despite a minor regression in validation set mAP.

## Rollback Instructions
To rollback to V2:
1. Update `config/model_versions.yaml` to point the `production` version back to `v2` (`edgevision_v2/models/ppe_best.pt`).
2. Verify that the production pipeline loads the `ppe_best.pt` model.
3. Rerun the end-to-end regression test to confirm V2 behavior.
