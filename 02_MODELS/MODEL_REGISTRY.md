# MODEL REGISTRY

| Model | Version | Purpose | Classes | Status | Metrics (mAP50) | Source Path | SHA256 |
|---|---|---|---|---|---|---|---|
| **V3-HN** | ppe_v3_hn_best.pt | Production Inference | `person`, `helmet`, `no_helmet`, `vest` | **PRODUCTION** | 0.842 | `02_MODELS/production/ppe_v3_hn_best.pt` | (See `_submission_backup_manifest.json`) |
| **V2** | ppe_v2_backup.pt | Rollback Baseline | `person`, `helmet`, `no_helmet`, `vest` | **ROLLBACK** | ~0.840 | `02_MODELS/rollback/ppe_v2_backup.pt` | (See `_submission_backup_manifest.json`) |
| **V3-BOOTS** | ppe_v3_boots_best.pt | Experimental / Regression | `person`, `helmet`, `no_helmet`, `vest`, `boots` | **EXPERIMENTAL** | - | `02_MODELS/experimental/ppe_v3_boots_best.pt` | (See `_submission_backup_manifest.json`) |
| **Harness** | N/A | Unsupported class | `harness`, `lanyard`, `hook` | **UNSUPPORTED / NO TRAINED MODEL** | N/A | N/A | N/A |

*Note: The YOLOv8n person detection model (`yolov8n.pt`) is preserved in its original location as required by the ByteTrack pipeline. All unsupported classes are strictly marked as untrained.*
