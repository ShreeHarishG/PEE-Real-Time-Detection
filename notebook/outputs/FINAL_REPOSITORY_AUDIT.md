# EdgeVision Repository Audit & Cleanup Strategy

## 1. Current Structure
The repository is currently organized as a hybrid ML-experimentation directory (`W:\3 projects\Building\Tfrenzy\notebook`) with remnants of older testing phases.
- **`models/`**: Contains validated production models (`ppe_v3_hn_best.pt`, `ppe_v2_backup.pt`, `yolov8n.pt`, `yolo26n.pt`).
- **`src/`**: Contains the currently consolidated inference pipeline script.
- **`scripts/`**: Contains 28 legacy scripts (e.g. `phase3a_threshold_calibration.py`, `run_exp1.ps1`) from various testing and validation phases.
- **`outputs/`**: Contains validation outputs from Phase 3 and Phase 4, along with some legacy tracking data.
- **`runs/`**: Contains raw Ultralytics YOLO training artifacts.
- **`fp_crops/`, `harness-1/`**: Appears to be legacy dataset artifacts or false positive mining folders.
- **`dashboard/`**: Contains a basic Streamlit dashboard MVP.
- **`datasets/`**: Core datasets for reproducibility.

## 2. What is Required (To be Built/Integrated)
To meet the PRD, the following must be introduced:
- **`frontend/`**: Next.js (TypeScript, Tailwind) web dashboard.
- **`backend/`**: FastAPI REST API handling the database, live events, and zone rules.
- **`ml/`**: Refactored existing `src/pipeline.py` and models into a cleaner ML module that outputs events to the backend DB instead of CSVs.
- **`docker-compose.yml`**: PostgreSQL database and optionally backend integration.
- **`backend/migrations/`**: Alembic migrations for DB schema.

## 3. What Must Be Preserved
- `models/ppe_v3_hn_best.pt` (Frozen V3-HN production candidate)
- `models/ppe_v2_backup.pt` (Rollback)
- `datasets/` (Reproducibility)
- Selected validation scripts in `scripts/` (e.g., `phase3i`, `phase3h` scripts used for final sign-off)
- `outputs/v4/` (The final report from the autonomous ML lab)
- The existing inference logic (YOLO person -> tracking -> YOLO PPE -> IoA -> Temporal Validation).

## 4. What is Obsolete / Can Be Deleted
- `runs/` (except the specific `train/weights` if they are the source of `v3_hn`, but they are already copied to `models/`). Safe to delete to save space since `models/` contains the frozen weights.
- `fp_crops/` (intermediate false-positive mining artifacts from earlier phases).
- `harness-1/` (legacy/experimental dataset fragment).
- The basic `dashboard/` Streamlit MVP (once the Next.js frontend is operational).
- Redundant phase testing scripts that do not form the final audit trail.

## 5. Cleanup Actions
1. Re-organize `src/` to `ml/inference/` and adapt the pipeline to POST to the FastAPI backend instead of writing to CSV.
2. Delete `runs/`, `fp_crops/`, `harness-1/`.
3. Scaffold `backend/` and `frontend/`.

## 6. Risks
- Removing older dataset fragments might break some legacy `phase3` scripts if they hardcoded those paths. We will preserve `datasets/` but remove obviously useless temp folders.
- Modifying `pipeline.py` to talk to a database could inadvertently introduce latency. We must ensure DB inserts are asynchronous or batched to maintain the 16.2 FPS requirement.
