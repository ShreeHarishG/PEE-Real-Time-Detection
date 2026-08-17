# EdgeVision V3-HN — Handover to TFrenzy Reviewer

---

## PROJECT
**EdgeVision PPE Compliance and Work-at-Height Safety Platform**

## MODEL
`ppe_v3_hn_best.pt` — YOLOv8n, fine-tuned with hard-negative mining strategy

## MODEL STATUS
**FROZEN** — do not retrain, modify weights, or change association thresholds.

## ROLLBACK
`models/ppe_v2_backup.pt` — V2 baseline is available for immediate rollback.

---

## Validated Desktop Performance

| Metric | Value |
| :--- | :--- |
| mAP50 | 84.20% |
| mAP50-95 | 48.77% |
| Helmet Recall | 82.33% |
| Vest Recall | 73.76% |
| Real-World Helmet FP | 0 |
| Real-World Vest FP | 0 |
| Warm FPS (RTX 4050) | 16.2 |
| P95 Latency | 134.63 ms |
| Confirmed Violations | 14 |
| Duplicate Events | 0 |
| Association Failures | 0 |

**12 FPS requirement:** ✅ PASS on validated desktop environment

---

## Infrastructure

| Component | Technology | Status |
| :--- | :--- | :--- |
| Container | Docker | PASS |
| Database | PostgreSQL 15 | PASS |
| Backend | FastAPI (Python) | PASS |
| Frontend | Next.js / React / Tailwind | PASS |
| ML Pipeline | Ultralytics YOLOv8 + ByteTrack | PASS |
| Edge Export | ONNX script ready | PASS |
| TensorRT | Build scripts ready | READY FOR HARDWARE |
| Jetson | Deployment docs complete | PENDING HARDWARE |

---

## Architecture
```
Camera / Video File
    ↓  (yolov8n person detection)
Person Filtering & ByteTrack Tracking
    ↓  (V3-HN PPE detection)
Spatial IoA Association
    ↓
Zone Rule Engine  (helmet + vest per zone, configurable via API)
    ↓
Temporal Validator  (2-sec / 8-of-10-frame hysteresis)
    ↓
Violation Event + Evidence Crop
    ↓  (async REST POST)
FastAPI Backend → PostgreSQL
    ↓
Next.js Dashboard (Live Monitoring, Violations, History, Compliance)
```

---

## Quick Start

### Windows (Development)
```bash
# 1. Start database
cd notebook
docker-compose up -d db

# 2. Initialise DB tables (first run only)
cd backend
python scripts/init_db.py

# 3. Start backend
python -m uvicorn app.main:app --port 8000

# 4. Start frontend (new terminal)
cd frontend
npm install
npm run dev

# 5. Run inference demo (new terminal, from notebook root)
python src/pipeline.py
```

### Ubuntu / Jetson (Production)
```bash
# Install JetPack 5.1+
# Clone repository to /opt/edgevision/notebook

pip install -r requirements.txt
docker-compose up -d db
python backend/scripts/init_db.py
python -m uvicorn backend/app/main:app --port 8000 &
npm --prefix frontend run build && npm --prefix frontend start &

# Export to ONNX
python scripts/export_onnx.py --model models/ppe_v3_hn_best.pt --imgsz 512

# Build TensorRT engine (on Jetson only)
./scripts/build_tensorrt.sh

# Install as system service
sudo cp scripts/edgevision.service /etc/systemd/system/
sudo systemctl enable --now edgevision
```

---

## Demo Procedure
See `docs/DEMO_GUIDE.md` for the full step-by-step reviewer demonstration.

**Quick steps:**
1. Start all 4 components (see Quick Start above).
2. Open `http://localhost:3000`
3. Let the pipeline generate a violation.
4. Watch it appear in Active Violations.
5. Confirm in PostgreSQL via:  
   `docker exec -it edgevision_db psql -U edgevision -d edgevision -c "SELECT * FROM violation_events LIMIT 5;"`

---

## Documentation Index

| Document | Location |
| :--- | :--- |
| Architecture | `docs/ARCHITECTURE.md` |
| API Reference | `docs/API.md` |
| Database Schema | `docs/DATABASE.md` |
| Training Reproducibility | `docs/TRAINING.md` |
| Jetson Deployment | `docs/JETSON_DEPLOYMENT.md` |
| DeepStream Direction | `docs/DEEPSTREAM.md` |
| Testing | `docs/TESTING.md` |
| Setup | `docs/SETUP.md` |
| Troubleshooting | `docs/TROUBLESHOOTING.md` |
| User Guide | `docs/USER_GUIDE.md` |
| Demo Procedure | `docs/DEMO_GUIDE.md` |
| Model Details | `docs/MODEL.md` |

---

## Known Limitations

1. **Unsupported PPE classes:** `boots`, `harness`, `lanyard`, `hook`, `anchor-point` are NOT detected by V3-HN. These are shown in the UI with an "UNTRAINED" badge. Do not claim they work.
2. **Temporary tracking IDs:** Worker identity is session-scoped ByteTrack IDs. No biometric persistence across sessions.
3. **Dense crowd cross-association:** Very tightly packed workers may cause momentary PPE cross-association. The 2-second temporal hysteresis significantly reduces false alarms.
4. **Positive-video detection rate:** Positive-video helmet detection at 63.8%, vest at 66.7% — these reflect that some frames show compliant workers briefly occluded or viewed at severe angles.

---

## Jetson Status
**PHYSICAL JETSON VALIDATION: PENDING HARDWARE**

All software preparation is complete:
- ONNX export script: `scripts/export_onnx.py`
- TensorRT FP16 build script: `scripts/build_tensorrt.sh`
- Benchmark script: `scripts/benchmark_tensorrt.sh`
- Systemd startup service: `scripts/edgevision.service`
- Full deployment documentation: `docs/JETSON_DEPLOYMENT.md`

The following are **NOT available** because hardware testing has not been performed:
- Jetson FPS numbers
- Jetson thermal readings
- Jetson power consumption
- 8-hour stability results

---

## Rollback Procedure
If V3-HN must be reverted to V2:

1. Edit `config/model_versions.yaml`:
   ```yaml
   production:
     version: V2
   ```
2. Restart the ML pipeline — it will automatically load `models/ppe_v2_backup.pt`.
