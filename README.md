# EdgeVision PPE Compliance Platform

Real-time PPE compliance and work-at-height safety monitoring, powered by YOLOv8 and the validated V3-HN model.

---

## Architecture
```
Camera / Video → Person Detection (YOLOv8n) → ByteTrack Tracking
→ V3-HN PPE Detection → Spatial Association
→ Zone Rule Engine → Temporal Validator
→ FastAPI Backend → PostgreSQL
→ Next.js Dashboard
```

---

## Quick Start (Windows — Development)

### Prerequisites
- Python 3.9+, Node.js 18+, Docker Desktop

### 1. Start the Database
```bash
cd notebook
docker-compose up -d db
```

### 2. Install Backend Dependencies & Initialise DB (first run only)
```bash
cd backend
pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic-settings
python scripts/init_db.py
```

### 3. Start FastAPI Backend
```bash
# In notebook/backend/
python -m uvicorn app.main:app --port 8000
```
Verify: http://localhost:8000/api/v1/health  
API docs: http://localhost:8000/docs

### 4. Start Next.js Frontend
```bash
# In notebook/frontend/
npm install
npm run dev
```
Dashboard: http://localhost:3000

### 5. Run the ML Demo Pipeline
```bash
# In notebook/ root
python src/pipeline.py
```

---

## Quick Start (Linux / Ubuntu / Jetson)

To deploy the fully automated pipeline on an NVIDIA Jetson device (JetPack 5.x/6.x):

1. **Transfer Files**: Copy the entire project folder to your Jetson device.
2. **Install Dependencies**: Open a terminal on the Jetson and navigate to the deployment folder:
   ```bash
   cd notebook/deployment
   sudo bash install_jetson.sh
   ```
   *This script automatically installs DeepStream, PyTorch, creates necessary directories, and installs the `edgevision` systemd service.*
3. **Start the Service**: 
   The pipeline will now run automatically on boot. To start it immediately:
   ```bash
   sudo systemctl start edgevision
   ```
4. **Compile TensorRT Engine**: To achieve maximum FPS, you must compile the ONNX model into a TensorRT engine on the Jetson itself. Follow the step-by-step instructions in:
   [`notebook/deployment/tensorrt_instructions.md`](notebook/deployment/tensorrt_instructions.md)
5. **View Dashboard**: Access the live stream and statistics from any device on the network by pointing a browser to `http://<JETSON_IP>:3000`.

---

## Models

| Model | File | Status |
| :--- | :--- | :--- |
| V3-HN (Production) | `models/ppe_v3_hn_best.pt` | FROZEN |
| V2 (Rollback) | `models/ppe_v2_backup.pt` | AVAILABLE |

---

## Key Metrics (V3-HN, Warm, RTX 4050)

| Metric | Value |
| :--- | :--- |
| mAP50 | 84.20% |
| Helmet Recall | 82.33% |
| Vest Recall | 73.76% |
| Real-World FP | 0 / 0 |
| Warm FPS | 16.2 |
| P95 Latency | 134.63 ms |

---

## Known Limitations
- `boots`, `harness`, `lanyard`, `hook` are **NOT trained** in V3-HN — clearly marked in UI as UNTRAINED.
- Jetson TensorRT benchmarking **PENDING PHYSICAL HARDWARE**.

---

## Documentation

| Document | Path |
| :--- | :--- |
| Handover (read first) | `docs/HANDOVER_TO_TFRENZY.md` |
| Setup Guide | `docs/SETUP.md` |
| Architecture | `docs/ARCHITECTURE.md` |
| API Reference | `docs/API.md` |
| Model Details | `docs/MODEL.md` |
| Jetson Deployment | `docs/JETSON_DEPLOYMENT.md` |
| Demo Procedure | `docs/DEMO_GUIDE.md` |
| User Guide | `docs/USER_GUIDE.md` |
| Training | `docs/TRAINING.md` |
| Database | `docs/DATABASE.md` |
| Testing | `docs/TESTING.md` |
| Troubleshooting | `docs/TROUBLESHOOTING.md` |
