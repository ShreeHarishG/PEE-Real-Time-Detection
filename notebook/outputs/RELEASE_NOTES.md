# EdgeVision V3 - Release Notes

## Final Model Version
- **Model**: `ppe_v3_hn_best.pt`
- **Base Architecture**: YOLOv8n
- **Resolution**: `imgsz=512`
- **Training Epochs**: 10
- **Strategy**: Hard-negative mining with 73 empty label frames to suppress reflective/machinery noise.

## Final Validated Metrics
- **mAP50**: 84.20%
- **mAP50-95**: 48.77%
- **Helmet Recall**: 82.33%
- **Vest Recall**: 73.76%
- **False Positives (Real-world)**: 0
- **Warm Inference FPS (Desktop RTX 4050)**: 16.2
- **P95 Latency**: 134 ms

## Software Stack
- **Frontend**: Next.js 15, React, Tailwind CSS, Lucide Icons
- **Backend**: FastAPI (Python), Uvicorn, SQLAlchemy
- **Database**: PostgreSQL 15 (Dockerized)
- **ML Inference**: Ultralytics YOLOv8, ByteTrack tracker, Custom TemporalValidator
- **Deployment Scripting**: ONNX, trtexec (TensorRT)

## Known Limitations
- The model is currently heavily optimized for helmets and vests. Secondary PPE such as boots, hooks, and harnesses are currently displayed as **UNTRAINED** placeholders on the frontend.
- Bounding-box spatial containment is used for Person-to-PPE association. Extremely dense crowds may cause temporary cross-association, though the 2-second `TemporalValidator` hysteresis mostly mitigates this.

## Jetson Pending Status
- The TensorRT `.engine` generation instructions and validation scripts are provided (`scripts/build_tensorrt.sh` and `scripts/benchmark_tensorrt.sh`).
- A systemd startup script (`scripts/edgevision.service`) has been created.
- **Pending Action**: The physical Jetson Orin Nano hardware requires flashing via JetPack, loading the `.onnx` file, and running the 8-hour continuous thermal benchmark.

## Exact Demo Startup Commands
Run these in 4 separate terminals:

```bash
# Terminal 1: Database
docker-compose up -d db

# Terminal 2: Backend
cd backend
python -m uvicorn app.main:app --port 8000

# Terminal 3: Frontend
cd frontend
npm run dev

# Terminal 4: ML Pipeline
python src/pipeline.py
```
