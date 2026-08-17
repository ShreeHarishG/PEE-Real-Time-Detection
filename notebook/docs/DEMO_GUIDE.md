# EdgeVision Demo Guide

## Before You Start
Ensure the full stack is running (see README.md for startup commands).

```
[Terminal 1] docker-compose up -d db
[Terminal 2] uvicorn app.main:app --port 8000       (from backend/)
[Terminal 3] npm run dev                             (from frontend/)
[Terminal 4] python src/pipeline.py                 (from notebook root)
```

---

## Demo Sequence

### Step 1 — Open the Dashboard
Navigate to `http://localhost:3000`. The **Live Monitoring** page loads by default.

### Step 2 — Show Live Monitoring
Point out:
- The simulation video feed labelled **DEMO / VIDEO SIMULATION**
- The FPS indicator (validated: **16.2 FPS warm**)
- The active violation counter
- The compliant worker counter

### Step 3 — Start the Inference Demo
The pipeline (`src/pipeline.py`) is already running in Terminal 4. Let it process a few seconds of the test video.

You will see console output like:
```
2026-08-12 [INFO] Confirmed violation for worker ID 83 — missing: helmet, vest
```

### Step 4 — Show Active Violations
Navigate to `/violations`. The violation generated in Step 3 should appear within seconds.

Highlight:
- Event ID (UUID)
- Worker Tracking ID
- Zone assignment
- Missing PPE items (e.g. `HELMET`, `VEST`)
- Confidence score
- Model version (`V3-HN`)
- Timestamp

### Step 5 — Open Evidence
Click the evidence image link. The crop of the violating worker is stored under `outputs/evidence/<event_id>.jpg`.

### Step 6 — Acknowledge
Click **Acknowledge** to mark the event as reviewed. Confirm it disappears from the Active Violations list and is flagged in the database.

### Step 7 — Event History
Navigate to `/history`. All past events (including the just-acknowledged one) are listed here. Demonstrate the date filter.

### Step 8 — Worker Compliance
Navigate to `/workers`. Point out the Temporary Worker Tracking IDs and explain they are session-scoped. Show compliance rate calculation.

### Step 9 — Zone Configuration
Navigate to `/zones`. Demonstrate the Construction Zone rule (helmet + vest). Highlight that `harness`, `boots`, `hook` are labelled **UNTRAINED** — be transparent that V3-HN does not detect these.

### Step 10 — Model Monitoring
Navigate to `/models`. Show the validated offline benchmark metrics:
- mAP50: **84.20%**
- Helmet Recall: **82.33%**
- Real-world FP: **0 / 0**
- Warm FPS: **16.2**

Clearly label these as **VALIDATED BENCHMARK METRICS** from desktop validation — not Jetson results.

### Step 11 — PostgreSQL Event Verification
Open a terminal and connect to the running database to show a real row:
```bash
docker exec -it edgevision_db psql -U edgevision -d edgevision -c "SELECT event_id, worker_tracking_id, missing_ppe, confidence, timestamp FROM violation_events ORDER BY timestamp DESC LIMIT 5;"
```

### Step 12 — ONNX Export Path
Show:
```bash
python scripts/export_onnx.py --model models/ppe_v3_hn_best.pt --imgsz 512
```
Explain this generates `ppe_v3_hn_best.onnx` for Jetson transfer. Then the TensorRT engine is compiled on-device via `scripts/build_tensorrt.sh`.

### Step 13 — Explain Jetson Status
Refer to `docs/JETSON_DEPLOYMENT.md`. Clearly state:
- All scripts and documentation are complete
- **Physical Jetson validation is PENDING HARDWARE**
- Do not invent benchmark numbers

---

## Fallback Procedure (No RTSP Camera)
The demo is entirely self-contained using the local video file (`docs/test.mp4`). No RTSP hardware is required for a complete demonstration. The video simulation mode is explicitly labelled in the dashboard.
