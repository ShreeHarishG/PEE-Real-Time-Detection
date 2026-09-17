<div align="center">
  <img src="assets/edgevision_banner.png" alt="EdgeVision Banner" width="100%" />

  # 🛡️ EdgeVision

  **Real-time PPE Compliance and Work-at-Height Safety Monitoring**  
  *Powered by YOLOv8, ByteTrack, and the validated V5-Harness model.*

  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
  [![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg?logo=node.js&logoColor=white)](https://nodejs.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![Next.js](https://img.shields.io/badge/Next.js-Dashboard-000000.svg?logo=next.js&logoColor=white)](https://nextjs.org/)
  [![YOLOv8](https://img.shields.io/badge/YOLOv8-Computer%20Vision-FF9900.svg?logo=ultralytics&logoColor=white)](https://ultralytics.com/)
</div>

---

## 🌟 Overview

EdgeVision is an end-to-end, real-time safety compliance platform designed for industrial and construction environments. It automatically detects workers, tracks them across zones, and verifies whether they are wearing required Personal Protective Equipment (PPE) such as helmets and vests.

### ✨ Key Features
- **Real-Time AI Detection**: Uses highly optimized YOLOv8 and V5-Harness models for instance segmentation and classification.
- **Robust Tracking**: Integrates ByteTrack for consistent temporal association and trajectory monitoring.
- **Dynamic Rule Engine**: Define custom spatial zones and apply specific PPE rules to each area.
- **Modern Dashboard**: A sleek Next.js React frontend for live monitoring, statistics, and configuration.
- **Edge Deployment Ready**: Designed to run optimally on NVIDIA Jetson edge devices using TensorRT.

---

## 📸 Live Inference Demo

EdgeVision accurately identifies workers and validates required PPE in real-time, highlighting compliance violations instantly:

<div align="center">
  <img src="assets/demo_screenshot.jpg" alt="Live Camera Inference Screenshot" width="800" />
</div>

> *Live camera output captured from `outputs/live_camera` and full video pipeline results are generated in `outputs/results`.*

---

## 🏗️ Architecture

The EdgeVision pipeline is designed for low-latency, high-throughput processing.

```mermaid
graph LR
    A[Camera / Video] -->|Frames| B(Person Detection<br>YOLOv8n)
    B --> C(Tracking<br>ByteTrack)
    A -->|Frames| D(PPE Detection<br>V5-Harness Model)
    C --> E(Spatial Association<br>& Zone Rules)
    D --> E
    E --> F(Temporal Validator)
    F -->|Violations & Stats| H(FastAPI Backend)
    H -->|Stores Data| G[(PostgreSQL)]
    H <--> I[Next.js Dashboard]
    
    style A fill:#2d3748,stroke:#4a5568,color:#fff
    style B fill:#3182ce,stroke:#2b6cb0,color:#fff
    style C fill:#3182ce,stroke:#2b6cb0,color:#fff
    style D fill:#805ad5,stroke:#6b46c1,color:#fff
    style E fill:#dd6b20,stroke:#c05621,color:#fff
    style F fill:#e53e3e,stroke:#c53030,color:#fff
    style G fill:#38a169,stroke:#2f855a,color:#fff
    style H fill:#009688,stroke:#00796b,color:#fff
    style I fill:#000000,stroke:#333333,color:#fff
```

---

## 🚀 Quick Start (Windows — Development)

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker Desktop

### 1️⃣ Start the Database
```bash
docker-compose up -d db
```

### 2️⃣ Initialize Backend
*(First run only)*
```bash
cd backend
pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic-settings
python scripts/init_db.py
```

### 3️⃣ Start FastAPI Backend
```bash
# In backend/
python -m uvicorn app.main:app --port 8000
```
- **Health Check**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4️⃣ Start Next.js Frontend
```bash
# In frontend/
npm install
npm run dev
```
- **Dashboard**: [http://localhost:3000](http://localhost:3000)

### 5️⃣ Run the ML Demo Pipeline
```bash
# In the root directory
python src/pipeline.py
```

---

## 🐧 Edge Deployment (Linux / Jetson)

EdgeVision is built for high-performance edge inference on NVIDIA Jetson devices (JetPack 5.x/6.x).

1. **Transfer Files**: Copy the project to your Jetson device.
2. **Install Dependencies**:
   ```bash
   cd deployment
   sudo bash install_jetson.sh
   ```
3. **Start the Service**:
   ```bash
   sudo systemctl start edgevision
   ```
4. **Compile TensorRT Engine**: For maximum FPS, compile the ONNX model to a TensorRT engine. Instructions are in `10_DOCUMENTATION/USER_GUIDE.md` or the deployment docs.
5. **View Dashboard**: Navigate to `http://<JETSON_IP>:3000` from any device on the network.

---

## 📊 Models & Performance

### Included Models
| Model | File | Status |
| :--- | :--- | :--- |
| **V5-Harness** (Production) | `models/ppe_v5_harness.pt` | 🟢 **ACTIVE** |
| **V3-HN** (Rollback) | `models/ppe_v3_hn_best.pt` | 🟡 **AVAILABLE** |

### Key Metrics (V5-Harness, Warm, RTX 4050)
| Metric | Value |
| :--- | :--- |
| **mAP50** | `84.20%` |
| **Helmet Recall** | `82.33%` |
| **Vest Recall** | `73.76%` |
| **Real-World FP** | `0 / 0` |
| **Warm FPS** | `16.2` |
| **P95 Latency** | `134.63 ms` |

> [!WARNING]  
> **Known Limitations**
> - The `lanyard` and `hook` classes are **NOT trained** in the V5-Harness model (marked as UNTRAINED in the UI).
> - Jetson TensorRT benchmarking is **PENDING PHYSICAL HARDWARE**.

---

## 📚 Documentation Reference

For more detailed technical documentation, please refer to the files in the `docs` folder:

- 📖 **[Handover Document](docs/HANDOVER.md)** — Read first for project handovers.
- ⚙️ **[Setup Guide](docs/SETUP.md)** — Detailed environment setup.
- 🧑‍💻 **[User Guide](docs/USER_GUIDE.md)** — How to operate the platform.
- 📄 **[PRD Specification](docs/PRD.pdf)** — Product requirements and architectural constraints.

---
<div align="center">
  <sub>Built with ❤️ for workplace safety.</sub>
</div>
