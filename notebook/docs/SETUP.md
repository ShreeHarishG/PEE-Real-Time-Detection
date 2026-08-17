# EdgeVision Setup Guide

## Prerequisites

| Component | Minimum Version |
| :--- | :--- |
| Python | 3.9+ |
| Node.js | 18+ |
| Docker Desktop | 4.0+ |
| CUDA (Optional) | 11.8+ (for GPU inference) |
| NVIDIA GPU (Optional) | GTX 1060 / RTX series (for full FPS) |

---

## Step 1 — Clone / Open Repository
```bash
cd "W:\3 projects\Building\Tfrenzy\notebook"   # Windows
# or
cd /opt/edgevision/notebook                      # Linux / Jetson
```

## Step 2 — Create Python Environment
```bash
python -m venv ppe-env
# Windows
ppe-env\Scripts\activate
# Linux
source ppe-env/bin/activate

pip install -r requirements.txt
```

## Step 3 — Install Backend Dependencies
```bash
pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic-settings alembic
```

## Step 4 — Configure Environment
```bash
cp .env.example .env
# Edit .env if you need custom database credentials
```

## Step 5 — Start PostgreSQL
```bash
docker-compose up -d db
```

## Step 6 — Initialise Database Tables
```bash
cd backend
python scripts/init_db.py
cd ..
```

## Step 7 — Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

## Step 8 — Start All Services

Open 3 separate terminals from the `notebook/` directory:

**Terminal A — Backend:**
```bash
cd backend
python -m uvicorn app.main:app --port 8000 --reload
```

**Terminal B — Frontend:**
```bash
cd frontend
npm run dev
```

**Terminal C — ML Pipeline:**
```bash
python src/pipeline.py
```

## Step 9 — Verify
- Backend health: http://localhost:8000/api/v1/health
- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:3000
