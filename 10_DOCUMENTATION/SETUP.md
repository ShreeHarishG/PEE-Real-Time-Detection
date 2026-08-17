# Setup Guide

## Requirements
- Python 3.11+
- Node.js 18+
- PostgreSQL 14+
- CUDA 11.4+ (for GPU acceleration)

## Installation

```bash
# 1. Environment
python -m venv ppe-env
.\ppe-env\Scripts\activate
pip install -r notebook/requirements.txt

# 2. Database
python 05_DATABASE/init_db.py

# 3. Backend
cd notebook/backend
pip install -r requirements.txt
# Set DATABASE_URL in .env if not using defaults
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 4. Frontend
cd ../frontend
npm install
npm run dev
```

For detailed architecture, see `11_ARCHITECTURE/`.
