# Testing Guide

## 1. Backend Testing
The backend is built with FastAPI. Unit tests validate the database models, API routing, and the violation ingestion logic.

Run backend tests:
```bash
cd backend
pytest tests/
```

## 2. ML Pipeline Testing
The `TemporalValidator` and bounding box association logic can be tested locally using the automated regression scripts.

Run validation scripts:
```bash
python scripts/phase3h_association_audit.py
```

## 3. Continuous Integration
All pushes to the main branch should trigger a CI action that:
1. Validates the `docker-compose.yml` build.
2. Runs the Python `pytest` suite.
3. Performs a sanity check on the ONNX export script.
