# Database Schema & Migrations

## Overview
EdgeVision uses PostgreSQL to persist long-term metrics and violation events. The schema is optimized for fast inserts during ML pipeline execution, and fast retrieval for the React frontend.

## Core Entities
1. **Zones:** Logical areas mapping physical spaces to specific safety rules (e.g., Construction Zone requires helmet + vest).
2. **Cameras:** Physical video sources linked to a specific Zone.
3. **Violation Events:** Individual instances where the ML model confirmed a missing PPE requirement for a tracked worker.
4. **Inference Metrics:** A log of system FPS and latency for health monitoring.

## Evidence Storage
Evidence images and videos are **NOT** stored in the database as BLOBS. They are saved directly to the filesystem under `outputs/evidence/`. The database stores the relative URI (e.g., `outputs/evidence/uuid.jpg`).

## Migrations
The repository uses Alembic for schema migrations.

### Initializing the DB
```bash
docker-compose up -d db
cd backend
alembic upgrade head
```
