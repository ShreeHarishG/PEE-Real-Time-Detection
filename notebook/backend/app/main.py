from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, timedelta
import os
import time
import uuid
import subprocess
import shutil
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from . import models, schemas, database

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="EdgeVision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, use env BACKEND_CORS_ORIGINS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
import time
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

# --- Health ---
@app.get("/api/v1/health")
def read_health():
    return {"status": "ok"}

# Serve outputs/evidence folder statically
outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs"))
os.makedirs(os.path.join(outputs_dir, "evidence"), exist_ok=True)
app.mount("/evidence", StaticFiles(directory=os.path.join(outputs_dir, "evidence")), name="evidence")

import asyncio

async def generate_frames():
    frame_path = os.path.join(outputs_dir, "latest_frame.jpg")
    last_mtime = 0
    while True:
        try:
            if os.path.exists(frame_path):
                mtime = os.path.getmtime(frame_path)
                if mtime > last_mtime:
                    with open(frame_path, "rb") as f:
                        frame = f.read()
                    last_mtime = mtime
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            await asyncio.sleep(0.03)
        except Exception:
            await asyncio.sleep(0.03)

@app.get("/api/v1/stream")
async def video_stream():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# --- Zones ---
@app.get("/api/v1/zones", response_model=List[schemas.Zone])
def get_zones(skip: int = 0, limit: int = 100, db: Session = Depends(database.get_db)):
    return db.query(models.Zone).offset(skip).limit(limit).all()

@app.post("/api/v1/zones", response_model=schemas.Zone)
def create_zone(zone: schemas.ZoneCreate, db: Session = Depends(database.get_db)):
    db_zone = models.Zone(**zone.dict())
    db.add(db_zone)
    db.commit()
    db.refresh(db_zone)
    return db_zone

@app.put("/api/v1/zones/{zone_id}", response_model=schemas.Zone)
def update_zone(zone_id: int, zone_update: schemas.ZoneUpdate, db: Session = Depends(database.get_db)):
    db_zone = db.query(models.Zone).filter(models.Zone.id == zone_id).first()
    if not db_zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    
    update_data = zone_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_zone, key, value)
        
    db.commit()
    db.refresh(db_zone)
    return db_zone

# --- Cameras ---
@app.get("/api/v1/cameras", response_model=List[schemas.Camera])
def get_cameras(skip: int = 0, limit: int = 100, db: Session = Depends(database.get_db)):
    return db.query(models.Camera).offset(skip).limit(limit).all()

@app.post("/api/v1/cameras", response_model=schemas.Camera)
def create_camera(camera: schemas.CameraCreate, db: Session = Depends(database.get_db)):
    db_camera = models.Camera(**camera.dict())
    db.add(db_camera)
    db.commit()
    db.refresh(db_camera)
    return db_camera

# --- Videos & Jobs ---
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/v1/videos/upload", response_model=schemas.Video)
def upload_video(file: UploadFile = File(...), db: Session = Depends(database.get_db)):
    file_extension = os.path.splitext(file.filename)[1]
    if file_extension.lower() not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    video_id = str(uuid.uuid4())
    filename = f"{video_id}{file_extension}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    db_video = models.Video(id=video_id, filename=file.filename, filepath=filepath)
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    return db_video

# Store active processes in memory so we can terminate them
active_processes = {}

def run_ml_pipeline(job_id: str, video_path: str):
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        src_dir = os.path.join(root_dir, "notebook", "src")
        
        # Look for virtual environment python
        venv_python = os.path.join(root_dir, "ppe-env", "Scripts", "python.exe")
        import sys
        python_exe = venv_python if os.path.exists(venv_python) else sys.executable
        
        process = subprocess.Popen(
            [python_exe, "pipeline.py", "--video", video_path, "--job-id", job_id],
            cwd=src_dir
        )
        active_processes[job_id] = process
        process.wait()
    except Exception as e:
        print(f"Error running pipeline: {e}")
    finally:
        if job_id in active_processes:
            del active_processes[job_id]

@app.post("/api/v1/jobs/{video_id}/process", response_model=schemas.ProcessingJob)
def process_video(video_id: str, background_tasks: BackgroundTasks, db: Session = Depends(database.get_db)):
    if video_id.startswith("demo_"):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        filename_prefix = video_id[5:]
        filename = f"{filename_prefix}.mp4"
        
        if filename_prefix == "vidssave":
            video_path = os.path.join(root_dir, "docs", "positive_test", "vidssave.com PPE Safety Video.✅#safetyfirst #ppe #video 👷_♂️👷✅💟 #viralvideo 720P.mp4")
            filename = "vidssave.mp4"
        else:
            video_path = os.path.join(root_dir, "docs", filename)
            
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f"Demo video not found at {video_path}")
        
        # Ensure demo video exists in DB to satisfy foreign key constraint
        db_demo_video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if not db_demo_video:
            db_demo_video = models.Video(id=video_id, filename=filename, filepath=video_path)
            db.add(db_demo_video)
            db.commit()
    else:
        db_video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if not db_video:
            raise HTTPException(status_code=404, detail="Video not found")
        video_path = db_video.filepath
        
    job_id = str(uuid.uuid4())
    db_job = models.ProcessingJob(
        id=job_id, 
        video_id=video_id, 
        status="queued",
        started_at=datetime.utcnow()
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    
    background_tasks.add_task(run_ml_pipeline, job_id, video_path)
    return db_job

@app.get("/api/v1/jobs/{job_id}", response_model=schemas.ProcessingJob)
def get_job(job_id: str, db: Session = Depends(database.get_db)):
    db_job = db.query(models.ProcessingJob).filter(models.ProcessingJob.id == job_id).first()
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
    return db_job

from pydantic import BaseModel
class JobProgressUpdate(BaseModel):
    progress: int
    total_frames: int
    fps: float
    status: Optional[str] = None

@app.put("/api/v1/jobs/{job_id}/progress")
def update_job_progress(job_id: str, update: JobProgressUpdate, db: Session = Depends(database.get_db)):
    db_job = db.query(models.ProcessingJob).filter(models.ProcessingJob.id == job_id).first()
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    db_job.progress = update.progress
    db_job.total_frames = update.total_frames
    db_job.fps = update.fps
    if update.status:
        db_job.status = update.status
        if update.status in ["completed", "failed", "stopped"]:
            db_job.completed_at = datetime.utcnow()
            
    db.commit()
    return {"status": "ok", "job_status": db_job.status}

@app.post("/api/v1/jobs/{job_id}/stop")
def stop_job(job_id: str, db: Session = Depends(database.get_db)):
    db_job = db.query(models.ProcessingJob).filter(models.ProcessingJob.id == job_id).first()
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    db_job.status = "stopped"
    db_job.completed_at = datetime.utcnow()
    db.commit()
    
    if job_id in active_processes:
        try:
            active_processes[job_id].terminate()
            active_processes[job_id].wait(timeout=2)
        except Exception:
            try:
                active_processes[job_id].kill()
            except Exception:
                pass
        del active_processes[job_id]
        
    return {"status": "stopped"}

# --- Violations ---
@app.get("/api/v1/violations", response_model=List[schemas.ViolationEvent])
def get_violations(
    skip: int = 0, 
    limit: int = 100, 
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledged status"),
    db: Session = Depends(database.get_db)
):
    query = db.query(models.ViolationEvent)
    if acknowledged is not None:
        query = query.filter(models.ViolationEvent.acknowledged == acknowledged)
    return query.order_by(models.ViolationEvent.timestamp.desc()).offset(skip).limit(limit).all()

@app.post("/api/v1/violations", response_model=schemas.ViolationEvent)
def create_violation(event: schemas.ViolationEventCreate, db: Session = Depends(database.get_db)):
    db_event = models.ViolationEvent(**event.dict())
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event

@app.patch("/api/v1/violations/{id}/acknowledge", response_model=schemas.ViolationEvent)
def acknowledge_violation(id: int, db: Session = Depends(database.get_db)):
    db_event = db.query(models.ViolationEvent).filter(models.ViolationEvent.id == id).first()
    if not db_event:
        raise HTTPException(status_code=404, detail="Event not found")
    db_event.acknowledged = True
    db.commit()
    db.refresh(db_event)
    return db_event

@app.patch("/api/v1/violations/{id}/feedback", response_model=schemas.ViolationEvent)
def provide_feedback(id: int, feedback: schemas.ViolationFeedbackUpdate, db: Session = Depends(database.get_db)):
    db_event = db.query(models.ViolationEvent).filter(models.ViolationEvent.id == id).first()
    if not db_event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    db_event.feedback_correct = feedback.feedback_correct
    db_event.feedback_helmet = feedback.feedback_helmet
    db_event.feedback_vest = feedback.feedback_vest
    
    # Optionally automatically acknowledge if feedback is provided
    db_event.acknowledged = True
    
    db.commit()
    db.refresh(db_event)
    return db_event

# --- Metrics ---
@app.get("/api/v1/metrics")
def get_metrics(db: Session = Depends(database.get_db)):
    metric = db.query(models.InferenceMetric).order_by(models.InferenceMetric.timestamp.desc()).first()
    if metric:
        return {
            "fps": metric.fps,
            "latency_ms": metric.latency_ms,
            "model_version": metric.model_version,
            "timestamp": metric.timestamp.isoformat()
        }
    return {}

@app.post("/api/v1/metrics")
def post_metrics(data: dict, db: Session = Depends(database.get_db)):
    metric = models.InferenceMetric(
        fps=data.get("fps", 0),
        latency_ms=data.get("latency_ms", 0),
        model_version=data.get("model_version", "unknown")
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return {"id": metric.id, "status": "ok"}

# --- Dashboard Stats ---
@app.get("/api/v1/stats")
def get_stats(db: Session = Depends(database.get_db)):
    """Aggregated stats for the dashboard."""
    total_violations = db.query(models.ViolationEvent).count()
    active_violations = db.query(models.ViolationEvent).filter(
        models.ViolationEvent.acknowledged == False
    ).count()
    acknowledged = db.query(models.ViolationEvent).filter(
        models.ViolationEvent.acknowledged == True
    ).count()
    
    # Unique workers
    unique_workers = db.query(
        func.count(func.distinct(models.ViolationEvent.worker_tracking_id))
    ).scalar() or 0
    
    # Compliance rate (acknowledged / total)
    resolution_rate = round((acknowledged / total_violations * 100), 1) if total_violations > 0 else 100.0

    # Latest metric
    metric = db.query(models.InferenceMetric).order_by(models.InferenceMetric.timestamp.desc()).first()

    return {
        "total_violations": total_violations,
        "active_violations": active_violations,
        "acknowledged": acknowledged,
        "resolution_rate": resolution_rate,
        "unique_workers": unique_workers,
        "latest_fps": metric.fps if metric else None,
        "latest_latency_ms": metric.latency_ms if metric else None,
        "model_version": metric.model_version if metric else "V3-HN",
    }
