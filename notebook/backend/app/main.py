from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import os
import time
import uuid
import subprocess
import shutil
import logging
import re
import threading
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

os.makedirs(os.path.join(outputs_dir, "results"), exist_ok=True)
app.mount("/results", StaticFiles(directory=os.path.join(outputs_dir, "results")), name="results")

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# The API owns process lifecycle; browser clients only consume status/frames.
active_processes = {}
active_camera_jobs = {}

demo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "14_DEMO"))
if os.path.exists(demo_dir):
    app.mount("/demo", StaticFiles(directory=demo_dir), name="demo")

import cv2
from fastapi.responses import Response

import asyncio

logger = logging.getLogger("edgevision.api")

def _live_frame_path(job_id: Optional[str]) -> str:
    if job_id and re.fullmatch(r"[0-9a-fA-F-]{8,64}", job_id):
        return os.path.join(outputs_dir, "live_camera", f"{job_id}.jpg")
    return os.path.join(outputs_dir, "latest_frame.jpg")

async def generate_frames(request: Request, job_id: Optional[str]):
    frame_path = _live_frame_path(job_id)
    last_mtime = 0
    while True:
        try:
            if await request.is_disconnected():
                logger.info("[STREAM] client disconnected job_id=%s", job_id)
                return
            if os.path.exists(frame_path):
                mtime = os.path.getmtime(frame_path)
                if mtime > last_mtime:
                    with open(frame_path, "rb") as f:
                        frame = f.read()
                    if not frame:
                        raise ValueError("empty JPEG frame")
                    last_mtime = mtime
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            await asyncio.sleep(0.03)
        except asyncio.CancelledError:
            return
        except (FileNotFoundError, PermissionError, OSError, ValueError) as exc:
            logger.debug("[STREAM] job_id=%s waiting for frame: %s", job_id, exc)
            await asyncio.sleep(0.03)

@app.get("/api/v1/stream")
async def video_stream(request: Request, jobId: Optional[str] = Query(None)):
    return StreamingResponse(generate_frames(request, jobId), media_type="multipart/x-mixed-replace; boundary=frame")

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

def resolve_camera_source(source: str):
    """Keep device indexes as indexes; only filesystem sources are resolved."""
    if source.strip().isdigit():
        return source.strip()
    if source.startswith(("rtsp://", "http://", "https://")) or os.path.isabs(source):
        return source
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(root_dir, source)

# --- Zones ---
@app.get("/api/v1/zones", response_model=List[schemas.Zone])
def get_zones(db: Session = Depends(database.get_db)):
    zones = db.query(models.Zone).all()
    if not zones:
        # Create default zone if none exists
        default_zone = models.Zone(
            name="construction",
            type="construction",
            required_ppe=["helmet", "vest"],
            confidence_threshold=0.25,
            min_seconds_in_zone=2.0,
            is_active=True
        )
        db.add(default_zone)
        db.commit()
        db.refresh(default_zone)
        return [default_zone]
    return zones

@app.put("/api/v1/zones/{id}", response_model=schemas.Zone)
def update_zone(id: int, zone_update: schemas.ZoneUpdate, db: Session = Depends(database.get_db)):
    db_zone = db.query(models.Zone).filter(models.Zone.id == id).first()
    if not db_zone:
        raise HTTPException(status_code=404, detail="Zone not found")
        
    update_data = zone_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_zone, key, value)
        
    db.commit()
    db.refresh(db_zone)
    return db_zone

# --- Videos & Jobs ---

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

def _get_video_path(video_id: str, db: Session):
    if video_id.startswith("demo_"):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        filename_prefix = video_id[5:]
        filename = f"{filename_prefix}.mp4"
        if filename_prefix == "vidssave":
            video_path = os.path.join(root_dir, "14_DEMO", "vidssave.com PPE Safety Video.✅#safetyfirst #ppe #video 👷_♂️👷✅💟 #viralvideo 720P.mp4")
        else:
            video_path = os.path.join(root_dir, "14_DEMO", filename)
            if not os.path.exists(video_path):
                video_path = os.path.join(root_dir, "docs", filename)
        return video_path
    
    db_video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if db_video:
        return db_video.filepath
    return None

@app.get("/api/v1/videos/{video_id}/snapshot")
def get_video_snapshot(video_id: str, db: Session = Depends(database.get_db)):
    video_path = _get_video_path(video_id, db)
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
        
    cap = cv2.VideoCapture(video_path)
    # Seek to 1 second in to avoid black frames
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        # Return a fallback image instead of 500
        import numpy as np
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, "VIDEO READ ERROR", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
    # Resize to max 1280 to save bandwidth
    h, w = frame.shape[:2]
    if w > 1280:
        scale = 1280 / w
        frame = cv2.resize(frame, (1280, int(h * scale)))
        
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.get("/api/v1/cameras/{camera_id}/snapshot")
def get_camera_snapshot(camera_id: int, db: Session = Depends(database.get_db)):
    db_camera = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
    if not db_camera:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    source = db_camera.rtsp_url if db_camera.rtsp_url else db_camera.local_video_path
    if not source:
        raise HTTPException(status_code=400, detail="Camera has no video source")
        
    active_job_id = active_camera_jobs.get(camera_id)
    if active_job_id:
        frame_path = _live_frame_path(active_job_id)
        if os.path.exists(frame_path):
            with open(frame_path, "rb") as frame_file:
                return Response(content=frame_file.read(), media_type="image/jpeg")
        # Never open a competing camera connection while a live job owns it.
        import numpy as np
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, "CONNECTING CAMERA", (350, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 180, 255), 3)
        _, buffer = cv2.imencode('.jpg', frame)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    source = resolve_camera_source(source)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    # Try standard open without forcing DSHOW
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
        
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        # Return a fallback image instead of 500
        import numpy as np
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, "STREAM OFFLINE", (420, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
    h, w = frame.shape[:2]
    if w > 1280:
        scale = 1280 / w
        frame = cv2.resize(frame, (1280, int(h * scale)))
        
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.delete("/api/v1/cameras/{camera_id}")
def delete_camera(camera_id: int, db: Session = Depends(database.get_db)):
    db_camera = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
    if not db_camera:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    db.delete(db_camera)
    db.commit()
    return {"status": "success"}


def run_ml_pipeline(job_id: str, video_path: str, camera_id: Optional[int] = None):
    process = None
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        src_dir = os.path.join(root_dir, "notebook", "src")
        
        # Add src_dir to sys path dynamically so we can import platform_resolver if not already there
        import sys
        if src_dir not in sys.path:
            sys.path.append(src_dir)
        import platform_resolver
        
        python_exe = platform_resolver.get_python_exe(root_dir)
        
        command = [python_exe, "pipeline.py", "--video", video_path, "--job-id", job_id]
        if camera_id is not None:
            command.extend(["--camera-id", str(camera_id)])
        log_dir = os.path.join(root_dir, "notebook", "outputs", "live_camera_logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"{job_id}.log"), "a", encoding="utf-8") as job_log:
            process = subprocess.Popen(command, cwd=src_dir, stdout=job_log, stderr=subprocess.STDOUT)
            active_processes[job_id] = process
            return_code = process.wait()
        if return_code != 0:
            db = database.SessionLocal()
            try:
                job = db.query(models.ProcessingJob).filter(models.ProcessingJob.id == job_id).first()
                if job and job.status not in {"stopped", "camera_unavailable", "completed"}:
                    job.status = "failed"
                    job.error_message = f"Pipeline exited with code {return_code}; see outputs/live_camera_logs/{job_id}.log"
                    job.completed_at = datetime.utcnow()
                    db.commit()
            finally:
                db.close()
    except Exception as e:
        logger.exception("[JOB] pipeline launch failed job_id=%s: %s", job_id, e)
        db = database.SessionLocal()
        try:
            job = db.query(models.ProcessingJob).filter(models.ProcessingJob.id == job_id).first()
            if job and job.status != "stopped":
                job.status = "failed"
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()
    finally:
        if job_id in active_processes:
            del active_processes[job_id]
        if camera_id is not None and active_camera_jobs.get(camera_id) == job_id:
            del active_camera_jobs[camera_id]

@app.post("/api/v1/jobs/{video_id}/process", response_model=schemas.ProcessingJob)
def process_video(video_id: str, background_tasks: BackgroundTasks, db: Session = Depends(database.get_db)):
    if video_id.startswith("demo_"):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        filename_prefix = video_id[5:]
        filename = f"{filename_prefix}.mp4"
        
        if filename_prefix == "vidssave":
            video_path = os.path.join(root_dir, "14_DEMO", "vidssave.com PPE Safety Video.✅#safetyfirst #ppe #video 👷_♂️👷✅💟 #viralvideo 720P.mp4")
            filename = "vidssave.mp4"
        else:
            video_path = os.path.join(root_dir, "14_DEMO", filename)
            
        if not os.path.exists(video_path):
            # Fallback to older docs folder if it was placed there
            video_path = os.path.join(root_dir, "docs", filename)
            
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f"Demo video not found at {video_path}")
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

@app.post("/api/v1/cameras/{camera_id}/process", response_model=schemas.ProcessingJob)
def process_camera(camera_id: int, background_tasks: BackgroundTasks, db: Session = Depends(database.get_db)):
    db_camera = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
    if not db_camera:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    source = db_camera.rtsp_url if db_camera.rtsp_url else db_camera.local_video_path
    if not source:
        raise HTTPException(status_code=400, detail="Camera has no source video URL")
        
    source = resolve_camera_source(source)
        
    job_id = str(uuid.uuid4())
    
    # We create a dummy Video record to satisfy the foreign key, or just use the camera.
    # Since processing_jobs requires a video_id, let's create a virtual video record for this stream session.
    video_id = f"stream_{camera_id}_{int(time.time())}"
    db_video = models.Video(id=video_id, filename=f"Camera {camera_id} Stream", filepath=source)
    db.add(db_video)
    db.commit()
    
    db_job = models.ProcessingJob(
        id=job_id, 
        video_id=video_id, 
        status="connecting",
        started_at=datetime.utcnow()
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    
    active_camera_jobs[camera_id] = job_id
    background_tasks.add_task(run_ml_pipeline, job_id, source, camera_id)
    return db_job

@app.get("/api/v1/jobs/{job_id}", response_model=schemas.ProcessingJob)
def get_job(job_id: str, db: Session = Depends(database.get_db)):
    db_job = db.query(models.ProcessingJob).filter(models.ProcessingJob.id == job_id).first()
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
    return db_job

class JobProgressUpdate(BaseModel):
    progress: int
    total_frames: int
    fps: float
    status: Optional[str] = None
    workers_detected: Optional[int] = 0
    violations_detected: Optional[int] = 0
    error_message: Optional[str] = None

@app.put("/api/v1/jobs/{job_id}/progress")
def update_job_progress(job_id: str, update: JobProgressUpdate, db: Session = Depends(database.get_db)):
    db_job = db.query(models.ProcessingJob).filter(models.ProcessingJob.id == job_id).first()
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    # A late capture-thread update must never resurrect a user-stopped job.
    if db_job.status == "stopped" and update.status not in (None, "stopped"):
        return {"status": "ok", "job_status": db_job.status}

    db_job.progress = update.progress
    db_job.total_frames = update.total_frames
    db_job.fps = update.fps
    db_job.workers_detected = update.workers_detected
    db_job.violations_detected = update.violations_detected
    if update.error_message is not None:
        db_job.error_message = update.error_message
    if update.status:
        db_job.status = update.status
        if update.status in ["completed", "failed", "stopped", "camera_unavailable"]:
            db_job.completed_at = datetime.utcnow()
            if update.status == "completed":
                db_job.output_video_path = os.path.join("outputs", "results", f"{job_id}.mp4")
            
    db.commit()
    return {"status": "ok", "job_status": db_job.status}

@app.get("/api/v1/jobs/{job_id}/violations", response_model=List[schemas.ViolationEvent])
def get_job_violations(job_id: str, db: Session = Depends(database.get_db)):
    return db.query(models.ViolationEvent).filter(models.ViolationEvent.job_id == job_id).order_by(models.ViolationEvent.timestamp.asc()).all()

@app.get("/api/v1/jobs/{job_id}/video")
def get_job_video(job_id: str, db: Session = Depends(database.get_db)):
    db_job = db.query(models.ProcessingJob).filter(models.ProcessingJob.id == job_id).first()
    if not db_job or not db_job.output_video_path:
        raise HTTPException(status_code=404, detail="Video not found")
    filename = os.path.basename(db_job.output_video_path)
    return {"url": f"http://localhost:8000/results/{filename}"}

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
            active_processes[job_id].wait(timeout=3)
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
        "model_version": metric.model_version if metric else "V4-Boots",
    }
