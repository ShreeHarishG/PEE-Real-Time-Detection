from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
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

# --- Health ---
@app.get("/api/v1/health")
def read_health():
    return {"status": "ok"}

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

# --- Violations ---
@app.get("/api/v1/violations", response_model=List[schemas.ViolationEvent])
def get_violations(skip: int = 0, limit: int = 100, db: Session = Depends(database.get_db)):
    return db.query(models.ViolationEvent).order_by(models.ViolationEvent.timestamp.desc()).offset(skip).limit(limit).all()

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

# --- Metrics ---
@app.get("/api/v1/metrics")
def get_metrics(db: Session = Depends(database.get_db)):
    # Return latest inference metric
    metric = db.query(models.InferenceMetric).order_by(models.InferenceMetric.timestamp.desc()).first()
    return metric if metric else {}
