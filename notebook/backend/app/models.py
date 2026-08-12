from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
import datetime
from .database import Base

class Camera(Base):
    __tablename__ = "cameras"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    source_type = Column(String) # RTSP, LOCAL, SIMULATION
    rtsp_url = Column(String, nullable=True)
    local_video_path = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    zone_id = Column(Integer, ForeignKey("zones.id"), nullable=True)
    
    zone = relationship("Zone", back_populates="cameras")
    events = relationship("ViolationEvent", back_populates="camera")

class Zone(Base):
    __tablename__ = "zones"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    type = Column(String) # construction, general, height, restricted
    required_ppe = Column(JSON) # e.g. ["helmet", "vest"]
    confidence_threshold = Column(Float, default=0.25)
    min_seconds_in_zone = Column(Float, default=2.0)
    is_active = Column(Boolean, default=True)
    
    cameras = relationship("Camera", back_populates="zone")
    events = relationship("ViolationEvent", back_populates="zone")

class ViolationEvent(Base):
    __tablename__ = "violation_events"
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String, unique=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"))
    zone_id = Column(Integer, ForeignKey("zones.id"))
    worker_tracking_id = Column(Integer)
    violation_type = Column(String) # missing_ppe, restricted_zone
    missing_ppe = Column(JSON) # e.g. ["helmet"]
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    evidence_image_path = Column(String)
    evidence_video_path = Column(String, nullable=True)
    acknowledged = Column(Boolean, default=False)
    model_version = Column(String)
    
    camera = relationship("Camera", back_populates="events")
    zone = relationship("Zone", back_populates="events")

class InferenceMetric(Base):
    __tablename__ = "inference_metrics"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    fps = Column(Float)
    latency_ms = Column(Float)
    model_version = Column(String)
