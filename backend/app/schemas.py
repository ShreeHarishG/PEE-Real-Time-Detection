from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime

class ZoneBase(BaseModel):
    name: str
    type: str
    required_ppe: List[str]
    polygon: Optional[List[List[float]]] = None
    confidence_threshold: float
    min_seconds_in_zone: float
    is_active: bool

class ZoneCreate(ZoneBase):
    pass

class ZoneUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    required_ppe: Optional[List[str]] = None
    polygon: Optional[List[List[float]]] = None
    confidence_threshold: Optional[float] = None
    min_seconds_in_zone: Optional[float] = None
    is_active: Optional[bool] = None

class Zone(ZoneBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class VideoBase(BaseModel):
    filename: str
    filepath: str

class Video(VideoBase):
    id: str
    uploaded_at: datetime
    model_config = ConfigDict(from_attributes=True)

class ProcessingJobBase(BaseModel):
    video_id: str

class ProcessingJob(ProcessingJobBase):
    id: str
    status: str
    progress: int
    total_frames: int
    fps: float
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    workers_detected: Optional[int] = 0
    violations_detected: Optional[int] = 0
    output_video_path: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)

class CameraBase(BaseModel):
    name: str
    source_type: str
    rtsp_url: Optional[str] = None
    local_video_path: Optional[str] = None
    is_active: bool = True
    zone_id: Optional[int] = None

class CameraCreate(CameraBase):
    pass

class Camera(CameraBase):
    id: int
    zone: Optional[Zone] = None
    model_config = ConfigDict(from_attributes=True)

class ViolationEventBase(BaseModel):
    event_id: str
    camera_id: int
    zone_id: int
    worker_tracking_id: int
    violation_type: str
    missing_ppe: List[str]
    confidence: float
    evidence_image_path: str
    evidence_video_path: Optional[str] = None
    model_version: str
    feedback_correct: Optional[bool] = None
    feedback_helmet: Optional[bool] = None
    feedback_vest: Optional[bool] = None
    feedback_boots: Optional[bool] = None
    feedback_harness: Optional[bool] = None
    job_id: Optional[str] = None
    video_timestamp_sec: Optional[float] = None

class ViolationEventCreate(ViolationEventBase):
    pass

class ViolationFeedbackUpdate(BaseModel):
    feedback_correct: bool
    feedback_helmet: bool
    feedback_vest: bool
    feedback_boots: bool
    feedback_harness: bool

class ViolationEvent(ViolationEventBase):
    id: int
    timestamp: datetime
    acknowledged: bool
    camera: Optional[Camera] = None
    zone: Optional[Zone] = None
    model_config = ConfigDict(from_attributes=True)

class WorkerCompliance(BaseModel):
    worker_tracking_id: int
    total_violations: int
    violation_history: List[ViolationEvent]

