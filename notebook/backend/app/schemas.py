from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime

class ZoneBase(BaseModel):
    name: str
    type: str
    required_ppe: List[str]
    confidence_threshold: float
    min_seconds_in_zone: float
    is_active: bool

class ZoneCreate(ZoneBase):
    pass

class Zone(ZoneBase):
    id: int
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

class ViolationEventCreate(ViolationEventBase):
    pass

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

