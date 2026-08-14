"""
EdgeVision V3 - Core Inference Pipeline
Detects persons and PPE, associates them, validates temporally, and logs violations.
"""
import os
import sys
import time
import uuid
import logging
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Set, Optional

import cv2
import numpy as np
import pandas as pd
import yaml
import argparse
from ultralytics import YOLO

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION LOADING
# ==============================================================================
def load_config() -> dict:
    """Load configuration from config/model_versions.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'model_versions.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

CONFIG = load_config()

# Pipeline configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ACTIVE_VERSION = CONFIG.get("production", {}).get("version", "V3-HN")
MODEL_CFG = CONFIG.get(ACTIVE_VERSION.lower(), {})  # looks up 'v3-hn' key
PPE_MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_CFG.get("path", "models/ppe_v3_hn_best.pt"))
PERSON_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8n.pt")


DEFAULT_INPUT_VIDEO = os.path.join(PROJECT_ROOT, "..", "docs", "test.mp4")
OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "outputs", "results", "annotated_output_functional.mp4")
OUTPUT_LOG = os.path.join(PROJECT_ROOT, "outputs", "results", "violations_functional.csv")
EVIDENCE_DIR = os.path.join(PROJECT_ROOT, "outputs", "evidence")

CONF_THRESHOLD = 0.25
MIN_ASSOC_SCORE = 0.40
IMGSZ = MODEL_CFG.get("imgsz", 512)
TRACKER_CONFIG = "bytetrack.yaml"
PERSON_CLASS_ID = 0
UNIFIED_CLASSES = ["person", "helmet", "no_helmet", "vest", "boots", "harness"]

ZONE_RULES = {
    "construction": {"required": ["helmet", "vest"]}
}
ACTIVE_ZONE = "construction"

# Ensure output directories exist
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_LOG), exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# ==============================================================================
# PIPELINE COMPONENTS
# ==============================================================================
class TemporalValidator:
    """Validates violations using temporal hysteresis to prevent alert fatigue."""
    def __init__(self, window_size: int = 10, violation_threshold: int = 8, min_seconds_in_zone: float = 2.0, fps: int = 25):
        self.window_size = window_size
        self.violation_threshold = violation_threshold
        self.min_frames_in_zone = int(min_seconds_in_zone * fps)
        self.history = defaultdict(lambda: deque(maxlen=window_size))
        self.frames_in_zone = defaultdict(int)
        self.confirmed = set()

    def update(self, track_id: int, has_violation: bool, confidence: float, conf_threshold: float = 0.4) -> bool:
        self.frames_in_zone[track_id] += 1
        is_violation = has_violation and confidence >= conf_threshold
        self.history[track_id].append(is_violation)

        if (sum(self.history[track_id]) >= self.violation_threshold and
                self.frames_in_zone[track_id] >= self.min_frames_in_zone and
                track_id not in self.confirmed):
            self.confirmed.add(track_id)
            return True
        return False

def check_violations(worn_ppe: Set[str], zone: str = ACTIVE_ZONE) -> List[str]:
    """Check what required PPE is missing."""
    required = ZONE_RULES[zone]["required"]
    return [item for item in required if item not in worn_ppe]

def box_center(box: List[float]) -> Tuple[float, float]:
    """Calculate center coordinates of a bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def associate_ppe_to_persons(person_boxes: List[List[float]], person_ids: List[int], 
                             ppe_boxes: List[List[float]], ppe_classes: List[str], margin: int = 50) -> Dict[int, List[str]]:
    """Associate PPE detections to person tracks using spatial containment."""
    assignments = defaultdict(list)
    for p_box, p_class in zip(ppe_boxes, ppe_classes):
        cx, cy = box_center(p_box)
        best_id, best_dist = None, float("inf")
        for person_box, pid in zip(person_boxes, person_ids):
            x1, y1, x2, y2 = person_box
            if (x1 - margin) <= cx <= (x2 + margin) and (y1 - margin) <= cy <= (y2 + margin):
                pcx, pcy = box_center(person_box)
                dist = (pcx - cx) ** 2 + (pcy - cy) ** 2
                if dist < best_dist:
                    best_dist, best_id = dist, pid
        if best_id is not None:
            assignments[best_id].append(p_class)
    return assignments

# ==============================================================================
# MAIN INFERENCE LOOP
# ==============================================================================
import requests

def get_zone_rules_from_api(api_url="http://localhost:8000/api/v1/zones"):
    try:
        response = requests.get(api_url, timeout=2)
        if response.status_code == 200:
            zones = response.json()
            return {z["name"]: {"required": z["required_ppe"], "id": z["id"], "polygon": z.get("polygon")} for z in zones}
    except Exception as e:
        logger.warning(f"Failed to fetch zones from API: {e}. Falling back to default.")
    return {"construction": {"required": ["helmet", "vest"], "id": 1, "polygon": None}}

def post_violation(event_data, api_url="http://localhost:8000/api/v1/violations"):
    try:
        requests.post(api_url, json=event_data, timeout=1)
    except Exception as e:
        logger.error(f"Failed to POST violation event: {e}")

def post_metrics(metrics_data, api_url="http://localhost:8000/api/v1/metrics"):
    try:
        requests.post(api_url, json=metrics_data, timeout=1)
    except Exception as e:
        logger.debug(f"Failed to POST metrics: {e}")

def update_job_progress(job_id, progress, total_frames, fps, status=None, api_url="http://localhost:8000/api/v1/jobs"):
    try:
        payload = {"progress": progress, "total_frames": total_frames, "fps": fps}
        if status:
            payload["status"] = status
        response = requests.put(f"{api_url}/{job_id}/progress", json=payload, timeout=1)
        if response.status_code == 200:
            return response.json().get("job_status")
    except Exception as e:
        logger.debug(f"Failed to update job progress: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description="EdgeVision V3 Inference Pipeline")
    parser.add_argument("--video", type=str, default=DEFAULT_INPUT_VIDEO, help="Path to input video file")
    parser.add_argument("--job-id", type=str, default=None, help="Job ID for tracking progress via DB")
    args = parser.parse_args()
    
    input_video_path = os.path.abspath(args.video)

    if not os.path.exists(input_video_path):
        logger.error(f"Input video not found: {input_video_path}")
        sys.exit(1)

    if not os.path.exists(PPE_MODEL_PATH):
        logger.error(f"Model not found at {PPE_MODEL_PATH}. Please ensure models are correctly placed.")
        sys.exit(1)

    logger.info(f"Starting EdgeVision Pipeline (Model: {ACTIVE_VERSION})")
    
    # Load Models
    try:
        ppe_model = YOLO(PPE_MODEL_PATH).to('cuda')
        person_model = YOLO(PERSON_MODEL_PATH).to('cuda')
    except Exception as e:
        logger.error(f"Failed to load models. Ensure CUDA is available: {e}")
        sys.exit(1)

    # Initialize Video
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    validator = TemporalValidator(fps=fps)
    log_rows = []
    
    # Fetch Zones
    dynamic_zones = get_zone_rules_from_api()
    zone_data = dynamic_zones.get(ACTIVE_ZONE, {})
    zone_id = zone_data.get("id", 1)
    polygon = zone_data.get("polygon")
    
    if args.job_id:
        update_job_progress(args.job_id, 0, total_frames, 0, status="processing")
    
    # Metrics
    metrics = {
        "frames": 0,
        "frames_zero_people": 0,
        "person_detections": 0,
        "unique_track_ids": set()
    }
    
    logger.info("Processing video stream...")
    start_time = time.time()
    last_metric_time = time.time()

    while True:
        # Removed real-time skipping to ensure smooth processing (even if slower)
            
        frame_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        metrics["frames"] += 1

        # Person Detection & Tracking
        p_result = person_model.track(frame, persist=True, classes=[PERSON_CLASS_ID],
                                      tracker=TRACKER_CONFIG, conf=CONF_THRESHOLD, verbose=False, imgsz=IMGSZ, half=True)[0]
        
        person_boxes = p_result.boxes.xyxy.cpu().numpy().tolist() if p_result.boxes.id is not None else []
        person_ids = p_result.boxes.id.int().cpu().tolist() if p_result.boxes.id is not None else []

        if len(person_boxes) == 0:
            metrics["frames_zero_people"] += 1
            
        metrics["person_detections"] += len(person_boxes)
        metrics["unique_track_ids"].update(person_ids)

        # PPE Detection
        ppe_result = ppe_model(frame, conf=CONF_THRESHOLD, verbose=False, imgsz=IMGSZ, half=True)[0]
        ppe_boxes = ppe_result.boxes.xyxy.cpu().numpy().tolist()
        ppe_classes = [UNIFIED_CLASSES[int(c)] for c in ppe_result.boxes.cls.cpu().numpy()]
        ppe_confs = ppe_result.boxes.conf.cpu().numpy().tolist()

        # Association
        assignments = associate_ppe_to_persons(person_boxes, person_ids, ppe_boxes, ppe_classes)
        
        # Draw zone polygon if exists
        abs_polygon = None
        if polygon:
            abs_polygon = np.array([[int(p[0] * w), int(p[1] * h)] for p in polygon], np.int32)
            cv2.polylines(frame, [abs_polygon], True, (0, 255, 255), 2)

        # Rule validation & Evidence Generation
        for p_box, pid in zip(person_boxes, person_ids):
            worn = set(assignments.get(pid, []))
            # Use dynamic rules if available
            required = dynamic_zones.get(ACTIVE_ZONE, {}).get("required", ZONE_RULES["construction"]["required"])
            missing = [item for item in required if item not in worn]
            has_violation = len(missing) > 0
            avg_conf = np.mean(ppe_confs) if ppe_confs else 0.5
            
            x1, y1, x2, y2 = map(int, p_box)
            
            # Spatial Check
            in_zone = True
            if abs_polygon is not None:
                bottom_center = (int((x1 + x2) / 2), int(y2))
                in_zone = cv2.pointPolygonTest(abs_polygon, bottom_center, False) >= 0
                if not in_zone:
                    has_violation = False
            
            # Rendering
            color = (0, 0, 255) if has_violation else ((0, 255, 0) if in_zone else (128, 128, 128))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text_prefix = f"ID:{pid}" if in_zone else f"ID:{pid} [OUTSIDE]"
            cv2.putText(frame, f"{text_prefix} {worn}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Temporal Confirmation
            if validator.update(pid, has_violation, avg_conf):
                event_id = str(uuid.uuid4())[:8]
                rel_ev_path = os.path.join("outputs", "evidence", f"{event_id}.jpg")
                abs_ev_path = os.path.join(PROJECT_ROOT, rel_ev_path)
                
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size > 0:
                    cv2.imwrite(abs_ev_path, crop)
                    
                event_payload = {
                    "event_id": event_id,
                    "camera_id": 1, # Default mock camera
                    "zone_id": zone_id,
                    "worker_tracking_id": pid,
                    "violation_type": "missing_ppe",
                    "missing_ppe": missing,
                    "confidence": round(float(avg_conf), 2),
                    "evidence_image_path": rel_ev_path,
                    "evidence_video_path": None,
                    "model_version": ACTIVE_VERSION
                }
                
                import threading
                threading.Thread(target=post_violation, args=(event_payload,)).start()

        writer.write(frame)
        try:
            final_frame_path = os.path.join(PROJECT_ROOT, "outputs", "latest_frame.jpg")
            # Write directly to avoid Windows os.replace lock issues
            cv2.imwrite(final_frame_path, frame)
        except Exception as e:
            # Safely skip if FastAPI is actively reading
            pass
        
        frame_latency_ms = (time.time() - frame_start_time) * 1000
        
        # Post metrics every 30 frames
        if metrics["frames"] % 30 == 0:
            current_time = time.time()
            current_fps = 30 / (current_time - last_metric_time)
            last_metric_time = current_time
            import threading
            metrics_payload = {
                "fps": current_fps,
                "latency_ms": frame_latency_ms,
                "model_version": ACTIVE_VERSION
            }
            threading.Thread(target=post_metrics, args=(metrics_payload,)).start()
            
            if args.job_id:
                # Update job progress and check for early stop
                current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                job_status = update_job_progress(args.job_id, current_pos, total_frames, current_fps)
                if job_status == "stopped":
                    logger.info("Job was stopped via API.")
                    break

    cap.release()
    writer.release()
    
    elapsed = time.time() - start_time
    avg_fps = metrics["frames"] / elapsed if elapsed > 0 else 0
    
    if args.job_id:
        # Avoid overriding 'stopped' if it was cancelled
        current_status = update_job_progress(args.job_id, total_frames, total_frames, avg_fps)
        if current_status != "stopped":
            update_job_progress(args.job_id, total_frames, total_frames, avg_fps, status="completed")

    logger.info("Pipeline processing completed.")
    logger.info(f"Total Frames: {metrics['frames']}")
    logger.info(f"Average FPS: {avg_fps:.2f}")
    logger.info(f"Total Unique IDs: {len(metrics['unique_track_ids'])}")

if __name__ == "__main__":
    main()

