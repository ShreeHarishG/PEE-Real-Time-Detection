"""
EdgeVision V3 - Core Inference Pipeline
Detects persons and PPE, associates them, validates temporally, and logs violations.
"""
import os
import sys
import time
import uuid
import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Set, Optional

import cv2
import numpy as np
import pandas as pd
import yaml
import argparse
from ultralytics import YOLO
from camera_capture import CameraCapture, CameraCaptureConfig, normalize_camera_source

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


DEFAULT_INPUT_VIDEO = os.path.join(PROJECT_ROOT, "..", "14_DEMO", "test.mp4")
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
os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "results"), exist_ok=True)
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
import os

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

def get_zone_rules_from_api(api_url=f"{API_BASE_URL}/api/v1/zones"):
    try:
        response = requests.get(api_url, timeout=2)
        if response.status_code == 200:
            zones = response.json()
            return {z["id"]: {"required": z["required_ppe"], "name": z["name"], "polygon": z.get("polygon")} for z in zones}
    except Exception as e:
        logger.warning(f"Failed to fetch zones from API: {e}. Falling back to default.")
    return {1: {"required": ["helmet", "vest"], "name": "construction", "polygon": None}}

def post_violation(event_data, api_url=f"{API_BASE_URL}/api/v1/violations"):
    try:
        requests.post(api_url, json=event_data, timeout=1)
    except Exception as e:
        logger.error(f"Failed to POST violation event: {e}")

def post_metrics(metrics_data, api_url=f"{API_BASE_URL}/api/v1/metrics"):
    try:
        requests.post(api_url, json=metrics_data, timeout=1)
    except Exception as e:
        logger.debug(f"Failed to POST metrics: {e}")

def update_job_progress(job_id, progress, total_frames, fps, workers_detected=0, violations_detected=0, status=None, api_url=f"{API_BASE_URL}/api/v1/jobs"):
    try:
        payload = {
            "progress": progress, 
            "total_frames": total_frames, 
            "fps": fps,
            "workers_detected": workers_detected,
            "violations_detected": violations_detected
        }
        if status:
            payload["status"] = status
        response = requests.put(f"{api_url}/{job_id}/progress", json=payload, timeout=1)
        if response.status_code == 200:
            return response.json().get("job_status")
    except Exception as e:
        logger.debug(f"Failed to update job progress: {e}")
    return None

def configure_live_job_logging(job_id: Optional[str]) -> Optional[logging.Handler]:
    """Add a job-specific rotating log without changing upload-job logging."""
    if not job_id:
        return None
    log_dir = os.path.join(PROJECT_ROOT, "outputs", "live_camera_logs")
    os.makedirs(log_dir, exist_ok=True)
    handler = RotatingFileHandler(
        os.path.join(log_dir, f"{job_id}.log"), maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return handler

def publish_live_frame(frame: np.ndarray, job_id: Optional[str]) -> bool:
    """Publish an MJPEG frame atomically so a reader never receives a partial JPEG."""
    if not job_id:
        return False
    frame_dir = os.path.join(PROJECT_ROOT, "outputs", "live_camera")
    os.makedirs(frame_dir, exist_ok=True)
    target = os.path.join(frame_dir, f"{job_id}.jpg")
    # Keep the temporary extension simple. Some Windows endpoint-protection
    # policies deny Python writes to compound names such as ``.jpg.<pid>.tmp``.
    temporary = os.path.join(frame_dir, f"{job_id}.tmp")
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        logger.warning("[STREAM] job_id=%s JPEG encoding failed", job_id)
        return False
    try:
        wrote_temporary = False
        for _ in range(3):
            try:
                with open(temporary, "wb") as output:
                    output.write(encoded.tobytes())
                wrote_temporary = True
                break
            except PermissionError:
                time.sleep(0.01)
        if not wrote_temporary:
            logger.warning("[STREAM] job_id=%s temporary frame write was locked", job_id)
            return False
        # A browser can briefly retain a Windows read handle. Retrying the
        # replacement leaves capture and inference independent of the reader.
        for _ in range(3):
            try:
                os.replace(temporary, target)
                return True
            except PermissionError:
                time.sleep(0.01)
        logger.warning("[STREAM] job_id=%s atomic frame replacement was locked", job_id)
    except OSError as exc:
        logger.warning("[STREAM] job_id=%s frame publish failed: %s", job_id, exc)
    finally:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass
    return False

def main():
    parser = argparse.ArgumentParser(description="EdgeVision V3 Inference Pipeline")
    parser.add_argument("--video", type=str, default=DEFAULT_INPUT_VIDEO, help="Path to input video file")
    parser.add_argument("--job-id", type=str, default=None, help="Job ID for tracking progress via DB")
    parser.add_argument("--camera-id", type=int, default=1, help="Database camera ID for live violation records")
    args = parser.parse_args()
    
    is_live_stream = False
    if args.video.isdigit():
        input_video_path = int(args.video)
        is_live_stream = True
    elif args.video.startswith(("rtsp://", "http://", "https://")):
        input_video_path = args.video
        is_live_stream = True
    else:
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

    live_log_handler = configure_live_job_logging(args.job_id) if is_live_stream else None
    live_capture = None
    cap = None
    if is_live_stream:
        def report_camera_state(state, details):
            if not args.job_id:
                return
            # The callback runs on the capture thread. Keep HTTP/database work
            # off that thread so it cannot interrupt frame acquisition.
            def report():
                detail = None
                if state == "reconnecting":
                    detail = f"Camera interrupted; reconnect attempt {details.get('reconnects', 0)}/{live_capture.config.reconnect_attempts}"
                elif state == "camera_unavailable":
                    detail = details.get("last_error") or "Camera recovery attempts exhausted"
                update_job_progress(args.job_id, metrics["frames"], 0, metrics.get("current_fps", 0.0),
                                    workers_detected=len(metrics["unique_track_ids"]),
                                    violations_detected=metrics["confirmed_violations"], status=state,
                                    error_message=detail)
            import threading
            threading.Thread(target=report, daemon=True).start()

        # Metrics is initialised before the callback can report a state.
        metrics = {"frames": 0, "frames_zero_people": 0, "person_detections": 0,
                   "unique_track_ids": set(), "confirmed_violations": 0, "current_fps": 0.0}
        live_capture = CameraCapture(
            CameraCaptureConfig(source=normalize_camera_source(input_video_path)),
            logger=logger,
            on_state_change=report_camera_state,
        )
        logger.info("[JOB] job_id=%s camera_id=%s source=%r starting live capture", args.job_id, args.camera_id, input_video_path)
        if not live_capture.start():
            logger.error("[CAMERA] source=%r unavailable: %s", input_video_path, live_capture.stats["last_error"])
            if args.job_id:
                update_job_progress(args.job_id, 0, 0, 0, status="camera_unavailable")
            if live_log_handler:
                logger.removeHandler(live_log_handler)
                live_log_handler.close()
            return
        fps = live_capture.stats["camera_fps"] or 25
        total_frames = 0
        w, h = live_capture.stats["width"], live_capture.stats["height"]
    else:
        # The established upload/video path remains synchronous and unchanged.
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {input_video_path}")
            if args.job_id:
                update_job_progress(args.job_id, 0, 0, 0, status="failed")
            sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = os.path.join(PROJECT_ROOT, "outputs", "results", f"{args.job_id}.mp4") if args.job_id else os.path.join(PROJECT_ROOT, "outputs", "results", "annotated_output_functional.mp4")
    writer = None
    if not is_live_stream:
        writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))

    validator = TemporalValidator(fps=fps)
    log_rows = []
    
    evidence_dir_rel = os.path.join("outputs", "evidence", args.job_id) if args.job_id else os.path.join("outputs", "evidence")
    evidence_dir_abs = os.path.join(PROJECT_ROOT, evidence_dir_rel)
    os.makedirs(evidence_dir_abs, exist_ok=True)
    
    # Fetch Zones
    dynamic_zones = get_zone_rules_from_api()
    zone_data = dynamic_zones.get(1, {}) # Use zone ID 1
    zone_id = 1
    polygon = zone_data.get("polygon")
    
    if args.job_id and not is_live_stream:
        update_job_progress(args.job_id, 0, total_frames, 0, status="processing")
    
    # Metrics
    if not is_live_stream:
        metrics = {
            "frames": 0,
            "frames_zero_people": 0,
            "person_detections": 0,
            "unique_track_ids": set(),
            "confirmed_violations": 0,
            "current_fps": 0.0,
        }
    
    logger.info("Processing video stream...")
    start_time = time.time()
    last_metric_time = time.time()

    last_live_sequence = 0
    terminal_status = None
    while True:
        frame_start_time = time.time()
        if is_live_stream:
            last_live_sequence, frame = live_capture.get_latest(last_live_sequence, timeout_s=0.5)
            if frame is None:
                if live_capture.state == "camera_unavailable":
                    terminal_status = "camera_unavailable"
                    logger.error("[CAMERA] job_id=%s source=%r exhausted reconnect attempts: %s", args.job_id, input_video_path, live_capture.stats["last_error"])
                    break
                continue
        else:
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
            required = dynamic_zones.get(1, {}).get("required", ZONE_RULES["construction"]["required"])
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
                metrics["confirmed_violations"] += 1
                event_id = str(uuid.uuid4())[:8]
                rel_ev_path = os.path.join(evidence_dir_rel, f"{event_id}.jpg")
                abs_ev_path = os.path.join(PROJECT_ROOT, rel_ev_path)
                
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size > 0:
                    cv2.imwrite(abs_ev_path, crop)
                    
                video_timestamp_sec = metrics["frames"] / float(fps) if is_live_stream else float(cap.get(cv2.CAP_PROP_POS_FRAMES)) / float(fps)
                
                event_payload = {
                    "event_id": event_id,
                    "camera_id": args.camera_id,
                    "zone_id": zone_id,
                    "worker_tracking_id": pid,
                    "violation_type": "missing_ppe",
                    "missing_ppe": missing,
                    "confidence": round(float(avg_conf), 2),
                    "evidence_image_path": rel_ev_path.replace("\\", "/"),
                    "evidence_video_path": None,
                    "model_version": ACTIVE_VERSION,
                    "job_id": args.job_id,
                    "video_timestamp_sec": round(video_timestamp_sec, 3)
                }
                
                import threading
                threading.Thread(target=post_violation, args=(event_payload,)).start()

        if writer is not None:
            writer.write(frame)
        if is_live_stream:
            output_frame_written = publish_live_frame(frame, args.job_id)
            logger.debug("[STREAM] job_id=%s frame=%s output_frame_written=%s", args.job_id, metrics["frames"], output_frame_written)
        else:
            # Retain the legacy output for completed-video UI compatibility.
            try:
                cv2.imwrite(os.path.join(PROJECT_ROOT, "outputs", "latest_frame.jpg"), frame)
            except Exception:
                pass
        
        frame_latency_ms = (time.time() - frame_start_time) * 1000
        
        # Post metrics every 30 frames
        if metrics["frames"] % 30 == 0:
            current_time = time.time()
            current_fps = 30 / (current_time - last_metric_time)
            last_metric_time = current_time
            metrics["current_fps"] = current_fps
            import threading
            metrics_payload = {
                "fps": current_fps,
                "latency_ms": frame_latency_ms,
                "model_version": ACTIVE_VERSION
            }
            threading.Thread(target=post_metrics, args=(metrics_payload,)).start()
            
            if args.job_id:
                # Update job progress and check for early stop
                current_pos = metrics["frames"]
                workers_detected = len(metrics["unique_track_ids"])
                violations_detected = metrics["confirmed_violations"]
                job_status = update_job_progress(
                    args.job_id, current_pos, total_frames, current_fps, 
                    workers_detected=workers_detected, violations_detected=violations_detected,
                    status="live" if is_live_stream else None,
                )
                if job_status == "stopped":
                    logger.info("Job stopped via API.")
                    terminal_status = "stopped"
                    break

    if live_capture is not None:
        live_capture.stop()
    elif cap is not None:
        cap.release()
    if writer is not None:
        writer.release()
        
    if metrics["frames"] == 0 and args.job_id and not is_live_stream:
        logger.error("No frames were read from the video source.")
        update_job_progress(args.job_id, 0, 0, 0, status="failed")
        
    elapsed = time.time() - start_time
    avg_fps = metrics["frames"] / elapsed if elapsed > 0 else 0
    
    if args.job_id:
        workers_detected = len(metrics["unique_track_ids"])
        violations_detected = metrics["confirmed_violations"]
        if terminal_status == "camera_unavailable":
            update_job_progress(args.job_id, metrics["frames"], 0, avg_fps,
                                workers_detected=workers_detected, violations_detected=violations_detected,
                                status="camera_unavailable")
            logger.error("[JOB] job_id=%s ended because camera recovery was exhausted", args.job_id)
        elif terminal_status == "stopped":
            logger.info("[JOB] job_id=%s stopped by user", args.job_id)
        else:
            update_job_progress(
                args.job_id, total_frames, total_frames, avg_fps,
                workers_detected=workers_detected, violations_detected=violations_detected,
                status="completed"
            )
            logger.info(f"Job {args.job_id} completed successfully.")

    if live_log_handler:
        logger.removeHandler(live_log_handler)
        live_log_handler.close()

    logger.info("Pipeline processing completed.")
    logger.info(f"Total Frames: {metrics['frames']}")
    logger.info(f"Average FPS: {avg_fps:.2f}")
    logger.info(f"Total Unique IDs: {len(metrics['unique_track_ids'])}")

if __name__ == "__main__":
    main()
