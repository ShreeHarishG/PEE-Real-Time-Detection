"""Standalone forensic probe for an EdgeVision camera source."""
import argparse
import os
import platform
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))

from camera_capture import CameraCapture, CameraCaptureConfig, normalize_camera_source


def run_probe(source: str, duration: float, backend: str | None) -> dict:
    """Exercise capture only: no browser, API, database, or inference."""
    normalized = normalize_camera_source(source)
    events: list[tuple[float, str, dict]] = []
    started = time.monotonic()

    def event(state: str, details: dict) -> None:
        events.append((time.monotonic() - started, state, details.copy()))
        print(f"[{time.strftime('%H:%M:%S')}] state={state} details={details}")

    capture = CameraCapture(CameraCaptureConfig(source=normalized, backend=backend), on_state_change=event)
    opened = capture.start(startup_timeout_s=min(10.0, duration))
    first_frame_s = None
    first_failure_s = None
    sequence = 0
    frames = 0
    deadline = time.monotonic() + duration
    while opened and time.monotonic() < deadline:
        sequence, frame = capture.get_latest(sequence, timeout_s=0.5)
        if frame is not None:
            frames += 1
            if first_frame_s is None:
                first_frame_s = time.monotonic() - started
        elif capture.state != "live" and first_failure_s is None:
            first_failure_s = time.monotonic() - started
        if capture.state == "camera_unavailable":
            break
    elapsed = time.monotonic() - started
    stats = capture.stats
    capture.stop()
    return {
        "source": source, "backend_requested": backend or "auto", "opened": opened,
        "first_frame_s": first_frame_s, "first_failure_s": first_failure_s,
        "frames": frames, "elapsed": elapsed, "capture_fps": frames / elapsed if elapsed else 0.0,
        "events": events, **stats,
    }


def print_report(result: dict) -> None:
    print("\nCAMERA DIAGNOSTIC\n-----------------")
    print(f"Source: {result['source']}")
    print(f"Requested backend: {result['backend_requested']}")
    print(f"Opened: {'YES' if result['opened'] else 'NO'}")
    print(f"Backend used: {result['backend'] or 'N/A'}")
    print(f"Resolution: {result['width']}x{result['height']}")
    print(f"Camera FPS reported: {result['camera_fps']:.2f}")
    print(f"First successful frame: {result['first_frame_s'] if result['first_frame_s'] is not None else 'N/A'}")
    print(f"First failure: {result['first_failure_s'] if result['first_failure_s'] is not None else 'N/A'}")
    print(f"Frames read: {result['frames']}")
    print(f"Consecutive failures: {result['consecutive_failures']}")
    print(f"Reconnects: {result['reconnects']}")
    print(f"Final state: {result['state']}")
    print(f"Last OpenCV/capture error: {result['last_error'] or 'None'}")
    print(f"Capture FPS: {result['capture_fps']:.2f}")
    print("Processing FPS: N/A (capture-only diagnostic)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose a webcam/IP camera without the frontend")
    parser.add_argument("--source", required=True, help="OpenCV source: 0, RTSP/HTTP URL, or local test file")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--backend", help="Optional OpenCV backend constant, for example CAP_DSHOW")
    args = parser.parse_args()

    requested_backends = [args.backend]
    if args.source.strip().isdigit() and not args.backend:
        requested_backends = (
            ["CAP_DSHOW", "CAP_MSMF", "CAP_ANY"]
            if platform.system().lower() == "windows"
            else ["CAP_V4L2", "CAP_GSTREAMER", "CAP_ANY"]
        )
    exit_code = 0
    for backend in requested_backends:
        result = run_probe(args.source, args.duration, backend)
        print_report(result)
        if not result["opened"]:
            exit_code = 1
    raise SystemExit(exit_code)
