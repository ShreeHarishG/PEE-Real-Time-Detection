"""Resilient, single-owner capture for live camera sources.

The worker intentionally keeps only the newest decoded frame.  This prevents a
slow inference or API call from backing up camera frames and, critically, keeps
all ``VideoCapture`` access out of the inference loop.
"""
from __future__ import annotations

import logging
import os
import platform
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import cv2
import numpy as np


Source = Union[int, str]
StateCallback = Callable[[str, dict], None]


@dataclass(frozen=True)
class CameraCaptureConfig:
    source: Source
    source_type: str = "auto"
    backend: Optional[str] = None
    read_timeout_ms: int = int(os.getenv("CAMERA_READ_TIMEOUT", "3000"))
    failure_threshold: int = int(os.getenv("CAMERA_READ_FAILURE_THRESHOLD", "3"))
    reconnect_attempts: int = int(os.getenv("CAMERA_RECONNECT_ATTEMPTS", "5"))
    reconnect_delay_s: float = float(os.getenv("CAMERA_RECONNECT_DELAY", "1.0"))


def normalize_camera_source(source: Union[str, int]) -> Source:
    """Convert only a numeric device index; do not confuse it with a DB id."""
    if isinstance(source, str) and source.strip().isdigit():
        return int(source.strip())
    return source


class CameraCapture:
    """A threaded latest-frame camera reader with bounded recovery attempts."""

    def __init__(
        self,
        config: CameraCaptureConfig,
        *,
        logger: Optional[logging.Logger] = None,
        on_state_change: Optional[StateCallback] = None,
    ) -> None:
        self.config = CameraCaptureConfig(
            **{**config.__dict__, "source": normalize_camera_source(config.source)}
        )
        self.logger = logger or logging.getLogger(__name__)
        self.on_state_change = on_state_change
        self._stop = threading.Event()
        self._frame_ready = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._sequence = 0
        self._state = "connecting"
        self._last_error: Optional[str] = None
        self._reconnects = 0
        self._failures = 0
        self._frames_read = 0
        self._opened_backend: Optional[str] = None
        self._camera_fps = 0.0
        self._width = 0
        self._height = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def stats(self) -> dict:
        return {
            "state": self._state,
            "frames_read": self._frames_read,
            "consecutive_failures": self._failures,
            "reconnects": self._reconnects,
            "last_error": self._last_error,
            "backend": self._opened_backend,
            "camera_fps": self._camera_fps,
            "width": self._width,
            "height": self._height,
        }

    def start(self, startup_timeout_s: float = 10.0) -> bool:
        self._thread = threading.Thread(target=self._run, name="edgevision-camera-capture", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + startup_timeout_s
        while time.monotonic() < deadline:
            if self._state == "live":
                return True
            if self._state == "camera_unavailable":
                return False
            time.sleep(0.05)
        self._last_error = f"no frame within {startup_timeout_s:.1f}s"
        self._set_state("camera_unavailable")
        self.stop()
        return False

    def get_latest(self, after_sequence: int, timeout_s: float = 0.5) -> Tuple[int, Optional[np.ndarray]]:
        """Wait briefly for a newer frame, returning no frame on timeout/state change."""
        with self._frame_ready:
            if self._sequence <= after_sequence and not self._stop.is_set():
                self._frame_ready.wait(timeout_s)
            if self._sequence <= after_sequence or self._latest_frame is None:
                return after_sequence, None
            return self._sequence, self._latest_frame.copy()

    def stop(self) -> None:
        self._stop.set()
        cap = self._cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        with self._frame_ready:
            self._frame_ready.notify_all()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, self.config.read_timeout_ms / 1000 + 1.0))
        if self._state != "camera_unavailable":
            self._set_state("stopped")

    def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        payload = self.stats
        self.logger.info("[CAMERA] state=%s source=%r details=%s", state, self.config.source, payload)
        if self.on_state_change:
            try:
                self.on_state_change(state, payload)
            except Exception:
                self.logger.exception("[CAMERA] state callback failed")

    def _backend_candidates(self) -> list[tuple[str, int]]:
        if self.config.backend:
            value = getattr(cv2, self.config.backend, None)
            if isinstance(value, int):
                return [(self.config.backend, value)]
        system = platform.system().lower()
        if isinstance(self.config.source, int):
            if system == "windows":
                # DSHOW is typically the most stable Windows webcam backend;
                # MSMF remains a fallback for devices not exposed through it.
                candidates = [("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", cv2.CAP_MSMF), ("CAP_ANY", cv2.CAP_ANY)]
            elif system == "linux":
                # V4L2 is the native webcam path on Ubuntu and Jetson; some
                # CSI/USB camera builds expose GStreamer instead.
                candidates = [("CAP_V4L2", cv2.CAP_V4L2), ("CAP_GSTREAMER", cv2.CAP_GSTREAMER), ("CAP_ANY", cv2.CAP_ANY)]
            else:
                candidates = [("CAP_ANY", cv2.CAP_ANY)]
        else:
            if system == "linux":
                candidates = [("CAP_GSTREAMER", cv2.CAP_GSTREAMER), ("CAP_FFMPEG", cv2.CAP_FFMPEG), ("CAP_ANY", cv2.CAP_ANY)]
            else:
                candidates = [("CAP_FFMPEG", cv2.CAP_FFMPEG), ("CAP_ANY", cv2.CAP_ANY)]
        return candidates

    def _open(self) -> bool:
        for backend_name, backend in self._backend_candidates():
            if self._stop.is_set():
                return False
            cap = cv2.VideoCapture(self.config.source, backend)
            # OpenCV backends ignore unsupported properties; setting them is
            # still valuable for FFmpeg streams and harmless elsewhere.
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.config.read_timeout_ms)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.config.read_timeout_ms)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if cap.isOpened():
                self._cap = cap
                self._opened_backend = cap.getBackendName() or backend_name
                self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                self._camera_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                self.logger.info(
                    "[CAMERA] source=%r opened backend=%s resolution=%sx%s camera_fps=%.2f",
                    self.config.source, self._opened_backend, self._width, self._height, self._camera_fps,
                )
                return True
            cap.release()
        self._last_error = "VideoCapture could not open any configured backend"
        return False

    def _run(self) -> None:
        attempts = 0
        while not self._stop.is_set():
            if self._cap is None and not self._open():
                attempts += 1
                self._reconnects = attempts
                if attempts > self.config.reconnect_attempts:
                    self._set_state("camera_unavailable")
                    return
                self._set_state("reconnecting")
                self.logger.warning("[RECONNECT] attempt=%s/%s source=%r error=%s", attempts, self.config.reconnect_attempts, self.config.source, self._last_error)
                self._stop.wait(self.config.reconnect_delay_s)
                continue

            cap = self._cap
            if cap is None:
                continue
            read_started = time.monotonic()
            try:
                ok, frame = cap.read()
            except cv2.error as exc:
                ok, frame = False, None
                self._last_error = f"OpenCV read error: {exc}"
            read_elapsed_ms = (time.monotonic() - read_started) * 1000
            if ok and frame is not None:
                self._frames_read += 1
                self._failures = 0
                attempts = 0
                if self._state != "live":
                    self._set_state("live")
                with self._frame_ready:
                    self._latest_frame = frame
                    self._sequence += 1
                    self._frame_ready.notify_all()
                continue

            self._failures += 1
            self._last_error = self._last_error or f"empty frame (read={read_elapsed_ms:.0f}ms)"
            self.logger.warning("[CAPTURE] source=%r consecutive_failure=%s read_ms=%.0f error=%s", self.config.source, self._failures, read_elapsed_ms, self._last_error)
            if self._failures < self.config.failure_threshold:
                self._stop.wait(0.05)
                continue

            self._set_state("reconnecting")
            try:
                cap.release()
            except Exception:
                pass
            self._cap = None
            attempts += 1
            self._reconnects = attempts
            if attempts > self.config.reconnect_attempts:
                self._set_state("camera_unavailable")
                return
            self._failures = 0
            self._stop.wait(self.config.reconnect_delay_s)
