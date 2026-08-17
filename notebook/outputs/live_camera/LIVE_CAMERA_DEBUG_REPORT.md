# Live Camera Forensic Diagnostic Report

Generated: 2026-08-17 (pre-change audit)

## Root cause evidence

1. **Local webcam source is converted to a path.** Camera ID `10` stores
   `local_video_path="0"`. `process_camera()` treats every non-URL source as a
   relative file path, so it launches the pipeline with
   `<repository>/0`, not OpenCV source `0`. This prevents the subprocess from
   opening the webcam at all.
2. **A live read failure is treated as normal completion.** `pipeline.py`
   performs `ret, frame = cap.read()` in its inference loop and executes
   `break` on the first `ret == False`. Its finalisation then marks the job
   `completed` even when no live frame can be read. There is no read-failure
   counter, reconnect attempt, or `camera_unavailable` state.
3. **Capture is blocked by all per-frame work.** Capture, two model calls,
   evidence writes, and HTTP progress work are serial in the same loop. A slow
   model/API call leaves the camera buffer unserviced; this is a plausible
   contributor to the observed 5--6 second failure, but it was not reproducible
   while the configured camera was offline.
4. **MJPEG has a file race.** The pipeline overwrites `latest_frame.jpg`
   directly while the stream generator reads it. The generator discards every
   exception, so a locked/partial file presents as a stalled browser feed with
   no diagnostic signal.

## Reproduction results

| Source | Result | Evidence |
| --- | --- | --- |
| Local webcam `0` | Cannot open on this host | OpenCV: `Camera index out of range` |
| Configured IP source (camera ID 9) | Cannot open on current network | OpenCV TCP connection to `192.168.137.205:4747` failed (`-138`) |

Camera IDs 11 and 12 are also configured as network sources but were not
probed because the requested configured source is camera ID 9.

## Audit flow

`/live` UI -> `POST /api/v1/cameras/{camera_id}/process` -> virtual video/job
record -> `subprocess.Popen(pipeline.py)` -> `cv2.VideoCapture` -> inference ->
`outputs/latest_frame.jpg` -> `GET /api/v1/stream` -> browser MJPEG image.

The upload path uses `POST /api/v1/jobs/{video_id}/process` and will remain
separate from the live capture lifecycle.

## Planned minimal repair

- Preserve numeric webcam sources during camera-job creation.
- Add a dedicated, bounded-reconnect capture worker with a one-frame latest
  buffer; inference consumes that buffer and never owns the camera.
- Add per-job structured logs and live job states.
- Atomically publish per-job JPEG frames and make MJPEG a consumer only.
- Keep video-file processing and the V3-HN configuration untouched.

## Limitations before validation

Neither a local webcam nor the configured IP camera was available from this
machine at test time. Five-minute physical-camera and failure-injection tests
cannot be claimed until one is reachable.
