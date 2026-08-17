# EdgeVision API Documentation

The EdgeVision backend uses FastAPI. Interactive Swagger UI documentation is available at `http://localhost:8000/docs`.

## Base URL
`http://localhost:8000/api/v1`

## Endpoints

### System
- `GET /health`
  Returns the health status of the API.
- `GET /stats`
  Returns aggregated statistics (active violations, resolution rate, total workers, etc.).

### Videos & Jobs
- `POST /videos/upload`
  Upload a new video for processing.
- `POST /jobs/{video_id}/process`
  Start ML pipeline processing on the specified video.
- `POST /jobs/{job_id}/stop`
  Stop an actively processing job.
- `GET /jobs/{job_id}`
  Get the status and progress of a job.

### Violations
- `GET /violations`
  Retrieve a paginated list of violation events.
  - Query Params: `skip` (default: 0), `limit` (default: 100)
- `POST /violations`
  (Internal) Record a new violation event from the ML pipeline.
- `PATCH /violations/{id}/acknowledge`
  Mark a violation as acknowledged/resolved.
- `PATCH /violations/{id}/feedback`
  Submit Human-In-The-Loop feedback for model training (correctness, helmet, vest presence).

### Zones
- `GET /zones`
  Get all configured safety zones and their required PPE.
- `POST /zones`
  Create a new safety zone.
- `PUT /zones/{zone_id}`
  Update a safety zone's polygon coordinates and required PPE.

### Cameras
- `GET /cameras`
  Retrieve a list of connected RTSP/IP cameras.
- `POST /cameras`
  Add a new camera stream.

### Live Stream
- `GET /stream`
  Returns a `multipart/x-mixed-replace` live MJPEG stream of the active inference pipeline.
