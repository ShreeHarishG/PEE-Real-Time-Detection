# EdgeVision REST API

The EdgeVision backend exposes a FastAPI REST application. It can be accessed interactively via Swagger UI at `http://localhost:8000/docs`.

## Base URL
`/api/v1`

## Endpoints

### Health
- `GET /health`: Returns the operational status of the API.

### Zones
- `GET /zones`: Retrieves all configured safety zones and their required PPE lists.
- `POST /zones`: Create a new zone rule configuration.

### Cameras
- `GET /cameras`: Retrieve a list of all active video sources.
- `POST /cameras`: Register a new video source (RTSP or local file).

### Violations
- `GET /violations`: Retrieves an ordered history of violation events.
- `POST /violations`: Internal endpoint used by the ML pipeline to register a confirmed violation. Requires an `event_id`, `camera_id`, `zone_id`, `missing_ppe` array, and an `evidence_image_path`.
- `PATCH /violations/{id}/acknowledge`: Marks a violation as reviewed by a safety officer.

### Metrics
- `GET /metrics`: Retrieves the latest live inference metrics (FPS, Latency) posted by the ML pipeline.
