# EdgeVision User Guide

## Overview
EdgeVision is a real-time PPE compliance and work-at-height safety platform. It uses computer vision to detect workers and their Personal Protective Equipment, validate compliance against configurable zone rules, and log violations into a web dashboard for safety officers.

---

## Accessing the Dashboard
Once the stack is running, open a browser and navigate to:

```
http://localhost:3000
```

---

## Dashboard Pages

### 1. Live Monitoring (`/`)
Displays the real-time or simulation video feed with annotated detections. Shows workers, bounding boxes, PPE detections, and tracking IDs.
- Simulation mode is clearly labelled **DEMO / VIDEO SIMULATION** when no RTSP camera is connected.
- Shows current FPS, active violations, and compliant worker count.

### 2. Active Violations (`/violations`)
Lists all PPE violations that have not yet been acknowledged by a safety officer.
- Each row shows: Event ID, Camera, Zone, Worker Tracking ID, Missing PPE, Timestamp.
- Click **Acknowledge** to mark an event as reviewed. This action persists to the database.

### 3. Event History (`/history`)
A searchable audit log of all violation events. Use the date selector and zone filter to narrow results.
- Evidence images are stored under `outputs/evidence/`.

### 4. Worker Compliance (`/workers`)
Shows aggregated per-tracking-ID compliance summaries.
> **Note:** These are **Temporary Worker Tracking IDs** assigned per session. EdgeVision does not implement persistent facial recognition or biometric identification.

### 5. Zone Configuration (`/zones`)
Configure safety zones and their required PPE.
- **Supported (TRAINED + VALIDATED):** helmet, vest
- **Not supported in current model (UNTRAINED):** boots, harness, hook, lanyard, anchor point — shown with "UNTRAINED" badge.

### 6. Camera Management (`/cameras`)
Add and configure video sources.
- Source types: `LOCAL_VIDEO` (for demo/simulation), `RTSP` (for live cameras).
- Cameras are assigned to zones so that zone rules apply correctly.

### 7. Reports (`/reports`)
Daily, weekly, or monthly compliance summaries. Export to CSV for management review.

### 8. Model Monitoring (`/models`)
Displays the frozen V3-HN offline benchmark metrics alongside live inference performance.

---

## Running the Demo Inference
See `docs/DEMO_GUIDE.md` for the step-by-step procedure to generate a real violation event end-to-end.
