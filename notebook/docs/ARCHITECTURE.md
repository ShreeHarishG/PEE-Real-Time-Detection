# System Architecture

The EdgeVision platform uses a hybrid edge-to-cloud architecture. Heavy inference runs on the edge device (NVIDIA Jetson) while the backend API serves the lightweight web application.

## Architecture Diagram

```mermaid
graph TD
    %% Define styles
    classDef edge fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef backend fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;
    classDef db fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef frontend fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;

    subgraph Edge Device [NVIDIA Jetson Edge Node]
        C[IP Camera / RTSP Stream] -->|Video Frames| DS[Inference Pipeline]
        DS -->|ByteTrack| PT[Person Tracker]
        DS -->|YOLOv8 FP16| PPE[PPE Detector]
        PT --> ASSOC[Person-to-PPE Association]
        PPE --> ASSOC
        ASSOC --> RE[Rule Engine & Temporal Validator]
    end

    subgraph Backend Server [FastAPI Backend]
        RE -->|HTTP POST| API[REST API]
        API --> DB[(PostgreSQL)]
        API --> FS[Evidence Image Storage]
    end

    subgraph Client [Web Dashboard]
        API -->|JSON Stats/Events| UI[Next.js Dashboard]
        FS -->|HTTP GET| UI
        DS -.->|MJPEG Live Stream| UI
    end

    class Edge Device edge;
    class Backend Server backend;
    class DB db;
    class Client frontend;
```

## Component Breakdown

1. **Edge Node (Inference Pipeline)**: 
   - A Python/DeepStream process running YOLOv8.
   - Extracts frames, detects people, detects PPE, and associates them.
   - Applies temporal validation to filter out flickering detections.
   - Sends confirmed violation events (JSON) and cropped evidence images to the backend.

2. **FastAPI Backend**:
   - Manages CRUD operations for Zones, Cameras, and Violations.
   - Interacts with the PostgreSQL database using SQLAlchemy ORM.
   - Stores image evidence files on the local filesystem.
   - Exposes endpoints for the Next.js frontend to poll.

3. **PostgreSQL Database**:
   - Stores structured relational data.
   - Tables include: `cameras`, `zones`, `violation_events`, `processing_jobs`.

4. **Next.js Web Dashboard**:
   - React-based frontend providing a live monitoring view.
   - Polls the backend API for real-time statistics and alerts.
   - Features a Human-in-the-Loop (HITL) interface to provide ground-truth feedback on model detections.
