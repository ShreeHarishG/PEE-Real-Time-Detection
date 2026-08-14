# Source Code Directory

Per the final submission guidelines, the working application is preserved in its original validated structure to prevent path breakage or deployment failures.

The actual runnable source code is located in the root `notebook/` directory of this repository:

- **Backend (FastAPI)**: `../notebook/backend/`
- **Frontend (Next.js)**: `../notebook/frontend/`
- **Inference Pipeline (YOLOv8 + ByteTrack)**: `../notebook/src/pipeline.py`
- **Configuration**: `../notebook/config/`
- **Tests**: `../notebook/backend/tests/`

The logic for **Rules**, **Tracking**, and **Association** is heavily integrated into the Temporal Validator and spatial containment logic within `../notebook/src/pipeline.py`.
