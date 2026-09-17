# User Guide

## Running the Application
The application is pre-configured for local testing.

1. Navigate to the root directory `w:\3 projects\Building\Tfrenzy`.
2. Start the FastApi backend:
   ```bash
   cd notebook/backend
   uvicorn app.main:app --reload
   ```
3. Start the Next.js frontend:
   ```bash
   cd notebook/frontend
   npm run dev
   ```
4. Access the dashboard at `http://localhost:3000`.

## Architecture Overview
See `11_ARCHITECTURE/` for system flow diagrams.
The pipeline utilizes ByteTrack for cross-frame association to enable temporal validation.

## Known Limitations
- The system struggles to accurately classify `boots`, `lanyard`, and `harness` due to bounding box inconsistency on small objects. These classes are disabled (UNTRAINED).
- The TensorRT engine requires physical validation on the target Jetson hardware.
