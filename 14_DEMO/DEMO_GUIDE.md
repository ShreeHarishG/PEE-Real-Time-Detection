# Demo Guide

To evaluate the EdgeVision platform:

1. Ensure the backend and frontend are running (see `10_DOCUMENTATION/SETUP.md`).
2. Navigate to `http://localhost:3000`.
3. In the Web Dashboard, configure a Construction Zone and require "helmet" and "vest".
4. Upload `test.mp4` (or `test1.mp4`) located in this `14_DEMO` directory via the dashboard.
5. Initiate processing.
6. The dashboard will live-update with violation events and evidence snapshots as the pipeline validates temporal rule breaches.
