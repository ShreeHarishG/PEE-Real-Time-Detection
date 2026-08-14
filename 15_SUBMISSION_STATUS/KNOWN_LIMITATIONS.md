# Known Limitations

- **Unsupported PPE Classes**: Boots, Harness, Lanyard, and Hooks are NOT trained in the V3-HN production model due to severe dataset limitations (small object occlusion and bounding box inconsistency). They are marked as UNTRAINED in the UI.
- **Hardware Metrics**: Exact FPS, temperature, and memory utilization for the target Jetson hardware are pending physical validation. Current performance benchmarks are derived from development workstations.
- **Extreme Distances**: Highly occluded or extremely distant small PPE might be occasionally missed in a single frame, though the temporal validator effectively mitigates alert spam.
