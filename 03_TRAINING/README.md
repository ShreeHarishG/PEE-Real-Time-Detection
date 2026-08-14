# Training Directory

This directory contains the artifacts used to train and evaluate the EdgeVision V3-HN production model and its experimental variants.

- **scripts/**: The Python scripts used for dataset splitting, YOLOv8 training, and post-training evaluation loops.
- **evaluation/**: Output metrics, CSV files, and confusion matrices generated during the V3 vs V4 architectural sweeps and V3-BOOTS regression checks.
- **configs/**: (Reference) `../notebook/config/model_versions.yaml` contains the live inference thresholds and image sizes.

> Note: Massive PyTorch cache files and intermediate checkpoints have been deliberately excluded from this submission package to ensure a lightweight and reproducible state.
