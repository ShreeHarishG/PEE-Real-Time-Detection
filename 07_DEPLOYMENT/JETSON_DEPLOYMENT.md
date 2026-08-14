# Jetson Deployment Guide

**STATUS**: PENDING TARGET HARDWARE VALIDATION

The current EdgeVision V3-HN model has been thoroughly validated on the development workstation. Physical deployment to the Jetson target hardware is pending access to the device.

## Prerequisites
- Jetson Orin Nano / NX (Target)
- JetPack 5.1.1+
- CUDA 11.4+
- TensorRT 8.5+
- DeepStream 6.2+

## Installation Steps (Pending Validation)
1. Clone the repository to the Jetson device.
2. Run the installation script:
```bash
sudo ./scripts/install_jetson.sh
```
3. Install the systemd service:
```bash
sudo cp systemd/edgevision.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable edgevision
sudo systemctl start edgevision
```
