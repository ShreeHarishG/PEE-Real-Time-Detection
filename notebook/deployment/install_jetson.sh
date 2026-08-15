#!/bin/bash
# Jetson Installation Script for EdgeVision Pipeline
# Requires: JetPack 5.x / 6.x

set -e

echo "============================================"
echo "Installing EdgeVision Dependencies on Jetson"
echo "============================================"

# Update system
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev libv4l-dev cmake gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav

# Install PyTorch and Ultralytics (Optimized for Jetson)
echo "Installing Python dependencies..."
pip3 install -r ../requirements.txt

# Create necessary directories
mkdir -p /opt/edgevision/models
mkdir -p /opt/edgevision/outputs/evidence
mkdir -p /var/log/edgevision

# Copy application files
echo "Deploying application to /opt/edgevision..."
sudo cp -r ../src /opt/edgevision/
sudo cp -r ../config /opt/edgevision/
sudo cp -r ../models /opt/edgevision/
sudo cp ../deployment/edgevision.service /etc/systemd/system/

echo "Setup complete. To start the service, run:"
echo "sudo systemctl daemon-reload"
echo "sudo systemctl enable edgevision.service"
echo "sudo systemctl start edgevision.service"
