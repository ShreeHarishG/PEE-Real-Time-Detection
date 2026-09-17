import shutil
import os

src = os.path.join("runs", "detect", "runs", "detect", "ppe_v5_harness-3", "weights", "best.pt")
dst = os.path.join("models", "ppe_v5_harness.pt")

if os.path.exists(src):
    shutil.copy(src, dst)
    print(f"Successfully copied {src} to {dst}")
else:
    print(f"Error: {src} not found!")
