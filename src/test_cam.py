import cv2
import time

print("Testing MSMF (default)...")
cap1 = cv2.VideoCapture(0)
if cap1.isOpened():
    ret, frame = cap1.read()
    print("MSMF Default: Opened successfully. Frame read:", ret)
else:
    print("MSMF Default: Failed to open.")
cap1.release()

time.sleep(2)

print("\nTesting DirectShow...")
cap2 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if cap2.isOpened():
    ret, frame = cap2.read()
    print("DirectShow: Opened successfully. Frame read:", ret)
else:
    print("DirectShow: Failed to open.")
cap2.release()
