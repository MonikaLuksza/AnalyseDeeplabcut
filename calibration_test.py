import cv2
import numpy as np
import sys
import os

# --------- CONFIG ---------
CHECKERBOARD = (8, 6)  # 5x5 squares = 4x4 inner corners
image_path = "C:/Users/Usagers/2-Project1-Monika-2025-05-07/ANIPOSE/calibration_test6.png"  #"C:/Users/Usagers/install-Monika/CALIBRATION/cam1/img_0638.jpg"  # <-- Replace this with your actual image path
# --------------------------

# 3. Read image
if not os.path.exists(image_path):
    print(f"❌ Image not found at: {image_path}")
    exit(1)

img = cv2.imread(image_path)
if img is None:
    print("❌ Could not read image")
    exit(1)

# 4. Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Optional: Apply histogram equalization
gray = cv2.equalizeHist(gray)

# 5. Try to find corners
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

print(f"✅ Checkerboard detected: {ret}")

if ret:
    cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
    cv2.imshow("Detected Corners", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ Could not detect corners. Try adjusting lighting or printing a flatter board.")
