import numpy as np

# Paths
DATASET_PATH = "../VO_dataset_SLAM_HW3/rgbd_dataset_freiburg2_pioneer_slam3"

# Camera calibration
FX, FY, CX, CY = 520.9, 521.0, 325.1, 249.7
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
DEPTH_SCALE = 5000.0

# General Tuning parameters
SCALE_FIX = 1.06
ALIGN_ANGLE = -45.0

# VO Optimization Parameters
RANSAC_THRESH = 3.4
GN_ITERS = 10

# Thresholds
THRESH_MIN_MATCHES = 20
THRESH_TELEPORT = 0.5

# Loop closure
THRESH_LOOP_DIST = 1.0
THRESH_LOOP_MATCHES = 40
THRESH_LOOP_FRAMES = 200  # Ignore recent frames to prevent false positives