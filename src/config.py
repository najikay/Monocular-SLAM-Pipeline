"""
Configuration parameters for the Visual Odometry SLAM system.
"""
import numpy as np

# Dataset paths
DATASET_PATH = "../VO_dataset_SLAM_HW3/rgbd_dataset_freiburg2_pioneer_slam3"

# Camera intrinsic parameters
FX, FY, CX, CY = 520.9, 521.0, 325.1, 249.7
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
DEPTH_SCALE = 5000.0

# General tuning parameters
SCALE_FIX = 1.06
ALIGN_ANGLE = -45.0

# Optimization parameters
RANSAC_THRESH = 3.4
GN_ITERS = 10

# Tracking and mapping thresholds
THRESH_MIN_MATCHES = 20
THRESH_TELEPORT = 0.5

# Loop closure thresholds
THRESH_LOOP_DIST = 1.0
THRESH_LOOP_MATCHES = 40
THRESH_LOOP_FRAMES = 200