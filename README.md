# Visual Odometry & 3D Reconstruction (Pure SLAM)

A robust, Pure Visual Odometry (VO) and SLAM pipeline built from scratch in Python. This system estimates camera trajectory and reconstructs a 3D Sparse Point Cloud environment relying **strictly on visual data** (RGB-D) without the use of IMU sensors, designed to handle aggressive camera motion and motion blur.

![SLAM Demo](assets/demo.png)
*(Screenshot of the Pangolin 3D viewer showing the estimated green trajectory, red ground truth, and the triangulated white point cloud).*

## Key Features & Algorithmic Design

* **Decoupled Keyframe Mapping:** Tracking and Mapping are separated. 3D triangulation only occurs between temporally distant Keyframes (Baseline > 20cm) to prevent epipolar geometric collapse (the "ray" effect) during pure camera rotation.
* **Epipolar Triangulation:** 3D environment generation using mathematical triangulation rather than relying on raw depth sensor mapping.
* **Sub-Pixel Feature Refinement:** Extracts ORB keypoints and refines them to sub-pixel accuracy (`cv2.cornerSubPix`) for highly stable tracking.
* **Reprojection Error Map Filtering:** A rigorous map-cleaning mechanism. 3D points are reprojected back onto their source 2D camera planes; only points with a reprojection error of `< 2.0 pixels` survive, resulting in a crisp, noise-free point cloud.
* **Non-Linear Optimization:** Implements custom Gauss-Newton optimization over EPnP + RANSAC inliers to minimize reprojection errors and refine the camera pose.
* **Loop Closure Detection:** Spatial and temporal scanning combined with Bag-of-Words visual matching to detect circular path completions.
* **SVD Trajectory Evaluation:** Evaluates drift using standard Singular Value Decomposition (SVD) to align the estimated path with Ground Truth, calculating the Absolute Trajectory Error (ATE).

## Performance Metrics
Tested on the highly volatile TUM dataset (`freiburg2_pioneer_slam3`), the system achieved excellent accuracy for a pure VO pipeline lacking Global Bundle Adjustment:

* **Estimated Distance:** 20.09 meters
* **Ground Truth Distance:** 18.80 meters
* **Absolute Trajectory Error (ATE):** `0.3399 meters` (~1.8% drift rate)

## Project Structure

The codebase follows a clean, modular Object-Oriented design:
* `src/main.py` - Main event loop, dataset synchronization, and tracking execution.
* `src/tracker.py` - The algorithmic core (ORB, EPnP, Gauss-Newton, Triangulation, Loop Closure).
* `src/viewer.py` - Real-time 3D OpenGL rendering using Pangolin.
* `src/map.py`, `src/frame.py`, `src/point.py` - Data structures managing spatial history and the global map.
* `src/dataset.py` - Ground Truth parsing and RGB-D temporal association.
* `src/tune_slam.py` - Headless auto-tuning script to optimize Scale and RANSAC thresholds.

## 💻 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/najikay/Monocular-SLAM-Pipeline.git](https://github.com/najikay/Monocular-SLAM-Pipeline.git)
   cd Monocular-SLAM-Pipeline
Install dependencies:

Bash
pip install -r requirements.txt
Dataset Setup:
Download the TUM RGB-D freiburg2_pioneer_slam3 dataset and place it in the path defined in src/config.py.

Run the Tracker:

Bash
cd src
python main.py
Academic Context
This project was developed as the final assignment for the "Navigation, Mapping, and Pose Estimation" course at the University of Haifa.

Author: Naji Kayal
