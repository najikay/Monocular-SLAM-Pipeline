# Pure Visual Odometry (VO) SLAM Pipeline

![gif-video](https://github.com/user-attachments/assets/8a0b35fb-60c6-498a-b82b-70fbd3fc0682)

A monocular Simultaneous Localization and Mapping (SLAM) system built entirely from scratch in Python. This project implements a Pure Visual Odometry pipeline capable of handling challenging environments with severe camera shake and motion blur, achieving a highly accurate trajectory without the use of IMU or external inertial sensors.

## Technical Highlights
* **Architecture:** Implements a continuous "Time Machine" (sliding window) tracking loop to recover from severe motion blur and track loss.
* **Frontend:** Extracts and matches ORB keypoints using Lowe's Ratio Test. Pose estimation is calculated via the EPnP algorithm wrapped in an adaptive RANSAC filter.
* **Optimization:** Custom-built non-linear Gauss-Newton optimization minimizes Reprojection Error to finely tune camera poses.
* **Loop Closure:** Detects previously visited locations using spatial distance thresholds and visual descriptor verification to correct accumulated scale drift.

## Performance & Results
Tested on the highly volatile `rgbd_dataset_freiburg2_pioneer_slam3` dataset.

* **Total Ground Truth Distance:** 18.80 meters
* **Total Estimated Distance:** 19.43 meters
* **Absolute Trajectory Error (ATE):** 0.4238 meters

*(ATE calculated via SVD alignment)*

## Project Structure
* `main.py`: Main execution loop, data synchronization, and system integration.
* `tracker.py`: The core algorithmic engine (Feature extraction, RANSAC + EPnP, Gauss-Newton optimization).
* `viewer.py`: Real-time 3D OpenGL rendering using Pangolin.
* `config.py`: Centralized configuration for camera intrinsics and heuristic parameters.

## Running the Project
1. Clone the repository.
2. Ensure you have the required dependencies (`opencv-python`, `numpy`, `pypangolin`).
3. Download the TUM RGB-D `freiburg2_pioneer_slam3` dataset and update the `DATASET_PATH` inside `config.py`.
4. Run `python3 main.py` to initialize the tracking loop and real-time 3D viewer.
