"""
Main entry point for the Visual Odometry SLAM system.
Handles data loading, synchronization, the main VO tracking loop, and evaluation.
"""
import os
# Suppress OpenCV and Qt warnings on Linux
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

import cv2
import numpy as np

import config
import dataset
from frame import Frame
from map import Map
from point import Point
from tracker import Tracker
from viewer import Viewer


def smooth_recent_trajectory(trajectory, window=10):
    """Applies a moving average filter to remove high-frequency visual jitter."""
    if len(trajectory) < window:
        return

    subset = np.array(trajectory[-window:])
    smoothed_subset = []

    for i in range(len(subset)):
        start = max(0, i - 2)
        end = min(len(subset), i + 3)
        avg_pt = np.mean(subset[start:end], axis=0)
        smoothed_subset.append(avg_pt)

    for i in range(window):
        trajectory[-(window - i)] = smoothed_subset[i]


def calculate_svd_trajectory_alignment(est_traj, gt_traj):
    """
    Calculates Absolute Trajectory Error (ATE) using Singular Value Decomposition (SVD).
    This mathematically aligns the estimated trajectory to the ground truth
    before calculating RMSE, which is the academic standard for SLAM evaluation.
    """
    est = np.array(est_traj)
    gt = np.array(gt_traj)

    # Center trajectories
    mu_est = np.mean(est, axis=0)
    mu_gt = np.mean(gt, axis=0)
    est_cen = est - mu_est
    gt_cen = gt - mu_gt

    # Calculate covariance matrix and SVD
    W = np.dot(est_cen.T, gt_cen)
    U, S, Vt = np.linalg.svd(W)

    # Calculate rotation matrix
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Calculate translation
    t = mu_gt - np.dot(R, mu_est)

    # Apply optimal transform
    aligned_est = np.dot(est, R.T) + t

    # Calculate final RMSE
    err = aligned_est - gt
    rmse = np.sqrt(np.mean(np.sum(err**2, axis=1)))

    return rmse


def main():
    # Force OpenCV window to initialize safely before OpenGL
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 640, 480)

    # Data loading and synchronization
    rgb_list = dataset.read_file_list(os.path.join(config.DATASET_PATH, "rgb.txt"))
    depth_list = dataset.read_file_list(os.path.join(config.DATASET_PATH, "depth.txt"))
    gt_data = dataset.load_ground_truth(os.path.join(config.DATASET_PATH, "groundtruth.txt"))
    matches = dataset.associate_data(rgb_list, depth_list)

    gt_synchronized = []

    # Only sync ground truth if the file was found
    if gt_data:
        start_time = matches[0][0]
        gt_timestamps = np.array(sorted(gt_data.keys()))

        start_idx_gt = np.abs(gt_timestamps - start_time).argmin()
        gt_start_pos = gt_data[gt_timestamps[start_idx_gt]]

        for i in range(len(matches)):
            timestamp = matches[i][0]
            idx = np.searchsorted(gt_timestamps, timestamp)
            if idx >= len(gt_timestamps):
                idx = len(gt_timestamps) - 1
            gt_synchronized.append(gt_data[gt_timestamps[idx]] - gt_start_pos)

    # System initialization
    vo_map = Map()
    tracker = Tracker()
    viewer = Viewer()

    first_rgb = cv2.imread(os.path.join(config.DATASET_PATH, matches[0][1]), cv2.IMREAD_GRAYSCALE)
    prev_depth_data = cv2.imread(os.path.join(config.DATASET_PATH, matches[0][3]), cv2.IMREAD_UNCHANGED)

    last_keyframe = Frame(0, first_rgb, matches[0][0], config.K)
    tracker.compute_features(last_keyframe)
    vo_map.add_frame(last_keyframe)

    trajectory = [np.zeros(3)]
    # Safely initialize ground truth drawing list
    gt_traj_draw = [gt_synchronized[0]] if gt_synchronized else []

    frame_history = [last_keyframe]
    depth_history = [prev_depth_data]
    skipped_frames = 0
    max_skip = 3

    for i in range(1, len(matches)):
        if viewer.should_quit():
            break

        curr_rgb = cv2.imread(os.path.join(config.DATASET_PATH, matches[i][1]), cv2.IMREAD_GRAYSCALE)
        curr_depth = cv2.imread(os.path.join(config.DATASET_PATH, matches[i][3]), cv2.IMREAD_UNCHANGED)
        if curr_rgb is None:
            continue

        curr_frame = Frame(i, curr_rgb, matches[i][0], config.K)
        tracker.compute_features(curr_frame)

        # Visual tracking against history
        history_window = frame_history[-5:]
        depth_window = depth_history[-5:]
        T_motion, obj_pts, _, valid_matches, shake_reason = tracker.match_and_track(history_window, curr_frame, depth_window)

        should_update_keyframe = False
        display_text = "Tracking"
        color = (0, 255, 0)

        # State update
        if T_motion is not None and shake_reason is None:
            T_motion[:3, 3] *= config.SCALE_FIX
            curr_frame.set_pose(last_keyframe.pose @ T_motion)
            vo_map.add_frame(curr_frame)

            # Loop closure check
            if i % 10 == 0:
                T_correction, loop_id = tracker.detect_loop(curr_frame, vo_map.frames)
                if T_correction is not None:
                    display_text = f"Loop closed: {loop_id}"
                    color = (255, 0, 255)

            # Extract global viewer position directly from visual pose
            aligned_pos = viewer.get_aligned_pos(curr_frame.pose)
            trajectory.append(aligned_pos)

            if i % 2 == 0:
                for pt in obj_pts[::50]:
                    global_pt = (curr_frame.pose[:3, :3] @ pt) + curr_frame.pose[:3, 3]
                    aligned_pt = viewer.get_aligned_pos(np.eye(4)) + (viewer.vis_align[:3, :3] @ global_pt)
                    vo_map.add_point(Point(0, aligned_pt))

            should_update_keyframe = True
            skipped_frames = 0
        else:
            # Coast (duplicate last known position) since tracking failed
            trajectory.append(trajectory[-1] if len(trajectory) > 0 else np.zeros(3))
            skipped_frames += 1
            display_text = f"Lost Track: {shake_reason}"
            color = (0, 0, 255)

            if skipped_frames > max_skip:
                should_update_keyframe = True
                skipped_frames = 0
                display_text = "Force resync"
                color = (0, 255, 255)
            else:
                curr_frame.pose = last_keyframe.pose

        # Trajectory smoothing and visualization
        smooth_recent_trajectory(trajectory, window=15)

        # Safely append ground truth data
        if gt_synchronized:
            gt_traj_draw.append(gt_synchronized[i])

        viewer.update(vo_map, trajectory, gt_traj_draw, last_keyframe.pose)

        # Draw tracking window showing current keypoints
        try:
            vis = cv2.cvtColor(curr_frame.image, cv2.COLOR_GRAY2BGR)
            # Draw keypoints as small circles
            if curr_frame.keypoints:
                cv2.drawKeypoints(curr_frame.image, curr_frame.keypoints, vis, color=(0, 255, 0), flags=0)

            vis = cv2.resize(vis, (640, 480))
            cv2.putText(vis, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Tracking", vis)
            cv2.waitKey(1)
        except Exception:
            pass

        if should_update_keyframe:
            last_keyframe = curr_frame
            prev_depth_data = curr_depth
            frame_history.append(curr_frame)
            depth_history.append(curr_depth)
            if len(frame_history) > 20:
                frame_history.pop(0)
                depth_history.pop(0)

    # Final evaluation (Skip if no ground truth provided)
    if gt_traj_draw:
        print("SLAM evaluation summary:")

        est_dist = sum(np.linalg.norm(trajectory[i] - trajectory[i-1]) for i in range(1, len(trajectory)))
        gt_dist = sum(np.linalg.norm(gt_traj_draw[i] - gt_traj_draw[i-1]) for i in range(1, len(gt_traj_draw)))

        print(f"Total distance (Estimated):    {est_dist:.2f} meters")
        print(f"Total distance (Ground truth): {gt_dist:.2f} meters")

        # Calculate absolute trajectory error
        min_len = min(len(trajectory), len(gt_traj_draw))
        ate_rmse = calculate_svd_trajectory_alignment(trajectory[:min_len], gt_traj_draw[:min_len])

        print(f"Final ATE (SVD aligned):       {ate_rmse:.4f} meters")
    else:
        print("SLAM run complete. (Evaluation skipped: No ground truth provided)")


if __name__ == "__main__":
    main()