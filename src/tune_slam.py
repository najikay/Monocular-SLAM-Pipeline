"""
Auto-tuning script to find optimal VO parameters.
Runs multiple headless SLAM simulations to minimize the SVD-aligned ATE.
"""
import numpy as np
import os
import cv2
import config
import dataset
from frame import Frame
from tracker import Tracker

TEST_SCALES = [1.04, 1.06, 1.08] # 1.06
TEST_RANSAC = [3.2, 3.4, 3.6] # 3.4
# Locked
TEST_GN_ITERS = [10]


def calculate_svd_trajectory_alignment(est_traj, gt_traj):
    """Calculates Absolute Trajectory Error (ATE) using SVD."""
    est = np.array(est_traj)
    gt = np.array(gt_traj)

    mu_est = np.mean(est, axis=0)
    mu_gt = np.mean(gt, axis=0)
    est_cen = est - mu_est
    gt_cen = gt - mu_gt

    W = np.dot(est_cen.T, gt_cen)
    U, S, Vt = np.linalg.svd(W)

    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = mu_gt - np.dot(R, mu_est)
    aligned_est = np.dot(est, R.T) + t

    err = aligned_est - gt
    rmse = np.sqrt(np.mean(np.sum(err ** 2, axis=1)))
    return rmse


def run_simulation(scale_val, ransac_val, gn_val, num_frames=600):
    """Executes a headless simulation of the VO pipeline."""
    # Apply test parameters to config
    config.SCALE_FIX = scale_val
    config.RANSAC_THRESH = ransac_val
    config.GN_ITERS = gn_val

    rgb_list = dataset.read_file_list(os.path.join(config.DATASET_PATH, "rgb.txt"))
    depth_list = dataset.read_file_list(os.path.join(config.DATASET_PATH, "depth.txt"))
    gt_data = dataset.load_ground_truth(os.path.join(config.DATASET_PATH, "groundtruth.txt"))
    matches = dataset.associate_data(rgb_list, depth_list)

    gt_timestamps = np.array(sorted(gt_data.keys()))
    start_time = matches[0][0]

    tracker = Tracker()

    first_rgb = cv2.imread(os.path.join(config.DATASET_PATH, matches[0][1]), cv2.IMREAD_GRAYSCALE)
    prev_depth_data = cv2.imread(os.path.join(config.DATASET_PATH, matches[0][3]), cv2.IMREAD_UNCHANGED)

    last_keyframe = Frame(0, first_rgb, matches[0][0], config.K)
    tracker.compute_features(last_keyframe)

    traj_est = [np.zeros(3)]
    traj_gt = []

    idx_gt_start = np.abs(gt_timestamps - start_time).argmin()
    gt_start_pos = gt_data[gt_timestamps[idx_gt_start]]
    traj_gt.append(np.zeros(3))

    history_frames = [last_keyframe]
    history_depths = [prev_depth_data]

    # Use a dummy static alignment matrix so we don't need the Viewer class
    theta = np.deg2rad(config.ALIGN_ANGLE)
    c, s = np.cos(theta), np.sin(theta)
    R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R_base = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    vis_align = np.eye(4)
    vis_align[:3, :3] = R_yaw @ R_base

    for i in range(1, min(num_frames, len(matches))):
        curr_rgb = cv2.imread(os.path.join(config.DATASET_PATH, matches[i][1]), cv2.IMREAD_GRAYSCALE)
        curr_depth = cv2.imread(os.path.join(config.DATASET_PATH, matches[i][3]), cv2.IMREAD_UNCHANGED)
        if curr_rgb is None:
            continue

        curr_frame = Frame(i, curr_rgb, matches[i][0], config.K)
        tracker.compute_features(curr_frame)

        T_motion, _, _, _, shake_reason = tracker.match_and_track(history_frames[-3:], curr_frame, history_depths[-3:])

        if T_motion is not None and shake_reason is None:
            T_motion[:3, 3] *= config.SCALE_FIX
            curr_frame.set_pose(last_keyframe.pose @ T_motion)

            # Align and append
            aligned_pos = (vis_align @ curr_frame.pose)[:3, 3]
            traj_est.append(aligned_pos)

            last_keyframe = curr_frame
            prev_depth_data = curr_depth
            history_frames.append(curr_frame)
            history_depths.append(curr_depth)

            if len(history_frames) > 5:
                history_frames.pop(0)
                history_depths.pop(0)
        else:
            # Coast
            traj_est.append(traj_est[-1])
            curr_frame.pose = last_keyframe.pose

        # Sync Ground truth
        idx = np.searchsorted(gt_timestamps, matches[i][0])
        if idx < len(gt_timestamps):
            traj_gt.append(gt_data[gt_timestamps[idx]] - gt_start_pos)
        else:
            traj_gt.append(traj_gt[-1])

    min_len = min(len(traj_est), len(traj_gt))
    return calculate_svd_trajectory_alignment(traj_est[:min_len], traj_gt[:min_len])


def main():
    print("VO Auto-Tuner Initialized")
    total_tests = len(TEST_SCALES) * len(TEST_RANSAC) * len(TEST_GN_ITERS)
    print(f"Testing {total_tests} configurations on the dataset...\n")

    best_ate = float('inf')
    best_params = None

    for s in TEST_SCALES:
        for r in TEST_RANSAC:
            for g in TEST_GN_ITERS:
                print(f"Testing Scale={s:.2f}, RANSAC={r:.1f}px, GN_Iters={g}...", end=" ", flush=True)

                # FORCE IT TO USE ALL FRAMES BY PASSING A HUGE NUMBER
                ate = run_simulation(s, r, g, num_frames=9999)

                print(f"ATE: {ate:.4f}m")

                if ate < best_ate:
                    best_ate = ate
                    best_params = (s, r, g)

    print("\n" + "=" * 40)
    print("  TUNING COMPLETE  ")
    print(f"Best ATE: {best_ate:.4f} meters")
    print("Optimal Parameters for config.py:")
    print(f"  SCALE_FIX     = {best_params[0]}")
    print(f"  RANSAC_THRESH = {best_params[1]}")
    print(f"  GN_ITERS      = {best_params[2]}")
    print("=" * 40)


if __name__ == "__main__":
    main()