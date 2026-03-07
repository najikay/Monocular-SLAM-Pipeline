"""
Tracker module responsible for visual odometry.
Handles ORB feature extraction, matching, 3D back-projection,
solvePnP estimation, Gauss-Newton optimization, and loop detection.
"""
import cv2
import numpy as np
import config


class Tracker:
    def __init__(self):
        """Initializes ORB detector and brute-force matcher."""
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def compute_features(self, frame):
        """Detects and computes ORB keypoints and descriptors for a given frame."""
        frame.keypoints, frame.descriptors = self.orb.detectAndCompute(frame.image, None)

    def detect_loop(self, curr_frame, all_frames):
        """
        Scans global map history to detect if the robot has returned to a previously
        mapped location (loop closure).

        Args:
            curr_frame (Frame): The current camera frame.
            all_frames (list): List of all previously tracked frames.
        Returns:
            T_correction (np.array): Corrective transformation matrix, or None.
            cand.id (int): ID of the matched loop frame.
        """
        # Ignore recent history to prevent false positive loop closures
        if len(all_frames) < config.THRESH_LOOP_FRAMES + 10:
            return None, None

        curr_pos = curr_frame.pose[:3, 3]
        candidates = []

        # Check spatial distance against all frames except the immediate past
        for old_frame in all_frames[:-config.THRESH_LOOP_FRAMES]:
            dist = np.linalg.norm(curr_pos - old_frame.pose[:3, 3])
            if dist < config.THRESH_LOOP_DIST:
                candidates.append(old_frame)

        candidates.sort(key=lambda f: np.linalg.norm(curr_pos - f.pose[:3, 3]))

        # Visually verify the top spatial candidates
        for cand in candidates[:3]:
            matches = self.bf.knnMatch(cand.descriptors, curr_frame.descriptors, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) > config.THRESH_LOOP_MATCHES:
                T_correction = cand.pose @ np.linalg.inv(curr_frame.pose)
                return T_correction, cand.id

        return None, None

    def match_and_track(self, history_frames, curr_frame, history_depths):
        """
        Multi-frame tracker strategy.
        Iterates backward through the history window if the immediate previous frame
        fails tracking (e.g., due to motion blur).

        Args:
            history_frames (list): List of recent Frame objects.
            curr_frame (Frame): The current frame to evaluate.
            history_depths (list): Corresponding depth matrices for history frames.
        Returns:
            Tuple containing: Transformation matrix, object points, None, valid matches, and failure reason.
        """
        if curr_frame.descriptors is None:
            return None, None, None, [], "No descriptors found"

        attempts = min(3, len(history_frames))
        best_inliers = 0
        best_result = (None, None, None, [], "All history frames failed")

        for i in range(attempts):
            ref_frame = history_frames[-(i + 1)]
            ref_depth = history_depths[-(i + 1)]

            if ref_frame.descriptors is None:
                continue

            # Feature matching using Lowe's ratio test
            raw_matches = self.bf.knnMatch(ref_frame.descriptors, curr_frame.descriptors, k=2)
            good_matches = []
            for m, n in raw_matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # Extract 3D-2D correspondences
            obj_pts, img_pts, valid_matches = [], [], []
            for m in good_matches:
                u1, v1 = ref_frame.keypoints[m.queryIdx].pt
                u2, v2 = curr_frame.keypoints[m.trainIdx].pt

                if int(v1) >= ref_depth.shape[0] or int(u1) >= ref_depth.shape[1]:
                    continue
                d = ref_depth[int(v1), int(u1)]

                if 0 < d < 50000:
                    # Back-project 2D pixel to 3D point using camera intrinsics
                    z = d / config.DEPTH_SCALE
                    p3d = np.array([(u1 - config.CX) * z / config.FX, (v1 - config.CY) * z / config.FY, z])
                    obj_pts.append(p3d)
                    img_pts.append((u2, v2))
                    valid_matches.append(m)

            if len(obj_pts) < config.THRESH_MIN_MATCHES:
                continue

            # Solve perspective-n-point with RANSAC using the config threshold
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.array(obj_pts), np.array(img_pts), config.K, None,
                confidence=0.99, reprojectionError=config.RANSAC_THRESH, flags=cv2.SOLVEPNP_EPNP
            )

            if success and inliers is not None and len(inliers) > best_inliers:
                best_inliers = len(inliers)

                # Refine pose via Gauss-Newton optimization
                inlier_obj = [obj_pts[k] for k in inliers.flatten()]
                inlier_img = [img_pts[k] for k in inliers.flatten()]
                rvec, tvec = self.optimize_pose_gn(rvec, tvec, np.array(inlier_obj), np.array(inlier_img))

                T_motion = np.linalg.inv(self.get_transform_matrix(rvec, tvec))

                # Normalize motion delta relative to the absolute last frame
                global_pose_curr = ref_frame.pose @ T_motion
                last_frame = history_frames[-1]
                T_rel_last_curr = np.linalg.inv(last_frame.pose) @ global_pose_curr

                best_result = (T_rel_last_curr, inlier_obj, None, valid_matches, None)

                # Early exit if highly confident
                if best_inliers > 40:
                    break

        return best_result

    def get_transform_matrix(self, rvec, tvec):
        """Converts rotation and translation vectors to a 4x4 homogeneous matrix."""
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T

    def optimize_pose_gn(self, rvec, tvec, obj_pts, img_pts):
        """
        Gauss-Newton nonlinear optimization to refine camera pose.
        Minimizes the reprojection error of 3D points onto the 2D image plane.
        """
        curr_rvec, curr_tvec = rvec.copy(), tvec.copy()
        # Use the config variable for iterations
        for _ in range(config.GN_ITERS):
            projected_pts, jacobian = cv2.projectPoints(obj_pts, curr_rvec, curr_tvec, config.K, None)
            projected_pts = projected_pts.reshape(-1, 2)

            error = img_pts - projected_pts
            residuals = error.ravel()
            J = jacobian[:, :6]

            try:
                H = J.T @ J + 1e-6 * np.eye(6)
                b = J.T @ residuals
                delta = np.linalg.solve(H, b)

                curr_rvec += delta[:3].reshape(3, 1)
                curr_tvec += delta[3:].reshape(3, 1)
            except np.linalg.LinAlgError:
                break

        return curr_rvec, curr_tvec