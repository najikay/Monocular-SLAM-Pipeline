"""
Tracker module responsible for visual odometry.
Handles ORB feature extraction, 3D back-projection, PnP estimation,
Gauss-Newton optimization, and loop detection.
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
        """Detects and computes ORB keypoints with sub-pixel refinement."""
        frame.keypoints, frame.descriptors = self.orb.detectAndCompute(frame.image, None)

        if frame.keypoints:
            pts = np.array([kp.pt for kp in frame.keypoints], dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            pts = cv2.cornerSubPix(frame.image, pts, (3, 3), (-1, -1), criteria)
            for i, kp in enumerate(frame.keypoints):
                kp.pt = tuple(pts[i])

    def detect_loop(self, curr_frame, all_frames):
        """Scans global map history to detect loop closures."""
        if len(all_frames) < config.THRESH_LOOP_FRAMES + 10:
            return None, None

        curr_pos = curr_frame.pose[:3, 3]
        candidates = []

        for old_frame in all_frames[:-config.THRESH_LOOP_FRAMES]:
            dist = np.linalg.norm(curr_pos - old_frame.pose[:3, 3])
            if dist < config.THRESH_LOOP_DIST:
                candidates.append(old_frame)

        candidates.sort(key=lambda f: np.linalg.norm(curr_pos - f.pose[:3, 3]))

        for cand in candidates[:3]:
            matches = self.bf.knnMatch(cand.descriptors, curr_frame.descriptors, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good) > config.THRESH_LOOP_MATCHES:
                T_correction = cand.pose @ np.linalg.inv(curr_frame.pose)
                return T_correction, cand.id

        return None, None

    def match_and_track(self, history_frames, curr_frame, history_depths):
        """Calculates camera pose iteratively via Perspective-n-Point (PnP)."""
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

            raw_matches = self.bf.knnMatch(ref_frame.descriptors, curr_frame.descriptors, k=2)
            good_matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

            obj_pts, img_pts, valid_matches = [], [], []
            for m in good_matches:
                u1, v1 = ref_frame.keypoints[m.queryIdx].pt
                u2, v2 = curr_frame.keypoints[m.trainIdx].pt

                if int(v1) >= ref_depth.shape[0] or int(u1) >= ref_depth.shape[1]:
                    continue
                d = ref_depth[int(v1), int(u1)]

                if 0 < d < 50000:
                    z = d / config.DEPTH_SCALE
                    p3d = np.array([(u1 - config.CX) * z / config.FX, (v1 - config.CY) * z / config.FY, z])
                    obj_pts.append(p3d)
                    img_pts.append((u2, v2))
                    valid_matches.append(m)

            if len(obj_pts) < config.THRESH_MIN_MATCHES:
                continue

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.array(obj_pts), np.array(img_pts), config.K, None,
                confidence=0.99, reprojectionError=config.RANSAC_THRESH, flags=cv2.SOLVEPNP_EPNP
            )

            if success and inliers is not None and len(inliers) > best_inliers:
                best_inliers = len(inliers)
                inlier_obj = [obj_pts[k] for k in inliers.flatten()]
                inlier_img = [img_pts[k] for k in inliers.flatten()]

                rvec, tvec = self.optimize_pose_gn(rvec, tvec, np.array(inlier_obj), np.array(inlier_img))
                T_motion = np.linalg.inv(self.get_transform_matrix(rvec, tvec))

                global_pose_curr = ref_frame.pose @ T_motion
                T_rel = np.linalg.inv(history_frames[-1].pose) @ global_pose_curr

                best_result = (T_rel, None, None, valid_matches, None)

                if best_inliers > 40:
                    break

        return best_result

    def triangulate_keyframes(self, kf1, kf2):
        """
        Calculates 3D map points from two 2D views using epipolar geometry.
        Applies physical bounds and reprojection error checks.
        """
        raw_matches = self.bf.knnMatch(kf1.descriptors, kf2.descriptors, k=2)
        good_matches = []
        for m, n in raw_matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 10:
            return []

        pts1 = [kf1.keypoints[m.queryIdx].pt for m in good_matches]
        pts2 = [kf2.keypoints[m.trainIdx].pt for m in good_matches]

        pose1 = kf1.pose
        pose2 = kf2.pose
        K = config.K

        T1 = np.linalg.inv(pose1)[:3, :]
        T2 = np.linalg.inv(pose2)[:3, :]
        P1 = K @ T1
        P2 = K @ T2

        pts1_idx = np.array(pts1, dtype=np.float32).T
        pts2_idx = np.array(pts2, dtype=np.float32).T

        points_4d = cv2.triangulatePoints(P1, P2, pts1_idx, pts2_idx)

        valid_3d_points = []
        for i in range(points_4d.shape[1]):
            w = points_4d[3, i]
            if abs(w) < 1e-5:
                continue

            p3d = points_4d[:3, i] / w

            # Physical room constraints
            p_cam1 = T1 @ np.append(p3d, 1.0)
            if not (0.1 < p_cam1[2] < 3.5 and -1.5 < p3d[1] < 1.5):
                continue

            # Reprojection error validation
            p3d_homo = np.append(p3d, 1.0)

            proj1 = P1 @ p3d_homo
            proj1 = proj1[:2] / proj1[2]
            err1 = np.linalg.norm(proj1 - pts1_idx[:, i])

            proj2 = P2 @ p3d_homo
            proj2 = proj2[:2] / proj2[2]
            err2 = np.linalg.norm(proj2 - pts2_idx[:, i])

            if err1 < 2.0 and err2 < 2.0:
                valid_3d_points.append(p3d)

        return valid_3d_points

    def get_transform_matrix(self, rvec, tvec):
        """Converts rotation and translation vectors into a 4x4 matrix."""
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T

    def optimize_pose_gn(self, rvec, tvec, obj_pts, img_pts):
        """Refines camera pose using Gauss-Newton nonlinear optimization."""
        curr_rvec, curr_tvec = rvec.copy(), tvec.copy()
        for _ in range(config.GN_ITERS):
            projected_pts, jacobian = cv2.projectPoints(obj_pts, curr_rvec, curr_tvec, config.K, None)
            projected_pts = projected_pts.reshape(-1, 2)
            error = img_pts - projected_pts
            J = jacobian[:, :6]
            try:
                H = J.T @ J + 1e-6 * np.eye(6)
                b = J.T @ error.ravel()
                delta = np.linalg.solve(H, b)
                curr_rvec += delta[:3].reshape(3, 1)
                curr_tvec += delta[3:].reshape(3, 1)
            except np.linalg.LinAlgError:
                break
        return curr_rvec, curr_tvec