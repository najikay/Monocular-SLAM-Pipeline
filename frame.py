"""
Represents a single visual snapshot in the SLAM system.
Encapsulates the image data, intrinsic parameters, spatial pose, and extracted keypoints.
"""
import numpy as np


class Frame:
    def __init__(self, frame_id, image, timestamp, K):
        """
        Initializes a tracking frame.

        Args:
            frame_id (int): Sequential identifier for the frame.
            image (np.array): Grayscale image data.
            timestamp (float): Time the frame was recorded.
            K (np.array): Camera intrinsic matrix (3x3).
        """
        self.id = frame_id
        self.image = image
        self.timestamp = timestamp
        self.K = K
        self.pose = np.eye(4)
        self.keypoints = []
        self.descriptors = None

    def set_pose(self, pose_matrix):
        """Updates the global 4x4 transformation pose of this frame."""
        self.pose = pose_matrix