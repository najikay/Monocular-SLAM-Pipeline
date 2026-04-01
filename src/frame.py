"""
Defines the Frame class representing a visual snapshot.
"""
import numpy as np

class Frame:
    def __init__(self, frame_id, image, timestamp, K):
        """
        Initializes a tracking frame.

        Args:
            frame_id (int): Sequential identifier.
            image (np.array): Grayscale image data.
            timestamp (float): Time the frame was recorded.
            K (np.array): Camera intrinsic matrix.
        """
        self.id = frame_id
        self.image = image
        self.timestamp = timestamp
        self.K = K
        self.pose = np.eye(4)
        self.keypoints = []
        self.descriptors = None

    def set_pose(self, pose_matrix):
        """Updates the global 4x4 transformation pose of the frame."""
        self.pose = pose_matrix