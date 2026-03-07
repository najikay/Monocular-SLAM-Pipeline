"""
Represents a discrete 3D spatial landmark triangulated from the environment.
"""


class Point:
    def __init__(self, point_id, coordinates):
        """
        Initializes a map point.

        Args:
            point_id (int): Unique identifier for the point.
            coordinates (np.array): 3D spatial coordinates [x, y, z].
        """
        self.id = point_id
        self.point = coordinates