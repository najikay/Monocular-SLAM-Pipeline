"""
Defines the Point class representing a 3D spatial landmark.
"""

class Point:
    def __init__(self, point_id, coordinates):
        """
        Initializes a 3D map point.

        Args:
            point_id (int): Unique identifier for the point.
            coordinates (np.array): 3D spatial coordinates [x, y, z].
        """
        self.id = point_id
        self.point = coordinates