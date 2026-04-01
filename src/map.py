"""
Defines the Map class representing the 3D environment and trajectory.
"""

class Map:
    def __init__(self):
        """Initializes lists for tracking frames and 3D map points."""
        self.frames = []
        self.points = []

    def add_frame(self, frame_obj):
        """Registers a tracked keyframe into the map."""
        self.frames.append(frame_obj)

    def add_point(self, point_obj):
        """Registers a triangulated 3D point into the map."""
        self.points.append(point_obj)