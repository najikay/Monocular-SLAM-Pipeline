"""
Global map data structure.
Stores the history of recorded frames and 3D spatial points for visualization and tracking.
"""

class Map:
    def __init__(self):
        """Initializes empty lists for trajectory frames and spatial map points."""
        self.frames = []
        self.points = []

    def add_frame(self, frame_obj):
        """Registers a newly tracked keyframe into the global map."""
        self.frames.append(frame_obj)

    def add_point(self, point_obj):
        """Registers a triangulated 3D point into the map for visualization."""
        self.points.append(point_obj)