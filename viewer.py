"""
3D visualization module using Pangolin and OpenGL.
Renders the map points, the estimated trajectory, the ground truth trajectory,
and the current camera pose in real-time.
"""
import pypangolin as pangolin
import OpenGL.GL as gl
import numpy as np
import config


class Viewer:
    def __init__(self):
        """Initializes the Pangolin window, OpenGL context, and camera view state."""
        pangolin.CreateWindowAndBind('VO SLAM', 960, 720)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin.ModelViewLookAt(-2.0, -2.0, -2.0, 0, 0, 0, pangolin.AxisDirection.AxisZ)
        )

        self.handler = pangolin.Handler3D(self.scam)
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(
            pangolin.Attach(0.0), pangolin.Attach(1.0),
            pangolin.Attach(0.0), pangolin.Attach(1.0),
            -640.0 / 480.0
        )
        self.dcam.SetHandler(self.handler)

        # Calculate alignment matrix once
        self.vis_align = self._calculate_alignment_matrix()

    def _calculate_alignment_matrix(self):
        """Calculates the static alignment matrix to correct coordinate frame orientations."""
        # Base camera frame correction
        R_base = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])

        # Restore the Yaw Alignment from Config
        theta = np.deg2rad(config.ALIGN_ANGLE)
        c, s = np.cos(theta), np.sin(theta)
        R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        align = np.eye(4)
        align[:3, :3] = R_yaw @ R_base
        return align

    def should_quit(self):
        """Checks if the user has requested to close the visualization window."""
        return pangolin.ShouldQuit()

    def update(self, vo_map, trajectory, gt_traj, current_pose):
        """
        Refreshes the OpenGL display with updated map and trajectory data.

        Args:
            vo_map (Map): The global map containing 3D points.
            trajectory (list): List of estimated 3D trajectory positions.
            gt_traj (list): List of ground truth 3D trajectory positions.
            current_pose (np.array): The current 4x4 camera pose matrix.
        """
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.dcam.Activate(self.scam)

        # Draw spatial points (white)
        gl.glPointSize(1)
        gl.glColor3f(1.0, 1.0, 1.0)
        if len(vo_map.points) > 0:
            pangolin.glDrawPoints([p.point for p in vo_map.points[-3000:]])

        # Draw estimated trajectory (green)
        gl.glLineWidth(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.glDrawLineStrip(trajectory)

        # Draw ground truth trajectory (red)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.glDrawLineStrip(gt_traj)

        # Draw current camera pose indicator (yellow cone)
        gl.glPushMatrix()
        aligned_pose = self.vis_align @ current_pose
        gl.glMultMatrixd(aligned_pose.T)
        gl.glLineWidth(1)
        gl.glColor3f(1.0, 1.0, 0.0)
        self._draw_camera_cone()
        gl.glPopMatrix()

        pangolin.FinishFrame()

    def get_aligned_pos(self, pose):
        """Returns the 3D position vector aligned to the visualization coordinate frame."""
        return (self.vis_align @ pose)[:3, 3]

    def _draw_camera_cone(self):
        """Draws a wireframe representation of the camera frustum."""
        w, h, z = 0.1, 0.075, 0.2
        gl.glBegin(gl.GL_LINES)
        vertices = [
            (0, 0, 0), (w, h, z), (0, 0, 0), (w, -h, z),
            (0, 0, 0), (-w, -h, z), (0, 0, 0), (-w, h, z),
            (w, h, z), (w, -h, z), (w, -h, z), (-w, -h, z),
            (-w, -h, z), (-w, h, z), (-w, h, z), (w, h, z)
        ]
        for v in vertices:
            gl.glVertex3f(*v)
        gl.glEnd()