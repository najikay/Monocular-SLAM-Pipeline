"""
3D visualization module using Pangolin and OpenGL.
Renders map points, estimated trajectory, and optional ground truth.
"""
import pypangolin as pangolin
import OpenGL.GL as gl
import numpy as np
import config

class Viewer:
    def __init__(self):
        """Initializes the Pangolin window and OpenGL context."""
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
        self.vis_align = self._calculate_alignment_matrix()

    def _calculate_alignment_matrix(self):
        """Calculates static alignment matrix to correct coordinate frame orientations."""
        R_base = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
        theta = np.deg2rad(config.ALIGN_ANGLE)
        c, s = np.cos(theta), np.sin(theta)
        R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        align = np.eye(4)
        align[:3, :3] = R_yaw @ R_base
        return align

    def should_quit(self):
        """Checks if the user has requested to close the window."""
        return pangolin.ShouldQuit()

    def update(self, vo_map, trajectory, gt_traj, current_pose):
        """Refreshes the OpenGL display with updated map and trajectory data."""
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.dcam.Activate(self.scam)

        # Draw 3D spatial map points (white)
        gl.glPointSize(1)
        gl.glColor3f(1.0, 1.0, 1.0)
        if len(vo_map.points) > 0:
            pangolin.glDrawPoints([p.point for p in vo_map.points[-100000:]])

        # Draw estimated trajectory (green)
        gl.glLineWidth(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        if trajectory:
            pangolin.glDrawLineStrip(trajectory)

        # Draw ground truth trajectory if provided (red)
        gl.glColor3f(1.0, 0.0, 0.0)
        if gt_traj:
            pangolin.glDrawLineStrip(gt_traj)

        # Draw current camera pose indicator
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