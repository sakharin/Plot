#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from Geometry import Geometry


class Arrow3D(FancyArrowPatch):
    #http://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector
    def __init__(self, pt1, pt2, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        x = [pt1[0, 0], pt2[0, 0]]
        y = [pt1[1, 0], pt2[1, 0]]
        z = [pt1[2, 0], pt2[2, 0]]
        self._verts3d = x, y, z

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class Plot(object):
    def __init__(self):
        self.fig = plt.figure()
        plt.ioff()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # For plotting scale
        self.xmx, self.xmn = -1e6, 1e6
        self.ymx, self.ymn = -1e6, 1e6
        self.zmx, self.zmn = -1e6, 1e6

        self.geo = Geometry()

    def show(self, elev=0., azim=0.):
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.view_init(elev=elev, azim=azim)

        # calculate scale
        rangeX = self.xmx - self.xmn
        rangeY = self.ymx - self.ymn
        rangeZ = self.zmx - self.zmn

        maxRange = max(rangeX, rangeY, rangeZ)
        md = maxRange / 2.
        cx = (self.xmx + self.xmn) / 2.
        cy = (self.ymx + self.ymn) / 2.
        cz = (self.zmx + self.zmn) / 2.

        self.ax.auto_scale_xyz([cx - md, cx + md],
                               [cy - md, cy + md],
                               [cz - md, cz + md])
        plt.show()

    def updateRegion(self, pt):
        x, y, z = pt[:3, 0]
        self.xmn = min(self.xmn, x)
        self.xmx = max(self.xmx, x)

        self.ymn = min(self.ymn, y)
        self.ymx = max(self.ymx, y)

        self.zmn = min(self.zmn, z)
        self.zmx = max(self.zmx, z)

    def plotPoint(self, pt1, color='.r'):
        self.updateRegion(pt1)
        self.ax.plot([pt1[0, 0]],
                     [pt1[1, 0]],
                     [pt1[2, 0]], color)

    def plotLine(self, pt1, pt2, color='-r'):
        self.updateRegion(pt1)
        self.updateRegion(pt2)
        self.ax.plot([pt1[0, 0], pt2[0, 0]],
                     [pt1[1, 0], pt2[1, 0]],
                     [pt1[2, 0], pt2[2, 0]], color)

    def plotPlane(self, P, color='-r'):
        f = lambda a, b: -(a * P[0, 0] + b * P[1, 0] + P[3, 0]) / P[2, 0]
        pt1 = np.matrix(np.array([[-0.5, 0.5, f(-0.5, 0.5), 1.0]]).T)
        pt2 = np.matrix(np.array([[0.5, 0.5, f(0.5, 0.5), 1.0]]).T)
        pt3 = np.matrix(np.array([[0.5, -0.5, f(0.5, -0.5), 1.0]]).T)
        pt4 = np.matrix(np.array([[-0.5, -0.5, f(-0.5, -0.5), 1.0]]).T)
        self.plotLine(pt1, pt2, color)
        self.plotLine(pt2, pt3, color)
        self.plotLine(pt3, pt4, color)
        self.plotLine(pt4, pt1, color)

    def plotBox(self, pt1, pt2, pt3=None, pt4=None, pt5=None, pt6=None, pt7=None, pt8=None, color='-r'):
        if pt3 is None:
            x1, y1, z1 = pt1[:, 0]
            x2, y2, z2 = pt2[:, 0]

            pt1 = np.array([[x1, y1, z1, 1]]).T
            pt2 = np.array([[x1, y2, z1, 1]]).T
            pt3 = np.array([[x2, y2, z1, 1]]).T
            pt4 = np.array([[x2, y1, z1, 1]]).T

            pt5 = np.array([[x1, y1, z2, 1]]).T
            pt6 = np.array([[x1, y2, z2, 1]]).T
            pt7 = np.array([[x2, y2, z2, 1]]).T
            pt8 = np.array([[x2, y1, z2, 1]]).T
        self.plotLine(pt1, pt2, color)
        self.plotLine(pt2, pt3, color)
        self.plotLine(pt3, pt4, color)
        self.plotLine(pt4, pt1, color)

        self.plotLine(pt5, pt6, color)
        self.plotLine(pt6, pt7, color)
        self.plotLine(pt7, pt8, color)
        self.plotLine(pt8, pt5, color)

        self.plotLine(pt1, pt5, color)
        self.plotLine(pt2, pt6, color)
        self.plotLine(pt3, pt7, color)
        self.plotLine(pt4, pt8, color)

    def plotCam(self, param0, param1=None, param2=None, param3=None,
                param4=None, color='-k'):
        msg = "usage:\n" \
            "    :plotCam(T, [A[, (w, h)[, s]]][, color])\n" \
            "     T         : 3x3, 3x4, 4x4 transformation matrix\n" \
            "     A         : 3x3 intrinsic parameter\n" \
            "     (w, h)    : width and height of an image in pixel\n" \
            "     s         : camera scale\n" \
            "     color     : matplotlib color\n" \
            "\n" \
            "    :plotCam(pt0, pt1, pt2, pt3, pt4[, color])\n" \
            "     pt1 - pt5 : 3x1, 4x1 points\n" \
            "     color     : matplotlib color\n"
        if not isinstance(param0, np.ndarray):
            raise TypeError(msg)
        if len(param0.shape) != 2:
            raise TypeError(msg)

        M, N = param0.shape
        if M != 3 and M != 4:
            raise TypeError(msg)

        if N == 1:
            # Check other points
            cond2 = not isinstance(param1, np.ndarray)
            cond3 = not isinstance(param2, np.ndarray)
            cond4 = not isinstance(param3, np.ndarray)
            cond5 = not isinstance(param4, np.ndarray)
            if cond2 or cond3 or cond4 or cond5:
                raise TypeError(msg)

            cond2 = len(param1.shape) != 2
            cond3 = len(param2.shape) != 2
            cond4 = len(param3.shape) != 2
            cond5 = len(param4.shape) != 2
            if cond2 or cond3 or cond4 or cond5:
                raise TypeError(msg)

            M2, N2 = param1.shape
            M3, N3 = param2.shape
            M4, N4 = param3.shape
            M5, N5 = param4.shape
            cond2 = M != M2 or N != N2
            cond3 = M != M3 or N != N3
            cond4 = M != M4 or N != N4
            cond5 = M != M5 or N != N5
            if cond2 or cond3 or cond4 or cond5:
                raise TypeError(msg)

            pt0 = param0
            pt1 = param1
            pt2 = param2
            pt3 = param3
            pt4 = param4

        if N >= 3 and N <= 4:
            # Plot using transformation matrix
            fx, fy = 1., 1.
            cx, cy = 0.5, 0.5
            w, h = 1., 1.
            d = 1

            if param1 is not None:
                if not isinstance(param1, np.ndarray):
                    raise TypeError(msg)
                if len(param1.shape) != 2:
                    raise TypeError(msg)
                if param1.shape[0] != 3 or param1.shape[1] != 3:
                    raise TypeError(msg)
                fx = param1[0, 0]
                fy = param1[1, 1]
                cx, cy = param1[:2, 2]

            if param2 is not None:
                if not isinstance(param2, tuple):
                    raise TypeError(msg)
                if len(param2) != 2:
                    raise TypeError(msg)
                w, h = param2

            if param3 is not None:
                if not isinstance(param3, (int, float, long)):
                    raise TypeError(msg)
                d = param3

            T = np.zeros((4, 4))
            T[:M, :N] = param0[:, :]
            T[3, 3] = 1

            h1 = -cy * d / fy
            w1 = -cx * d / fx
            h2 = (h - cy) * d / fy
            w2 = (w - cx) * d / fx

            pt0 = T.dot(np.array([[0, 0, 0, 1]]).T)
            pt1 = T.dot(np.array([[w1, h1, d, 1]]).T)
            pt2 = T.dot(np.array([[w2, h1, d, 1]]).T)
            pt3 = T.dot(np.array([[w1, h2, d, 1]]).T)
            pt4 = T.dot(np.array([[w2, h2, d, 1]]).T)

        # Plot using 5 pts
        self.plotLine(pt0, pt1, color)
        self.plotLine(pt0, pt2, color)
        self.plotLine(pt0, pt3, color)
        self.plotLine(pt0, pt4, color)

        self.plotLine(pt1, pt2, color)
        self.plotLine(pt1, pt3, color)
        self.plotLine(pt2, pt4, color)
        self.plotLine(pt3, pt4, color)

        self.plotPoint(pt1, '.r')
        self.plotPoint(pt2, '.g')

    def plotAxis(self, R=None, scale=1.):
        pt0 = np.matrix(np.array([[0., 0., 0., 1.]]).T)
        ptx = np.matrix(np.array([[scale, 0., 0., 1.]]).T)
        pty = np.matrix(np.array([[0., scale, 0., 1.]]).T)
        ptz = np.matrix(np.array([[0., 0., scale, 1.]]).T)

        if R is not None:
            pt0 = R * pt0
            ptx = R * ptx
            pty = R * pty
            ptz = R * ptz

        self.plotLine(pt0, ptx, '-r')
        self.plotLine(pt0, pty, '-g')
        self.plotLine(pt0, ptz, '-b')

    def plotArrow(self, pt1, pt2, color='r'):
        a = Arrow3D(pt1, pt2, mutation_scale=20, lw=1, arrowstyle="-|>", color=color)
        self.ax.add_artist(a)
        self.updateRegion(pt1)
        self.updateRegion(pt2)

    def plotRay(self, R, color='r', scale=1.):
        pt1 = R[:3, :]
        T = self.geo.getRMatrixEulerAngles(0, 0, R[4, 0])
        T = T.dot(self.geo.getRMatrixEulerAngles(0, R[3, 0], 0))
        pt2 = T.dot(np.array([[0, 0, scale]]).T) + pt1
        self.updateRegion(pt1)
        self.updateRegion(pt2)
        self.plotArrow(pt1, pt2, color)

    def plotVec(self, vec, origin=None, color='r'):
        if origin is None:
            origin = np.zeros((3, 1))
        ray = np.zeros((5, 1))
        ray[:3, :] = origin
        norm = np.sqrt((vec ** 2).sum())
        if norm == 0:
            ray[3, 0] = 0
            ray[4, 0] = 0
        elif vec[0, 0] == 0:
            ray[3, 0] = np.arccos(vec[2, 0] / norm)
            if vec[1, 0] == 0:
                ray[4, 0] = 0
            elif vec[1, 0] > 0:
                ray[4, 0] = np.pi / 2.
            else:
                ray[4, 0] = 3 * np.pi / 2.
        else:
            ray[3, 0] = np.arccos(vec[2, 0] / norm)
            ray[4, 0] = np.arctan2(vec[1, 0], vec[0, 0])
        self.plotRay(ray, color, norm)

    def plotAirplane(self, R=None, scale=1.):
        # plot an airplane centered at (0, 0, 0) heading to x direction

        # Left elevator tip
        pt1 = np.matrix(np.array([[-1.0, 0.5, 0.0, 1.0]]).T)
        # Tail
        pt2 = np.matrix(np.array([[-1.0, 0.0, 0.0, 1.0]]).T)
        # Right elevator tip
        pt3 = np.matrix(np.array([[-1.0, -0.5, 0.0, 1.0]]).T)

        # Front of elevator
        pt4 = np.matrix(np.array([[-.5, 0.0, 0.0, 1.0]]).T)

        # Rear of wing
        pt5 = np.matrix(np.array([[0.0, 0.0, 0.0, 1.0]]).T)

        # Right wing
        pt6 = np.matrix(np.array([[0.0, 1.0, 0.0, 1.0]]).T)

        # Left wing
        pt7 = np.matrix(np.array([[0.0, -1.0, 0.0, 1.0]]).T)

        pt8 = np.matrix(np.array([[1.0, 0.0, 0.0, 1.0]]).T)

        pt9 = np.matrix(np.array([[-1.0, 0.0, 0.5, 1.0]]).T)

        if R is None:
            R = np.eye(4)
        pt1 = R * pt1
        pt2 = R * pt2
        pt3 = R * pt3
        pt4 = R * pt4
        pt5 = R * pt5
        pt6 = R * pt6
        pt7 = R * pt7
        pt8 = R * pt8
        pt9 = R * pt9

        # Elevator
        self.plotLine(pt1, pt2)
        self.plotLine(pt2, pt3)
        self.plotLine(pt1, pt4)
        self.plotLine(pt2, pt4)
        self.plotLine(pt3, pt4)

        # Body
        self.plotLine(pt4, pt5)
        self.plotLine(pt5, pt8)

        # Left wing
        self.plotLine(pt5, pt6)
        self.plotLine(pt6, pt8)

        # Right wing
        self.plotLine(pt5, pt7)
        self.plotLine(pt7, pt8)

        # Rudder
        self.plotLine(pt2, pt9)
        self.plotLine(pt4, pt9)

        # Lights
        self.plotPoint(pt6, 'or')
        self.plotPoint(pt7, 'og')
        self.plotPoint(pt9, 'ow')


if __name__ == "__main__":
    p = Plot()
    for i in range(10):
        pt = np.random.uniform(0, 10, 3).reshape((3, 1))
        p.plotPoint(pt)

        pt1 = np.random.uniform(-5, 0, 3).reshape((3, 1))
        pt2 = np.random.uniform(-5, 0, 3).reshape((3, 1))
        p.plotLine(pt1, pt2, 'g-')

        pt1 = np.random.uniform(-10, -5, 3).reshape((3, 1))
        pt2 = np.random.uniform(-10, -5, 3).reshape((3, 1))
        p.plotBox(pt1, pt2, color='b-')
    p.plotAxis(scale=1)

    R = np.eye(4) * 3
    R[:3, 3] = np.array([5, -5, -2])
    R[3, 3] = 1
    p.plotAirplane(R, scale=5)

    thetas = np.pi / 16 * np.arange(32)
    phis = np.pi / 4 * np.arange(5)
    for theta in thetas:
        for phi in phis:
            R = np.array([[-10, 5, 2, theta, phi]]).T
            p.plotRay(R, 'c', 3)

            vec = np.zeros((3, 1))
            vec[0, 0] = np.sin(theta)
            vec[2, 0] = np.cos(theta)
            vec[1, 0] = vec[0, 0] * np.sin(phi)
            vec[0, 0] = vec[0, 0] * np.cos(phi)
            vec *= 2
            pt = np.array([[0, -10, 0]]).T
            p.plotVector(vec, pt, 'm')

    p.show(50, -75)
