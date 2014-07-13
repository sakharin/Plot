#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Plot(object):
    def __init__(self):
        self.fig = plt.figure()
        plt.ioff()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # For plotting scale
        self.xmx, self.xmn = -1e6, 1e6
        self.ymx, self.ymn = -1e6, 1e6
        self.zmx, self.zmn = -1e6, 1e6

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

    def plotCam(self, pt1, pt2, pt3, pt4, pt5, color='-k'):
        self.plotLine(pt1, pt2, color)
        self.plotLine(pt1, pt3, color)
        self.plotLine(pt1, pt4, color)
        self.plotLine(pt1, pt5, color)

        self.plotLine(pt2, pt3, color)
        self.plotLine(pt3, pt4, color)
        self.plotLine(pt4, pt5, color)
        self.plotLine(pt5, pt2, color)

        self.plotPoint(pt2, '.r')
        self.plotPoint(pt3, '.g')

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
    p.show(50, -75)
