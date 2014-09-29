#!/usr/bin/env python
import cv2
import numpy as np

from Plot import Plot
from Geometry import Geometry


class Ray(object):
    def __init__(self, pt0=None, pt1=None, pt2=None):
        self.val = np.zeros((5, 1))
        self.geo = Geometry()

        if pt0 is not None and pt1 is not None and pt2 is None:
            self.from2Points(pt0, pt1)
        elif pt0 is not None and pt1 is not None and pt2 is not None:
            self.from3Points(pt0, pt1, pt2)

    def from2Points(self, pt1, pt2):
        self.val[:3, :] = pt1[:3, :]

        diff = pt2[:3, :] - pt1[:3, :]

        # phi = arctan(y / x)
        phi = np.arctan2(diff[1, 0], diff[0, 0])
        self.val[4, :] = phi

        # theta : arccos(z / dist): dist is the length of the ray
        diffSq = diff ** 2
        dist = np.sqrt(diffSq.sum())
        theta = np.arccos(diff[2, 0] / dist)
        self.val[3, :] = theta

    def from3Points(self, pt0, pt1, pt2):
        self.from2Points(pt1, pt2)
        self.val[:3, :] = pt0[:3, :]

    def toRot(self):
        theta = self.val[3, 0]
        phi = self.val[4, 0]

        rot0 = self.geo.getRMatrixEulerAngles(0, 0, np.deg2rad(-90))
        rot1 = self.geo.getRMatrixEulerAngles(0, theta, 0)
        rot2 = self.geo.getRMatrixEulerAngles(0, 0, phi)
        return rot2.dot(rot1.dot(rot0))

    def toT(self, c_W):
        R_WC = self.toRot()
        R_CW = np.linalg.inv(R_WC)
        t_C = -R_CW.dot(c_W)
        t_W = -R_WC.dot(t_C)

        T_CW = np.zeros((4, 4))
        T_CW[:3, :3] = R_CW
        T_CW[:3, 3:4] = t_C
        T_CW[3, 3] = 1.

        T_WC = np.zeros((4, 4))
        T_WC[:3, :3] = R_WC
        T_WC[:3, 3:4] = t_W
        T_WC[3, 3] = 1.
        return T_WC, T_CW


if __name__ == "__main__":
    p = Plot()
    geo = Geometry()
    d = 10
    pt0 = np.array([[0], [0], [0]])
    p.plotPoint(pt0)
    for i in range(10):
        angle = np.deg2rad(10 * i)
        pt = np.array([[d * np.cos(angle)],
                       [-d * np.sin(angle)],
                       [i]])
        if i == 0:
            p.plotPoint(pt, 'g.')
        else:
            p.plotPoint(pt)

        r = Ray(pt0, pt)
        T_WC, T_CW = r.toT(pt)
        p.plotCam(T_WC)
        p.plotRay(r.val, scale=10)

        r = Ray(pt, pt0, pt)
        p.plotRay(r.val, 'g', scale=10)
    p.show(65, 40)
