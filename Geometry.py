#!/usr/bin/env python
import cv2
import numpy as np


class Geometry(object):
    def __init__(self):
        pass

    def getRMatrixEulerAngles(self, A=0, B=0, C=0):
        # Graphic Gems IV, Paul S. Heckbert, 1994
        cA, sA = np.cos(A), np.sin(A)
        cB, sB = np.cos(B), np.sin(B)
        cC, sC = np.cos(C), np.sin(C)

        rC = np.array([[cC, -sC, 0.],
                       [sC, cC, 0.],
                       [0., 0., 1.]])
        rB = np.array([[cB, 0., sB],
                       [0., 1., 0.],
                       [-sB, 0., cB]])
        rA = np.array([[1., 0., 0.],
                       [0., cA, -sA],
                       [0., sA, cA]])
        return rA.dot(rB).dot(rC)

    def projectPointToPlane(self, pts, P):
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html
        # Find a point on the plane
        z = -1. * (P[3, 0]) / P[2, 0]
        orgPt = np.array([[0], [0], [z], [1]])

        # 1
        v = pts - orgPt
        v[:, 3, 0] = 1

        # 2
        dist = np.tensordot(v[:, :3, :], P[:3, :], axes=([1, 0])).reshape(-1)

        # 3
        N = pts.shape[0]
        ppts = np.zeros((N, 4, 1))
        for i in range(N):
            ppts[i, :, :] = pts[i, :, :]
            ppts[i, :3, 0] -= dist[i] * P[:3, 0]
        return ppts

    def twoPointsToRay(self, pt1, pt2):
        R = np.zeros((5, 1))
        R[:3, :] = pt1[:3, :]

        diff = pt2[:3, :] - pt1[:3, :]

        # theta = arctan(y / x)
        theta = np.arctan2(diff[1, 0], diff[0, 0])
        R[3, :] = theta

        # phi : arctan(z / dist): dist is the projected ray
        #       on x-y plane
        dist = np.sqrt(diff[0, 0] ** 2 + diff[1, 0] ** 2)
        phi = -np.arctan2(diff[2, 0], dist)
        R[4, :] = phi

        return R

    # Calculate semi inverse function of projection transformation
    def uvToXYZ(self, KInv, T_WC, uv1, F=1):
        XYZ1 = np.ones((4, 1))
        XYZ1[:3, :] = (KInv * F).dot(uv1)
        return T_WC.dot(XYZ1)


if __name__ == "__main__":
    geo = Geometry()
