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

        # phi : arctan(dist / z): dist is the length of the ray
        diffSq = diff ** 2
        dist = np.sqrt(diffSq.sum())
        phi = np.arctan2(dist, diff[2, 0])
        R[4, :] = phi
        return R

    # Calculate semi inverse function of projection transformation
    def uvToXYZ(self, K, T_CW, uv):
        XYZ1 = np.ones((4, 1))
        uvDot = np.ones((3, 1))
        uvDot[:2, 0] = uv[:2, 0]

        T = T_CW[:3, :3]
        C = T_CW[:3, 3:4]

        KInv = np.linalg.inv(K)
        TInv = np.linalg.inv(T)

        XYZ1[:3, :1] = TInv.dot(KInv.dot(uvDot) - C)
        return XYZ1


if __name__ == "__main__":
    geo = Geometry()
