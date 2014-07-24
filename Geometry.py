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

    # Calculate semi inverse function of projection transformation
    def uvToXYZ(self, KInv, T_WC, uv1, F=1):
        XYZ1 = np.ones((4, 1))
        XYZ1[:3, :] = (KInv * F).dot(uv1)
        return T_WC.dot(XYZ1)


if __name__ == "__main__":
    geo = Geometry()
