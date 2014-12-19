#!/usr/bin/env python
import cv2
import numpy as np

from Plot import Plot
from Geometry import Geometry

PI = np.pi

if __name__ == "__main__":
    p = Plot()
    geo = Geometry()

    R = 18.1
    T = 42.8

    PO = np.zeros((3, 1))

    # Define test points
    PC = np.array([[0], [0], [R]])
    h, w = 3, 15
    Ppx = np.zeros((h, w, 3))
    Ppx[:, :, 0] = np.arange(-(h / 2), (h / 2) + 1).reshape((h, 1))
    Ppx[:, :, 1] = -np.arange(-(w / 2), (w / 2) + 1).reshape((1, w))
    Ppx[:, :, 2] = 1.2 * R

    # Define rotation angles
    Ppx_theta = np.pi / 2
    Ppx_phi = 3 * np.pi / 16

    # Rotate points
    m = geo.getRMatrixEulerAngles(0, 0, Ppx_phi)
    m = m.dot(geo.getRMatrixEulerAngles(0, -Ppx_theta + PI, 0))
    Ppx_rotated = np.zeros((h, w, 3))
    Ppx_rotated[:, :, 0] = m[0, 0] * Ppx[:, :, 0] + \
        m[0, 1] * Ppx[:, :, 1] + \
        m[0, 2] * Ppx[:, :, 2]
    Ppx_rotated[:, :, 1] = m[1, 0] * Ppx[:, :, 0] + \
        m[1, 1] * Ppx[:, :, 1] + \
        m[1, 2] * Ppx[:, :, 2]
    Ppx_rotated[:, :, 2] = m[2, 0] * Ppx[:, :, 0] + \
        m[2, 1] * Ppx[:, :, 1] + \
        m[2, 2] * Ppx[:, :, 2]
    PC_rotated = m.dot(PC)

    # Define radius vector
    vecR = geo.twoPts2Vec(PO, PC_rotated) * R
    vecR_size = geo.normVec(vecR)

    # Define viewing vectors
    vec = geo.twoPts2Vec(PC_rotated, Ppx_rotated)

    # Call function
    vecView = geo.projectVecs2Depth(T, vecR, vec)
    vecView_size = geo.normVec(vecView)

    # Update viewing vectors
    vec = vecView - vecR.reshape((1, 1, 3))
    vec_size = geo.normVec(vec)

    # Plot
    p.plotVec(vecR, PO, 'r', vecR_size)
    p.plotPoint(PC_rotated, '.g')
    for i in range(h):
        for j in range(w):
            p.plotVec(vecView[i, j, :].reshape((3, 1)),
                      PO, 'm',
                      vecView_size[i, j])
            p.plotVec(vec[i, j, :].reshape((3, 1)),
                      PC_rotated, 'g',
                      vec_size[i, j])
    p.plotAxis()
    p.show(90, 180)
