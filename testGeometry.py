#!/usr/bin/env python
import cv2
import numpy as np

import Plot
import Geometry


if __name__ == "__main__":
    geo = Geometry.Geometry()
    p = Plot.Plot()

    P0 = np.zeros((3, 1))

    xyz = np.zeros((1, 34, 3))
    # Conners
    xyz[0, 0, :] = [1, 1, 1]
    xyz[0, 1, :] = [1, -1, 1]
    xyz[0, 2, :] = [-1, 1, 1]
    xyz[0, 3, :] = [-1, -1, 1]
    xyz[0, 4, :] = [1, 1, -1]
    xyz[0, 5, :] = [1, -1, -1]
    xyz[0, 6, :] = [-1, 1, -1]
    xyz[0, 7, :] = [-1, -1, -1]

    # Faces
    xyz[0, 8, :] = [1, 0, 0]
    xyz[0, 9, :] = [-1, 0, 0]
    xyz[0, 10, :] = [0, 1, 0]
    xyz[0, 11, :] = [0, -1, 0]
    xyz[0, 12, :] = [0, 0, 1]
    xyz[0, 13, :] = [0, 0, -1]

    # Edge
    xyz[0, 14, :] = [1, 1, 0]
    xyz[0, 15, :] = [-1, 1, 0]
    xyz[0, 16, :] = [1, -1, 0]
    xyz[0, 17, :] = [-1, -1, 0]

    xyz[0, 18, :] = [1, 0, 1]
    xyz[0, 19, :] = [-1, 0, 1]
    xyz[0, 20, :] = [1, 0, -1]
    xyz[0, 21, :] = [-1, 0, -1]

    xyz[0, 22, :] = [0, 1, 1]
    xyz[0, 23, :] = [0, -1, 1]
    xyz[0, 24, :] = [0, 1, -1]
    xyz[0, 25, :] = [0, -1, -1]

    xyz[0, 26, :] = [1, 0, 1]
    xyz[0, 27, :] = [-1, 0, 1]
    xyz[0, 28, :] = [1, 0, -1]
    xyz[0, 29, :] = [-1, 0, -1]

    xyz[0, 30, :] = [0, 1, 1]
    xyz[0, 31, :] = [0, -1, 1]
    xyz[0, 32, :] = [0, 1, -1]
    xyz[0, 33, :] = [0, -1, -1]

    vec1 = geo.twoPts2Vec(P0, xyz)
    phi2, theta2 = geo.vec2Angs(vec1)
    vec3 = geo.angs2Vec(phi2, theta2)
    ray4 = geo.vec2Ray(vec3, P0)
    vec5 = geo.ray2Vec(ray4)

    theta6, phi6 = geo.twoPts2Ang(P0, xyz)
    ray7 = geo.twoPts2Ray(P0, xyz)
    ray8 = geo.ptAngles2Ray(P0, theta6, phi6)

    p.plotPoint(P0, '.b')
    for i in range(34):
        p.plotPoint(xyz[0, i, :].reshape((3, 1)))
        if False:
            p.plotVec(vec1[0, i, :].reshape((3, 1)), None, 'r')
            p.plotVec(vec3[0, i, :].reshape((3, 1)), None, 'g')
            p.plotVec(vec5[0, i, :].reshape((3, 1)), None, 'b')

        if True:
            p.plotRay(ray4[0, i, :].reshape((5, 1)), 'r')
            p.plotRay(ray7[0, i, :].reshape((5, 1)), 'g')
            p.plotRay(ray8[0, i, :].reshape((5, 1)), 'b')
    p.plotAxis()
    p.show(90, 180)

if __name__ == "__main__1":
    # Test minMaxAng function
    errorMax = 1e-6
    for i in range(1000):
        geo = Geometry.Geometry()
        ang1 = np.random.randint(360)
        ang2 = ang1 + np.random.randint(180)
        angs = np.deg2rad(np.arange(ang1, ang2 + 1)) % (2 * np.pi)
        minAng, maxAng = np.rad2deg(geo.minMaxAng(angs))
        if abs((minAng % 360) - (ang1 % 360)) > errorMax or abs((maxAng % 360) - (ang2 % 360)) > errorMax:
            print "Error !"
            print minAng, ang1, maxAng, ang2
    print "Tested."

if __name__ == "__main__":
    A = geo.getRMatrixEulerAngles(0, 0, np.deg2rad(30))
    geo.checkRMatrix(A)
    B = np.eye(3)
    geo.checkRMatrix(B)
    C = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]])
    geo.checkRMatrix(C)
    D = np.array([[0, 0, 1, 3],
                  [1, 0, 0, 1],
                  [0, 1, 0, 2]])
    E = np.array([[0, 0, 1, 3],
                  [1, 0, 0, 1],
                  [0, 1, 0, 2],
                  [0, 0, 0, 1]])
    geo.checkTMatrix(A)
    geo.checkTMatrix(B)
    geo.checkTMatrix(C)
    geo.checkTMatrix(D)
    geo.checkTMatrix(E)
