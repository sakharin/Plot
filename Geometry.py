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

    def angleDiff(self, angle1, angle2):
        # http://stackoverflow.com/questions/12234574/calculating-if-an-angle-is-between-two-angles
        # Return diff in range [-pi, pi]
        return (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi

    def lineCrossingPlane(self, P0, u, V0, n):
        # http://geomalgorithms.com/a05-_intersect-1.html
        # L: x = P0 + s u
        # P: 0 = n.(x - V0)
        shape = u.shape
        if len(shape) == 2:
            Si = n.T.dot(V0 - P0) / n.T.dot(u)
            return P0 + Si * u
        else:
            num = n.T.dot(V0 - P0)
            den = (n.reshape(1, 1, 3) * u).sum(axis=2)
            Si = num / den
            points = P0.reshape((1, 1, 3)) + \
                Si.reshape((shape[0], shape[1], 1)) * u
            return points

    def distPoint2Line(self, P, P0, u):
        # http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        P0Shape = P0.shape
        uShape = u.shape
        if len(uShape) == 2:
            diff = P0 - P
            return np.sqrt(((diff - diff.T.dot(u) * u) ** 2).sum())
        elif len(P0Shape) == 2:
            diff = P0 - P
            diffTDotu = (diff.reshape((1, 1, 3)) * u).sum(axis=2)
            diffTDotuTimesu = diffTDotu.reshape((uShape[0], uShape[1], 1)) * u
            diffMinusdiffTDotuTimesu = diff.reshape((1, 1, 3)) - diffTDotuTimesu
            return np.sqrt((diffMinusdiffTDotuTimesu ** 2).sum(axis=2))
        elif len(P0Shape) == 3:
            diff = P0 - P.reshape((1, 1, 3))
            diffTDotu = (diff * u).sum(axis=2)
            diffTDotuTimesu = diffTDotu.reshape((uShape[0], uShape[1], 1)) * u
            diffMinusdiffTDotuTimesu = diff - diffTDotuTimesu
            return np.sqrt((diffMinusdiffTDotuTimesu ** 2).sum(axis=2))

    def projectPoint2Plane(self, pts, P):
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

    def twoPts2Vec(self, P1, P2):
        sP1 = P1.shape
        sP2 = P2.shape
        if len(sP1) == 2 and len(sP2) == 2:
            diff = P2 - P1
            norm = np.sqrt((diff ** 2).sum())
            vec = diff / norm
            return vec
        elif len(sP1) == 2 and len(sP2) == 3:
            diff = P2 - P1.reshape((1, 3))
            norm = np.sqrt((diff ** 2).sum(axis=2))
            vec = diff / norm.reshape(sP2[0], sP2[1], 1)
            return vec

    def vec2Angs(self, vec):
        shape = vec.shape
        if len(shape) == 2:
            diff = vec
            norm = np.sqrt((diff ** 2).sum())
            theta, phi = 0, 0
            if norm == 0:
                theta = 0
                phi = 0
            elif diff[0, 0] == 0:
                theta = np.arccos(diff[2, 0] / norm)
                if diff[1, 0] == 0:
                    phi[4, 0] = 0
                elif diff[1, 0] > 0:
                    phi[4, 0] = np.pi / 2.
                else:
                    phi[4, 0] = 3 * np.pi / 2.
            else:
                theta = np.arccos(diff[2, 0] / norm)
                phi = np.arctan2(diff[1, 0], diff[0, 0])
            return theta, phi
        else:
            diff = vec
            norm = np.sqrt((diff ** 2).sum(axis=2))

            m1 = norm == 0
            m2 = diff[:, :, 0] == 0
            m3 = diff[:, :, 1] > 0

            acos = np.arccos(diff[:, :, 2] / norm)
            atan = np.arctan2(diff[:, :, 1], diff[:, :, 0])

            theta = np.zeros((shape[0], shape[1]))
            phi = np.zeros((shape[0], shape[1]))

            theta += m2 * acos
            phi += m2 * m3 * (np.pi / 2.)
            phi += m2 * (1 - m3) * (3 * np.pi / 2.)

            theta += (1 - m1) * (1 - m2) * acos
            phi += (1 - m1) * (1 - m2) * atan

            return theta, phi

    def angs2Vec(self, thetas, phis):
        h, w = thetas.shape[:2]
        res = np.zeros((h, w, 3))
        res[:, :, 0] = np.sin(thetas)
        res[:, :, 2] = np.cos(thetas)
        res[:, :, 1] = res[:, :, 0] * np.sin(phis)
        res[:, :, 0] = res[:, :, 0] * np.cos(phis)
        return res

    def vec2Ray(self, vec, P1):
        shape = vec.shape
        if len(shape) == 2:
            ray = np.zeros((5, 1))
            ray[:3, 0:1] = P1
            theta, phi = self.vec2Angs(vec)
            ray[3, 0] = theta
            ray[4, 0] = phi
            return ray
        else:
            rays = np.zeros((shape[0], shape[1], 5))
            if len(P1.shape) == 2:
                rays[:, :, :3] = P1[:, 0]
            else:
                rays[:, :, :3] = P1[:, :, :]
            theta, phi = self.vec2Angs(vec)
            rays[:, :, 3] = theta
            rays[:, :, 4] = phi
            return rays

    def ray2Vec(self, Rs):
        shape = Rs.shape
        if len(shape) == 2:
            thetas = Rs[3, 0]
            phis = Rs[4, 0]
            return self.angs2Vec(thetas, phis)
        elif len(shape) == 3:
            thetas = Rs[:, :, 3]
            phis = Rs[:, :, 4]
            return self.angs2Vec(thetas, phis)

    def twoPts2Ang(self, P1, P2):
        shape = P2.shape
        if len(shape) == 2:
            diff = P2 - P1
            return self.vec2Angs(diff)
        else:
            diff = P2 - P1.reshape((1, 3))
            return self.vec2Angs(diff)

    def twoPts2Ray(self, P1, P2):
        shape = P2.shape
        theta, phi = self.twoPts2Ang(P1, P2)
        if len(shape) == 2:
            ray = np.zeros((5, 1))
            ray[0:3, 0:1] = P1
            ray[3, 0] = theta
            ray[4, 0] = phi
            return ray
        else:
            rays = np.zeros((shape[0], shape[1], 5))
            rays[:, :, :3] = P1.reshape((1, 3))
            rays[:, :, 3] = theta
            rays[:, :, 4] = phi
            return rays

    def ptAngles2Ray(self, pt, theta, phi):
        if isinstance(theta, (int, long, float)):
            ray = np.zeros((5, 1))
            ray[:3, 0] = pt[:, 0]
            ray[3, 0] = theta
            ray[4, 0] = phi
            return ray
        else:
            shape = theta.shape
            ray = np.zeros((shape[0], shape[1], 5))
            if len(pt.shape) == 2:
                ray[:, :, :3] = pt.reshape((1, 1, 3))
            elif len(pt.shape) == 3:
                ray[:, :, :3] = pt[:, :, :3]
            ray[:, :, 3] = theta
            ray[:, :, 4] = phi
            return ray

    def minMaxAng(self, ang):
        # Assume that all angs are within a half circle
        ang = ang.reshape(-1)
        mean = np.mean(ang)
        if np.any(np.abs(ang - mean) > np.pi):
            ang = (ang + 0.5 * np.pi) % (2 * np.pi)
            minAng = np.min(ang) - 0.5 * np.pi
            maxAng = np.max(ang) - 0.5 * np.pi
        else:
            minAng = np.min(ang - mean) + mean
            maxAng = np.max(ang - mean) + mean
        return minAng, maxAng

    # Calculate semi inverse function of projection transformation
    def uv2XYZ(self, K, T_CW, uv):
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
