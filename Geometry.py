#!/usr/bin/env python
import cv2
import inspect
import numpy as np

PI = np.pi
TWOPI = 2 * PI


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

    def checkRMatrix(self, R):
        assert(isinstance(R, np.ndarray))
        assert(R.shape == (3, 3))

        # https://www.fastgraph.com/makegames/3drotation/
        # 1.R is normalized: the squares of the elements in any row or column
        # sum to 1.
        for i in range(3):
            assert(np.allclose(np.linalg.norm(R[i, :], 2), 1.))
            assert(np.allclose(np.linalg.norm(R[:, i], 2), 1.))
        # R is orthogonal: the dot product of any pair of rows or any pair of
        # columns is 0.
        assert(np.allclose(R[0, :].dot(R[1, :]), 0))
        assert(np.allclose(R[0, :].dot(R[2, :]), 0))
        assert(np.allclose(R[1, :].dot(R[2, :]), 0))
        assert(np.allclose(R[:, 0].dot(R[:, 1]), 0))
        assert(np.allclose(R[:, 0].dot(R[:, 2]), 0))
        assert(np.allclose(R[:, 1].dot(R[:, 2]), 0))
        return R

    def checkTMatrix(self, T):
        self.checkRMatrix(T[:3, :3])
        assert(T.shape[0] >= 3 and T.shape[0] <= 4)
        if T.shape[1] == 4:
            return T[:3, :4]
        if T.shape[0] == 3:
            S = np.zeros((3, 4))
            S[:3, :3] = T[:, :]
            return S

    def angleDiff(self, angle1, angle2):
        # http://stackoverflow.com/questions/12234574/calculating-if-an-angle-is-between-two-angles
        # Return diff in range [-pi, pi]
        return (angle1 - angle2 + PI) % TWOPI - PI

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

    def distPts(self, P0, P1):
        shapeP0 = P0.shape
        if len(shapeP0) == 2:
            shapeP1 = P1.shape
            if len(shapeP1) == 2:
                return np.sqrt(((P0 - P1) ** 2).sum())
            elif len(shapeP1) == 3:
                return np.sqrt(((P0.reshape((1, 1, 3)) - P1) ** 2).sum(axis=2))

    def distPt2Line(self, P, P0, u):
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

    def projectPt2Plane(self, p0, V0, n):
        # http://stackoverflow.com/questions/8942950/how-do-i-find-the-orthogonal-projection-of-a-point-onto-a-plane
        # point p0 = (x, y, z)T
        # plane point V0 = (a, b, c)T
        # plane normal n = (d, e, f)T
        return p0 - (p0 - V0).reshape(-1).dot(n.reshape(-1)) * n

    def calLnLnIntersection(self, p1, v1, p2, v2):
        a = np.linalg.norm(np.cross((p2 - p1).T, v2.T)) / \
            np.linalg.norm(np.cross(v1.T, v2.T))
        return p1 + a * v1

    def calPlLnIntersection(self, p0, n, l0, l):
        # http://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
        lDotn = l.T.dot(n)
        if lDotn == 0:
            # parallel case
            if(p0 - l0).T.dot(n) == 0:
                # line is on the plane
                return False
            else:
                # no intersection
                return False
        d = (p0 - l0).T.dot(n) / lDotn
        return d * l + l0

    def projectVecs2Depth(self, T, vA, vB):
        shape = vB.shape
        if len(shape) == 2:
            vC = vA + vB
            A = np.linalg.norm(vA)
            B = np.linalg.norm(vB)
            C = np.linalg.norm(vC)

            vADotvB = (vA * vB).sum()
            vBDotvC = (vB * -vC).sum()
            vADotvC = (-vC * vA).sum()
            alpha = np.arccos(vADotvB / (A * B))
            beta = np.arccos(vBDotvC / (B * C))
            gamma = np.arccos(vADotvC / (A * C))

            if alpha == PI:
                return vA / A * T
            if alpha == 0:
                return vA / A * -T
            if alpha + beta + gamma != PI:
                alpha = PI - alpha

            beta = np.arcsin(A * np.sin(alpha) / T)
            gamma = PI - alpha - beta
            B_new = np.sin(gamma) * T / np.sin(alpha)
            vB = vB / B * B_new
            vC = vA + vB
            return vC
        if len(shape) == 3:
            h, w, d = shape

            vA = vA.reshape((1, 1, 3))
            vC = vA + vB
            A = self.normVec(vA)
            B = self.normVec(vB)
            C = self.normVec(vC)

            vADotvB = (vA * vB).sum(axis=2)
            vBDotvC = (vB * -vC).sum(axis=2)
            vADotvC = (-vC * vA).sum(axis=2)
            alpha = np.arccos(vADotvB / (A * B))
            beta = np.arccos(vBDotvC / (B * C))
            gamma = np.arccos(vADotvC / (A * C))

            mask1 = alpha == 0
            mask2 = alpha + beta + gamma != PI
            alpha = alpha * np.bitwise_not(mask2) + \
                PI - alpha * mask2
            # Avoid division by zero
            alpha += 1 * mask1
            beta = np.arcsin(A * np.sin(alpha) / T)
            gamma = PI - alpha - beta
            B_new = np.sin(gamma) * T / np.sin(alpha)
            vB = vB * (B_new / B).reshape((h, w, 1))
            vC = vA + vB
            vC = vC * np.bitwise_not(mask1).reshape((h, w, 1)) + \
                (vA / A * T * mask1.reshape((h, w, 1)))
            return vC

    def getPerpendicularVector2D(self, v):
        if v[0, 0] == 0 and v[1, 0] == 0:
            if v[2, 0] == 0:
                # v is Vector(0, 0, 0)
                raise ValueError('Zero Vector')

            # v is Vector(0, 0, v.z)
            return np.array([[0], [1], [0]])
        return np.array([[-v[1, 0]], [v[0, 0]], [0]])

    def normVec(self, vec):
        shape = vec.shape
        if len(shape) == 2:
            return np.sqrt((vec ** 2).sum())
        elif len(shape) == 3:
            return np.sqrt((vec ** 2).sum(axis=2))

    def normalizedVec(self, vec):
        return vec / self.normVec(vec)

    def getOrthogonalVecs(self, vec):
        vec_ = vec / self.normVec(vec)
        vA = None
        u1, v1, w1 = vec_[:, 0]
        if u1 != 0:
            v2 = 1
            w2 = 0
            u2 = (-v1 * v2 - w1 * w2) / u1
            vA = np.array([[u2], [v2], [w2]])
        elif v1 != 0:
            u2 = 1
            w2 = 0
            v2 = (-u1 * u2 - w1 * w2) / v1
            vA = np.array([[u2], [v2], [w2]])
        elif w1 != 0:
            u1 = 1
            v1 = 0
            w2 = (-u1 * u2 - v1 * v2) / w1
            vA = np.array([[u2], [v2], [w2]])

        vA = self.normalizedVec(vA)
        vB = np.cross(vec.reshape(-1), vA.reshape(-1)).reshape((3, 1))
        vB = self.normalizedVec(vB)
        return vA, vB

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
                    phi = 0
                elif diff[1, 0] > 0:
                    phi = PI / 2.
                else:
                    phi = 3 * PI / 2.
            else:
                theta = np.arccos(diff[2, 0] / norm)
                phi = np.arctan2(diff[1, 0], diff[0, 0])
            phi %= TWOPI
            return phi, theta
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
            phi += m2 * m3 * (PI / 2.)
            phi += m2 * (1 - m3) * (3 * PI / 2.)

            theta += (1 - m1) * (1 - m2) * acos
            phi += (1 - m1) * (1 - m2) * atan
            phi %= TWOPI

            return phi, theta

    def angs2Vec(self, phi, theta):
        if isinstance(theta, (int, long, float)):
            res = np.zeros((3, 1))
            res[0] = np.sin(theta)
            res[2] = np.cos(theta)
            res[1] = res[0] * np.sin(phi)
            res[0] = res[0] * np.cos(phi)
            return res
        else:
            h, w = theta.shape[:2]
            res = np.zeros((h, w, 3))
            res[:, :, 0] = np.sin(theta)
            res[:, :, 2] = np.cos(theta)
            res[:, :, 1] = res[:, :, 0] * np.sin(phi)
            res[:, :, 0] = res[:, :, 0] * np.cos(phi)
            return res

    def vec2Ray(self, vec, P1):
        shape = vec.shape
        if len(shape) == 2:
            ray = np.zeros((5, 1))
            ray[:3, 0:1] = P1
            phi, theta = self.vec2Angs(vec)
            ray[3, 0] = theta
            ray[4, 0] = phi
            return ray
        else:
            rays = np.zeros((shape[0], shape[1], 5))
            if len(P1.shape) == 2:
                rays[:, :, :3] = P1[:, 0]
            else:
                rays[:, :, :3] = P1[:, :, :]
            phi, theta = self.vec2Angs(vec)
            rays[:, :, 3] = theta
            rays[:, :, 4] = phi
            return rays

    def ray2Vec(self, Rs):
        shape = Rs.shape
        if len(shape) == 2:
            thetas = Rs[3, 0]
            phis = Rs[4, 0]
            return self.angs2Vec(phis, thetas)
        elif len(shape) == 3:
            thetas = Rs[:, :, 3]
            phis = Rs[:, :, 4]
            return self.angs2Vec(phis, thetas)

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
        ang = ang.reshape(-1) % (2 * PI)
        x = np.mean(np.cos(ang))
        y = np.mean(np.sin(ang))
        ang0 = np.arctan2(y, x)
        ang = ((ang - ang0) % TWOPI + PI) % TWOPI
        minAng = (ang.min() + ang0 - PI) % TWOPI
        maxAng = (ang.max() + ang0 - PI) % TWOPI
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

    def _getCallerClass(self):
        stack = inspect.stack()
        return stack[2][0].f_locals["self"]

    def u2Phi(self, u, W=None):
        if W is None:
            # get W of the caller class
            W = self._getCallerClass().W
        return u * -2 * np.pi / W + 2 * np.pi

    def v2Theta(self, v, H=None):
        if H is None:
            H = self._getCallerClass().H
        return v * np.pi / H

    def phi2u(self, phi, W=None):
        if W is None:
            W = self._getCallerClass().W
        return ((phi - 2 * np.pi) * W / (-2 * np.pi)) % W

    def theta2v(self, theta, H=None):
        if H is None:
            H = self._getCallerClass().H
        return theta * H / np.pi

    def normalizedQuat(self, quat):
        return quat / np.linalg.norm(quat)

    def quat2AngleVector(self, quat):
        quatNorm = self.normalizedQuat(quat)

        qw = quatNorm[0]
        qx = quatNorm[1]
        qy = quatNorm[2]
        qz = quatNorm[3]

        ang = 2 * np.arccos(qw)
        s = np.sqrt(1 - qw * qw)
        # avoid division by zero
        if (s < 0.001):
            x = 1
            y = 0
            z = 0
        else:
            x = qx / s
            y = qy / s
            z = qz / s
        vec = np.array([[x], [y], [z]])
        return ang, vec

    def quatConj(self, quat):
        quatNorm = self.normalizedQuat(quat)
        return np.array([quatNorm[0],
                         -quatNorm[1],
                         -quatNorm[2],
                         -quatNorm[3]])

    def quatMul(self, quat1, quat2):
        w1, x1, y1, z1 = quat1
        w2, x2, y2, z2 = quat2
        w3, x3, y3, z3 = 0, 0, 0, 0

        w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z3 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w3, x3, y3, z3])

    def quat2RMatrix(self, quat):
        # world -> object
        qX = np.array([0, 1, 0, 0])
        qY = np.array([0, 0, 1, 0])
        qZ = np.array([0, 0, 0, 1])

        qx = self.quatMul(self.quatMul(quat, qX), self.quatConj(quat))
        qy = self.quatMul(self.quatMul(quat, qY), self.quatConj(quat))
        qz = self.quatMul(self.quatMul(quat, qZ), self.quatConj(quat))

        R = np.eye(3)
        R[:, 0] = qx[1:]
        R[:, 1] = qy[1:]
        R[:, 2] = qz[1:]

        return R


if __name__ == "__main__":
    geo = Geometry()
