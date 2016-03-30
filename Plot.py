#!/usr/bin/env python
import numpy as np
import random
import string

from Color import ColorSet01
from Geometry import Geometry


class Plot(object):
    def __init__(self):
        self.geo = Geometry()
        self.colors = ColorSet01().colors
        self.setColors()

        self.pO = np.array([[0], [0], [0]])
        self.vX = np.array([[1], [0], [0]])
        self.vY = np.array([[0], [1], [0]])
        self.vZ = np.array([[0], [0], [1]])

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def setColors(self):
        for c in self.colors:
            setattr(self, 'C' + c['name'],
                    self.cvtColor(c['R'], c['G'], c['B']))

    def cvtColor(self, r, g, b, base=255):
        raise NotImplemented("You should implemented this.")

    def setDefaultParamsPoint(self, **params):
        if params.get('color') is None:
            params.update({'color': self.Cblack})

        if params.get('point_size') is None:
            params.update({'point_size': 3})

        return params

    def setDefaultParamsLine(self, **params):
        if params.get('color') is None:
            params.update({'color': self.Cblack})

        if params.get('line_width') is None:
            params.update({'line_width': 1})

        return params

    def setDefaultParamsArrow(self, **params):
        if params.get('color') is None:
            params.update({'color': self.Cblack})

        if params.get('line_width') is None:
            params.update({'line_width': 1})

        if params.get('head_size') is None:
            params.update({'head_size': 0.05})

        return params

    def setDefaultParamsText(self, **params):
        if params.get('color') is None:
            params.update({'color': self.Cwhite})

        if params.get('bgColor') is None:
            params.update({'bGColor': self.Cblack})

        if params.get('id') is None:
            size = 6
            chars = string.ascii_uppercase + string.digits
            id = ''.join(random.choice(chars) for _ in range(size))
            params.update({'id': id})

    def plotPoint(self, pt1, **params):
        pass

    def plotLine(self, pt1, pt2, **params):
        pass

    def plotArrow(self, pt1, pt2, **params):
        pass

    def genQuaternion(self, quat, **params):
        quatNorm = quat / np.linalg.norm(quat)
        ang, vec = self.geo.quat2AngleVector(quat)
        othoVec = self.geo.getOrthogonalVecs(vec)[0]
        othoVecQ = np.array([0, othoVec[0, 0], othoVec[1, 0], othoVec[2, 0]])
        quatConj = self.geo.quatConj(quat)
        res = self.geo.quatMul(self.geo.quatMul(quatNorm, othoVecQ), quatConj)
        return vec, othoVec, res[1:].reshape((3, 1))

    def plotQuaternion(self, pt1, quat, **params):
        vec, othoVec, othoVec2 = self.genQuaternion(quat, **params)
        self.plotArrow(pt1, pt1 + vec, **params)
        params.update({'color': self.Cgreen})
        self.plotArc(pt1, 0.25, othoVec, othoVec2, **params)

    def genAxis(self, pt1, R, scale, **params):
        if pt1 is None:
            pt1 = self.pO

        vX_ = self.vX
        vY_ = self.vY
        vZ_ = self.vZ
        if R is not None:
            if R.shape[1] == 4:
                self.pt1_ = R[:3, 3:4]
            vX_ = R[:3, :3].dot(self.vX)
            vY_ = R[:3, :3].dot(self.vY)
            vZ_ = R[:3, :3].dot(self.vZ)
        return (pt1, vX_, vY_, vZ_)

    def plotAxis(self, pt1=None, R=None, scale=1, **params):
        pt1, vX_, vY_, vZ_ = \
            self.genAxis(pt1, R, scale, **params)
        params.update({'color': self.Cred})
        self.plotArrow(pt1, pt1 + scale * vX_, **params)
        params.update({'color': self.Cgreen})
        self.plotArrow(pt1, pt1 + scale * vY_, **params)
        params.update({'color': self.Cblue})
        self.plotArrow(pt1, pt1 + scale * vZ_, **params)

    def genPlane(self, pt1, R, vN, w, h, **params):
        if pt1 is None:
            pt1 = self.pO

        vA = self.vX
        vB = self.vY
        if R is not None:
            R = self.geo.checkRMatrix(R)
            vA = R.dot(self.vX)
            vB = R.dot(self.vY)
            vN = R.dot(self.vZ)
        elif vN is not None:
            # Find an orthogonal vector
            vA, vB = self.geo.getOrthogonalVecs(vN)
        else:
            vN = self.vZ

        w2 = w / 2.
        h2 = h / 2.
        p1 = pt1 - w2 * vA - h2 * vB
        p2 = pt1 + w2 * vA - h2 * vB
        p3 = pt1 + w2 * vA + h2 * vB
        p4 = pt1 - w2 * vA + h2 * vB
        return (p1, p2, p3, p4)

    def plotPlane(self, pt1=None,
                  R=None, vN=None,
                  w=1, h=1, **params):
        p1, p2, p3, p4 = \
            self.genPlane(pt1, R, vN, w, h, **params)
        self.plotLine(p1, p2, **params)
        self.plotLine(p2, p3, **params)
        self.plotLine(p3, p4, **params)
        self.plotLine(p4, p1, **params)

    def genCam(self, pt1, R, vU, vE,
               camSizeH, camSizeW, camF, camScale,
               **params):
        if pt1 is None:
            pt1 = self.pO
        vA = self.vX
        vB = self.vY
        vC = self.vZ
        if R is not None:
            R = self.geo.checkRMatrix(R)
            vA = R.dot(self.vX)
            vB = R.dot(self.vY)
            vC = R.dot(self.vZ)
        if vU is not None and vE is not None:
            vC = vE
            vA = np.cross(vC.reshape(-1), vU.reshape(-1)).reshape((3, 1))
            vB = np.cross(vC.reshape(-1), vA.reshape(-1)).reshape((3, 1))
        camF = camF * camScale
        w2 = camSizeW / 2. * camScale
        h2 = camSizeH / 2. * camScale
        p0 = pt1
        p1 = pt1 - w2 * vA - h2 * vB + camF * vC
        p2 = pt1 + w2 * vA - h2 * vB + camF * vC
        p3 = pt1 + w2 * vA + h2 * vB + camF * vC
        p4 = pt1 - w2 * vA + h2 * vB + camF * vC
        return (p0, p1, p2, p3, p4)

    def plotCam(self, pt1=None,
                R=None, vU=None, vE=None,
                camSizeH=0.024,
                camSizeW=0.036,
                camF=0.035,
                camScale=10,
                **params):
        p0, p1, p2, p3, p4 = \
            self.genCam(pt1, R, vU, vE, camSizeH, camSizeW, camF, camScale, **params)
        self.plotLine(p0, p1, **params)
        self.plotLine(p0, p2, **params)
        self.plotLine(p0, p3, **params)
        self.plotLine(p0, p4, **params)

        self.plotLine(p1, p2, **params)
        self.plotLine(p2, p3, **params)
        self.plotLine(p3, p4, **params)
        self.plotLine(p4, p1, **params)

        params.update({'color': self.Cred})
        self.plotPoint(p1, **params)
        params.update({'color': self.Cgreen})
        self.plotPoint(p2, **params)

    def genAirplane(self, pt1, R, vU, vE, scale, **params):
        if pt1 is None:
            pt1 = self.pO
        vA = self.vX
        vB = self.vY
        vC = self.vZ
        if R is not None:
            R = self.geo.checkRMatrix(R)
            vA = R.dot(self.vX)
            vB = R.dot(self.vY)
            vC = R.dot(self.vZ)
        if vU is not None and vE is not None:
            vC = vE
            vA = np.cross(vC.reshape(-1), vU.reshape(-1)).reshape((3, 1))
            vB = np.cross(vC.reshape(-1), vA.reshape(-1)).reshape((3, 1))

        vA = vA * scale
        vB = vB * scale
        vC = vC * scale

        p0 = pt1 + 0.0 * vA + 0.0 * vB + 1.0 * vC
        p1 = pt1 - 1.0 * vA + 0.0 * vB + 0.0 * vC
        p2 = pt1 + 0.0 * vA + 0.0 * vB + 0.0 * vC
        p3 = pt1 + 1.0 * vA + 0.0 * vB + 0.0 * vC
        p4 = pt1 + 0.0 * vA + 0.0 * vB - 0.5 * vC
        p5 = pt1 - 0.5 * vA + 0.0 * vB - 1.0 * vC
        p6 = pt1 + 0.0 * vA + 0.0 * vB - 1.0 * vC
        p7 = pt1 + 0.5 * vA + 0.0 * vB - 1.0 * vC
        p8 = pt1 + 0.0 * vA - 0.5 * vB - 1.0 * vC
        return (p0, p1, p2, p3, p4, p5, p6, p7, p8)

    def plotAirplane(self, pt1=None,
                     R=None, vU=None, vE=None,
                     scale=1., **params):
        p0, p1, p2, p3, p4, p5, p6, p7, p8 = \
            self.genAirplane(pt1, R, vU, vE, scale, **params)
        self.plotLine(p0, p1, **params)
        self.plotLine(p0, p2, **params)
        self.plotLine(p0, p3, **params)
        self.plotLine(p1, p2, **params)
        self.plotLine(p2, p3, **params)

        self.plotLine(p2, p4, **params)

        self.plotLine(p4, p5, **params)
        self.plotLine(p4, p6, **params)
        self.plotLine(p4, p7, **params)
        self.plotLine(p5, p6, **params)
        self.plotLine(p6, p7, **params)

        self.plotLine(p4, p8, **params)
        self.plotLine(p6, p8, **params)

        params.update({'color': self.Cred})
        self.plotPoint(p1, **params)
        params.update({'color': self.Cgreen})
        self.plotPoint(p3, **params)
        params.update({'color': self.Cwhite})
        self.plotPoint(p8, **params)

    def genCircle(self, pt1, r, R, vN, numSegments):
        if pt1 is None:
            pt1 = np.copy(self.pO)
        if vN is None:
            vN = np.copy(self.vZ)
        m = None
        if R is not None:
            m = self.geo.checkRMatrix(R)
        else:
            phi, theta = self.geo.vec2Angs(vN)
            m = self.geo.getRMatrixEulerAngles(0, 0, phi)
            m = m.dot(self.geo.getRMatrixEulerAngles(0, theta, 0))
        pts = np.zeros((3, numSegments))
        for i in range(numSegments):
            ang1 = i * 2 * np.pi / numSegments
            p1 = pt1 + m.dot(r * np.array([[np.cos(ang1)], [np.sin(ang1)], [0]]))
            pts[:, i:i + 1] = p1[:, :]
        return pts

    def plotCircle(self, pt1=None, r=1, R=None, vN=None, numSegments=64, isDash=False, **params):
        pts = self.genCircle(pt1=pt1, r=r, R=R, vN=vN, numSegments=numSegments)
        for i in range(numSegments):
            p1 = pts[:, i:i + 1]
            if i == numSegments - 1:
                p2 = pts[:, 0:1]
            else:
                p2 = pts[:, i + 1:i + 2]
            self.plotLine(p1, p2, **params)

    def genArc(self, pt1, r, vStart, vEnd, numSegments):
        if pt1 is None:
            pt1 = np.copy(self.pO)
        if vStart is None:
            vStart = np.copy(self.vZ)
        if vEnd is None:
            vEnd = np.copy(self.vX)

        vN = np.cross(vStart[:, 0], vEnd[:, 0]).reshape((3, 1))
        vN = vN / self.geo.normVec(vN)

        phi, theta = self.geo.vec2Angs(vN)
        m = self.geo.getRMatrixEulerAngles(0, 0, phi)
        m = m.dot(self.geo.getRMatrixEulerAngles(0, theta, 0))

        mInv = np.linalg.inv(m)
        phi1, _ = self.geo.vec2Angs(mInv.dot(vStart))
        phi2, _ = self.geo.vec2Angs(mInv.dot(vEnd))
        angDiff = self.geo.angleDiff(phi2, phi1)

        pts = np.zeros((3, numSegments + 1))
        for i in range(numSegments + 1):
            ang1 = i * angDiff / numSegments + phi1
            p1 = pt1 + m.dot(r * np.array([[np.cos(ang1)], [np.sin(ang1)], [0]]))
            pts[:, i:i + 1] = p1[:, :]
        return pts

    def plotArc(self, pt1=None, r=1, vStart=None, vEnd=None, numSegments=64, **params):
        pts = self.genArc(pt1=pt1, r=r, vStart=vStart, vEnd=vEnd, numSegments=numSegments)

        for i in range(numSegments):
            p1 = pts[:, i:i + 1]
            p2 = pts[:, i + 1:i + 2]
            self.plotLine(p1, p2, **params)

    def plotSphere1(self, pt1=None, r=1, numSegments=(16, 16), **params):
        if pt1 is None:
            pt1 = np.copy(self.pO)
        vX = r * np.copy(self.vX)
        for i in range(numSegments[0]):
            y = 2. * r * (i + 1) / (numSegments[0] + 1) - r
            pos = np.array([[0],
                            [y],
                            [0]])
            rLocal = np.sqrt(r ** 2 - y ** 2)
            self.plotCircle(pt1 + pos, rLocal, vN=self.vY, **params)

        for i in range(numSegments[1]):
            ang = np.pi * i / numSegments[1]
            m = self.geo.getRMatrixEulerAngles(0, ang, 0)
            vN = m.dot(vX)
            self.plotCircle(pt1, r, vN=vN, **params)

    def genSphere(self, pt1, r, subDivition):
        if pt1 is None:
            pt1 = np.copy(self.pO)

        pTop = -self.vY
        pBottom = self.vY
        pFront = self.vZ
        m = self.geo.getRMatrixEulerAngles(0, np.deg2rad(120), 0)
        pRight = m.dot(self.vZ)
        m = self.geo.getRMatrixEulerAngles(0, np.deg2rad(-120), 0)
        pLeft = m.dot(self.vZ)

        ps = list()
        ps.append(pTop)
        ps.append(pFront)
        ps.append(pRight)
        ps.append(pLeft)
        ps.append(pBottom)

        es = list()
        es.append((0, 1))
        es.append((1, 2))
        es.append((2, 0))

        es.append((0, 2))
        es.append((2, 3))
        es.append((3, 0))

        es.append((0, 3))
        es.append((3, 1))
        es.append((1, 0))

        es.append((1, 4))
        es.append((4, 2))
        es.append((2, 1))

        es.append((2, 4))
        es.append((4, 3))
        es.append((3, 2))

        es.append((3, 4))
        es.append((4, 1))
        es.append((1, 3))

        # Sub triangle
        for j in range(subDivition):
            N = len(es)
            for i in range(N / 3):
                M = len(ps)

                # Delete 3 edges
                e1 = es.pop(0)
                e2 = es.pop(0)
                e3 = es.pop(0)

                # Get 3 points
                p1 = ps[e1[0]]
                p2 = ps[e2[0]]
                p3 = ps[e3[0]]

                # Get 3 new points
                p12 = self.geo.normalizedVec(p1 + p2)
                p23 = self.geo.normalizedVec(p2 + p3)
                p31 = self.geo.normalizedVec(p3 + p1)
                ps.append(p12)
                ps.append(p23)
                ps.append(p31)

                # Add 12 edges
                es.append((e1[0], M + 0))
                es.append((M + 0, M + 2))
                es.append((M + 2, e1[0]))

                es.append((M + 0, M + 1))
                es.append((M + 1, M + 2))
                es.append((M + 2, M + 0))

                es.append((M + 0, e2[0]))
                es.append((e2[0], M + 1))
                es.append((M + 1, M + 0))

                es.append((M + 2, M + 1))
                es.append((M + 1, e3[0]))
                es.append((e3[0], M + 2))

        return [pt1 + r * p for p in ps], es

    def plotSphere(self, pt1=None, r=1, subDivition=3, **params):
        ps, es = self.genSphere(pt1, r, subDivition)

        for e in es:
            self.plotLine(ps[e[0]], ps[e[1]], **params)

    def draw(self):
        pass

    def show(self):
        self.draw()
