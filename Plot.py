#!/usr/bin/env python
import numpy as np

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

    def plotPoint(self, pt1, **params):
        pass

    def plotLine(self, pt1, pt2, **params):
        pass

    def plotArrow(self, pt1, pt2, **params):
        pass

    def genAxis(self, pt1, scale, R, **params):
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

    def plotAxis(self, pt1=None, scale=1, R=None, **params):
        pt1, vX_, vY_, vZ_ = \
            self.genAxis(pt1, scale, R, **params)
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
            theta, phi = self.geo.vec2Angs(vN)
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

        theta, phi = self.geo.vec2Angs(vN)
        m = self.geo.getRMatrixEulerAngles(0, 0, phi)
        m = m.dot(self.geo.getRMatrixEulerAngles(0, theta, 0))

        mInv = np.linalg.inv(m)
        _, phi1 = self.geo.vec2Angs(mInv.dot(vStart))
        _, phi2 = self.geo.vec2Angs(mInv.dot(vEnd))
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

    def draw(self):
        pass

    def show(self):
        self.draw()
