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

    def plotAxis(self, pt1=None, scale=1, R=None, **params):
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
        params.update({'color': self.Cred})
        self.plotArrow(pt1, pt1 + scale * vX_, **params)
        params.update({'color': self.Cgreen})
        self.plotArrow(pt1, pt1 + scale * vY_, **params)
        params.update({'color': self.Cblue})
        self.plotArrow(pt1, pt1 + scale * vZ_, **params)

    def plotPlane(self, pt1=None,
                  R=None, vN=None,
                  w=1, h=1, **params):
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
        self.plotLine(p1, p2, **params)
        self.plotLine(p2, p3, **params)
        self.plotLine(p3, p4, **params)
        self.plotLine(p4, p1, **params)

    def draw(self):
        pass

    def show(self):
        self.draw()
