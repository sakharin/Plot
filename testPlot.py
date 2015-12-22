#!/usr/bin/env python
import numpy as np
from PlotGL import PlotGL
from PlotMatPlotLib import PlotMatPlotLib
from PlotSVG import PlotSVG


class testPlot(PlotGL):
#class testPlot(PlotMatPlotLib):
#class testPlot(PlotSVG):
    def __init__(self):
        super(testPlot, self).__init__()
        self.N = 10
        self.pointPoint = \
            [np.array([np.random.uniform(-2.0, -1.0, 1),
                       np.random.uniform(-1.5, -0.5, 1),
                       np.random.uniform(-0.5, 0.5, 1)])
                for i in range(self.N)]
        self.sizePoint = \
            [np.random.uniform(1, 5, 1)[0] for i in range(self.N)]

        self.pointLine = \
            [np.array([np.random.uniform(-1.0, 0.0, 1),
                       np.random.uniform(-1.5, -0.5, 1),
                       np.random.uniform(-0.5, 0.5, 1)])
                for i in range(2 * self.N)]
        self.sizeLine = \
            [np.random.uniform(1, 5, 1)[0] for i in range(self.N)]

        self.pointArrow = \
            [np.array([np.random.uniform(0.0, 1.0, 1),
                       np.random.uniform(-1.5, -0.5, 1),
                       np.random.uniform(-0.5, 0.5, 1)])
                for i in range(2 * self.N)]
        self.sizeArrow = \
            [np.random.uniform(1, 5, 1)[0] for i in range(self.N)]

        self.pointAxis = \
            [np.array([np.random.uniform(1.0, 2.0, 1),
                       np.random.uniform(-1.5, -0.5, 1),
                       np.random.uniform(-0.5, 0.5, 1)])
                for i in range(self.N)]
        self.scaleAxis = \
            [np.random.uniform(0.01, 0.2, 1)[0] for i in range(self.N)]
        thetas = np.random.uniform(0, np.pi, 2 * self.N)
        phis = np.random.uniform(0, 2 * np.pi, 2 * self.N)
        self.rotationAxis = \
            [self.geo.getRMatrixEulerAngles(0, 0, phis[i]).dot(self.geo.getRMatrixEulerAngles(0, thetas[i], 0))
                for i in range(self.N)]

        self.pointPlane = \
            [np.array([np.random.uniform(-2.0, -1.0, 1),
                       np.random.uniform(-0.5, 0.5, 1),
                       np.random.uniform(-0.5, 0.5, 1)])
                for i in range(self.N)]
        self.wPlane = \
            [np.random.uniform(0.05, 0.2, 1)[0] for i in range(self.N)]
        self.hPlane = \
            [np.random.uniform(0.05, 0.2, 1)[0] for i in range(self.N)]
        self.rotationPlane = \
            [self.geo.getRMatrixEulerAngles(0, 0, phis[i]).dot(self.geo.getRMatrixEulerAngles(0, thetas[i], 0))
                for i in range(self.N)]

        self.pointCam = \
            [np.array([np.random.uniform(-1.0, 0.0, 1),
                       np.random.uniform(-0.5, 0.5, 1),
                       np.random.uniform(-0.5, 0.5, 1)])
                for i in range(self.N)]
        self.wCam = \
            [np.random.uniform(0.05, 0.2, 1)[0] for i in range(self.N)]
        self.hCam = \
            [np.random.uniform(0.05, 0.2, 1)[0] for i in range(self.N)]
        self.rotationCam = \
            [self.geo.getRMatrixEulerAngles(0, 0, phis[i]).dot(self.geo.getRMatrixEulerAngles(0, thetas[i], 0))
                for i in range(self.N)]

        self.pointAirplane = \
            [np.array([np.random.uniform(0.0, 1.0, 1),
                       np.random.uniform(-0.5, 0.5, 1),
                       np.random.uniform(-0.5, 0.5, 1)])
                for i in range(self.N)]
        self.rotationAirplane = \
            [self.geo.getRMatrixEulerAngles(0, 0, phis[i]).dot(self.geo.getRMatrixEulerAngles(0, thetas[i], 0))
                for i in range(self.N)]
        self.scaleAirplane = \
            [np.random.uniform(0.05, 0.2, 1)[0] for i in range(self.N)]

        self.pointCircle = \
            [np.array([np.random.uniform(1.0, 2.0, 1),
                       np.random.uniform(-0.5, 0.5, 1),
                       np.random.uniform(-0.5, 0.5, 1)])
                for i in range(self.N)]
        self.rCircle = \
            [np.random.uniform(0.01, 0.2, 1)[0] for i in range(self.N)]
        self.rotationCircle = \
            [self.geo.getRMatrixEulerAngles(0, 0, phis[i]).dot(self.geo.getRMatrixEulerAngles(0, thetas[i], 0))
                for i in range(self.N)]

        self.pointArc = \
            [np.array([np.random.uniform(-2.0, -1.0, 1),
                       np.random.uniform(0.5, 1.5, 1),
                       np.random.uniform(-0.5, 0.5, 1)])
                for i in range(self.N)]
        self.rArc = \
            [np.random.uniform(0.05, 0.2, 1)[0] for i in range(self.N)]
        self.rotationArc = \
            [self.geo.getRMatrixEulerAngles(0, 0, phis[i]).dot(self.geo.getRMatrixEulerAngles(0, thetas[i], 0))
                for i in range(self.N * 2)]

    def draw(self):
        # Test plotPoint
        if True:
            for i in range(self.N):
                pt = self.pointPoint[i]
                size = self.sizePoint[i]
                self.plotPoint(pt, color=self.Cred, point_size=size)

        # Test plotLine
        if True:
            for i in range(self.N):
                pt1 = self.pointLine[i * 2]
                pt2 = self.pointLine[i * 2 + 1]
                size = self.sizeLine[i]
                self.plotLine(pt1, pt2, color=self.Cgreen, line_width=size)

        # Test plotArrow
        if True:
            for i in range(self.N):
                pt1 = self.pointArrow[i * 2]
                pt2 = self.pointArrow[i * 2 + 1]
                size = self.sizeArrow[i]
                self.plotArrow(pt1, pt2, color=self.Cblue, line_width=size)

        # Test plotAxis
        if True:
            self.plotAxis()
            for i in range(self.N):
                pt = self.pointAxis[i]
                scale = self.scaleAxis[i]
                R = self.rotationAxis[i]
                self.plotAxis(pt, scale=scale, R=R)

        # Test plotPlane
        if True:
            for i in range(self.N):
                pt = self.pointPlane[i]
                R = self.rotationPlane[i]
                w = self.wPlane[i]
                h = self.hPlane[i]
                self.plotPlane(pt, R=R, w=w, h=h, color=self.Corange)

                vN = R.dot(self.vZ)
                self.plotPlane(pt, vN=vN, w=w, h=h, color=self.Clightgray)

        # Test plotCam
        if True:
            for i in range(self.N):
                pt = self.pointCam[i]
                R = self.rotationCam[i]
                w = self.wCam[i]
                h = self.hCam[i]
                self.plotCam(pt, R=R, camSizeW=w, camSizeH=h, camScale=2, color=self.Corange)

                pt2 = pt + 0.1 * self.vX
                vU = R.dot(-self.vY)
                vE = R.dot(self.vZ)
                self.plotCam(pt2, vU=vU, vE=vE, camSizeW=w, camSizeH=h, camScale=2, color=self.Clightgray)

        # Test plotAirplane
        if True:
            for i in range(self.N):
                pt = self.pointAirplane[i]
                R = self.rotationAirplane[i]
                scale = self.scaleAirplane[i]
                self.plotAirplane(pt, R=R, scale=scale, color=self.Corange)

                pt2 = pt + 0.1 * self.vX
                vU = R.dot(-self.vY)
                vE = R.dot(self.vZ)
                self.plotAirplane(pt2, vU=vU, vE=vE, scale=scale, color=self.Clightgray)

        # Test plotCircle
        if True:
            for i in range(self.N):
                pt = self.pointCircle[i]
                r = self.rCircle[i]
                R = self.rotationCircle[i]
                vN = R.dot(self.vZ)
                self.plotCircle(pt, r=r, R=R, color=self.Corange)

                pt2 = pt + 0.1 * self.vX
                vN = R.dot(self.vZ)
                self.plotCircle(pt2, r=r, vN=vN, color=self.Clightgray)

        # Test plotArc
        if True:
            for i in range(self.N):
                pt = self.pointArc[i]
                r = self.rArc[i]
                R1 = self.rotationArc[i * 2]
                R2 = self.rotationArc[i * 2 + 1]
                vS = R1.dot(self.vZ)
                vE = R2.dot(self.vZ)

                self.plotArrow(pt, pt + 1.5 * r * vS, color=self.Cred)
                self.plotArrow(pt, pt + 1.5 * r * vE, color=self.Cblue)
                self.plotArc(pt, r=r, vStart=vS, vEnd=vE, color=self.Clightgray)


if __name__ == "__main__":
    with testPlot() as p:
        p.show()
