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
        thetas = np.random.uniform(0, np.pi, self.N)
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
                self.plotPlane(pt, vN=vN, w=w, h=h, color=self.Cwhite)

        # Test plotCam
        if True:
            for i in range(self.N):
                pt = self.pointCam[i]
                R = self.rotationCam[i]
                w = self.wCam[i]
                h = self.hCam
                self.plotCam(pt, R=R, w=w, h=h, camScale=2, color=self.Corange)

                pt2 = pt + 0.1 * self.vX
                vU = R.dot(-self.vY)
                vE = R.dot(self.vZ)
                self.plotCam(pt2, vU=vU, vE=vE, w=w, h=h, camScale=2, color=self.Cwhite)

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
                self.plotAirplane(pt2, vU=vU, vE=vE, scale=scale, color=self.Cwhite)

if __name__ == "__main__":
    with testPlot() as p:
        p.show()
