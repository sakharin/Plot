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
            [np.array([np.random.uniform(1, 2, 1),
                       np.random.uniform(0, 1, 1),
                       np.random.uniform(-1, 1, 1)])
                for i in range(self.N)]
        self.sizePoint = \
            [np.random.uniform(0, 10, 1)[0] for i in range(self.N)]

        self.pointLine = \
            [np.array([np.random.uniform(0, 1, 1),
                       np.random.uniform(0, 1, 1),
                       np.random.uniform(-1, 1, 1)])
                for i in range(2 * self.N)]
        self.sizeLine = \
            [np.random.uniform(0, 5, 1)[0] for i in range(self.N)]

        self.pointArrow = \
            [np.array([np.random.uniform(-1, 0, 1),
                       np.random.uniform(0, 1, 1),
                       np.random.uniform(-1, 1, 1)])
                for i in range(2 * self.N)]
        self.sizeArrow = \
            [np.random.uniform(0, 5, 1)[0] for i in range(self.N)]

        self.pointAxis = \
            [np.array([np.random.uniform(1, 2, 1),
                       np.random.uniform(-1, 0, 1),
                       np.random.uniform(-0.5, 0.5, 1)])
                for i in range(self.N)]
        self.scaleAxis = \
            [np.random.uniform(0, 0.3, 1)[0] for i in range(self.N)]

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
                self.plotArrow(pt1, pt2, color=self.Cgreen, line_width=size)

        # Test plotAxis
        if True:
            for i in range(self.N):
                pt = self.pointAxis[i]
                scale = self.scaleAxis[i]
                self.plotAxis(pt, scale=scale)


if __name__ == "__main__":
    with testPlot() as p:
        p.show()
