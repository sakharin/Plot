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
        self.N = 100
        self.pointPoint = \
            [np.random.uniform(-1, 1, 3).reshape((3, 1)) for i in range(self.N)]
        self.sizePoint = \
            [np.random.uniform(0, 10, 1)[0] for i in range(self.N)]
        self.pointLine = \
            [np.random.uniform(-5, 0, 3).reshape((3, 1)) for i in range(2 * self.N)]
        self.sizeLine = \
            [np.random.uniform(0, 10, 1)[0] for i in range(self.N)]

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
                size = self.sizePoint[i]
                self.plotLine(pt1, pt2, color=self.Cgreen, line_width=size)


if __name__ == "__main__":
    p = testPlot()
    p.show()
