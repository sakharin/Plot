#!/usr/bin/env python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from Plot import Plot


class PlotMatPlotLib(Plot):
    def __init__(self):
        super(PlotMatPlotLib, self).__init__()
        self.fig = plt.figure()
        plt.ioff()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # For plotting scale
        self.xmx, self.xmn = -1e6, 1e6
        self.ymx, self.ymn = -1e6, 1e6
        self.zmx, self.zmn = -1e6, 1e6

        # For viewing direction
        self.elev = 0.
        self.azim = 0.

    def updateRegion(self, pt):
        x, y, z = pt[:3, 0]
        self.xmn = min(self.xmn, x)
        self.xmx = max(self.xmx, x)

        self.ymn = min(self.ymn, y)
        self.ymx = max(self.ymx, y)

        self.zmn = min(self.zmn, z)
        self.zmx = max(self.zmx, z)

    def cvtColor(self, r, g, b, base=255):
        return (r * 1. / base, g * 1. / base, b * 1. / base)

    def setDefaultParamsPoint(self, **params):
        c1 = params.get('marker') is None
        c2 = params.get('point_size') is None
        if c1 and not c2:
            if params.get('point_size') < 5:
                params.update({'marker': '.'})
            else:
                params.update({'marker': 'o'})
        elif c1 and c2:
            params.update({'marker': '.'})

        params = super(PlotMatPlotLib, self).setDefaultParamsPoint(**params)
        del params['point_size']

        return params

    def plotPoint(self, pt1, **params):
        super(PlotMatPlotLib, self).plotPoint(pt1, **params)
        params = self.setDefaultParamsPoint(**params)
        self.updateRegion(pt1)
        self.ax.plot([pt1[0, 0]],
                     [pt1[1, 0]],
                     [pt1[2, 0]], **params)

    def show(self):
        self.draw()
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.view_init(elev=self.elev, azim=self.azim)

        # calculate scale
        rangeX = self.xmx - self.xmn
        rangeY = self.ymx - self.ymn
        rangeZ = self.zmx - self.zmn

        maxRange = max(rangeX, rangeY, rangeZ)
        md = maxRange / 2.
        cx = (self.xmx + self.xmn) / 2.
        cy = (self.ymx + self.ymn) / 2.
        cz = (self.zmx + self.zmn) / 2.

        self.ax.auto_scale_xyz([cx - md, cx + md],
                               [cy - md, cy + md],
                               [cz - md, cz + md])
        plt.show()
