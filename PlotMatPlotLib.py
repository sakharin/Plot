#!/usr/bin/env python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from Plot import Plot


class Arrow3D(FancyArrowPatch):
    #http://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector
    def __init__(self, pt1, pt2, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        x = [pt1[0, 0], pt2[0, 0]]
        y = [pt1[1, 0], pt2[1, 0]]
        z = [pt1[2, 0], pt2[2, 0]]
        self._verts3d = x, y, z

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


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
        self.elev = -90.
        self.azim = -90.

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

    def setDefaultParamsLine(self, **params):
        params = super(PlotMatPlotLib, self).setDefaultParamsLine(**params)

        c1 = params.get('linestyle') is None
        c2 = params.get('line_width') is None
        if c1 and not c2:
            params.update({'linestyle': '-'})
        del params['line_width']

        return params

    def setDefaultParamsArrow(self, **params):
        params = super(PlotMatPlotLib, self).setDefaultParamsArrow(**params)
        return self.setDefaultParamsLine(**params)

    def plotPoint(self, pt1, **params):
        super(PlotMatPlotLib, self).plotPoint(pt1, **params)
        params = self.setDefaultParamsPoint(**params)
        self.updateRegion(pt1)
        self.ax.plot([pt1[0, 0]],
                     [pt1[1, 0]],
                     [pt1[2, 0]], **params)

    def plotLine(self, pt1, pt2, **params):
        params = self.setDefaultParamsLine(**params)
        self.updateRegion(pt1)
        self.updateRegion(pt2)
        self.ax.plot([pt1[0, 0], pt2[0, 0]],
                     [pt1[1, 0], pt2[1, 0]],
                     [pt1[2, 0], pt2[2, 0]],
                     **params)

    def plotArrow(self, pt1, pt2, **params):
        params = self.setDefaultParamsArrow(**params)
        a = Arrow3D(pt1, pt2, mutation_scale=20, lw=1, arrowstyle="-|>", color=params.get('color'))
        self.ax.add_artist(a)
        self.updateRegion(pt1)
        self.updateRegion(pt2)

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
