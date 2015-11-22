#!/usr/bin/env python
from Color import ColorSet01
from Geometry import Geometry


class Plot(object):
    def __init__(self):
        self.geo = Geometry()
        self.colors = ColorSet01().colors
        self.setColors()

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

    def plotPoint(self, pt1, **params):
        pass

    def plotLine(self, pt1, pt2, **params):
        pass

    def draw(self):
        pass

    def show(self):
        self.draw()
