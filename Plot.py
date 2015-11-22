#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from Geometry import Geometry
from Color import ColorSet01


class Plot(object):
    def __init__(self):
        self.geo = Geometry()
        self.colors = ColorSet01().colors
        self.setColors()

    def cvtColor(self, r, g, b, base=255):
        raise NotImplementedError("You must implement this!")
        return repr(r) + " " + repr(g) + " " + repr(b)

    def setColors(self):
        for c in self.colors:
            setattr(self, 'C' + c['name'],
                    self.cvtColor(c['R'], c['G'], c['B'], 255))

    def setDefaultParamsPoint(self, **params):
        if params.get('color') is None:
            params.update({'color': self.Cblack})

        if params.get('point_size') is None:
            params.update({'point_size': 5})
        return params

    def plotPoint(self, pt1, **params):
        pass

    def draw(self):
        raise NotImplementedError("You must implement this!")

    def show(self):
        self.draw()
