#!/usr/bin/env python
import numpy as np
import subprocess
import os
import svgwrite

from Geometry import Geometry
from Plot import Plot


class PlotSVG(Plot):
    def __init__(self, size=(600, 800), outFileName="test.svg"):
        super(PlotSVG, self).__init__()
        self.size = size
        self.outFileName = outFileName
        self.dwg = \
            svgwrite.Drawing(self.outFileName,
                             size=(self.size[1], self.size[0]),
                             profile='full')

        self.setCamera()

    def setCamera(self):
        fx = self.size[0] / 2.
        fy = fx
        cx = self.size[1] / 2.
        cy = self.size[0] / 2.
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
        self.R = self.geo.getRMatrixEulerAngles(np.deg2rad(180), 0, 0)
        self.R = self.geo.getRMatrixEulerAngles(0, 0, np.deg2rad(-90)).dot(self.R)
        self.T = np.array([[0], [0], [1]])

    def project(self, p):
        p_ = self.K.dot(self.R.dot(p) + self.T)
        return (p_[0, 0] / p_[2, 0], p_[1, 0] / p_[2, 0])

    def add(self, obj):
        self.dwg.add(obj)

    def fixTextPath(self):
        outLine = ''
        with open(self.outFileName, 'r') as inFile:
            for lineCount, line in enumerate(inFile):
                line = line.replace('<textPath', '<text><textPath')
                outLine = line.replace('</textPath>', '</textPath></text>')

        with open(self.outFileName, 'w') as outFile:
            outFile.write(outLine)

    def cvtColor(self, r, g, b, base=255):
        return svgwrite.rgb(r * 1. / base * 100, g * 1. / base * 100, b * 1. / base * 100, '%')

    def setDefaultParamsPoint(self, **params):
        c1 = params.get('stroke') is None
        c2 = params.get('color') is None
        if c1 and c2:
            params.update({'stroke': self.Cblack})
            params.update({'fill': self.Cblack})
        elif c1 and not c2:
            params.update({'stroke': params.get('color')})
            params.update({'fill': params.get('color')})

        params = super(PlotSVG, self).setDefaultParamsPoint(**params)
        del params['color']

        if params.get('stroke_width') is None:
            params.update({'stroke_width': params.get('point_size')})
            del params['point_size']

        if params.get('stroke_linecap') is None:
            params.update({'stroke_linecap': 'round'})

        if params.get('fill') is None:
            params.update({'fill': 'none'})

        return params

    def plotPoint(self, pt1, **params):
        super(PlotSVG, self).plotPoint(pt1, **params)
        params = self.setDefaultParamsPoint(**params)
        size = params.get('stroke_width')
        p1 = self.project(pt1)
        self.add(self.dwg.circle(p1, size, **params))

    def show(self):
        self.draw()
        self.dwg.save()
        self.fixTextPath()
        os.system("inkscape test.svg --export-area-drawing --export-pdf=test.pdf")
