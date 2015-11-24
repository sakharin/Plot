#!/usr/bin/env python
import numpy as np
import os
import svgwrite

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
        # Avoid division by zero
        e = 1e-6
        p_ = self.K.dot(self.R.dot(p) + self.T) + e
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

    def setDefaultParamsLine(self, **params):
        c1 = params.get('stroke') is None
        c2 = params.get('color') is None
        if c1 and c2:
            params.update({'stroke': self.black})
        elif c1 and not c2:
            params.update({'stroke': params.get('color')})
            del params['color']

        if params.get('stroke_width') is not None:
            params.update({'line_width': params.get('stroke_width')})
            del params['stroke_width']

        if params.get('stroke_linecap') is None:
            params.update({'stroke_linecap': 'round'})

        if params.get('fill') is None:
            params.update({'fill': 'none'})

        params = super(PlotSVG, self).setDefaultParamsLine(**params)

        params.update({'stroke_width': params.get('line_width')})
        del params['line_width']

        return params

    def setDefaultParamsArrow(self, **params):
        c1 = params.get('stroke') is None
        c2 = params.get('color') is None
        if c1 and c2:
            params.update({'stroke': self.black})
        elif c1 and not c2:
            params.update({'stroke': params.get('color')})
            del params['color']

        if params.get('stroke_width') is not None:
            params.update({'line_width': params.get('stroke_width')})
            del params['stroke_width']

        if params.get('stroke_linecap') is None:
            params.update({'stroke_linecap': 'round'})

        if params.get('fill') is None:
            params.update({'fill': params.get('stroke')})

        params = super(PlotSVG, self).setDefaultParamsArrow(**params)

        params.update({'stroke_width': params.get('line_width')})
        del params['line_width']

        return params

    def plotPoint(self, pt1, **params):
        super(PlotSVG, self).plotPoint(pt1, **params)
        params = self.setDefaultParamsPoint(**params)
        size = params.get('stroke_width')
        p1 = self.project(pt1)
        self.add(self.dwg.circle(p1, size, **params))

    def plotLine(self, pt1, pt2, **params):
        params = self.setDefaultParamsLine(**params)
        pt1_ = self.project(pt1)
        pt2_ = self.project(pt2)
        self.add(self.dwg.line(pt1_, pt2_, **params))

    def plotArrow(self, pt1, pt2, **params):
        params = self.setDefaultParamsArrow(**params)
        numSegments = 32
        headSize = params.get('head_size')
        del params['head_size']

        pt2_ = self.project(pt2)
        self.plotLine(pt1, pt2, **params)

        # Draw arrow head
        params.update({'stroke_dasharray': 'none'})
        params.update({'fill': params.get('stroke')})
        lineVec = pt2 - pt1
        lenVec = np.linalg.norm(lineVec)
        r = headSize / 2. * lenVec
        pt3 = pt2 - headSize * lineVec

        theta, phi = self.geo.vec2Angs(lineVec / lenVec)
        m = self.geo.getRMatrixEulerAngles(0, 0, phi)
        m = m.dot(self.geo.getRMatrixEulerAngles(0, theta, 0))
        for i in range(numSegments):
            angA = i * 2 * np.pi / numSegments
            angB = (i + 1) * 2 * np.pi / numSegments
            pA = r * np.array([[np.cos(angA)], [np.sin(angA)], [0]])
            pB = r * np.array([[np.cos(angB)], [np.sin(angB)], [0]])
            pA_ = self.project(pt3 + m.dot(pA))
            pB_ = self.project(pt3 + m.dot(pB))
            self.add(self.dwg.polygon([pt2_, pA_, pB_], **params))

    def show(self):
        self.draw()
        self.dwg.save()
        self.fixTextPath()
        os.system("inkscape test.svg --export-area-drawing --export-pdf=test.pdf")
