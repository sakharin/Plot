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

        self.scale = 0.25

    def project(self, p):
        A = np.array([[1, 0],
                      [0, 1],
                      [0, 0]])
        AT = A.T
        ATAI = np.linalg.inv(AT.dot(A))
        proj = A.dot(ATAI).dot(AT)
        #print proj
        q = proj.dot(p) * self.size[1] * self.scale
        return (q[0, 0] + self.size[1] / 2., q[1, 0] + self.size[0] / 2.)

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
        elif c1 and not c2:
            params.update({'stroke': params.get('color')})

        params = super(PlotSVG, self).setDefaultParamsPoint(**params)
        del params['color']

        if params.get('stroke_width') is None:
            params.update({'stroke_width': params.get('point_size')})
            del params['point_size']
        if params.get('point_size') is not None:
            del params['point_size']

        if params.get('stroke_linecap') is None:
            params.update({'stroke_linecap': 'round'})

        params.update({'fill': params.get('stroke')})

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
        size = params.get('stroke_width') / 2.
        p1 = self.project(pt1)
        self.add(self.dwg.circle(p1, size, **params))

    def plotLine(self, pt1, pt2, **params):
        params = self.setDefaultParamsLine(**params)
        pt1_ = self.project(pt1)
        pt2_ = self.project(pt2)
        self.add(self.dwg.line(pt1_, pt2_, **params))

    def plotArrow(self, pt1, pt2, **params):
        numSegments = 32

        params = self.setDefaultParamsArrow(**params)
        headSize = params.get('head_size')
        del params['head_size']

        pt1_ = self.project(pt1)
        pt2_ = self.project(pt2)

        g = self.dwg.g()
        g.add(self.dwg.line(pt1_, pt2_, **params))

        # Draw arrow head
        params.update({'stroke_dasharray': 'none'})
        params.update({'fill': params.get('stroke')})
        params.update({'style': 'stroke-linejoin:round'})
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
            g.add(self.dwg.polygon([pt2_, pA_, pB_], **params))
        self.add(g)

    def plotPlane(self, pt1=None,
                  R=None, vN=None,
                  w=1, h=1, **params):
        p1, p2, p3, p4 = \
            self.genPlane(pt1, R, vN, w, h, **params)

        p1_ = self.project(p1)
        p2_ = self.project(p2)
        p3_ = self.project(p3)
        p4_ = self.project(p4)

        g = self.dwg.g()
        params = self.setDefaultParamsLine(**params)
        g.add(self.dwg.line(p1_, p2_, **params))
        g.add(self.dwg.line(p2_, p3_, **params))
        g.add(self.dwg.line(p3_, p4_, **params))
        g.add(self.dwg.line(p4_, p1_, **params))
        self.add(g)

    def plotCam(self, pt1=None,
                R=None, vU=None, vE=None,
                camSizeH=0.024,
                camSizeW=0.036,
                camF=0.035,
                camScale=10,
                **params):
        p0, p1, p2, p3, p4 = \
            self.genCam(pt1, R, vU, vE, camSizeH, camSizeW, camF, camScale, **params)
        p0_ = self.project(p0)
        p1_ = self.project(p1)
        p2_ = self.project(p2)
        p3_ = self.project(p3)
        p4_ = self.project(p4)

        g = self.dwg.g()
        params = self.setDefaultParamsLine(**params)
        g.add(self.dwg.line(p0_, p1_, **params))
        g.add(self.dwg.line(p0_, p2_, **params))
        g.add(self.dwg.line(p0_, p3_, **params))
        g.add(self.dwg.line(p0_, p4_, **params))

        g.add(self.dwg.line(p1_, p2_, **params))
        g.add(self.dwg.line(p2_, p3_, **params))
        g.add(self.dwg.line(p3_, p4_, **params))
        g.add(self.dwg.line(p4_, p1_, **params))

        params = self.setDefaultParamsPoint(**params)
        size = params.get('stroke_width')
        params.update({'stroke': self.Cred})
        params.update({'fill': self.Cred})
        g.add(self.dwg.circle(p1_, size, **params))
        params.update({'stroke': self.Cgreen})
        params.update({'fill': self.Cgreen})
        g.add(self.dwg.circle(p2_, size, **params))
        self.add(g)

    def plotAirplane(self, pt1=None,
                     R=None, vU=None, vE=None,
                     scale=1., **params):
        p0, p1, p2, p3, p4, p5, p6, p7, p8 = \
            self.genAirplane(pt1, R, vU, vE, scale, **params)
        p0_ = self.project(p0)
        p1_ = self.project(p1)
        p2_ = self.project(p2)
        p3_ = self.project(p3)
        p4_ = self.project(p4)
        p5_ = self.project(p5)
        p6_ = self.project(p6)
        p7_ = self.project(p7)
        p8_ = self.project(p8)

        g = self.dwg.g()
        params = self.setDefaultParamsLine(**params)
        g.add(self.dwg.line(p0_, p1_, **params))
        g.add(self.dwg.line(p0_, p2_, **params))
        g.add(self.dwg.line(p0_, p3_, **params))
        g.add(self.dwg.line(p1_, p2_, **params))
        g.add(self.dwg.line(p2_, p3_, **params))

        g.add(self.dwg.line(p2_, p4_, **params))

        g.add(self.dwg.line(p4_, p5_, **params))
        g.add(self.dwg.line(p4_, p6_, **params))
        g.add(self.dwg.line(p4_, p7_, **params))
        g.add(self.dwg.line(p5_, p6_, **params))
        g.add(self.dwg.line(p6_, p7_, **params))

        g.add(self.dwg.line(p4_, p8_, **params))
        g.add(self.dwg.line(p6_, p8_, **params))

        params = self.setDefaultParamsPoint(**params)
        size = params.get('stroke_width')
        params.update({'stroke': self.Cred})
        params.update({'fill': self.Cred})
        g.add(self.dwg.circle(p1_, size, **params))
        params.update({'stroke': self.Cgreen})
        params.update({'fill': self.Cgreen})
        g.add(self.dwg.circle(p3_, size, **params))
        params.update({'stroke': self.Cwhite})
        params.update({'fill': self.Cwhite})
        g.add(self.dwg.circle(p8_, size, **params))
        self.add(g)

    def plotCircle(self, pt1=None, r=1, R=None, vN=None, numSegments=64, isDash=False, **params):
        if pt1 is None:
            pt1 = np.copy(self.pO)
        pts = self.genCircle(pt1=pt1, r=r, R=R, vN=vN, numSegments=numSegments)

        params = self.setDefaultParamsLine(**params)

        g = self.dwg.g()
        for i in range(numSegments):
            p1_ = self.project(pts[:, i:i + 1])
            if i == numSegments - 1:
                p2_ = self.project(pts[:, 0:1])
            else:
                p2_ = self.project(pts[:, i + 1:i + 2])
            g.add(self.dwg.line(p1_, p2_, **params))
        self.add(g)

    def show(self):
        self.draw()
        self.dwg.save()
        self.fixTextPath()
        os.system("inkscape test.svg --export-area-drawing --export-pdf=test.pdf")
