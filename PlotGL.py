#!/usr/bin/env python
import numpy as np

import pygame
import pygame.locals as pgl
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut

from Geometry import Geometry
from Plot import Plot


class PlotGL(Plot):
    def __init__(self, size=(600, 800), winName="Viewer"):
        super(PlotGL, self).__init__()
        self.size = size
        self.h, self.w = self.size

        pygame.init()
        glut.glutInit()
        pygame.display.set_caption(winName)
        self.window = pygame.display.set_mode((self.w, self.h), pgl.DOUBLEBUF | pgl.OPENGL)

        self.FOV = 45
        self.aspectRatio = 1. * self.w / self.h
        self.zNear = 0.01
        self.zFar = 30.

        self.zoomReset = 0.25
        self.zoomSpeedMax = 2.
        self.zoomSpeedMin = 1.01
        self.zoomSpeed = 1.05
        self.zoomMax = self.zoomReset * self.zoomSpeed ** 20
        self.zoomMin = self.zoomReset / self.zoomSpeed ** 20

        self.moveReset = [0., 0.]
        self.moveSpeedMax = 10.
        self.moveSpeedMin = 0.001
        self.moveSpeed = 0.1
        self.moveMax = 10.
        self.moveMin = 0.001

        self.viewAngleReset = [0., 0.]
        self.viewAngleSpeedMax = np.deg2rad(30.)
        self.viewAngleSpeedMin = np.deg2rad(0.001)
        self.viewAngleSpeed = np.deg2rad(10)

        self.setParams()
        self.isShowBg = False

    def setParams(self):
        self.zoom = self.zoomReset
        self.move = [self.moveReset[0], self.moveReset[1]]
        self.viewAngle = [self.viewAngleReset[0], self.viewAngleReset[1]]

        self.setView()
        self.setText()

    def setView(self):
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClearDepth(1.0)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glShadeModel(gl.GL_SMOOTH)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        #glu.gluPerspective(self.FOV, self.aspectRatio,
        #                   self.zNear, self.zFar)
        h, w = self.size
        self.xLeft = -0.5 / self.zoom
        self.xRight = 0.5 / self.zoom
        self.yUp = -0.5 / self.zoom * h / w
        self.yDown = 0.5 / self.zoom * h / w
        gl.glOrtho(self.xLeft, self.xRight,
                   self.yUp, self.yDown,
                   self.zNear, self.zFar)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def setText(self, size=24, color=None, bgColor=None):
        self.textSize = size
        self.textColor = color
        if self.textColor is None:
            self.textColor = self.Cwhite
        self.textBgColor = bgColor
        if self.textBgColor is None:
            self.textBgColor = self.Cblack

    def plotText(self, pos, text):
        font = pygame.font.Font(None, self.textSize)
        textSurface = font.render(text, True,
                                  self.textColor * 255,
                                  self.textBgColor * 255)
        textData = pygame.image.tostring(textSurface, "RGBA", True)
        gl.glRasterPos3d(*pos)
        gl.glDrawPixels(textSurface.get_width(),
                        textSurface.get_height(),
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, textData)

    def eventHandler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                self.keyboardEvents(event.key)

            if event.type == pygame.MOUSEBUTTONDOWN:
                self.mouseEvents(event.button)

    def keyboardEvents(self, key):
        if key == pygame.K_ESCAPE:
            pygame.quit()
            quit()

        self.keyboardEventsZoom(key)
        self.keyboardEventsNavigation(key)
        self.keyboardEventSave(key)

    def keyboardEventsNavigation(self, key):
        if key == pygame.K_LEFT:
            self.move[0] = (self.move[0] - self.moveSpeed)
        if key == pygame.K_RIGHT:
            self.move[0] = (self.move[0] + self.moveSpeed)

        if key == pygame.K_UP:
            self.move[1] = (self.move[1] - self.moveSpeed)
        if key == pygame.K_DOWN:
            self.move[1] = (self.move[1] + self.moveSpeed)

        if key == pygame.K_KP4 or key == pygame.K_h:
            self.viewAngle[0] += self.viewAngleSpeed
        if key == pygame.K_KP6 or key == pygame.K_l:
            self.viewAngle[0] -= self.viewAngleSpeed

        if key == pygame.K_KP8 or key == pygame.K_j:
            self.viewAngle[1] += self.viewAngleSpeed
        if key == pygame.K_KP2 or key == pygame.K_k:
            self.viewAngle[1] -= self.viewAngleSpeed
        self.viewAngle[0] %= 2. * np.pi
        self.viewAngle[1] %= 2. * np.pi

        if key == pygame.K_HOME or key == pygame.K_KP5:
            self.move = [self.moveReset[0], self.moveReset[1]]
            self.viewAngle = [self.viewAngleReset[0], self.viewAngleReset[1]]

    def keyboardEventsZoom(self, key):
        if key == pygame.K_PAGEUP:
            self.zoom *= self.zoomSpeed
        if key == pygame.K_PAGEDOWN:
            self.zoom /= self.zoomSpeed

        if key == pygame.K_HOME or key == pygame.K_KP5:
            self.zoom = self.zoomReset

        self.zoom = self.zoomMax if self.zoom > self.zoomMax else self.zoom
        self.zoom = self.zoomMin if self.zoom < self.zoomMin else self.zoom

    def keyboardEventSave(self, key):
        if key == pygame.K_F2:
            data = gl.glReadPixels(0, 0,
                                   self.w, self.h,
                                   gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
            surface = pygame.image.fromstring(str(data),
                                              (self.w, self.h), 'RGBA', True)
            fileName = "test.png"
            pygame.image.save(surface, fileName)

    def mouseEvents(self, button):
        pass

    def report(self):
        pass

    def update(self):
        pass

    def draw(self):
        pass

    def cvtColor(self, r, g, b, base=255):
        return np.array([[r * 1. / base], [g * 1. / base], [b * 1. / base]])

    def setDefaultParamsPoint(self, **params):
        params = super(PlotGL, self).setDefaultParamsPoint(**params)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glPointSize(params.get('point_size'))
        return params

    def setDefaultParamsLine(self, **params):
        params = super(PlotGL, self).setDefaultParamsLine(**params)
        lineWidth = params.get('line_width')
        gl.glLineWidth(lineWidth)
        return params

    def setDefaultParamsArrow(self, **params):
        params = super(PlotGL, self).setDefaultParamsArrow(**params)
        return self.setDefaultParamsLine(**params)

    def plotPoint(self, pt1, **params):
        super(PlotGL, self).plotPoint(pt1, **params)
        params = self.setDefaultParamsPoint(**params)
        color = params.get('color')
        gl.glBegin(gl.GL_POINTS)
        gl.glColor3f(*color)
        gl.glVertex3f(*pt1)
        gl.glEnd()
        if params.get('text') is not None:
            self.plotText(pt1, params.get('text'))

    def plotLine(self, pt1, pt2, **params):
        super(PlotGL, self).plotLine(pt1, pt2, **params)
        params = self.setDefaultParamsLine(**params)
        color = params.get('color')
        gl.glColor3f(*color)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(*pt1)
        gl.glVertex3f(*pt2)
        gl.glEnd()
        if params.get('text') is not None:
            pt3 = (pt1 + pt2) / 2
            self.plotPoint(pt3, color=color)
            self.plotText(pt3, params.get('text'))

    def plotArrow(self, pt1, pt2, **params):
        super(PlotGL, self).plotArrow(pt1, pt2, **params)
        params = self.setDefaultParamsArrow(**params)
        headSize = params.get('head_size')

        lineVec = pt2 - pt1
        lenVec = np.linalg.norm(lineVec)

        gl.glPushMatrix()
        try:
            self.plotLine(pt1, pt2, **params)
            phi, theta = self.geo.vec2Angs(pt2 - pt1)
            gl.glTranslatef(pt2[0, 0], pt2[1, 0], pt2[2, 0])
            gl.glRotatef(np.rad2deg(phi) + 90, 0, 0, 1)
            gl.glRotatef(np.rad2deg(theta), 1, 0, 0)
            glut.glutSolidCone(0.5 * headSize * lenVec, headSize * lenVec, 50, 10)
        finally:
            gl.glPopMatrix()

        if params.get('text') is not None:
            p2 = (pt1 + pt2) / 2
            self.plotPoint(p2, **params)
            self.plotText(p2, params.get('text'))

    def plotAxis(self, pt1=None, R=None, scale=1, isText=False, **params):
        pt1, vX_, vY_, vZ_ = \
            self.genAxis(pt1, R, scale, **params)
        params.update({'color': self.Cred})
        self.plotArrow(pt1, pt1 + scale * vX_, **params)
        params.update({'color': self.Cgreen})
        self.plotArrow(pt1, pt1 + scale * vY_, **params)
        params.update({'color': self.Cblue})
        self.plotArrow(pt1, pt1 + scale * vZ_, **params)

        if isText:
            self.plotText(pt1 + scale * vX_, "x")
            self.plotText(pt1 + scale * vY_, "y")
            self.plotText(pt1 + scale * vZ_, "z")

    def show(self):
        self.Clock = pygame.time.Clock()
        while True:
            self.Clock.tick()
            self.eventHandler()
            self.report()
            self.update()

            gl.glPushMatrix()
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            self.setView()

            if self.isShowBg:
                gl.glBegin(gl.GL_QUADS)
                gl.glColor3f(1, 1, 1)
                z = -self.zFar + 0.1
                gl.glVertex3f(self.xLeft, self.yUp, z)
                gl.glVertex3f(self.xRight, self.yUp, z)
                gl.glVertex3f(self.xRight, self.yDown, z)
                gl.glVertex3f(self.xLeft, self.yDown, z)
                gl.glEnd()

            gl.glPushMatrix()

            move = np.array([[self.move[1]], [-self.move[0]], [0]])
            eye = np.array([[0], [0], [-1]])
            center = np.array([[0], [0], [0]])
            up = np.array([[-1], [0], [0]])
            R = self.geo.getRMatrixEulerAngles(0, 0, self.viewAngle[0] + np.pi / 2.)
            R = R.dot(self.geo.getRMatrixEulerAngles(0, self.viewAngle[1], 0))
            m = R.dot(move)
            e = R.dot(eye) + m
            c = R.dot(center) + m
            u = R.dot(up)
            glu.gluLookAt(e[0, 0], e[1, 0], e[2, 0],
                          c[0, 0], c[1, 0], c[2, 0],
                          u[0, 0], u[1, 0], u[2, 0])

            self.draw()
            gl.glPopMatrix()
            gl.glPopMatrix()

            pygame.display.flip()
            pygame.time.wait(1)
