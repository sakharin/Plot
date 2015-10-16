#!/usr/bin/env python
import cv2
import numpy as np

import pygame
import pygame.locals as pgl
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut

from Geometry import Geometry


class PlotGL(object):
    def __init__(self, size=(600, 600), winName="Viewer"):
        self.h, self.w = size

        pygame.init()
        glut.glutInit()
        pygame.display.set_caption(winName)
        display = (self.w, self.h)
        self.window = pygame.display.set_mode(display, pgl.DOUBLEBUF | pgl.OPENGL)

        self.FOV = 45
        self.aspectRatio = 1. * self.w / self.h
        self.zNear = 0.01
        self.zFar = 30.

        self.zoomReset = 1. / 4.
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
        self.viewAngleSpeedMax = 30.
        self.viewAngleSpeedMin = 0.001
        self.viewAngleSpeed = 10
        self.viewAngleMax = 10.
        self.viewAngleMin = 0.001

        self.setParams()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def setParams(self):
        self.geo = Geometry()

        self.zoom = self.zoomReset
        self.move = [self.moveReset[0], self.moveReset[1]]
        self.viewAngle = [self.viewAngleReset[0], self.viewAngleReset[1]]

        self.setColor()
        self.setPoint()
        self.setLine()
        self.setText()
        self.setView()

    def setColor(self):
        self.red = np.array([[1], [0], [0]])
        self.green = np.array([[0], [1], [0]])
        self.darkGreen = np.array([[0], [100. / 255.], [0]])
        self.blue = np.array([[0], [0], [1]])
        self.cyan = np.array([[0], [1], [1]])
        self.white = np.array([[1], [1], [1]])
        self.orange = np.array([[1], [0.5], [0]])
        self.black = np.array([[0], [0], [0]])

    def setPoint(self, pointSize=5.):
        self.pointSize = pointSize
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glPointSize(self.pointSize)

    def setLine(self, lineWidth=1.):
        self.lineWidth = lineWidth
        gl.glLineWidth(self.lineWidth)

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
        gl.glOrtho(-0.5 / self.zoom, 0.5 / self.zoom, -0.5 / self.zoom, 0.5 / self.zoom, self.zNear, self.zFar)
        gl.glMatrixMode(gl.GL_MODELVIEW)

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
            self.move[0] = (self.move[0] + self.moveSpeed)
        if key == pygame.K_RIGHT:
            self.move[0] = (self.move[0] - self.moveSpeed)

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
        self.viewAngle[0] %= 360
        self.viewAngle[1] %= 360

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
            gl.glReadBuffer(gl.GL_FRONT)
            data = gl.glReadPixels(0, 0,
                                   self.w, self.h,
                                   gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            surface = pygame.image.fromstring(str(buffer(data)),
                                              (self.w, self.h), 'RGB', True)
            fileName = "res.png"
            pygame.image.save(surface, fileName)

    def mouseEvents(self, button):
        pass

    def report(self):
        pass

    def update(self):
        pass

    def draw(self):
        pass

    def plotPoint(self, p0, color=None, text=None):
        if color is None:
            color = self.red
        gl.glBegin(gl.GL_POINTS)
        gl.glColor3f(*color)
        gl.glVertex3f(*p0)
        gl.glEnd()
        if text is not None:
            self.plotText(p0, text)

    def plotLine(self, p0, p1, color=None, text=None):
        if color is None:
            color = self.red
        gl.glColor3f(*color)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(*p0)
        gl.glVertex3f(*p1)
        gl.glEnd()
        if text is not None:
            p2 = (p0 + p1) / 2
            self.plotPoint(p2, color=color)
            self.plotText(p2, text)

    def plotArrow(self, p0, p1, color=None, isInverse=False, text=None):
        if color is None:
            color = self.red

        lineVec = p1 - p0
        lenVec = np.linalg.norm(lineVec)

        gl.glPushMatrix()
        try:
            self.plotLine(p0, p1, color)
            pt = p1 - 0.1 * lineVec
            theta, phi = self.geo.vec2Angs(p1 - p0)
            gl.glTranslatef(pt[0, 0], pt[1, 0], pt[2, 0])
            gl.glRotatef(np.rad2deg(phi) + 90, 0, 0, 1)
            gl.glRotatef(np.rad2deg(theta), 1, 0, 0)
            glut.glutSolidCone(0.05 * lenVec, 0.1 * lenVec, 50, 10)
        finally:
            gl.glPopMatrix()

        if text is not None:
            p2 = (p0 + p1) / 2
            self.plotPoint(p2, color=color)
            self.plotText(p2, text)

    def plotAxis(self, scale=1., isText=False):
        pO = np.array([[0], [0], [0]])
        vX = np.array([[1], [0], [0]])
        vY = np.array([[0], [1], [0]])
        vZ = np.array([[0], [0], [1]])
        self.plotArrow(pO, pO + scale * vX, color=self.red)
        self.plotArrow(pO, pO + scale * vY, color=self.green)
        self.plotArrow(pO, pO + scale * vZ, color=self.blue)
        if isText:
            self.plotText(pO + scale * vX, "x")
            self.plotText(pO + scale * vY, "y")
            self.plotText(pO + scale * vZ, "z")

    def plotRectangle(self, p0, p1, color=None):
        if color is None:
            color = self.red
        pA = p0
        pB = p0
        pC = p1
        pD = p1

        gl.glBegin(gl.GL_TRIANGLES)
        gl.glColor3f(*color)
        gl.glVertex3f(*pA)
        gl.glVertex3f(*pB)
        gl.glVertex3f(*pC)
        gl.glEnd()

        gl.glBegin(gl.GL_TRIANGLES)
        gl.glColor3f(*color)
        gl.glVertex3f(*pB)
        gl.glVertex3f(*pC)
        gl.glVertex3f(*pD)
        gl.glEnd()

    def plotCircle(self, p0, r, color=None, isFill=False, numSegments=32):
        self.plotArc(p0, r, start=0, end=2 * np.pi, color=color,
                     isLoop=True, isFill=isFill, numSegments=numSegments)

    def plotArc(self, p0, r, start=0, end=np.pi, color=None, isLoop=False, isFill=False, numSegments=32):
        if color is None:
            color = self.red
        gl.glPushMatrix()
        gl.glTranslatef(*p0)
        if isFill:
            gl.glBegin(gl.GL_POLYGON)
        else:
            if isLoop:
                gl.glBegin(gl.GL_LINE_LOOP)
            else:
                gl.glBegin(gl.GL_LINE_STRIP)
        gl.glColor3f(*color)
        for i in range(numSegments):
            angRange = self.geo.angleDiff(start, end)
            if start != end and angRange == 0:
                angRange = 2 * np.pi
            theta = angRange * i / numSegments
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            gl.glVertex3f(x, y, 0)
        gl.glEnd()
        gl.glPopMatrix()

    def plotSphere(self, p0, r, color=None, numSegments=(16, 32)):
        vec = np.array([[0], [0], [1]])
        gl.glPushMatrix()
        gl.glTranslatef(*p0)

        for i in range(numSegments[0]):
            theta = i * np.pi / numSegments[0]
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glColor3f(*color)
            for j in range(numSegments[1]):
                phi = j * 2 * np.pi / numSegments[1]

                m = self.geo.getRMatrixEulerAngles(0, 0, phi)
                m = m.dot(self.geo.getRMatrixEulerAngles(0, theta, 0))

                rVec = r * m.dot(vec)
                gl.glVertex3f(*rVec)
            gl.glEnd()

        for j in range(numSegments[1]):
            phi = j * 2 * np.pi / numSegments[1]
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glColor3f(*color)
            for i in range(numSegments[0] + 1):
                theta = i * np.pi / numSegments[0]
                m = self.geo.getRMatrixEulerAngles(0, 0, phi)
                m = m.dot(self.geo.getRMatrixEulerAngles(0, theta, 0))

                rVec = r * m.dot(vec)
                gl.glVertex3f(*rVec)
            gl.glEnd()
        gl.glPopMatrix()

    def plotCamera(self, pC, vC, cameraSizeH=0.024, cameraSizeW=0.036, cameraF=0.035, cameraScale=10):
        vX = np.array([[1], [0], [0]])
        vY = np.array([[0], [1], [0]])
        vZ = np.array([[0], [0], [1]])
        pC0 = np.zeros((3, 1))
        pC1 = -cameraSizeW / 2 * vX - cameraSizeH / 2 * vY + cameraF * vZ
        pC2 = cameraSizeW / 2 * vX - cameraSizeH / 2 * vY + cameraF * vZ
        pC3 = -cameraSizeW / 2 * vX + cameraSizeH / 2 * vY + cameraF * vZ
        pC4 = cameraSizeW / 2 * vX + cameraSizeH / 2 * vY + cameraF * vZ

        m, n = vC.shape
        if m == 3 and n == 1:
            # Vector
            theta, phi = self.geo.vec2Angs(vC)
            M = self.geo.getRMatrixEulerAngles(0, 0, -np.pi / 2.)
            M = self.geo.getRMatrixEulerAngles(0, theta, 0).dot(M)
            M = self.geo.getRMatrixEulerAngles(0, 0, phi).dot(M)
        elif m >= 3 and m <= 4 and n >= 3 and n <= 4:
            # Rotation matrix
            M = vC[:3, :3]

        pC0 = cameraScale * M.dot(pC0) + pC
        pC1 = cameraScale * M.dot(pC1) + pC
        pC2 = cameraScale * M.dot(pC2) + pC
        pC3 = cameraScale * M.dot(pC3) + pC
        pC4 = cameraScale * M.dot(pC4) + pC

        self.plotLine(pC0, pC1, color=self.blue)
        self.plotLine(pC0, pC2, color=self.blue)
        self.plotLine(pC0, pC3, color=self.blue)
        self.plotLine(pC0, pC4, color=self.blue)

        self.plotLine(pC1, pC2, color=self.blue)
        self.plotLine(pC1, pC3, color=self.blue)
        self.plotLine(pC2, pC4, color=self.blue)
        self.plotLine(pC3, pC4, color=self.blue)
        self.plotPoint(pC1, color=self.red)
        self.plotPoint(pC2, color=self.green)

    def setText(self, size=24, color=None, bgColor=None):
        self.textSize = size
        self.textColor = color
        if self.textColor is None:
            self.textColor = self.white
        self.textBgColor = bgColor
        if self.textBgColor is None:
            self.textBgColor = self.black

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

            move = np.array([[self.move[1]], [-self.move[0]], [0]])
            eye = np.array([[0], [0], [1]])
            center = np.array([[0], [0], [0]])
            up = np.array([[-1], [0], [0]])
            R = self.geo.getRMatrixEulerAngles(0, 0, np.deg2rad(self.viewAngle[0] - 180))
            R = R.dot(self.geo.getRMatrixEulerAngles(0, np.deg2rad(-self.viewAngle[1]), 0))
            m = R.dot(move)
            e = R.dot(eye) + m
            c = R.dot(center) + m
            u = R.dot(up)
            glu.gluLookAt(e[0, 0], e[1, 0], e[2, 0],
                          c[0, 0], c[1, 0], c[2, 0],
                          u[0, 0], u[1, 0], u[2, 0])

            self.draw()
            gl.glPopMatrix()

            pygame.display.flip()
            pygame.time.wait(1)


class test(PlotGL):
    def draw(self):
        self.plotAxis()
        for i in range(10):
            px = np.array([[i * 0.1], [0], [0]])
            self.plotCircle(px, 1, self.red)
            py = np.array([[0], [i * 0.1], [0]])
            self.plotArc(py, 1, 0, np.pi, self.green)
            pz = np.array([[0], [0], [i * 0.1]])
            self.plotCircle(pz, 1, self.blue)


if __name__ == "__main__":
    p = test()
    p.show()
