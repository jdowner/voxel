import logging

import numpy
import OpenGL
OpenGL.ERROR_ON_COPY = True

from OpenGL.arrays import vbo
from OpenGL.GL import *
from OpenGL.GLUT import *

# PyOpenGL 3.0.1 introduces this convenience module...
import OpenGL.GL.shaders as glsl


LOGLEVEL = {
        'debug':    logging.DEBUG,
        'info':     logging.INFO,
        'warning':  logging.WARNING,
        'error':    logging.ERROR,
        'critical': logging.CRITICAL}

log = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
log.addHandler(sh)

def set_log_level(level):
    log.setLevel(LOGLEVEL[level.lower()])


class Voxel(object):
    def __init__(self, x, y, z, hx, hy, hz, r, g, b, a):
        self._vertices = [
                [x + hx, y - hy, z - hz, r, g, b, a],
                [x + hx, y + hy, z - hz, r, g, b, a],
                [x + hx, y + hy, z + hz, r, g, b, a],
                [x + hx, y - hy, z + hz, r, g, b, a],
                [x + hx, y - hy, z + hz, r, g, b, a],
                [x + hx, y + hy, z + hz, r, g, b, a],
                [x - hx, y + hy, z + hz, r, g, b, a],
                [x - hx, y - hy, z + hz, r, g, b, a],
                [x + hx, y + hy, z + hz, r, g, b, a],
                [x + hx, y + hy, z - hz, r, g, b, a],
                [x - hx, y + hy, z - hz, r, g, b, a],
                [x - hx, y + hy, z + hz, r, g, b, a],
                [x - hx, y - hy, z + hz, r, g, b, a],
                [x - hx, y + hy, z + hz, r, g, b, a],
                [x - hx, y + hy, z - hz, r, g, b, a],
                [x - hx, y - hy, z - hz, r, g, b, a],
                [x - hx, y - hy, z - hz, r, g, b, a],
                [x - hx, y + hy, z - hz, r, g, b, a],
                [x + hx, y + hy, z - hz, r, g, b, a],
                [x + hx, y - hy, z - hz, r, g, b, a],
                [x - hx, y - hy, z + hz, r, g, b, a],
                [x - hx, y - hy, z - hz, r, g, b, a],
                [x + hx, y - hy, z - hz, r, g, b, a],
                [x + hx, y - hy, z + hz, r, g, b, a],
                ]

    @property
    def vertices(self):
        return self._vertices


class Renderer(object):
    def __init__(self, width, height):
        self._program = 0
        self._vertex_shaders = []
        self._fragment_shaders = []
        self._voxels = []
        self._vbo_voxels = None

        glClearColor(0.2, 0.2, 0.2, 1.0)
        glClearDepth(1.0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(-1.0, 1.0, -1.0, 1.0, 1.0, 100.0)

        glMatrixMode(GL_MODELVIEW)


    def add_voxel(self, voxel):
        self._vbo_voxels = None
        self._voxels.append(voxel)


    def load_vertex_shader(self, filename):
        with open(filename) as fp:
            shader = fp.read()
        self._vertex_shaders.append(glsl.compileShader(shader, GL_VERTEX_SHADER))


    def load_fragment_shader(self, filename):
        with open(filename) as fp:
            shader = fp.read()
        self._vertex_shaders.append(glsl.compileShader(shader, GL_FRAGMENT_SHADER))


    def build_program(self):
        if self._program != 0:
            glDeleteProgram(self._program)

        self._program = glCreateProgram()

        for shader in self._vertex_shaders:
            glAttachShader(self._program, shader)

        for shader in self._fragment_shaders:
            glAttachShader(self._program, shader)

        glLinkProgram(self._program)
        glValidateProgram(self._program)


    def resize(self, width, height):
        assert width > 0
        assert height > 0

        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, 0, height, 0, 100)
        glMatrixMode(GL_MODELVIEW)


    def display(self):
        with glsl.ShaderProgram(self._program):
            try:
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glLoadIdentity()

                glEnable(GL_BLEND);
                glDepthMask(GL_FALSE);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

                glTranslatef(0,0,-10)

                if self._vbo_voxels is None:
                    self._construct_vbo_voxels()

                self._vbo_voxels.bind()
                try:
                    glEnableClientState(GL_VERTEX_ARRAY)
                    glEnableClientState(GL_COLOR_ARRAY)
                    glVertexPointer(4, GL_FLOAT, 28, self._vbo_voxels)
                    glColorPointer(4, GL_FLOAT, 28, self._vbo_voxels + 12)
                    glDrawArrays(GL_QUADS, 0, len(self._vbo_voxels))
                finally:
                    self._vbo_voxels.unbind()
                    glDisableClientState(GL_VERTEX_ARRAY)
                    glDisableClientState(GL_COLOR_ARRAY)
            except:
                # @todo need to print out the relevant information and terminate
                # execution.
                raise

        glutSwapBuffers()

    def _construct_vbo_voxels(self):
        vertices = [v for q in self._voxels for v in q.vertices]
        self._vbo_voxels = vbo.VBO(numpy.array(vertices, 'f'))

