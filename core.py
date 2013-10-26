import logging
import sys

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


class ShaderProgram(object):
    """
    This class encapsulates the shader program and provides some utilities and
    convenience functions for interacting with the shader program.
    """

    def __init__(self):
        """
        Create an instance of ShaderProgram.
        """
        self._vertex_shaders = []
        self._fragment_shaders = []
        self._program = None

    @property
    def vertex_shaders(self):
        """
        This list of vertex shaders associated with this program.
        """
        return self._vertex_shaders

    @property
    def fragment_shaders(self):
        """
        This list of fragment shaders associated with this program.
        """
        return self._fragment_shaders

    @property
    def handle(self):
        """
        Returns the handle/id of the shader program. Ideally we would not be
        exposing this. Consider it deprecated.
        """
        return self._program

    def load_vertex_shader(self, filename):
        """
        Loads a vertex shader.

        @param filename - the path to the file containing the vertex shader.
        """
        with open(filename) as fp:
            shader = fp.read()

        try:
            self._vertex_shaders.append(glsl.compileShader(shader, GL_VERTEX_SHADER))
        except RuntimeError as e:
            loc, shader, errcode = e.args
            print(errcode)
            print(loc)
            for number, line in enumerate(shader[0].split('\r\n')):
                print('%d: %s' % (number, line))
            sys.exit(1)

    def load_fragment_shader(self, filename):
        """
        Loads a fragment shader.

        @param filename - the path to the file containing the fragment shader.
        """
        with open(filename) as fp:
            shader = fp.read()
        self._fragment_shaders.append(glsl.compileShader(shader, GL_FRAGMENT_SHADER))

    def build(self):
        """
        Compiles and links that shader program. Note that repeated calls to this
        function can be made, but the exiting shader program (known by this
        object) will be destroyed first.
        """
        if self._program is not None:
            glDeleteProgram(self._program)

        # Create the shader program and attach all of the shaders
        self._program = glCreateProgram()

        for shader in self._vertex_shaders:
            glAttachShader(self._program, shader)

        for shader in self._fragment_shaders:
            glAttachShader(self._program, shader)

        # Linking stage
        glLinkProgram(self._program)

        # Check that the program in valid and throw and informative exception if
        # it is not.
        glValidateProgram(self._program)
        if not glGetProgramiv(self._program, GL_VALIDATE_STATUS):
            info = glGetProgramInfoLog(self._program)
            raise RuntimeError('Invalid program: %s' % (info,))

        # Find all of the active uniform variables in the program and bind them
        # to this object. This provides a nice interface for setting the uniform
        # variables.
        num_active_uniforms = glGetProgramiv(self._program, GL_ACTIVE_UNIFORMS)
        for index in xrange(num_active_uniforms):
            name, location, dtype = glGetActiveUniform(self._program, index)

            # @todo support for data types
            if dtype == GL_FLOAT_VEC3:
                def set_uniform(_, vals):
                    glUniform(location, *vals)

                setattr(self, name, set_uniform)

