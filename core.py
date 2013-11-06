import itertools
import logging
import sys
import math

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

# Set up the logger
log = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
log.addHandler(sh)

def set_log_level(level):
    log.setLevel(LOGLEVEL[level.lower()])


class Renderer(object):
    def __init__(self, program, (width, height)):
        self._program = program
        self._voxels = []
        self._vbo_voxels = None
        self._frustum = Frustum(width, height, 5000, 25.0)
        self._camera = Camera()
        self._camera.position = numpy.array([0,0,5000])
        self._camera.orientation = Quaternion.from_axis_angle(
                numpy.array([0, 1, 0]), 0)
        self._clear_color = Color(0.3, 0.3, 0.3, 1.0)

        self.set_viewport(width, height)

    @property
    def camera(self):
        return self._camera

    def add_voxel(self, voxel):
        self._vbo_voxels = None
        self._voxels.append(voxel)

    def set_viewport(self, width, height):
        assert width > 0
        assert height > 0

        glViewport(0, 0, width, height)

        self._frustum.resize((width, height))

    def resize(self, width, height):
        self.set_viewport(width, height)

    def _render_fiducials(self):
        length = 20
        sep = 100

        hues = (c / 10.0 for c in xrange(1,10))
        colors = [Color.from_hsv(hue, 0.5, 0.95) for hue in hues]
        centers = [x for x in itertools.product([-sep,0,sep], repeat=3)]
        for center, color in zip(centers, colors):
            glPushMatrix()
            glTranslate(*center)
            glBegin(GL_LINES)
            glColor3f(color.r, color.g, color.b)
            glVertex3f(+length, 0, 0)
            glVertex3f(-length, 0, 0)
            glVertex3f(0, +length, 0)
            glVertex3f(0, -length, 0)
            glVertex3f(0, 0, +length)
            glVertex3f(0, 0, -length)
            glEnd()
            glPopMatrix()

    def display(self):
        with glsl.ShaderProgram(self._program.handle):
            try:
                glClearColor(*self._clear_color)

                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                glEnable(GL_CULL_FACE)
                glEnable(GL_DEPTH_TEST)
                glDepthFunc(GL_LESS)

                orientation = self.camera.orientation
                axis = orientation.axis()
                angle = orientation.angle()

                glRotatef(-180.0 * angle / math.pi, axis[0], axis[1], axis[2])

                position = self.camera.position
                x = position[0]
                y = position[1]
                z = position[2]

                glTranslate(-x, -y, -z)

                #self._render_fiducials()

                self._program.light_position(0, 0, 5000)

                if self._vbo_voxels is None:
                    self._construct_vbo_voxels()

                self._vbo_voxels.bind()
                try:
                    glEnableClientState(GL_VERTEX_ARRAY)
                    glEnableClientState(GL_NORMAL_ARRAY)
                    glEnableClientState(GL_COLOR_ARRAY)
                    glVertexPointer(3, GL_FLOAT, 40, self._vbo_voxels)
                    glNormalPointer(GL_FLOAT, 40, self._vbo_voxels + 12)
                    glColorPointer(4, GL_FLOAT, 40, self._vbo_voxels + 24)
                    glDrawArrays(GL_QUADS, 0, len(self._vbo_voxels))
                finally:
                    self._vbo_voxels.unbind()
                    glDisableClientState(GL_VERTEX_ARRAY)
                    glDisableClientState(GL_NORMAL_ARRAY)
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
            name, _, dtype = glGetActiveUniform(self._program, index)
            location = glGetUniformLocation(self._program, name)

            # @todo support for data types
            if dtype == GL_FLOAT_VEC3:
                def set_uniform_3f(x, y, z):
                    glUniform3f(location, x, y, z)

                setattr(self, name, set_uniform_3f)


class Frustum(object):
    def __init__(self, height, width, depth, fov):
        self._height = height
        self._width = width
        self._depth = depth
        self._fov = fov

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def depth(self):
        return self._depth

    @property
    def fov(self):
        return self._fov

    def update(self):
        near = self.width * math.tan(self.fov * math.pi / 180.0)
        far = self.depth + near

        half_width = self.width / 2.0
        half_height = self.height / 2.0

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(-half_width, half_width, -half_height, half_height, near, far)

    def resize(self, (width, height)):
        self._width = width
        self._height = height
        self.update()


class Camera(object):
    """
    The camera class is used to define the viewers perspective.
    """

    def __init__(self):
        self._basis = numpy.diag((1.0,1.0,1.0))
        self._orientation = Quaternion.from_axis_angle(
                numpy.array([0.0, 1.0, 0.0]), 0.0)
        self._position = numpy.array([0,0,0], dtype=numpy.float32)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = position.astype(float)

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        self._orientation = orientation

    @property
    def up(self):
        return numpy.array(self._orientation.rotate(self._basis[:,1]))

    @property
    def down(self):
        return -self.up

    @property
    def forward(self):
        return -numpy.array(self._orientation.rotate(self._basis[:,2]))

    @property
    def backward(self):
        return -self.forward

    @property
    def left(self):
        return -numpy.array(self._orientation.rotate(self._basis[:,0]))

    @property
    def right(self):
        return -self.left

    def pitch(self, angle):
        self._orientation = Quaternion.from_axis_angle(self.right, angle) * self._orientation
        self._orientation.normalize()

    def roll(self, angle):
        self._orientation = Quaternion.from_axis_angle(self.forward, angle) * self._orientation
        self._orientation.normalize()

    def yaw(self, angle):
        self._orientation = Quaternion.from_axis_angle(self.up, angle) * self._orientation
        self._orientation.normalize()

    def move_forward(self, distance):
        self._position += distance * self.forward

    def move_backward(self, distance):
        self._position -= distance * self.forward

    def move_left(self, distance):
        self._position += distance * self.left

    def move_right(self, distance):
        self._position -= distance * self.left

    def move_up(self, distance):
        self._position += distance * self.up

    def move_down(self, distance):
        self._position -= distance * self.up


class Quaternion(object):
    def __init__(self, w, x, y, z):
        self._data = numpy.array([w, x, y, z], dtype = numpy.float32)

    def __iadd__(self, q):
        self._data = numpy.add(self._data, q._data)
        return self

    def __isub__(self, q):
        self._data = numpy.subtract(self._data, q._data)
        return self

    def __imul__(self, q):
        # (ua + va)(ub + vb) = (ua * ub - va * vb) + ua * vb + ub * va + va x vb

        u = self._data
        v = q._data

        r = u[0] * v[0] - numpy.dot(u[1:], v[1:])
        s = u[0] * v[1:] + v[0] * u[1:] + numpy.cross(u[1:], v[1:])

        self._data[0] = r
        self._data[1:] = s

        return self

    def __add__(self, q):
        r = self.clone()
        r += q
        return r

    def __sub__(self, q):
        r = self.clone()
        r -= q
        return r

    def __mul__(self, q):
        r = self.clone()
        r *= q
        return r

    def __eq__(self, q):
        return numpy.array_equal(self._data, q._data)

    def __ne__(self, q):
        return not numpy.array_equal(self._data, q._data)

    def __repr__(self):
        return repr((self.w, self.x, self.y, self.z))

    @property
    def w(self):
        return self._data[0]

    @property
    def x(self):
        return self._data[1]

    @property
    def y(self):
        return self._data[2]

    @property
    def z(self):
        return self._data[3]

    @property
    def length(self):
        return numpy.linalg.norm(self._data)

    @classmethod
    def from_axis_angle(cls, axis, angle):
        lensqr = numpy.dot(axis, axis)
        if abs(lensqr - 1.0) > 0.0001:
            axis = axis / math.sqrt(lensqr)

        c = math.cos(angle / 2.0)
        s = math.sin(angle / 2.0)
        return cls(c, s * axis[0], s * axis[1], s * axis[2])

    def clone(self):
        return Quaternion(self.w, self.x, self.y, self.z)

    def conjugate(self):
        self._data[1:] = -self._data[1:]
        return self

    def invert(self):
        self.conjugate()
        self._data = self._data / numpy.dot(self._data, self._data)
        return self

    def normalize(self):
        self._data = self._data / self.length
        return self

    def conjugated(self):
        return self.clone().conjugate()

    def inverted(self):
        return self.clone().invert()

    def normalized(self):
        return self.clone().normalize()

    def angle(self):
        return 2.0 * math.atan2(numpy.linalg.norm(self._data[1:]), self.w)

    def axis(self):
        s2 = abs((1.0 - self.w) * (1.0 + self.w))
        if s2 < 0.000001:
            x = 0
            y = 0
            z = 1
        else:
            norm = numpy.linalg.norm(self._data[1:])
            x = self.x / norm
            y = self.y / norm
            z = self.z / norm

        return (x, y, z)

    def matrix(self):
        w = self.w
        x = self.x
        y = self.y
        z = self.z

        R = numpy.zeros((3,3))
        R[0,0] = 1.0 - 2.0 * (y * y + z * z)
        R[1,1] = 1.0 - 2.0 * (x * x + z * z)
        R[2,2] = 1.0 - 2.0 * (x * x + y * y)
        R[0,1] = 2.0 * (x * y + w * z)
        R[0,2] = 2.0 * (x * z - w * y)
        R[1,2] = 2.0 * (y * z + w * x)
        R[1,0] = 2.0 * (x * y - w * z)
        R[2,0] = 2.0 * (x * z + w * y)
        R[2,1] = 2.0 * (y * z - w * x)

        return R

    def rotate(self, (x, y, z)):
        q = self * Quaternion(0, x, y, z) * self.inverted()
        return map(float, q.axis())


class Color(object):
    def __init__(self, r, g, b, a):
        self._color = (r, g, b, a)

    @property
    def r(self):
        return self._color[0]

    @property
    def g(self):
        return self._color[1]

    @property
    def b(self):
        return self._color[2]

    @property
    def a(self):
        return self._color[3]

    @r.setter
    def r(self):
        return self._color[0]

    @g.setter
    def g(self):
        return self._color[1]

    @b.setter
    def b(self):
        return self._color[2]

    @a.setter
    def a(self):
        return self._color[3]

    @classmethod
    def from_hsv(cls, h, s, v):
        hi = int(6 * h)
        f = 6 * h - hi
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        if hi == 0:
            r, g, b = v, t, p
        if hi == 1:
            r, g, b = q, v, p
        if hi == 2:
            r, g, b = p, v, t
        if hi == 3:
            r, g, b = p, q, v
        if hi == 4:
            r, g, b = t, p, v
        if hi == 5:
            r, g, b = v, p, q

        return cls(r, g, b, 1.0)

    def __iter__(self):
        return (c for c in self._color)

    def __getitem__(self, i):
        return self._color[i]

    def __setitem__(self, i, v):
        self._color[i] = v

    def __getslice__(self, i, j):
        return self._color[i:j]

    def __setslice__(self, i, j, v):
        self._color[i:j] = v

    def __repr__(self):
        return repr(self._color)


class Voxel(object):
    """
    This is a simple representation of a cubic, volume element that is used in
    rendering.
    """

    def __init__(self, x, y, z, hx, hy, hz, r, g, b, a):
        """
        Creates a voxel using (x, y, z) as the center of the voxel, (hx, hy, hz)
        as the half-widths of the voxel is the x, y, and z axis respectively,
        and (r, g, b, a) is a float32 RGBA color.

        """
        self._vertices = [
                [x + hx, y - hy, z - hz, 1, 0, 0, r, g, b, a],
                [x + hx, y + hy, z - hz, 1, 0, 0, r, g, b, a],
                [x + hx, y + hy, z + hz, 1, 0, 0, r, g, b, a],
                [x + hx, y - hy, z + hz, 1, 0, 0, r, g, b, a],

                [x + hx, y - hy, z + hz, 0, 0, 1, r, g, b, a],
                [x + hx, y + hy, z + hz, 0, 0, 1, r, g, b, a],
                [x - hx, y + hy, z + hz, 0, 0, 1, r, g, b, a],
                [x - hx, y - hy, z + hz, 0, 0, 1, r, g, b, a],

                [x + hx, y + hy, z + hz, 0, 1, 0, r, g, b, a],
                [x + hx, y + hy, z - hz, 0, 1, 0, r, g, b, a],
                [x - hx, y + hy, z - hz, 0, 1, 0, r, g, b, a],
                [x - hx, y + hy, z + hz, 0, 1, 0, r, g, b, a],

                [x - hx, y - hy, z + hz, -1, 0, 0, r, g, b, a],
                [x - hx, y + hy, z + hz, -1, 0, 0, r, g, b, a],
                [x - hx, y + hy, z - hz, -1, 0, 0, r, g, b, a],
                [x - hx, y - hy, z - hz, -1, 0, 0, r, g, b, a],

                [x - hx, y - hy, z - hz, 0, 0, -1, r, g, b, a],
                [x - hx, y + hy, z - hz, 0, 0, -1, r, g, b, a],
                [x + hx, y + hy, z - hz, 0, 0, -1, r, g, b, a],
                [x + hx, y - hy, z - hz, 0, 0, -1, r, g, b, a],

                [x - hx, y - hy, z + hz, 0, -1, 0, r, g, b, a],
                [x - hx, y - hy, z - hz, 0, -1, 0, r, g, b, a],
                [x + hx, y - hy, z - hz, 0, -1, 0, r, g, b, a],
                [x + hx, y - hy, z + hz, 0, -1, 0, r, g, b, a],
                ]

    @property
    def vertices(self):
        """
        The vertices of the voxel.

        """
        return self._vertices
