import itertools
import logging
import math
import sys

import numpy
import OpenGL

OpenGL.ERROR_ON_COPY = True
from OpenGL.arrays import vbo
from OpenGL.GL import *
from OpenGL.GLUT import *
import OpenGL.GL.shaders as glsl


log = logging.getLogger(__name__)


class Renderer(object):
    """
    This class is responsible for rendering the voxels to a window.

    """

    def __init__(self, program, size):
        """
        Creates a renderer using the provided shader program and window size.

        """
        width, height = size
        self._program = program
        self._voxels = []
        self._vbo_voxels = None
        self._frustum = Frustum(width, height, 5000, 25.0)
        self._camera = Camera()
        self._camera.position = numpy.array([0, 0, 5000])
        self._camera.orientation = Quaternion.from_axis_angle(numpy.array([0, 1, 0]), 0)
        self._clear_color = Color(0.3, 0.3, 0.3, 1.0)

        self.resize(width, height)

    @property
    def camera(self):
        """
        The camera that provides the scene view.

        """
        return self._camera

    @property
    def frustum(self):
        """
        The frustum that defines the viewing volume of the camera.

        """
        return self._frustum

    def add_voxel(self, voxel):
        """
        Add a voxel to the renderer.

        """
        # Setting the VBO to None will trigger its re-creation before rendering.
        self._vbo_voxels = None
        self._voxels.append(voxel)

    def add_voxels(self, voxels):
        """
        Adds an iterable of voxels to the renderer.

        """
        self._vbo_voxels = None
        self._voxels.extend(v for v in voxels)

    def resize(self, width, height):
        """
        Sets the dimensions of the viewport and ensure that the frustum matches.

        """
        assert width > 0
        assert height > 0

        glViewport(0, 0, width, height)

        self.frustum.resize((width, height))

    def _render_fiducials(self):
        """
        A debugging function that draws 9 equally-spaced crosses about the
        origin.

        """
        length = 20
        sep = 100

        hues = (c / 10.0 for c in range(1, 10))
        colors = [Color.from_hsv(hue, 0.5, 0.95) for hue in hues]
        centers = [x for x in itertools.product([-sep, 0, sep], repeat=3)]
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
        """
        This is the function that performs the actual rendering.

        """
        with glsl.ShaderProgram(self._program.handle):
            try:
                # Initialize the OpenGL state
                glClearColor(*self._clear_color)

                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                glEnable(GL_CULL_FACE)
                glEnable(GL_DEPTH_TEST)
                glDepthFunc(GL_LESS)

                # Determine the model-view transform from the camera
                orientation = self.camera.orientation
                axis = orientation.axis()
                angle = orientation.angle()

                glRotatef(-180.0 * angle / math.pi, axis[0], axis[1], axis[2])

                position = self.camera.position
                x = position[0]
                y = position[1]
                z = position[2]

                glTranslate(-x, -y, -z)

                self._program.light_position(0, 0, 5000)

                # Reconstruct the VBO if necessary
                if self._vbo_voxels is None:
                    self._construct_vbo_voxels()

                # Pass all of the data to the hardware
                if self._vbo_voxels is not None:
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
            except Exception:
                # @todo need to print out the relevant information and terminate
                # execution.
                raise

        glutSwapBuffers()

    def _construct_vbo_voxels(self):
        """
        Extract the vertices in the voxels and create the VBO.

        """
        if len(self._voxels):
            vertices = numpy.vstack(tuple(v.vertices for v in self._voxels))
            self._vbo_voxels = vbo.VBO(vertices)


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
        Loads and compiles the specified vertex shader.

        """
        self.compile_vertex_shader(open(filename).read())

    def compile_vertex_shader(self, shader):
        """
        Compiles a vertex shader

        """
        try:
            self._vertex_shaders.append(glsl.compileShader(shader, GL_VERTEX_SHADER))
        except RuntimeError as e:
            loc, shader, errcode = e.args
            log.error("%s" % (errcode,))
            log.error("%s" % (loc,))
            for number, line in enumerate(shader[0].split("\r\n")):
                log.error("%d: %s" % (number, line))
            sys.exit(1)

    def load_fragment_shader(self, filename):
        """
        Loads and compiles the specified fragment shader.

        """
        self.compile_fragment_shader(open(filename).read())

    def compile_fragment_shader(self, shader):
        """
        Compiles the specified fragment shader.

        """
        try:
            self._fragment_shaders.append(
                glsl.compileShader(shader, GL_FRAGMENT_SHADER)
            )
        except RuntimeError as e:
            loc, shader, errcode = e.args
            log.error("%s" % (errcode,))
            log.error("%s" % (loc,))
            for number, line in enumerate(shader[0].split("\r\n")):
                log.error("%d: %s" % (number, line))
            sys.exit(1)

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
            raise RuntimeError("Invalid program: %s" % (info,))

        # Find all of the active uniform variables in the program and bind them
        # to this object. This provides a nice interface for setting the uniform
        # variables.
        num_active_uniforms = glGetProgramiv(self._program, GL_ACTIVE_UNIFORMS)
        for index in range(num_active_uniforms):
            name, _, dtype = glGetActiveUniform(self._program, index)
            location = glGetUniformLocation(self._program, name)

            # @todo support for data types
            if dtype == GL_FLOAT_VEC3:

                def set_uniform_3f(x, y, z):
                    glUniform3f(location, x, y, z)

                setattr(self, name.decode("utf-8"), set_uniform_3f)


class Frustum(object):
    """
    The class represents a frustum and handles the perspective transformation
    for the renderer.

    """

    def __init__(self, height, width, depth, fov):
        """
        Creates a frustum.

        """
        self._height = height
        self._width = width
        self._depth = depth
        self._fov = fov

    @property
    def height(self):
        """
        The height of the frustum in the near plane.

        """
        return self._height

    @property
    def width(self):
        """
        The width of the frustum in the near plane.

        """
        return self._width

    @property
    def depth(self):
        """
        The distance from the near plane to the far plane.

        """
        return self._depth

    @property
    def near(self):
        """
        The distance to the near plane.

        """
        return self.width * math.tan(self.fov * math.pi / 180.0)

    @property
    def far(self):
        """
        The distance to the far plane.

        """
        return self.depth + self.near

    @property
    def fov(self):
        """
        The field of view (degrees).

        """
        return self._fov

    def resize(self, size):
        """
        Resizes the frustum using the provided dimensions.

        """
        self._width, self._height = size

        half_width = self.width / 2.0
        half_height = self.height / 2.0

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(
            -half_width, half_width, -half_height, half_height, self.near, self.far
        )


class Camera(object):
    """
    The camera class is used to define the viewers perspective.
    """

    def __init__(self):
        self._basis = numpy.diag((1.0, 1.0, 1.0))
        self._orientation = Quaternion.from_axis_angle(
            numpy.array([0.0, 1.0, 0.0]), 0.0
        )
        self._position = numpy.array([0, 0, 0], dtype=numpy.float32)

    @property
    def position(self):
        """
        The position of the camera in the world co-ordinates.

        """
        return self._position

    @position.setter
    def position(self, position):
        """
        Sets the position of the camera in world co-ordinates.

        """
        self._position = position.astype(float)

    @property
    def orientation(self):
        """
        The orientation of the camera (quaternion) in world co-ordinates.

        """
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        """
        Sets the orientation of the camera (quaternion) in world co-ordinates.

        """
        self._orientation = orientation

    @property
    def up(self):
        """
        The 'up' vector of the camera in world co-ordinates.

        """
        return numpy.array(self._orientation.rotate(self._basis[:, 1]))

    @property
    def down(self):
        """
        The 'down' vector of the camera in world co-ordinates.

        """
        return -self.up

    @property
    def forward(self):
        """
        The 'forward' vector of the camera in world co-ordinates.

        """
        return -numpy.array(self._orientation.rotate(self._basis[:, 2]))

    @property
    def backward(self):
        """
        The 'backward' vector of the camera in world co-ordinates.

        """
        return -self.forward

    @property
    def left(self):
        """
        The 'left' vector of the camera in world co-ordinates.

        """
        return -numpy.array(self._orientation.rotate(self._basis[:, 0]))

    @property
    def right(self):
        """
        The 'right' vector of the camera in world co-ordinates.

        """
        return -self.left

    def pitch(self, angle):
        """
        Rotates the pitch of the camera (radians).

        """
        self._orientation = (
            Quaternion.from_axis_angle(self.right, angle) * self._orientation
        )
        self._orientation.normalize()

    def roll(self, angle):
        """
        Rotates the roll of the camera (radians).

        """
        self._orientation = (
            Quaternion.from_axis_angle(self.forward, angle) * self._orientation
        )
        self._orientation.normalize()

    def yaw(self, angle):
        """
        Rotates the yaw of the camera (radians).

        """
        self._orientation = (
            Quaternion.from_axis_angle(self.up, angle) * self._orientation
        )
        self._orientation.normalize()

    def move_forward(self, distance):
        """
        Move the camera in the forward direction.

        """
        self._position += distance * self.forward

    def move_backward(self, distance):
        """
        Move the camera in the backward direction.

        """
        self._position -= distance * self.forward

    def move_left(self, distance):
        """
        Move the camera in the left direction.

        """
        self._position += distance * self.left

    def move_right(self, distance):
        """
        Move the camera in the right direction.

        """
        self._position -= distance * self.left

    def move_up(self, distance):
        """
        Move the camera in the up direction.

        """
        self._position += distance * self.up

    def move_down(self, distance):
        """
        Move the camera in the down direction.

        """
        self._position -= distance * self.up


class Quaternion(object):
    """
    This class represents a quaternion implemented using numpy arrays.
    """

    def __init__(self, w, x, y, z):
        """
        Creates a quaternion.

        """
        self._data = numpy.array([w, x, y, z], dtype=numpy.float32)

    def __iadd__(self, q):
        """
        Adds a quaternion to this instance. Returns a reference to this
        instance.

        """
        self._data = numpy.add(self._data, q._data)
        return self

    def __isub__(self, q):
        """
        Subtracts a quaternion from this instance. Returns a reference to this
        instance.

        """
        self._data = numpy.subtract(self._data, q._data)
        return self

    def __imul__(self, q):
        """
        Post-multiplies this quaternion by another quaternion and returned a
        reference to this instance.

        """
        # (ur + ui)(vr + vi) = (ur * vr - ui * vi) + ur * vi + vr * ui + ui x vi

        u = self._data
        v = q._data

        r = u[0] * v[0] - numpy.dot(u[1:], v[1:])
        s = u[0] * v[1:] + v[0] * u[1:] + numpy.cross(u[1:], v[1:])

        self._data[0] = r
        self._data[1:] = s

        return self

    def __add__(self, q):
        """
        Adds a quaternion to this quaternion and returns the result.

        """
        r = self.clone()
        r += q
        return r

    def __sub__(self, q):
        """
        Subtracts a quaternion from this quaternion and returns the result.

        """
        r = self.clone()
        r -= q
        return r

    def __mul__(self, q):
        """
        Post-multiplies this quaternion by a another quaternion and results the
        result.

        """
        r = self.clone()
        r *= q
        return r

    def __eq__(self, q):
        """
        Returns True if the provided quaternion is equal to this quaternion.

        """
        return numpy.array_equal(self._data, q._data)

    def __ne__(self, q):
        """
        Returns False if the provided quaternion is equal to this quaternion.

        """
        return not numpy.array_equal(self._data, q._data)

    def __repr__(self):
        return repr((self.w, self.x, self.y, self.z))

    @property
    def w(self):
        """
        The w component of the quaternion.

        """
        return self._data[0]

    @property
    def x(self):
        """
        The x component of the quaternion.

        """
        return self._data[1]

    @property
    def y(self):
        """
        The y component of the quaternion.

        """
        return self._data[2]

    @property
    def z(self):
        """
        The z component of the quaternion.

        """
        return self._data[3]

    @property
    def length(self):
        """
        The length of the quaternion.

        """
        return numpy.linalg.norm(self._data)

    @classmethod
    def from_axis_angle(cls, axis, angle):
        """
        Creates an instance of Quaternion using an axis (unit vector) and an
        angle (radians).

        """
        lensqr = numpy.dot(axis, axis)
        if abs(lensqr - 1.0) > 0.0001:
            axis = axis / math.sqrt(lensqr)

        c = math.cos(angle / 2.0)
        s = math.sin(angle / 2.0)
        return cls(c, s * axis[0], s * axis[1], s * axis[2])

    def clone(self):
        """
        Returns a copy of this quaternion.

        """
        return Quaternion(self.w, self.x, self.y, self.z)

    def conjugate(self):
        """
        Conjugates this quaternion and returns a reference to this instance.

        """
        self._data[1:] = -self._data[1:]
        return self

    def invert(self):
        """
        Inverts this quaternion and returns a reference to this instance.

        """
        self.conjugate()
        self._data = self._data / numpy.dot(self._data, self._data)
        return self

    def normalize(self):
        """
        Normalizes this quaternion and returns a reference to this instance.

        """
        self._data = self._data / self.length
        return self

    def conjugated(self):
        """
        Returns a conjugates copy of this quaternion.

        """
        return self.clone().conjugate()

    def inverted(self):
        """
        Returns an inverted copy of this quaternion.

        """
        return self.clone().invert()

    def normalized(self):
        """
        Returns a normalized copy of this quaternion.

        """
        return self.clone().normalize()

    def angle(self):
        """
        Returns the angle that this quaternion represents (assumes that this is
        a unit quaternion).

        """
        return 2.0 * math.atan2(numpy.linalg.norm(self._data[1:]), self.w)

    def axis(self):
        """
        Returns the axis that this quaternion represents (assumes that this is a
        unit quaternion).

        """
        s2 = abs((1.0 - self.w) * (1.0 + self.w))
        if s2 < 0.000001:
            # When s2 is effectively zero, the angle of the quaternion is
            # approximately zero, so the axis returned is arbitrary.
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
        """
        Returns a matrix representation of the quaternion.

        """
        w = self.w
        x = self.x
        y = self.y
        z = self.z

        R = numpy.zeros((3, 3))
        R[0, 0] = 1.0 - 2.0 * (y * y + z * z)
        R[1, 1] = 1.0 - 2.0 * (x * x + z * z)
        R[2, 2] = 1.0 - 2.0 * (x * x + y * y)
        R[0, 1] = 2.0 * (x * y + w * z)
        R[0, 2] = 2.0 * (x * z - w * y)
        R[1, 2] = 2.0 * (y * z + w * x)
        R[1, 0] = 2.0 * (x * y - w * z)
        R[2, 0] = 2.0 * (x * z + w * y)
        R[2, 1] = 2.0 * (y * z - w * x)

        return R

    def rotate(self, point):
        """
        Rotates a vector by this quaternion and returns the result.

        """
        x, y, z = point
        q = self * Quaternion(0, x, y, z) * self.inverted()
        return [float(v) for v in q.axis()]


class Color(object):
    """
    This class represents an RGBA color.
    """

    def __init__(self, r, g, b, a):
        """
        Creates a color.

        """
        self._color = (r, g, b, a)

    @property
    def r(self):
        """
        The red component of the color.

        """
        return self._color[0]

    @property
    def g(self):
        """
        The green component of the color.

        """
        return self._color[1]

    @property
    def b(self):
        """
        The blue component of the color.

        """
        return self._color[2]

    @property
    def a(self):
        """
        The alpha component of the color.

        """
        return self._color[3]

    @r.setter
    def r(self):
        """
        Sets the red component of the color.

        """
        return self._color[0]

    @g.setter
    def g(self):
        """
        Sets the green component of the color.

        """
        return self._color[1]

    @b.setter
    def b(self):
        """
        Sets the blue component of the color.

        """
        return self._color[2]

    @a.setter
    def a(self):
        """
        Sets the alpha component of the color.

        """
        return self._color[3]

    @classmethod
    def from_hsv(cls, h, s, v):
        """
        Creates a color from HSV values.

        """
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

    OFFSETS = numpy.array(
        [
            [+1, -1, -1, +1, 0, 0, 0, 0, 0, 0],
            [+1, +1, -1, +1, 0, 0, 0, 0, 0, 0],
            [+1, +1, +1, +1, 0, 0, 0, 0, 0, 0],
            [+1, -1, +1, +1, 0, 0, 0, 0, 0, 0],
            [+1, -1, +1, 0, 0, +1, 0, 0, 0, 0],
            [+1, +1, +1, 0, 0, +1, 0, 0, 0, 0],
            [-1, +1, +1, 0, 0, +1, 0, 0, 0, 0],
            [-1, -1, +1, 0, 0, +1, 0, 0, 0, 0],
            [+1, +1, +1, 0, +1, 0, 0, 0, 0, 0],
            [+1, +1, -1, 0, +1, 0, 0, 0, 0, 0],
            [-1, +1, -1, 0, +1, 0, 0, 0, 0, 0],
            [-1, +1, +1, 0, +1, 0, 0, 0, 0, 0],
            [-1, -1, +1, -1, 0, 0, 0, 0, 0, 0],
            [-1, +1, +1, -1, 0, 0, 0, 0, 0, 0],
            [-1, +1, -1, -1, 0, 0, 0, 0, 0, 0],
            [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0],
            [-1, -1, -1, 0, 0, -1, 0, 0, 0, 0],
            [-1, +1, -1, 0, 0, -1, 0, 0, 0, 0],
            [+1, +1, -1, 0, 0, -1, 0, 0, 0, 0],
            [+1, -1, -1, 0, 0, -1, 0, 0, 0, 0],
            [-1, -1, +1, 0, -1, 0, 0, 0, 0, 0],
            [-1, -1, -1, 0, -1, 0, 0, 0, 0, 0],
            [+1, -1, -1, 0, -1, 0, 0, 0, 0, 0],
            [+1, -1, +1, 0, -1, 0, 0, 0, 0, 0],
        ],
        dtype=numpy.float32,
    )

    def __init__(self, x, y, z, hx, hy, hz, r, g, b, a):
        """
        Creates a voxel using (x, y, z) as the center of the voxel, (hx, hy, hz)
        as the half-widths of the voxel is the x, y, and z axis respectively,
        and (r, g, b, a) is a float32 RGBA color.

        """
        scale = numpy.array([hx, hy, hz, 1, 1, 1, 1, 1, 1, 1], dtype=numpy.float32)
        shift = numpy.array([x, y, z, 0, 0, 0, r, g, b, a], dtype=numpy.float32)

        product = numpy.multiply(Voxel.OFFSETS, scale[numpy.newaxis, :])

        self._vertices = numpy.add(product, shift[numpy.newaxis, :])

    @property
    def vertices(self):
        """
        The vertices of the voxel.

        """
        return self._vertices
