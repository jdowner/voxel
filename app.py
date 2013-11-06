#!/usr/bin/python2

import logging
import math
import sys
import random

import h5py
import numpy
import OpenGL
OpenGL.ERROR_ON_COPY = True

from OpenGL.arrays import vbo
from OpenGL.GL import *
from OpenGL.GLUT import *

# PyOpenGL 3.0.1 introduces this convenience module...
import OpenGL.GL.shaders as glsl

import core


class App(object):
    """
    This class provides the an interface that can be called by GLUT and exposes
    the high level functions that allow a user to interact with the scene.
    """

    def __init__(self, datafile):
        # Initialize the key bindings and application state
        self._keys = {}
        self._keys['\x1b'] = self.exit
        self._keys['w'] = self.move_forward
        self._keys['s'] = self.move_backward
        self._keys['a'] = self.move_left
        self._keys['d'] = self.move_right

        self._keys['W'] = self.pitch_forward
        self._keys['S'] = self.pitch_backward
        self._keys['A'] = self.yaw_left
        self._keys['D'] = self.yaw_right

        self._last_mouse_press = None
        self._key_pressed = None

        # Build the shader program so that the renderer can be created.
        program = core.ShaderProgram();
        program.load_vertex_shader('basic.vert')
        program.load_fragment_shader('basic.frag')
        program.build()

        self._renderer = core.Renderer(program, (800, 600))

        # Now that the renderer has been created we can load the data.
        self.load_data(datafile)

    @property
    def renderer(self):
        return self._renderer

    def load_data(self, filename):
        """
        Loads the data contained contained in the specified file. The file is
        expected to be an HDF5 file.

        """
        with h5py.File(filename) as fp:
            for group in fp:
                r, g, b, a = core.Color.from_hsv(random.random(), 0.5, 0.95)
                res = 10.0
                def make_voxel(x, y, z):
                    self._renderer.add_voxel(
                            core.Voxel(x, y, z, res, res, res, r, g, b, a))

                dataset = fp[group]
                points = set()
                for i in xrange(dataset.shape[0]):
                    points.add(tuple(1000.0 * dataset[i,:3]))

                for x, y, z in points:
                    make_voxel(x, y, z)

    def resize(self, width, height):
        """
        Called when the window is resized. The height and width of the new
        window size are passed in.

        """
        self.renderer.resize(width, height)

    def idle(self):
        """
        Called when the GLUT program is idle. This is where key presses are
        handled and we also call display to refresh the scene.

        """
        key = self._key_pressed
        if key is not None and key in self._keys:
            self._keys[key]()
        self.display()

    def display(self):
        """
        Called to refresh the scene.

        """
        self.renderer.display()

    def exit(self):
        """
        Forces the program to exit.

        """
        sys.exit(0)

    def move_forward(self):
        """
        Moves the camera forward.

        """
        self.renderer.camera.move_forward(20.0)

    def move_backward(self):
        """
        Moves the camera backward.

        """
        self.renderer.camera.move_backward(20.0)

    def move_down(self):
        """
        Moves the camera down.

        """
        self.renderer.camera.move_down(20.0)

    def move_up(self):
        """
        Moves the camera up.

        """
        self.renderer.camera.move_up(20.0)

    def move_left(self):
        """
        Moves the camera left.

        """
        self.renderer.camera.move_left(20.0)

    def move_right(self):
        """
        Moves the camera right.

        """
        self.renderer.camera.move_right(20.0)

    def roll_left(self):
        """
        Rotates the camera to the left along the forward/backward axis.

        """
        self.renderer.camera.roll(math.pi / 120.0)

    def roll_right(self):
        """
        Rotates the camera to the right along the forward/backward axis.

        """
        self.renderer.camera.roll(-math.pi / 120.0)

    def pitch_forward(self):
        """
        Rotates the camera forward along the left/right axis

        """
        self.renderer.camera.pitch(-math.pi / 120.0)

    def pitch_backward(self):
        """
        Rotates the camera backward along the left/right axis

        """
        self.renderer.camera.pitch(math.pi / 120.0)

    def yaw_left(self):
        """
        Rotates the camera left along the vertical axis

        """
        self.renderer.camera.yaw(math.pi / 120.0)

    def yaw_right(self):
        """
        Rotates the camera right along the vertical axis

        """
        self.renderer.camera.yaw(-math.pi / 120.0)

    def keyboard(self, *args):
        """
        Called when a key is pressed.

        """
        self._key_pressed = args[0]

    def keyboard_up(self, *args):
        """
        Called when a key is released.

        """
        self._key_pressed = None

    def mouse_move(self, *args):
        """
        Called when the mouse moves. The args parameters contains the 2D
        position of the mouse in window co-ordinates.

        """
        xn, yn = args
        if self._last_mouse_press:
            x0, y0 = self._last_mouse_press
            delta_x = xn - x0
            delta_y = yn - y0
            self.renderer.camera.yaw(-0.001 * delta_x * math.pi / 360.0)
            self.renderer.camera.pitch(-0.001 * delta_y * math.pi / 360.0)

    def mouse_press(self, *args):
        """
        Called when a mouse button is pressed or released. The args parameter
        is a tuple with 4 elements,

            (button, pressed, x, y)

        The first element indicates which button is pressed, the second is a 0
        (released) or 1 (pressed), and final two are the position of the mouse
        in window co-ordinates.

        """
        button, up, x, y = args
        if button == 0:
            self._last_mouse_press = (x, y) if not up else None
