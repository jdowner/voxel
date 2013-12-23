#!/usr/bin/python2

import logging
import math
import sys
import random

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
    This class provides an interface that can be called by GLUT and exposes
    the high level functions that allow a user to interact with the scene.
    """

    def __init__(self, config):
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

        window = config.app.window
        self._renderer = core.Renderer(program, (window.width, window.height))

    @property
    def renderer(self):
        return self._renderer

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


class Config(object):
    """
    A utility class for represented a dictionary of nested data as an object.
    """

    def __init__(self, datadict):
        """
        Creates a Config object using the providing dictionary.

        """
        for k, v in datadict.items():
            try:
                setattr(self, k, Config(v))
            except:
                setattr(self, k, v)

    def items(self):
        """
        Returns a list of the key-value pairs.

        """
        return self.__dict__.items()


keymap = {
        'key_escape': '\x1b',
        'key_a': 'a',
        'key_b': 'b',
        'key_c': 'c',
        'key_d': 'd',
        'key_e': 'e',
        'key_f': 'f',
        'key_g': 'g',
        'key_h': 'h',
        'key_i': 'i',
        'key_j': 'j',
        'key_k': 'k',
        'key_l': 'l',
        'key_m': 'm',
        'key_n': 'n',
        'key_o': 'o',
        'key_p': 'p',
        'key_q': 'q',
        'key_r': 'r',
        'key_s': 's',
        'key_t': 't',
        'key_u': 'u',
        'key_v': 'v',
        'key_w': 'w',
        'key_x': 'x',
        'key_y': 'y',
        'key_z': 'z',
        'key_A': 'A',
        'key_B': 'B',
        'key_C': 'C',
        'key_D': 'D',
        'key_E': 'E',
        'key_F': 'F',
        'key_G': 'G',
        'key_H': 'H',
        'key_I': 'I',
        'key_J': 'J',
        'key_K': 'K',
        'key_L': 'L',
        'key_M': 'M',
        'key_N': 'N',
        'key_O': 'O',
        'key_P': 'P',
        'key_Q': 'Q',
        'key_R': 'R',
        'key_S': 'S',
        'key_T': 'T',
        'key_U': 'U',
        'key_V': 'V',
        'key_W': 'W',
        'key_X': 'X',
        'key_Y': 'Y',
        'key_Z': 'Z',
   }
