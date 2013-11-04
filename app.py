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
    def __init__(self):
        program = core.ShaderProgram();
        program.load_vertex_shader('basic.vert')
        program.load_fragment_shader('basic.frag')
        program.build()

        self._renderer = core.Renderer(program, (800, 600))

        random.seed(1)

        def make_voxel(x, y, z):
            r, g, b, a = core.Color.from_hsv(random.random(), 0.5, 0.95)
            self._renderer.add_voxel(core.Voxel(x, y, z, 10, 10, 10, r, g, b, a))

        points = set()
        for i in xrange(4000):
            x = random.randint(-1000, 1000)
            y = random.randint(-1000, 1000)
            z = random.randint(1000, 2000)
            x = math.floor(x / 20.0) * 20.0
            y = math.floor(y / 20.0) * 20.0
            z = math.floor(z / 20.0) * 20.0
            points.add((x, y, z))

        for x, y, z in points:
            make_voxel(x, y, z)

        self._keys = {}
        self._keys['\x1b'] = self.exit
        self._keys['k'] = self.move_forward
        self._keys['j'] = self.move_backward
        self._keys['J'] = self.move_down
        self._keys['K'] = self.move_up
        self._keys['h'] = self.move_left
        self._keys['l'] = self.move_right
        self._keys['H'] = self.yaw_left
        self._keys['L'] = self.yaw_right
        self._keys['i'] = self.info

    @property
    def renderer(self):
        return self._renderer

    def resize(self, width, height):
        self.renderer.resize(width, height)

    def display(self):
        self.renderer.display()

    def exit(self):
        sys.exit(0)

    def move_forward(self):
        self.renderer.camera.move_forward(20.0)

    def move_backward(self):
        self.renderer.camera.move_backward(20.0)

    def move_down(self):
        self.renderer.camera.move_down(20.0)

    def move_up(self):
        self.renderer.camera.move_up(20.0)

    def move_left(self):
        self.renderer.camera.move_left(20.0)

    def move_right(self):
        self.renderer.camera.move_right(20.0)

    def yaw_left(self):
        self.renderer.camera.yaw(math.pi / 120.0)

    def yaw_right(self):
        self.renderer.camera.yaw(-math.pi / 120.0)

    def info(self):
        print('angle',self.renderer.camera.orientation.angle())
        print('axis',self.renderer.camera.orientation.axis())
        print('position',self.renderer.camera.position)

    def keyboard(self, *args):
        key = args[0]
        if key in self._keys:
            self._keys[key]()

    def mouse_move(self, *args):
        pass

    def mouse_press(self, *args):
        pass
