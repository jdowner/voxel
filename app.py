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
    def __init__(self):
        program = core.ShaderProgram();
        program.load_vertex_shader('basic.vert')
        program.load_fragment_shader('basic.frag')
        program.build()

        self._renderer = core.Renderer(program, (800, 600))

        with h5py.File('clusters.hdf5') as fp:
            for group in fp:
                r, g, b, a = core.Color.from_hsv(random.random(), 0.5, 0.95)
                res = 10.0
                def make_voxel(x, y, z):
                    self._renderer.add_voxel(
                            core.Voxel(x, y, z, res, res, res, r, g, b, a))

                dataset = fp[group][...]
                points = set()

                for i in xrange(dataset.shape[0]):
                    x = 1000.0 * dataset[i,0]
                    y = 1000.0 * dataset[i,2]
                    z = 1000.0 * dataset[i,3]
                    points.add((x, y, z))

                for x, y, z in points:
                    make_voxel(x, y, z)

        self._keys = {}
        self._keys['\x1b'] = self.exit
        self._keys['w'] = self.move_forward
        self._keys['s'] = self.move_backward
        self._keys['a'] = self.move_left
        self._keys['d'] = self.move_right

        self._last_mouse_press = None
        self._key_pressed = None

    @property
    def renderer(self):
        return self._renderer

    def resize(self, width, height):
        self.renderer.resize(width, height)

    def idle(self):
        key = self._key_pressed
        if key is not None and key in self._keys:
            self._keys[key]()
        self.display()

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

    def roll_left(self):
        self.renderer.camera.roll(math.pi / 120.0)

    def roll_right(self):
        self.renderer.camera.roll(-math.pi / 120.0)

    def pitch_forward(self):
        self.renderer.camera.pitch(-math.pi / 120.0)

    def pitch_backward(self):
        self.renderer.camera.pitch(math.pi / 120.0)

    def yaw_left(self):
        self.renderer.camera.yaw(math.pi / 120.0)

    def yaw_right(self):
        self.renderer.camera.yaw(-math.pi / 120.0)

    def info(self):
        print('angle',self.renderer.camera.orientation.angle())
        print('axis',self.renderer.camera.orientation.axis())
        print('position',self.renderer.camera.position)

    def keyboard(self, *args):
        self._key_pressed = args[0]

    def keyboard_up(self, *args):
        self._key_pressed = None

    def mouse_move(self, *args):
        xn, yn = args
        if self._last_mouse_press:
            x0, y0 = self._last_mouse_press
            delta_x = xn - x0
            delta_y = yn - y0
            self.renderer.camera.yaw(-0.001 * delta_x * math.pi / 360.0)
            self.renderer.camera.pitch(-0.001 * delta_y * math.pi / 360.0)

    def mouse_press(self, *args):
        button, up, x, y = args
        if button == 0:
            self._last_mouse_press = (x, y) if not up else None


    def _random_voxels(self):
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

