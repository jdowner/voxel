#!/usr/bin/python2

import logging
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

    @property
    def renderer(self):
        return self._renderer

    def resize(self, width, height):
        self.renderer.resize(width, height)

    def display(self):
        self.renderer.display()

    def keyboard(self, *args):
        if args[0] == '\x1b':
            sys.exit()

    def mouse_move(self, *args):
        print(args)

    def mouse_press(self, *args):
        print(args)
