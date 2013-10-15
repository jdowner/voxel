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


class App(object):
    def __init__(self):
        self._renderer = core.Renderer(800, 600)
        self._renderer.load_vertex_shader('basic.vert')
        self._renderer.load_fragment_shader('basic.frag')
        self._renderer.build_program()

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

