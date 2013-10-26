#!/usr/bin/python2

import sys

from OpenGL.GL import (
        glGetString,
        GL_RENDERER,
        GL_VERSION,
        GL_VENDOR,
        GL_EXTENSIONS,
        )

from OpenGL.GLUT import *

import core
import app

def main():
    core.set_log_level('debug')

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutInitWindowPosition(0, 0)

    window = glutCreateWindow("--VOXEL--")
    a = app.App()

    glutDisplayFunc(a.display)
    glutIdleFunc(a.display)
    glutReshapeFunc(a.resize)
    glutKeyboardFunc(a.keyboard)
    glutMouseFunc(a.mouse_press)
    glutMotionFunc(a.mouse_move)

    for arg in sys.argv:
        if arg in ('-i', '--info'):
            print("GL_RENDERER   = %s" % (glGetString(GL_RENDERER),))
            print("GL_VERSION    = %s" % (glGetString(GL_VERSION),))
            print("GL_VENDOR     = %s" % (glGetString(GL_VENDOR),))
            print("GL_EXTENSIONS = ")
            for ext in sorted(glGetString(GL_EXTENSIONS).split()):
                print("  %s" % (ext,))

    glutMainLoop()

if __name__ == "__main__":
    main()
