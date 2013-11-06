#!/usr/bin/python2

import argparse
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
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('datafile')

    args = parser.parse_args()

    if args.verbose:
        core.set_log_level('debug')

    # Initialize the window
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutInitWindowPosition(0, 0)
    glutCreateWindow("--VOXEL--")
    glutIgnoreKeyRepeat(1)

    # Create application and bind functions to GLUT
    a = app.App(datafile = args.datafile)

    glutDisplayFunc(a.display)
    glutIdleFunc(a.idle)
    glutReshapeFunc(a.resize)
    glutKeyboardFunc(a.keyboard)
    glutKeyboardUpFunc(a.keyboard_up)
    glutMouseFunc(a.mouse_press)
    glutMotionFunc(a.mouse_move)

    # Log diagnostic information
    core.log.debug("GL_RENDERER   = %s" % (glGetString(GL_RENDERER),))
    core.log.debug("GL_VERSION    = %s" % (glGetString(GL_VERSION),))
    core.log.debug("GL_VENDOR     = %s" % (glGetString(GL_VENDOR),))
    core.log.debug("GL_EXTENSIONS = ")
    for ext in sorted(glGetString(GL_EXTENSIONS).split()):
        core.log.debug("  %s" % (ext,))

    glutMainLoop()

if __name__ == "__main__":
    main()
