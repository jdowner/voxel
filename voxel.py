#!/usr/bin/python2

from OpenGL.GLUT import *

import core
import app

def main():
    core.set_log_level('debug')

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutInitWindowPosition(0, 0)

    window = glutCreateWindow("--3d-Viewer--")
    a = app.App()

    glutDisplayFunc(a.display)
    glutIdleFunc(a.display)
    glutReshapeFunc(a.resize)
    glutKeyboardFunc(a.keyboard)
    glutMouseFunc(a.mouse_press)
    glutMotionFunc(a.mouse_move)

    glutMainLoop()

if __name__ == "__main__":
    main()
