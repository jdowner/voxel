#!/usr/bin/env python

import argparse
import logging
import sys

log = logging.getLogger('voxel.examples.box')

from OpenGL.GL import (
        glGetString,
        GL_RENDERER,
        GL_VERSION,
        GL_VENDOR,
        GL_EXTENSIONS,
        )

from OpenGL.GLUT import *

import yaml

import voxel.core
import voxel.app

fragment_shader = """
uniform vec3 light_position;
varying float intensity;

void main()
{
  vec4 color = gl_Color;
  color.x = intensity * color.x;
  color.y = intensity * color.y;
  color.z = intensity * color.z;

  gl_FragColor = color;
}
"""

vertex_shader = """
uniform vec3 light_position;
varying vec3 normal;
varying float intensity;

void main()
{
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
	gl_FrontColor = gl_Color;

  normal = normalize(gl_NormalMatrix * gl_Normal);
  intensity = min(max(dot(normalize(light_position), normal), 0.2), 0.9);
}

"""


class VoxelApp(voxel.app.App):
    def __init__(self):
        config = voxel.app.Config(yaml.load("""
            app:
                window:
                    height: 600
                    width: 800
                bindings:
                    key_escape: exit
                    key_w: move_forward
                    key_s: move_backward
                    key_a: move_left
                    key_d: move_right
                    key_W: pitch_forward
                    key_S: pitch_backward
                    key_A: yaw_left
                    key_D: yaw_right
                resolution: 100.0
                linear_speed: 20.0
                angular_speed: 0.02617993877
            """))

        # Create shader program
        program = voxel.core.ShaderProgram()
        program.compile_vertex_shader(vertex_shader)
        program.compile_fragment_shader(fragment_shader)
        program.build()

        # Define the window using configuration information
        window = config.app.window

        # Create the renderer
        renderer = voxel.core.Renderer(program, (window.width, window.height))

        super(VoxelApp, self).__init__(config, renderer)

        self.add_point(0, 0, 0, voxel.core.Color(1,1,1,1))

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    args = parser.parse_args()

    if args.verbose:
        level = logging.DEBUG if args.verbose else logging.ERROR
        logging.getLogger('voxel').setLevel(level)

    # Initialize the window
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutInitWindowPosition(0, 0)
    glutCreateWindow("--VOXEL--")
    glutIgnoreKeyRepeat(1)

    # Create application and bind functions to GLUT
    a = VoxelApp()

    glutDisplayFunc(a.display)
    glutIdleFunc(a.idle)
    glutReshapeFunc(a.resize)
    glutKeyboardFunc(a.keyboard)
    glutKeyboardUpFunc(a.keyboard_up)
    glutMouseFunc(a.mouse_press)
    glutMotionFunc(a.mouse_move)

    # Log diagnostic information
    log.debug("GL_RENDERER   = %s" % (glGetString(GL_RENDERER),))
    log.debug("GL_VERSION    = %s" % (glGetString(GL_VERSION),))
    log.debug("GL_VENDOR     = %s" % (glGetString(GL_VENDOR),))
    log.debug("GL_EXTENSIONS = ")
    for ext in sorted(glGetString(GL_EXTENSIONS).split()):
        voxel.core.log.debug("  %s" % (ext,))

    glutMainLoop()

if __name__ == "__main__":
    main()
