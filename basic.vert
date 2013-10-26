uniform vec3 light_position;
varying vec4 base_color;

void main()
{
	// specify the vertex position
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

  // convert vertex and light into camera co-ordinates
  vec3 vertex_position_camera = vec3(gl_ModelViewMatrix * gl_Vertex);
  vec3 light_position_camera = vec3(gl_ModelViewMatrix * vec4(light_position, 0.0));

  // calculate a light intensity based upon the displacement from the vertex to
  // the light source.
  vec3 displacement = vertex_position_camera - light_position_camera;
  float scale = clamp(100.0 / sqrt(dot(displacement, displacement)), 0.0, 1.0);

  // Set the base color
  base_color = vec4(scale, scale, scale, 1.0);
}

