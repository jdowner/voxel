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

