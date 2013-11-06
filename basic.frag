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

