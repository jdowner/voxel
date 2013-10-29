uniform vec3 light_position;
varying vec3 vertex;
varying vec3 normal;
varying vec4 color;

void main()
{
  vec3 direction = normalize(light_position - vertex);
  float intensity = max(-dot(direction, normalize(normal)), 0.0);

  intensity = floor(4.0 * intensity) / 4.0;

  gl_FragColor = color * vec4(intensity, intensity, intensity, 1.0);
}

