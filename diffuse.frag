vec4 diffuse_color = vec4(1.0, 0.0, 0.0, 1.0);
vec4 specular_color = vec4(1.0, 1.0, 1.0, 1.0);

varying float specular_intensity;
varying float diffuse_intensity;

void main()
{
	//gl_FragColor = gl_Color;
	gl_FragColor = diffuse_color * diffuse_intensity +
	  specular_color * specular_intensity;
}
