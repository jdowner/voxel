varying vec3 vertex;
varying vec3 normal;
varying vec4 color;

void main()
{
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

	vertex = vec3(gl_ModelViewMatrix * gl_Vertex);
	normal = normalize(vec3(gl_NormalMatrix * gl_Normal));
	color = gl_Color;
}

