"""
This module contains the base class for all objects
"""
# Import modules
import OpenGL.GL as gl
import numpy as np
from OpenGL.GL.shaders import compileShader
from ..camera import Camera, Projection


class BaseObject:
    """Base class for all objects that can be rendered"""

    # Class shader storage
    __shader_program: int = -1
    __class_initialized: bool = False

    __current_projection: np.ndarray = np.identity(4)

    def __init__(self):
        self.__initialized: bool = False

    @property
    def initialized(self):
        return self.__initialized

    @classmethod
    def class_initialized(cls):
        return cls.__class_initialized

    @property
    def shader_program(self) -> int:
        return self.__shader_program

    @staticmethod
    def getFragmentShaderSource() -> str:
        """Returns the source code for the fragment shader"""
        pass

    @staticmethod
    def getVertexShaderSource() -> str:
        """Returns the source code for the vertex shader"""
        return """\
        #version 420 core
        
        out vec3 FragPos;
        
        uniform mat4 projection;
        uniform mat4 view;
        
        void main()
        {
            gl_Position = projection * view * vec4(position, 1.0);
            FragPos = position;
        }
        """

    @classmethod
    def compileShaders(cls) -> int:
        """Recompile the shader programs used to draw objects of this type"""
        # Load and compile shader sources
        compiled_vertex_shader = compileShader(cls.getVertexShaderSource(), gl.GL_VERTEX_SHADER)
        compiled_fragment_shader = compileShader(cls.getFragmentShaderSource(), gl.GL_FRAGMENT_SHADER)
        shader_program = gl.glCreateProgram()
        gl.glAttachShader(shader_program, compiled_vertex_shader)
        gl.glAttachShader(shader_program, compiled_fragment_shader)

        # Validate program
        gl.glValidateProgram(shader_program)

        # Return program pointer
        return shader_program

    @classmethod
    def class_initialize(cls):
        """Initializes the class for OpenGL rendering"""
        if cls.class_initialized():  # Don't try to initialize more than once
            return

        # Compile shader programs
        cls.__shader_program = cls.compileShaders()

    def initialize(self):
        """Initializes the object/class in OpenGL if it isn't already"""
        if self.initialized:  # Don't try to initialize more than once
            return

        # Perform class initialization first, if not already done
        self.class_initialize()

    @classmethod
    def setProjectionUniform(cls, projection: Projection):
        """Sets the projection uniform to match the provided camera object"""
        matrix = projection.matrix
        # Don't bother if the projection hasn't changed
        if np.all(matrix == cls.__current_projection):
            return

        # Locate projection uniform in shader and set it
        gl.glUseProgram(cls.shader_program)
        loc = gl.glGetUniformLocation(cls.shader_program, 'projection')
        gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, matrix.flatten())

    @classmethod
    def setCameraUniform(cls, camera: Camera):
        """Sets the camera uniform to match the provided camera object"""
        gl.glUseProgram(cls.shader_program)
        loc = gl.glGetUniformLocation(cls.shader_program, 'view')
        gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, camera.matrix.flatten())

    def render(self, camera: Camera, projection: Projection):
        """
        Renders this objects onto the current OpenGL context

        Arguments:
            camera (Camera): Camera view object that the scene is rendered from the perspective of
            projection (Projection): Projection to use to case 3D space onto a 2D plane
        """
        # Use shader program
        gl.glUseProgram(self.shader_program)

        # Set camera/projection matrices
        self.setCameraUniform(camera)
        self.setProjectionUniform(projection)

        # Further rendering done in subclass
        pass
