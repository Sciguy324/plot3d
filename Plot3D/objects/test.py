"""
This module contains a class for image-based renderable objects
"""
# Import modules
import ctypes
import numpy as np
from .base import BaseObject
from ..camera import Camera, Projection
import OpenGL.GL as gl


class TestObject(BaseObject):
    """Class for renderable images"""

    # Class shader storage
    __shader_program: int = -1
    __class_initialized: bool = False
    __current_projection: np.ndarray = np.identity(4)

    def __init__(self):
        # Call super-class constructor
        super().__init__()

        # Generate triangles for rendering
        self.triangles = np.array([[0.0, 0.0, -0.2],
                                   [1.0, 0.0, -0.2],
                                   [1.0, 1.0, -0.2]], dtype=np.float32)

        self.__vao = -1
        self.__vbo = -1

    @property
    def effective_radius(self):
        """Effective radius used when focusing on this object"""
        return np.max(self.triangles)*np.sqrt(2)

    @property
    def vao(self) -> int:
        return self.__vao

    @property
    def vbo(self) -> int:
        return self.__vbo

    @staticmethod
    def getFragmentShaderSource() -> str:
        """Returns the source code for the fragment shader"""
        return """\
        #version 430 core
        // Shader inputs and outputs
        out vec4 FragColor;
        in vec4 FragPos;
        
        void main()
        {
            float dist = length(FragPos);
            FragColor = vec4(abs(FragPos.z*1.0), 1.0, 0.0, 1.0);
        }
        """

    @staticmethod
    def getVertexShaderSource() -> str:
        """Returns the source code for the vertex shader"""
        return """\
        #version 430 core
        // Inputs provided by buffer objects
        layout (location = 0) in vec3 position;

        // Shader outputs
        out vec4 FragPos;
        
        // Uniforms
        uniform mat4 projection;
        uniform mat4 view;
        void main()
        {
            gl_Position = projection * view * vec4(position, 1.0);
            //gl_Position = view * vec4(position.xyz, 1.0);
            //gl_Position = projection * vec4(position, 1.0);
            //gl_Position = vec4(position.xy, 0.0, 1.0);
            FragPos = projection * view * vec4(position, 1.0);
        }
        """

    def setBuffers(self):
        """
        Sets the internal buffer objects to the provided arrays
        """
        # Bind and set buffer to triangle data
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.triangles.size*self.triangles.itemsize, self.triangles.flatten(), gl.GL_STATIC_DRAW)

    def initialize(self):
        """Initializes the object/class in OpenGL if it isn't already"""
        if self.initialized:  # Don't try to initialize more than once
            return

        # Perform class initialization first, if not already done
        self.class_initialize()

        # Set buffer data
        self.createBuffers()
        self.setBuffers()

        # Mark as initialized
        self._initialized = True

    def createBuffers(self):
        """Creates the internal buffer objects used in the shader"""
        # Destroy existing buffers (if there are any)
        if self.vao != -1:
            gl.glDeleteVertexArrays(self.vao)
            self.__vao = -1
        if self.vbo != -1:
            gl.glDeleteBuffers(self.vbo)
            self.__vbo = -1

        # Create vertex objects
        self.__vao = gl.glGenVertexArrays(1)
        self.__vbo = gl.glGenBuffers(1)

        # Set blank mesh
        blank_mesh = np.zeros(4, dtype=np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 0, blank_mesh, gl.GL_STATIC_DRAW)

        # Configure vertex attributes
        gl.glBindVertexArray(self.vao)
        buffer_offset = ctypes.c_void_p
        stride = 3 * blank_mesh.itemsize

        # Position (x, y, z)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
        gl.glEnableVertexAttribArray(0)

    def render(self, camera: Camera, projection: Projection):
        """
        Renders this object onto the current OpenGL context

        Arguments:
            camera (Camera): Camera view object that the scene is rendered from the perspective of
            projection (Projection): Projection to use to case 3D space onto a 2D plane
        """
        # Use shader and set camera/projection uniforms
        super().render(camera, projection)

        # Bind appropriate buffers
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        # Draw triangles
        gl.glPointSize(20)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
