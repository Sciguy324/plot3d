"""
This module contains a class for crosshair objects
"""
# Import modules
from typing import Tuple
import ctypes
import numpy as np
from .base import BaseObject
from ..camera import Camera, Projection
from ..transform import translate, rotateX, rotateY, zoom
import OpenGL.GL as gl


class Crosshair(BaseObject):
    """Class for crosshairs"""

    # Class shader storage
    __shader_program: int = -1
    __class_initialized: bool = False

    __current_projection: np.ndarray = np.identity(4)

    def __init__(self, screen_position: Tuple[float, float] = (-0.5, -0.5), scale: float = 0.125, order: str = 'xyz'):
        """
        Arguments:
            screen_position (Tuple[float, float]): Position on screen, in normal coordinates to show crosshair
            scale (float): Factor to scale crosshair by
            order (str): Order that the axes should appear in
        """
        # Call super-class constructor
        super().__init__()

        # Save attributes and enforce type
        self.__position: Tuple[float, float] = float(screen_position[0]), float(screen_position[1])
        self.__scale: float = float(scale)

        # Generate lines for rendering
        # Format: X, Y, Z, R, G, B
        self.lines = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                               [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]],
                              dtype=np.float32)

        # Assign colors based on ordering
        color = {'x': [1.0, 0.0, 0.0],
                 'y': [0.0, 1.0, 0.0],
                 'z': [0.0, 0.0, 1.0]}

        for i, axis in enumerate(order):
            c = np.array(color.get(order[i], [0.0, 0.0, 0.0]))
            self.lines[2*i, 3:] = c
            self.lines[2*i+1, 3:] = c

        # Vertex buffer attributes
        self.__vao: int = -1
        self.__vbo: int = -1

    @property
    def effective_radius(self):
        """Effective radius used when focusing on this object"""
        return 1.0

    @property
    def position(self) -> Tuple[float, float]:
        return self.__position[0], self.__position[1]

    @position.setter
    def position(self, value: Tuple[float, float]):
        self.__position = float(value[0]), float(value[1])  # Enforce array size and element type

    @property
    def scale(self) -> float:
        return self.__scale

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
        #version 410 core
        // Shader inputs and outputs
        out vec4 Color;
        in vec3 FragColor;

        void main()
        {
            Color = vec4(FragColor, 1.0);
        }
        """

    @staticmethod
    def getVertexShaderSource() -> str:
        """Returns the source code for the vertex shader"""
        return """\
        #version 410 core
        // Inputs provided by buffer objects
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 color;

        // Shader outputs
        out vec3 FragColor;
        
        // Uniforms
        uniform mat4 projection;
        uniform mat4 view;
        uniform vec2 screen_pos;
        void main()
        {
            gl_Position = projection * (view * vec4(position, 1.0) + vec4(0.0, 0.0, -1.0, 0.0)) + vec4(screen_pos, 0.0, 0.0);
            FragColor = color;
        }
        """

    def setBuffers(self):
        """
        Sets the internal buffer objects to the provided arrays
        """
        # Bind and set buffer to triangle data
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.lines.size*self.lines.itemsize, self.lines.flatten(), gl.GL_STATIC_DRAW)

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
        stride = (3+3) * blank_mesh.itemsize

        # Position (x, y, z)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
        gl.glEnableVertexAttribArray(0)

        # Color (R, G, B)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, buffer_offset(3*blank_mesh.itemsize))
        gl.glEnableVertexAttribArray(1)

    def setCameraUniform(self, camera: Camera):
        """Sets the camera uniform to match the provided camera object"""
        gl.glUseProgram(self.shader_program)
        loc = gl.glGetUniformLocation(self.shader_program, 'view')
        if loc != -1:
            transform = np.identity(4, 'f')
            transform = transform @ rotateX(float(np.radians(camera.phi)))
            transform = transform @ rotateY(float(np.radians(camera.theta)))
            transform = transform @ zoom(self.scale)

            gl.glUniformMatrix4fv(loc, 1, gl.GL_TRUE, transform.flatten())

    def render(self, camera: Camera, projection: Projection):
        """
        Renders this object onto the current OpenGL context

        Arguments:
            camera (Camera): Camera view object that the scene is rendered from the perspective of
            projection (Projection): Projection to use to case 3D space onto a 2D plane
        """
        # Use shader and set camera/projection uniforms
        super().render(camera, projection)

        # Set offset
        pos_loc = gl.glGetUniformLocation(self.shader_program, 'screen_pos')
        if pos_loc != -1:
            gl.glUniform2fv(pos_loc, 1, np.array(self.position, dtype=np.float32))

        # Bind appropriate buffers
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        # Draw lines
        gl.glDrawArrays(gl.GL_LINES, 0, len(self.lines))
