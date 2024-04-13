"""
This module contains a class for image-based renderable objects
"""
# Import modules
from typing import Tuple
import ctypes
import numpy as np
from .base import BaseObject
from ..camera import Camera, Projection
import OpenGL.GL as gl


class ImageObject(BaseObject):
    """Class for renderable images"""

    # Class shader storage
    __shader_program: int = -1
    __class_initialized: bool = False

    __current_projection: np.ndarray = np.identity(4)

    def __init__(self, array: np.ndarray, normal: Tuple[float, float, float], offset: Tuple[float, float, float], width: float = None, height: float = None, vmin: float = None, vmax: float = None, cmap: Colormap = None):
        """
        Arguments:
            array (np.ndarray): A 2D numpy array containing the image to show
            normal (Tuple[float, float, float]): Normal vector that the image face is pointing in
            offset (Tuple[float, float, float]): Location of the center of the image
            width (float): Width of rendered image.  Defaults to the number of columns in array if not provided.
            height (float): Height of rendered image.  Defaults to the number of rows in array if not provided
            vmin (float): Minimum bound of colormap.  Defaults to array minimum if not provided
            vmax (float): Maximum bound of colormap.  Defaults to array maximum if not provided
            cmap (Colormap): Colormap used when displaying image
        """
        # Call super-class constructor
        super().__init__()

        # Verify attributes
        array = np.array(array, dtype=np.float32)
        if len(array.shape) != 2:
            raise ValueError(f"Expected a 2D array for 'array' (was {len(array.shape)}-dimensional)")

        # Save attributes
        self.array: np.ndarray = array
        self.normal: np.ndarray = np.array(normal)[:3]
        self.offset: np.ndarray = np.array(offset)[:3]

        self.width: float = array.shape[1] if width is None else width
        self.height: float = array.shape[0] if height is None else height

        self.vmin = np.min(self.array) if vmin is None else vmin
        self.vmax = np.min(self.array) if vmax is None else vmax

        # Generate triangles for rendering
        positions = np.array([[0.0, 0.0, 0.0],  # First triangle
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [1.0, 1.0, 0.0],  # Second triangle
                              [0.0, 1.0, 0.0],
                              [1.0, 0.0, 0.0]], dtype=np.float32)

        size = np.array([self.width, self.height, 0.0])
        positions = positions[:, ]*size
        positions = positions[:, ] - size*np.array([0.5, 0.5, 0.0]) + self.offset

        # Rotate positions to match normal vector
        uv = np.array([[0.0, 0.0],  # First triangle
                       [1.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 1.0],  # Second triangle
                       [0.0, 1.0],
                       [1.0, 0.0]], dtype=np.float32)

        triangles = np.zeros((len(positions), 5), dtype=np.float32)
        triangles[:, 0] = positions.T[0]
        triangles[:, 0] = positions.T[1]
        triangles[:, 0] = positions.T[2]
        self.triangles = triangles.flatten()

        # Vertex buffer attributes
        self.__vao: int = -1
        self.__vbo: int = -1

        self.colormap = cmap

    @property
    def vao(self) -> int:
        return self.__vao

    @property
    def vbo(self) -> int:
        return self.__vbo

    @property
    def cmap_tex(self) -> int:
        return self.colormap.tex

    @staticmethod
    def getFragmentShaderSource() -> str:
        """Returns the source code for the fragment shader"""
        return """\
        #version 430 core
        // Shader inputs and outputs
        out vec4 FragColor;
        in vec3 FragPos;
        in vec3 TexSize;
        
        // Uniforms
        uniform float vmin;
        uniform float vmax;
        uniform sampler1D cmap;
        uniform sampler2D tex;

        void main()
        {
            FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
        """

    @staticmethod
    def getVertexShaderSource() -> str:
        """Returns the source code for the vertex shader"""
        return """\
        #version 420 core
        // Inputs provided by buffer objects
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec2 uv;

        // Shader outputs
        out vec3 FragPos;
        
        // Uniforms
        uniform mat4 projection;
        uniform mat4 view;
        void main()
        {
            gl_Position = projection * view * vec4(position, 1.0);
            FragPos = uv;
        }
        """

    def setBuffers(self, *arrays: np.ndarray):
        """
        Sets the internal buffer objects to the provided arrays
        """
        # Bind and set buffer to triangle data
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 0, self.triangles, gl.GL_STATIC_DRAW)

    def initialize(self):
        super().initialize()
        self.setBuffers()

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
        stride = (3+2) * blank_mesh.itemsize

        # Position (x, y, z)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
        gl.glEnableVertexAttribArray(0)

        # UV position (u, v)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, buffer_offset(3*blank_mesh.itemsize))
        gl.glEnableVertexAttribArray(1)

    def render(self, camera: Camera, projection: Projection):
        """
        Renders this objects onto the current OpenGL context

        Arguments:
            camera (Camera): Camera view object that the scene is rendered from the perspective of
            projection (Projection): Projection to use to case 3D space onto a 2D plane
        """
        # Use shader and set camera/projection uniforms
        super().render(camera, projection)

        # Render triangles forming this object

