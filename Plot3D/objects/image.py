"""
This module contains a class for image-based renderable objects
"""
# Import modules
from typing import Tuple
import ctypes
import numpy as np
from .base import BaseObject
from ..colormaps import Colormap, viridis
from ..camera import Camera, Projection
import OpenGL.GL as gl


class ImageObject(BaseObject):
    """Class for renderable images"""

    # Class shader storage
    __shader_program: int = -1
    __class_initialized: bool = False

    __current_projection: np.ndarray = np.identity(4)

    def __init__(self, array: np.ndarray, offset: Tuple[float, float, float], normal: str = 'z', width: float = None, height: float = None, vmin: float = None, vmax: float = None, cmap: Colormap = None):
        """
        Arguments:
            array (np.ndarray): A 2D numpy array containing the image to show
            normal (str): Direction that face's normal vector points in.  Must be either, 'x', 'y', or 'z'
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
        self.__array: np.ndarray = array
        self.offset: np.ndarray = np.array(offset)[:3]

        self.width: float = array.shape[1] if width is None else width
        self.height: float = array.shape[0] if height is None else height

        self.vmin = np.min(self.array) if vmin is None else vmin
        self.vmax = np.max(self.array) if vmax is None else vmax

        # Generate triangles for rendering
        if normal == 'z':
            positions = np.array([[0.0, 0.0, 0.0],  # First triangle
                                  [1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [1.0, 1.0, 0.0],  # Second triangle
                                  [0.0, 1.0, 0.0],
                                  [1.0, 0.0, 0.0]], dtype=np.float32)
            positions -= np.array([0.5, 0.5, 0.0])
            positions *= np.array([self.width, self.height, 0.0])

        elif normal == 'x':
            positions = np.array([[0.0, 0.0, 0.0],  # First triangle
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0],
                                  [0.0, 1.0, 1.0],  # Second triangle
                                  [0.0, 0.0, 1.0],
                                  [0.0, 1.0, 0.0]], dtype=np.float32)
            positions -= np.array([0.0, 0.5, 0.5])
            positions *= np.array([0.0, self.width, self.height])

        elif normal == 'y':
            positions = np.array([[0.0, 0.0, 0.0],  # First triangle
                                  [1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0],
                                  [1.0, 0.0, 1.0],  # Second triangle
                                  [0.0, 0.0, 1.0],
                                  [1.0, 0.0, 0.0]], dtype=np.float32)
            positions -= np.array([0.5, 0.0, 0.5])
            positions *= np.array([self.width, 0.0, self.height])

        else:
            raise ValueError(f"Unexpected normal direction: {normal}.  Must be either 'x', 'y', or 'z'")

        positions += self.offset

        # Rotate positions to match normal vector
        uv = np.array([[0.0, 0.0],  # First triangle
                       [1.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 1.0],  # Second triangle
                       [0.0, 1.0],
                       [1.0, 0.0]], dtype=np.float32)

        triangles = np.zeros((len(positions), 5), dtype=np.float32)
        triangles[:, 0] = positions.T[0]
        triangles[:, 1] = positions.T[1]
        triangles[:, 2] = positions.T[2]
        triangles[:, 3] = uv.T[0]
        triangles[:, 4] = uv.T[1]
        self.triangles = triangles

        # Vertex buffer attributes
        self.__vao: int = -1
        self.__vbo: int = -1

        # Texture attributes
        self.__tex: int = -1
        self.colormap = viridis if cmap is None else cmap

    @property
    def effective_radius(self):
        """Effective radius used when focusing on this object"""
        return np.sqrt(self.width**2 + self.height**2)

    @property
    def array(self) -> np.ndarray:
        return self.__array

    @property
    def vao(self) -> int:
        return self.__vao

    @property
    def vbo(self) -> int:
        return self.__vbo

    @property
    def tex(self) -> int:
        return self.__tex

    @property
    def cmap_tex(self) -> int:
        return self.colormap.tex

    @staticmethod
    def getFragmentShaderSource() -> str:
        """Returns the source code for the fragment shader"""
        return """\
        #version 410 core
        // Shader inputs and outputs
        out vec4 FragColor;
        in vec2 FragPos;
        in vec3 TexSize;
        
        // Uniforms
        uniform sampler1D cmap;
        uniform sampler2D tex;

        vec3 colormap(float value)
        {
            // Clamp to allowed range [0.0-1.0]
            value = clamp(value, 0.0, 1.0);
            
            // Convert value to color using cmap
            return texture(cmap, value).xyz;
        }

        void main()
        {
            // Get value from tex at this position
            // float value = texture(tex, FragPos).x;
            float value = texture(tex, FragPos).x;
            
            // Convert to color using colormap and output
            FragColor = vec4(colormap(value), 1.0);
        }
        """

    @staticmethod
    def getVertexShaderSource() -> str:
        """Returns the source code for the vertex shader"""
        return """\
        #version 410 core
        // Inputs provided by buffer objects
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec2 uv;

        // Shader outputs
        out vec2 FragPos;
        
        // Uniforms
        uniform mat4 projection;
        uniform mat4 view;
        void main()
        {
            gl_Position = projection * view * vec4(position, 1.0);
            FragPos = uv;
        }
        """

    def setBuffers(self):
        """
        Sets the internal buffer objects to the provided arrays
        """
        # Bind and set buffer to triangle data
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.triangles.size*self.triangles.itemsize, self.triangles.flatten(), gl.GL_STATIC_DRAW)

    def createTexture(self):
        """
        Creates and configures the texture object used to store to the image
        """
        if self.tex is None or self.tex == -1:
            self.__tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

    def setTexture(self, array: np.ndarray):
        """
        Sets the internal image texture
        """
        # Prepare new texture if not set
        if self.tex is None or self.tex == -1:
            self.createTexture()

        # Set texture data
        data = ((np.array(array)-self.vmin)/(self.vmax-self.vmin)).flatten().astype(np.float32)
        height, width = array.shape
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, width, height, 0, gl.GL_RED, gl.GL_FLOAT, data)

    def initialize(self):
        """Initializes the object/class in OpenGL if it isn't already"""
        if self.initialized:  # Don't try to initialize more than once
            return

        # Perform class initialization first, if not already done
        self.class_initialize()

        # Set buffer data
        self.createBuffers()
        self.setBuffers()
        self.setTexture(self.array)

        # Set texture locations
        cmap_loc = gl.glGetUniformLocation(self.shader_program, "cmap")
        tex_loc = gl.glGetUniformLocation(self.shader_program, "tex")
        gl.glUniform1i(cmap_loc, 0)
        gl.glUniform1i(tex_loc, 1)

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
        stride = (3+2) * blank_mesh.itemsize

        # Position (x, y, z)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
        gl.glEnableVertexAttribArray(0)

        # UV position (u, v)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, buffer_offset(3*blank_mesh.itemsize))
        gl.glEnableVertexAttribArray(1)

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
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_1D, self.cmap_tex)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        # Draw triangles
        # data = np.zeros(5, dtype=np.float32)
        # gl.glGetBufferSubData(gl.GL_ARRAY_BUFFER, 0, 5, data)
        # data = gl.glGetBufferSubData(gl.GL_ARRAY_BUFFER, 0, 20*6)
        # print(data.view(np.float32).reshape(-1, 5))
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(self.triangles))
