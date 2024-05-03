"""
This module contains the rendering code for OpenGL-based rendering
"""
# Import modules
from typing import List, Tuple
from time import time
import numpy as np
import OpenGL.GL as gl
from .camera import Camera, Projection, PerspectiveProjection
from .objects.base import BaseObject
from .animation import Animation


class Canvas:
    """
    Main OpenGL renderer class for viewing/interaction
    """
    def __init__(self, window_width: int, window_height: int):
        self._is_initialized: bool = False

        # Screen size
        self.width = int(window_width)
        self.height = int(window_height)

        # Rendering settings
        self.__background_color: tuple = (0.0, 0.0, 0.0, 1.0)
        self.__vmin: float = 0.0
        self.__vmax: float = 1.0

        # Object data
        self.rendered_objects: List[BaseObject] = []
        self.animations: List[Animation] = []

        # Initialize view
        self.start_time: float = time()
        self._projection = None
        self.camera = None
        self.initView()

    def initView(self):
        """Initializes camera view/projection"""
        self._projection = PerspectiveProjection()
        self.camera = Camera()

    def resetView(self):
        """Resets the camera view"""
        radius = max(renderable.effective_radius for renderable in self.rendered_objects)
        self.camera.focusView([0.0, 0.0, 0.0], radius, self.projection)

    @property
    def projection(self) -> Projection:
        return self._projection

    @projection.setter
    def projection(self, new_projection):
        """Sets the projection used by the renderer"""
        self._projection = new_projection

    @property
    def background_color(self) -> Tuple[float, float, float, float]:
        return self.__background_color

    @background_color.setter
    def background_color(self, value: Tuple[float, float, float, float]):
        self.__background_color = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        # Pass along to OpenGL if initialized
        if self.initialized:
            gl.glClearColor(*self.background_color)

    @property
    def vmin(self) -> float:
        return self.__vmin

    @property
    def vmax(self) -> float:
        return self.__vmax

    @property
    def initialized(self):
        return self._is_initialized

    def initialize(self):
        """Initializes OpenGL settings"""
        if self.initialized:
            return

        # Enable depth buffer
        gl.glDepthFunc(gl.GL_LESS)
        gl.glEnable(gl.GL_DEPTH_TEST)

        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glClearColor(*self.background_color)

        # Configure viewport
        gl.glViewport(0, 0, self.width, self.height)

        # Print debug information
        version = gl.glGetString(gl.GL_VERSION)
        version = version.decode('ascii') if isinstance(version, bytes) else version
        print(f"Using OpenGL version '{version}'")

        # Mark as initialized
        self._is_initialized = True

    def addObject(self, renderable: BaseObject):
        """Adds a renderable object to the canvas"""
        self.rendered_objects.append(renderable)

    def clearObjects(self):
        """Removes all renderable objects from the canvas"""
        self.rendered_objects = []

    def addAnimation(self, animation: Animation):
        """Adds an animation function to the canvas"""
        self.animations.append(animation)

    def clearAnimations(self):
        """Clears all animation functions from the canvas"""
        self.animations = []

    def resize(self, width: float, height: float):
        """Resizes the internal viewport"""
        self.width, self.height = width, height
        gl.glViewport(0, 0, self.width, self.height)
        self.projection.setSize(height, width)

    def clearDrawn(self):
        """Clears the canvas of all rendered content"""
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    def render(self):
        """Renders the canvas to the current OpenGL context"""
        # Attempt to initialize if not already initialized
        if not self.initialized:
            self.initialize()

        # Clear screen
        self.clearDrawn()

        # Execute animation functions
        for animation in self.animations:
            animation(time() - self.start_time)

        # Draw all renderable objects
        for renderable in self.rendered_objects:
            renderable.render(self.camera, self.projection)
