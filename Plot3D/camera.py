"""
This module contains classes and methods for working with cameras in 3D space
"""
# Import modules
import numpy as np
from enum import Enum, IntFlag, auto
from dataclasses import dataclass


class Direction(IntFlag):
    """A simple enum representing a direction relative to the camera"""
    forward = auto(int)
    backward = auto(int)
    left = auto(int)
    right = auto(int)
    up = auto(int)
    down = auto(int)


class Projection:
    """Base class for all projections onto the camera's view"""

    @property
    def matrix(self) -> np.ndarray:
        """Returns the 4x4 matrix defining this projection"""
        return np.identity(4)


@dataclass
class PerspectiveProjection(Projection):
    """Represents a perspective projection"""
    fov: float = 45.0          # Field of view, in degrees
    aspect: float = 1.0        # Aspect ratio of view
    near_plane: float = 0.1    # Near cutoff plane
    far_plane: float = 150.0   # Far cutoff plane

    @property
    def matrix(self) -> np.ndarray:
        """Returns the 4x4 matrix defining this projection"""
        fov = np.radians(self.fov)
        return np.array([[1/(self.aspect * np.tan(fov/2)), 0.0, 0.0, 0.0],
                         [0.0, 1.0/np.tan(fov/2), 0.0, 0.0],
                         [0.0, 0.0, -(self.far_plane + self.near_plane)/(self.far_plane-self.near_plane), -1.0],
                         [0.0, 0.0, -2*self.far_plane*self.near_plane/(self.far_plane-self.near_plane), 0.0]],
                        dtype=np.float32)


@dataclass
class OrthographicProjection(Projection):
    """Represents an orthographic projection"""
    width: float = 1.0         # Width of view
    height: float = 1.0        # Height of view
    near_plane: float = 0.1    # Near cutoff plane
    far_plane: float = 150.0   # Far cutoff plane

    @property
    def matrix(self) -> np.ndarray:
        """Returns the 4x4 matrix defining this projection"""
        return np.array([[2/self.width, 0.0, 0.0, 0.0],
                         [0.0, 2/self.height, 0.0, 0.0],
                         [0.0, 0.0, -1//(self.far_plane-self.near_plane), 0.0],
                         [0.0, 0.0, -self.near_plane/(self.far_plane-self.near_plane), 1.0]],
                        dtype=np.float32)



