"""
This module contains a class for image-based renderable objects
"""
# Import modules
import numpy as np
from .base_surface_contour import BaseCrossSection


class InteriorBoxSection(BaseCrossSection):
    """Cube interior that displays a cross-section of a 3D array that it intersects"""

    @staticmethod
    def base_model() -> np.ndarray:
        """Returns the base model"""
        vertices = np.array([[0.0, 0.0, 0.5],   # 0
                             [0.0, 1.0, 0.5],   # 1
                             [1.0, 0.0, 0.5],   # 2
                             [1.0, 1.0, 0.5],   # 3
                             [0.0, 0.5, 0.0],   # 4
                             [0.0, 0.5, 1.0],   # 5
                             [1.0, 0.5, 0.0],   # 6
                             [1.0, 0.5, 1.0],   # 7
                             [0.5, 0.0, 0.0],   # 8
                             [0.5, 0.0, 1.0],   # 9
                             [0.5, 1.0, 0.0],   # 10
                             [0.5, 1.0, 1.0]])  # 11

        triangles = np.array([[0, 1, 2], [1, 3, 2],
                              [4, 5, 6], [5, 7, 6],
                              [8, 9, 10], [9, 11, 10]])

        return vertices[triangles].reshape(-1, 3)
