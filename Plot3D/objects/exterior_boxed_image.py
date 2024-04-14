"""
This module contains a class for image-based renderable objects
"""
# Import modules
import numpy as np
from .base_surface_contour import BaseCrossSection


class ExteriorBoxSection(BaseCrossSection):
    """Cube that displays a cross-section of a 3D array that it intersects"""

    @staticmethod
    def base_model() -> np.ndarray:
        """Returns the base model"""
        vertices = np.array([[0.0, 0.0, 0.0],   # 0
                             [0.0, 0.0, 1.0],   # 1
                             [0.0, 1.0, 1.0],   # 2
                             [0.0, 1.0, 0.0],   # 3
                             [1.0, 0.0, 0.0],   # 4
                             [1.0, 0.0, 1.0],   # 5
                             [1.0, 1.0, 1.0],   # 6
                             [1.0, 1.0, 0.0]])  # 7

        triangles = np.array([[0, 1, 2], [0, 2, 3],
                              [1, 5, 6], [1, 6, 2],
                              [0, 1, 5], [0, 4, 5],
                              [0, 4, 7], [0, 3, 7],
                              [2, 3, 6], [3, 6, 7],
                              [4, 5, 7], [5, 6, 7]])

        return vertices[triangles].reshape(-1, 3)
