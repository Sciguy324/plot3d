"""
This module contains a class for image-based renderable objects
"""
# Import modules
import numpy as np
from typing import Tuple
from .base import BaseObject


class ImageObject(BaseObject):

    def __init__(self, array: np.ndarray, normal: Tuple[float, float, float], width: float, height: float):
        """
        Arguments:
            array (np.ndarray): A 2D numpy array containing the image to show
            normal (Tuple[float, float, float]): Normal vector that the image face is pointing in
            width (float): Width of rendered image
            height (float): Height of rendered image
        """
        # Call super-class constructor
        super().__init__()

        # Save attributes
        self.array: np.ndarray = array
        self.normal: np.ndarray = np.array(normal)[:3]
        self.width: float = width
        self.height: float = height



