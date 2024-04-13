"""
This module contains some utility functions for generating various transformation matrices
"""
# Import modules
import numpy as np


def translate(dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Builds and returns a 4x4 translation matrix

    Arguments:
        dx (float): Translation in x-direction
        dy (float): Translation in y-direction
        dz (float): Translation in z-direction

    Returns (np.ndarray): A numpy array containing a 4x4 translation matrix
    """
    return np.array([[1.0, 0.0, 0.0, dx],
                     [0.0, 1.0, 0.0, dy],
                     [0.0, 0.0, 1.0, dz],
                     [0.0, 0.0, 0.0, 1.0]],
                    dtype=np.float32)


def rotateX(angle: float) -> np.ndarray:
    """
    Builds and returns a 4x4 rotation matrix about the x-axis

    Arguments:
        angle (float): Angle to rotate about the x-axis, in radians

    Returns (np.ndarray): A numpy array containing a 4x4 translation matrix
    """
    return np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, np.cos(angle), -np.sin(angle), 0.0],
                     [0.0, np.sin(angle), np.cos(angle), 0.0],
                     [0.0, 0.0, 0.0, 1.0]],
                    dtype=np.float32)


def rotateY(angle: float) -> np.ndarray:
    """
    Builds and returns a 4x4 rotation matrix about the y-axis

    Arguments:
        angle (float): Angle to rotate about the y-axis, in radians

    Returns (np.ndarray): A numpy array containing a 4x4 translation matrix
    """
    return np.array([[np.cos(angle), 0.0, np.sin(angle), 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [-np.sin(angle), 0.0, np.cos(angle), 0.0],
                     [0.0, 0.0, 0.0, 1.0]],
                    dtype=np.float32)


def rotateZ(angle: float) -> np.ndarray:
    """
    Builds and returns a 4x4 rotation matrix about the z-axis

    Arguments:
        angle (float): Angle to rotate about the z-axis, in radians

    Returns (np.ndarray): A numpy array containing a 4x4 translation matrix
    """
    return np.array([[np.cos(angle), -np.sin(angle), 0.0, 0.0],
                     [np.sin(angle), np.cos(angle), 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]],
                    dtype=np.float32)


def zoom(scale: float) -> np.ndarray:
    """
    Builds and returns a 4x4 scaling matrix

    Arguments:
        scale (float): Scale factor in all direction

    Returns (np.ndarray): A numpy array containing a 4x4 scaling matrix
    """
    return np.ndarray([[scale, 0.0, 0.0, 0.0],
                       [0.0, scale, 0.0, 0.0],
                       [0.0, 0.0, scale, 0.0],
                       [0.0, 0.0, 0.0, scale]],
                      dtype=np.float32)
