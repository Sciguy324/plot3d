"""
Plot3D is a Python library for rendering 3D plots using OpenGL
"""
# Imports
from typing import Tuple
from .window import QtWindow
from .canvas import Canvas

__all__ = ["figure"]


# Convenience functions
def figure() -> Tuple[QtWindow, Canvas]:
    """Creates and returns a window-canvas pair"""
    new_window = QtWindow()
    return new_window, new_window.view_widget.canvas

