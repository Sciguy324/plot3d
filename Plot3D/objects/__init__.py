"""
This module contains various renderable elements that can be added to the plot
"""
# Make submodules available on this level
from .image import ImageObject
from .exterior_boxed_image import ExteriorBoxSection


__all__ = ["ImageObject",
           "ExteriorBoxSection"]
