"""
Example script for creating and showing a 3D plot
"""
# Import modules
from Plot3D import figure
import PyQt5.QtWidgets as qt
from sys import argv
from Plot3D.camera import OrthographicProjection
from Plot3D.objects.test import TestObject
from Plot3D.objects import ImageObject
import numpy as np

# Qt application
app = qt.QApplication(argv)

# Create plot
window, canvas = figure()

# Add elements to plot
X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 200))
image = ImageObject(X, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
canvas.addObject(image)

# Show plot
window.show()
app.exec_()
