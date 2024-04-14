"""
Example script for creating and showing a 3D plot
"""
# Import modules
from Plot3D import figure
import PyQt5.QtWidgets as qt
from sys import argv
from Plot3D.objects import ImageObject, ExteriorBoxSection, InteriorBoxSection, Crosshair
import numpy as np

# Qt application
app = qt.QApplication(argv)

# Create plot
window, canvas = figure()

# Add elements to plot
X, Y, Z = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100), np.linspace(0, 1, 100))
F = (X-0.5)**2+(Y-0.5)**2+(Z-0.5)**2
# vmin, vmax = F.min(), F.max()
# canvas.addObject(ImageObject(F[50], (0.0, 0.0, 0.0), 'z', vmin=vmin, vmax=vmax))
# canvas.addObject(ImageObject(F[:, 50], (0.0, 0.0, 0.0), 'y', vmin=vmin, vmax=vmax))
# canvas.addObject(ImageObject(F[:, :, 50], (0.0, 0.0, 0.0), 'x', vmin=vmin, vmax=vmax))
canvas.addObject(InteriorBoxSection(Z, (0.0, 0.0, 0.0)))
canvas.addObject(Crosshair(order='zxy'))

# Show plot
window.show()
app.exec_()
