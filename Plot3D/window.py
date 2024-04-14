"""
This module contains the main window class used for containing OpenGL canvas instances
"""
# Import modules
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qt
import PyQt5.QtGui as qtg
from .camera import Direction
from .canvas import Canvas
from dataclasses import dataclass
from time import time


@dataclass
class MoveControlState:
    direction: Direction = 0    # Direction to move in as requested by keyboard input
    slow: bool = False          # Whether to move at half-speed
    pan_move: tuple = (0, 0)    # Direction to pan in relative to camera view
    pan_rotate: tuple = (0, 0)  # Direction to rotate in relative to camera view
    zoom: float = 0.0           # Amount to change zoom factor by


class QtGLWidget(qt.QOpenGLWidget):

    glInitializedEvent = qtc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        # Call super-class constructor
        super().__init__(*args, **kwargs)

        # Input/control variables
        self.old_mouse_position = None
        self.time = time()
        self.previous_time = time()
        self.move_state = MoveControlState()

        # OpenGL canvas
        self.canvas = Canvas(self.width(), self.height())

        # Declare event timer
        self.update_timer = qtc.QTimer(self)
        self.update_timer.setInterval(1)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start()

    @property
    def time_delta(self) -> float:
        return self.time - self.previous_time

    def mouseMoveEvent(self, event: qtg.QMouseEvent) -> None:
        """Event handler for moving the mouse"""
        pos = (event.x(), event.y())

        # Compute velocity
        if self.old_mouse_position is None:
            self.old_mouse_position = pos
            return
        else:
            vel = (i-j for i, j in zip(pos, self.old_mouse_position))

        # Convert input state to into an action
        left_clicking = event.buttons() & qtc.Qt.LeftButton
        middle_clicking = event.buttons() & qtc.Qt.MiddleButton
        alternative = event.modifiers() & qtc.Qt.ControlModifier

        primary_left_click = left_clicking and not alternative
        alternative_left_click = left_clicking and alternative

        rotating = primary_left_click
        panning = middle_clicking or alternative_left_click

        # Left-Click: pivot view
        if rotating:
            self.move_state.pan_rotate = vel

        # Middle-Click: pan view
        #   Alternative: Left-Click + Control Key
        if panning:
            self.move_state.pan_move = vel

        # Save mouse position for later and conclude the event
        self.old_mouse_position = pos
        event.accept()

    def mousePressEvent(self, event: qtg.QMouseEvent) -> None:
        """Set focus upon clicking widget"""
        self.old_mouse_position = (event.x(), event.y())
        self.setFocus()

    def keyPressEvent(self, event: qtg.QKeyEvent) -> None:
        """Handle key press events"""
        # Prevent repeated pressing due to auto-repeat
        if event.isAutoRepeat():
            return

        # Update key state table
        if event.key() == qtc.Qt.Key_W:
            self.move_state.direction |= Direction.forward
        elif event.key() == qtc.Qt.Key_A:
            self.move_state.direction |= Direction.left
        elif event.key() == qtc.Qt.Key_S:
            self.move_state.direction |= Direction.backward
        elif event.key() == qtc.Qt.Key_D:
            self.move_state.direction |= Direction.right
        elif event.key() == qtc.Qt.Key_Space:
            self.move_state.direction |= Direction.up
        elif event.key() == qtc.Qt.Key_Shift:
            self.move_state.direction |= Direction.down
        elif event.key() == qtc.Qt.Key_Q:
            self.move_state.slow = not self.move_state.slow

        # Print debug info
        elif event.key() == qtc.Qt.Key_P:
            print(self.canvas.camera)

        # Return to home view
        elif event.key() == qtc.Qt.Key_H:
            self.canvas.resetView()

        # Allow event to pass through
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: qtg.QKeyEvent) -> None:
        """Handle key release events"""
        # Prevent repeated releasing due to auto-repeat
        if event.isAutoRepeat():
            return

        # Update key state table
        if event.key() == qtc.Qt.Key_W:
            self.move_state.direction &= ~Direction.forward
        elif event.key() == qtc.Qt.Key_A:
            self.move_state.direction &= ~Direction.left
        elif event.key() == qtc.Qt.Key_S:
            self.move_state.direction &= ~Direction.backward
        elif event.key() == qtc.Qt.Key_D:
            self.move_state.direction &= ~Direction.right
        elif event.key() == qtc.Qt.Key_Space:
            self.move_state.direction &= ~Direction.up
        elif event.key() == qtc.Qt.Key_Shift:
            self.move_state.direction &= ~Direction.down

        # Release the focus if the user pressed 'escape'
        if event.key() == qtc.Qt.Key_Escape:
            self.clearFocus()

        # Allow the key release event to pass through
        super().keyReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        """Mouse scroll wheel event handler"""
        # Zoom in/out
        self.move_state.zoom = event.angleDelta().y()

    def applyMoveStates(self):
        """Applies the current movement control state"""
        # Cap time delta to 1.0
        dt = min(self.time_delta, 1.0)

        # Check for slow-move mode
        if self.move_state.slow:
            dt *= 0.5

        # Pivot view, reset pivot amount after
        self.canvas.camera.rotateView(*self.move_state.pan_rotate)
        self.move_state.pan_rotate = (0, 0)

        # Pan view, reset pan amount after
        self.canvas.camera.panView(*self.move_state.pan_move)
        self.move_state.pan_move = (0, 0)

        # Zoom view, reset zoom amount after
        self.canvas.camera.zoomView(self.move_state.zoom)
        self.move_state.zoom = 0.0

        # Move in direction
        self.canvas.camera.moveDirection(dt, self.move_state.direction)

    def initializeGL(self) -> None:
        self.canvas.initialize()
        self.glInitializedEvent.emit()

    def paintGL(self) -> None:
        # Handle movement
        self.applyMoveStates()

        # Do rendering
        self.canvas.render()

        # Update time
        self.previous_time = self.time
        self.time = time()

    def resizeGL(self, width, height):
        """Handle resizing of the widget"""
        # No negative sizes!
        if min(width, height) < 0:
            return

        # Update renderer
        self.canvas.resize(width, height)


class QtWindow(qt.QMainWindow):
    """Main window class"""
    PROGRAM_NAME = "Plot-3D"

    def __init__(self, *args, **kwargs):
        # Set default OpenGL format
        fmt = qtg.QSurfaceFormat()
        fmt.setVersion(4, 1)
        fmt.setProfile(qtg.QSurfaceFormat.CoreProfile)
        fmt.setSwapInterval(0)
        qtg.QSurfaceFormat.setDefaultFormat(fmt)

        # Call super-class constructor
        super().__init__(*args, **kwargs)

        # OpenGL widget
        self.view_widget = QtGLWidget(self)
        self.view_widget.setSizePolicy(qt.QSizePolicy.MinimumExpanding, qt.QSizePolicy.MinimumExpanding)
        self.setCentralWidget(self.view_widget)

        # Basic window setup
        self.setWindowTitle(self.PROGRAM_NAME)

    def sizeHint(self):
        return qtc.QSize(500, 500)

    def show(self):
        """Show the main viewing window"""
        self.view_widget.canvas.resetView()
        super().show()
