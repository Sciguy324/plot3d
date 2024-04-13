"""
This module contains classes and methods for working with cameras in 3D space
"""
# Import modules
import numpy as np
from typing import Tuple
from enum import Enum, IntFlag, auto
from dataclasses import dataclass
from .transform import translate, rotateX, rotateY, zoom


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


@dataclass
class Camera:
    """
    Represents a viewing camera in 3D space
    """
    x: float = 0.0               # Camera x-position
    y: float = 0.0               # Camera y-position
    z: float = 0.0               # Camera z-position
    theta: float = 0.0           # Camera rotation
    phi: float = 0.0             # Camera pitch
    zoom: float = 0.0            # Camera zoom exponent
    pivot_dist: float = 5.0      # Distance between camera and pivot point
    move_speed: float = 2.5      # Movement speed scale factor
    pan_speed: float = 0.01      # Movement speed from panning
    rotation_speed: float = 1.0  # Rotation speed

    @property
    def scale(self) -> float:
        """The scale factor used for scaling the 3D space.  Computed using 10^(zoom)"""
        return 10**self.zoom

    @property
    def cam_pos(self) -> Tuple[float, float, float]:
        """Returns the camera's position as a three-tuple"""
        return self.x, self.y, self.z

    @cam_pos.setter
    def cam_pos(self, other):
        other = [float(i) for i in other]  # Enforce float type
        self.x, self.y, self.z = other

    @property
    def cam_angle(self) -> Tuple[float, float]:
        """Returns the camera's rotation as a two-tuple of angles"""
        return self.theta, self.phi

    def panView(self, horizontal: float, vertical: float):
        """Pans the view relative to the camera's current direction"""
        # Transform into coordinate system relative to the camera's current direction
        theta, phi = np.radians(self.theta), np.radians(self.phi)
        transform = np.array([[0.0, np.cos(theta), np.sin(theta)*np.sin(phi)],
                              [0.0, 0.0, np.cos(phi)],
                              [0.0, np.sin(theta), -np.cos(theta)*np.sin(phi)]])
        transformed_delta = transform @ np.array([0.0, horizontal, -vertical])

        # Move by factor that's controlled by pan speed and current zoom factor
        factor = self.pan_speed * self.scale
        self.cam_pos -= transformed_delta*factor

    def zoomView(self, amount: float):
        """Adjusts the camera zoom about the pivot"""
        self.zoom += 0.001*amount

    def rotateView(self, delta_theta: float, delta_phi: float):
        """Adjusts the camera angle by the given amount"""
        # Scale by rotation speed and adjust angles
        self.theta += delta_theta * self.rotation_speed
        self.phi += delta_phi * self.rotation_speed
        # Clamp values
        self.theta %= 360
        if 90 < self.phi < 180:
            self.phi = 90.0
        elif 180 < self.phi < 270:
            self.phi = 270
        self.phi %= 360

    def moveDirection(self, delta: float, direction: Direction):
        """
        Moves the camera position relative to its view.
        Movement is scaled by the internal movements speed and scale factor.

        Arguments:
            delta (float): Amount to scale the movement by
            direction (Direction): Direction flags indicating which direction to move in
        """
        delta *= self.move_speed * self.scale
        cos_theta = np.cos(np.radians(self.theta))
        sin_theta = np.sin(np.radians(self.theta))

        # Move forwards
        if Direction.forward & direction:
            self.x += sin_theta*delta
            self.z -= cos_theta*delta
        # Move backwards
        if Direction.backward & direction:
            self.x -= sin_theta*delta
            self.z += cos_theta*delta
        # Move left
        if Direction.left & direction:
            self.x -= cos_theta*delta
            self.z -= sin_theta*delta
        # Move right
        if Direction.right & direction:
            self.x += cos_theta*delta
            self.z += sin_theta*delta
        # Move up
        if Direction.up & direction:
            self.y += delta
        # Move down
        if Direction.down & direction:
            self.y -= delta

    def focusView(self, center: Tuple[float, float, float], dimensions: Tuple[float, float, float]):
        """
        Resets the camera to focus on an object

        Arguments:
            center (Tuple[float, float, float]): 3-tuple of coordinates to focus pivot on
            dimensions (Tuple[float, float, float]): 3-tuple of coordinates to try to contain within view
        """
        # FIXME: Make this function more reliable
        # Move camera to focus on center of mass
        self.cam_pos = center

        # Compute a better zoom that fits the structure
        self.zoom = np.log10(np.max(dimensions)/2)

    @property
    def matrix(self) -> np.ndarray:
        """Builds and returns the transformation matrix for the camera view"""
        transform = np.identity(4)
        transform = transform @ translate(0.0, 0.0, -self.pivot_dist)
        transform = transform @ rotateX(np.radians(self.phi))
        transform = transform @ rotateY(np.radians(self.theta))
        transform = transform @ zoom(1/self.scale)
        transform = transform @ translate(0.0, 0.0, self.pivot_dist)
        transform = transform @ translate(-self.x, -self.y, -self.z)
        transform = transform @ translate(0.0, 0.0, -self.pivot_dist)
        return transform

    @classmethod
    def fromJSON(cls, data: dict):
        """
        Builds and returns a new camera object from a JSON-compatible dictionary

        Arguments:
            data (dict): A JSON-compatible dictionary

        Returns (Camera): A new camera object
        """
        return cls(x=float(data.get('x', 0.0)),
                   y=float(data.get('y', 0.0)),
                   z=float(data.get('z', 0.0)),
                   theta=float(data.get('theta', 0.0)),
                   phi=float(data.get('phi', 0.0)),
                   zoom=float(data.get('zoom', 0.0)),
                   pivot_dist=float(data.get('pivot_dist', 0.0)),
                   move_speed=float(data.get('move_speed', 0.0)),
                   pan_speed=float(data.get('pan_speed', 0.0)),
                   rotation_speed=float(data.get('rotation_speed', 0.0)))

    def toJSON(self) -> dict:
        """
        Converts the camera into a JSON-compatible dictionary object

        Returns (dict): A JSON-compatible dictionary
        """
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "theta": self.theta,
            "phi": self.phi,
            "zoom": self.zoom,
            "pivot_dist": self.pivot_dist,
            "move_speed": self.move_speed,
            "pan_speed": self.pan_speed,
            "rotation_speed": self.rotation_speed
        }

