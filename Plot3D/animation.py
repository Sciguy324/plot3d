"""
This module contains tools for creating animations in a 3D plot
"""
# Import modules
from typing import Callable


class Animation:
    """
    Represents an animation on a 3D plot
    """

    def __init__(self, func: Callable[[int], None], interval: int = 30, frames: int = None, loop: bool = True):
        """
        Constructor for an animation instance
        
        Arguments:
            func (Callable[[int], None]): Function called to update canvas state.  Must accept an integer timestep as a parameter
            interval (int): Number of milliseconds between successive animation updates
            frames (int): Number of frames the animation should play for.  If not provided, the animation will run with no upper bound on the timestep
            loop (bool): Whether to loop this animation or halt at the final frame
        """
        # Save parameters
        self.func: Callable[[int], None] = func
        self.interval: int = interval
        self.frames: int = frames
        self.loop: bool = loop

        # Timekeeping variables
        self.last_frame: int = None

    def __call__(self, time: float) -> None:
        """
        Executes this animation's update function

        Arguments:
            time (float): The current time since the start, in seconds
        """
        # Process timestep based on animation parameters
        time_ms = time * 1000
        frame = time_ms // self.interval

        # Either loop, stop at end, or continue indefinitely
        if self.frames is not None:
            if self.loop:
                frame %= self.frames
            else:
                frame = min(frame, self.frames)

        frame = int(frame)

        # Only apply update if frame number changed
        if frame != self.last_frame:
            self.last_frame = frame
            self.func(frame)
