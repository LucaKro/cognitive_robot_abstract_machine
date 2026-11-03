"""
Directions relative to furniture.
"""

# %% imports
from enum import StrEnum


# %% furniture directions
class FurnitureDirection(StrEnum):
    """
    Side of a piece of furniture as seen by the robot.
    """

    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BACK = "back"
