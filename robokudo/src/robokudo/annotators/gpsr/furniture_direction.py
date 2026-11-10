"""
Directions relative to furniture.
"""

# %% Imports
from enum import StrEnum


# %% furniture directions
class FurnitureDirection(StrEnum):
    """
    Side of a piece of furniture as seen by the robot.
    """

    LEFT = "left"
    """
    Left of the furniture, seen from the robot.
    """

    RIGHT = "right"
    """
    Right of the furniture, seen from the robot.
    """

    FRONT = "front"
    """
    In front of the furniture, seen from the robot.
    """

    BACK = "back"
    """
    Behind the furniture, seen from the robot.
    """
