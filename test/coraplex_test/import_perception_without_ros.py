"""
Imports :mod:`coraplex.perception` in an interpreter where ROS 2 cannot be imported.

Run as a script; it exits non-zero with an :class:`ImportError` if the module needs ROS
to import.
"""

import sys

# %% hide ROS 2

for ros_module in ("rclpy", "rclpy.node"):
    sys.modules[ros_module] = None

# %% import

import coraplex.perception  # noqa: E402,F401
