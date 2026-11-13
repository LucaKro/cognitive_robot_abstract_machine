from unittest.mock import patch

import rclpy

from semantic_digital_twin.adapters.ros import background_node

# %% shutting down


def test_the_spin_thread_has_ended_before_ros_is_shut_down() -> None:
    """
    Shutting down stops the spin thread itself, so that it no longer runs when the node
    it spins is destroyed and ROS is shut down below it.
    """
    background_node.init_node("node_shut_down_by_the_test")
    spin_thread = background_node.spinner_thread
    spin_thread_was_alive = []
    shut_ros_down = rclpy.shutdown

    def record_then_shut_ros_down() -> None:
        spin_thread_was_alive.append(spin_thread.is_alive())
        shut_ros_down()

    with patch.object(rclpy, "shutdown", side_effect=record_then_shut_ros_down):
        background_node.shutdown()

    assert spin_thread_was_alive == [False]
