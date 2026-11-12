#!/usr/bin/env python
"""
Runs the HSR kitchen trash can demo as a regression test, as fast as the computer
allows, from three start spots of the milk, and fails when the milk is not let go of
over the trash can's opening. Exits non-zero with a traceback on failure.

Run it from this folder, with ROS 2 and the ROS workspace sourced::

    python throw_trash_rviz_demo_test.py
"""

from __future__ import annotations

import numpy as np
from krrood_spatial_math.spatial_types import Pose2D
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
    TrashCan,
)

import throw_trash_rviz_demo as demo

# %% checking where the milk is let go of


def check_milk_is_let_go_of_over_the_trash_can(milk_start: Pose2D) -> None:
    """
    Run the demo with the milk starting at `milk_start`, and check that the hand let go
    of it centred over the trash can's opening, its bottom
    :attr:`~throw_trash_rviz_demo.FloorToTrashCanDemonstration.drop_clearance` above the
    rim, where a kinematic simulation without gravity leaves it.

    :param milk_start: Where the milk stands on the floor before the plan begins.
    """
    demonstration = demo.FloorToTrashCanDemonstration(
        milk_start=milk_start,
        search_spot=Pose2D.from_xy_yaw(x=milk_start.x, y=milk_start.y),
        real_time_factor=None,
    )

    world = demonstration.run()

    [milk] = world.get_semantic_annotations_by_type(Milk)
    [trash_can] = world.get_semantic_annotations_by_type(TrashCan)
    trash_can_box = trash_can.root.collision.as_bounding_box_collection_in_frame(
        world.root
    ).bounding_box()
    milk_box = milk.root.collision.as_bounding_box_collection_in_frame(
        milk.root
    ).bounding_box()
    expected_position = [
        (trash_can_box.min_x + trash_can_box.max_x) / 2,
        (trash_can_box.min_y + trash_can_box.max_y) / 2,
        trash_can_box.max_z + demonstration.drop_clearance - milk_box.min_z,
    ]
    position = world.compute_forward_kinematics_np(world.root, milk.root)[:3, 3]
    assert milk.root.parent_connection.parent is world.root
    assert np.allclose(position, expected_position, atol=0.01)


# %% running the checks


def main() -> None:
    """
    Check three start spots of the milk on the floor, one after the other.
    """
    for milk_start in [
        Pose2D.from_xy_yaw(x=1.7, y=4.6, yaw=0.3),
        Pose2D.from_xy_yaw(x=1.2, y=4.0, yaw=1.0),
        Pose2D.from_xy_yaw(x=0.4, y=4.3, yaw=2.5),
    ]:
        check_milk_is_let_go_of_over_the_trash_can(milk_start)


if __name__ == "__main__":
    main()
