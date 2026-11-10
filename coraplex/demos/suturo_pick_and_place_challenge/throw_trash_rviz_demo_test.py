#!/usr/bin/env python
"""
Runs the HSR kitchen trash can demo as a regression test, as fast as the computer
allows, from three start spots of the milk, and fails when the milk does not end up
inside the trash can. Exits non-zero with a traceback on failure.

Run it from this folder, with ROS 2 and the ROS workspace sourced::

    python throw_trash_rviz_demo_test.py
"""

from __future__ import annotations

import numpy as np
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
    TrashCan,
)

import throw_trash_rviz_demo as demo

# %% checking where the milk ends up


def check_milk_ends_inside_the_trash_can(milk_start: demo.FloorSpot) -> None:
    """
    Run the demo with the milk starting at `milk_start`, and check that it ends upright
    on the trash can's bottom, centred over the opening it was dropped through.

    :param milk_start: Where the milk stands on the floor before the plan begins.
    """
    demonstration = demo.FloorToTrashCanDemonstration(
        milk_start=milk_start,
        search_spot=demo.FloorSpot(x=milk_start.x, y=milk_start.y),
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
        trash_can_box.min_z + demonstration.trash_can_wall_thickness - milk_box.min_z,
    ]
    position = world.compute_forward_kinematics_np(world.root, milk.root)[:3, 3]
    assert milk.root.parent_connection.parent is trash_can.root
    assert np.allclose(position, expected_position, atol=0.01)


# %% running the checks


def main() -> None:
    """
    Check three start spots of the milk on the floor, one after the other.
    """
    for milk_start in [
        demo.FloorSpot(x=1.7, y=4.6, yaw=0.3),
        demo.FloorSpot(x=1.2, y=4.0, yaw=1.0),
        demo.FloorSpot(x=0.4, y=4.3, yaw=2.5),
    ]:
        check_milk_ends_inside_the_trash_can(milk_start)


if __name__ == "__main__":
    main()
