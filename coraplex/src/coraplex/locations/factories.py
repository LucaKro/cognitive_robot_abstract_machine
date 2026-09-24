from __future__ import annotations

from typing_extensions import Union

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ReachFraction
from coraplex.locations.base import Location
from coraplex.locations.costmaps import OccupancyCostmap, RingCostmap, VisibilityCostmap
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Drawer,
)
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


def occupancy_location(target_pose: Pose, context: Context) -> Location:
    """
    Where the robot can stand around a target without standing in anything.

    :param target_pose: The pose the standing poses are drawn around.
    :param context: The context in which to create the location.
    :returns: A location of poses clear of the surroundings.
    """
    return OccupancyCostmap.default_map(context=context, target=target_pose)


def reachability_location(
    target_pose: Pose,
    context: Context,
    arm: Arm,
    reach_fraction: float = ReachFraction.GRASPING,
) -> Location:
    """
    Where the robot can stand to reach a target with one arm.

    :param target_pose: The pose the arm is to reach.
    :param context: The context in which to create the location.
    :param arm: The arm with which to reach the target.
    :param reach_fraction: The fraction of the arm's length the robot stands off the
        target by.
    :returns: Standing poses clear of the surroundings, at the arm's reach distance
        around the target.
    """
    occupancy_costmap = OccupancyCostmap.default_map(
        context=context, target=target_pose
    )
    ring_costmap = RingCostmap.from_arm_reach_distance(
        context=context, arm=arm, origin=target_pose, reach_fraction=reach_fraction
    )
    return occupancy_costmap & ring_costmap


def accessing_location(
    container: Union[Drawer, Cabinet],
    context: Context,
    arm: Arm,
    reach_fraction: float = ReachFraction.ACCESSING,
) -> Location:
    """
    Where the robot can stand to open or close a container by its handle.

    The same location :func:`reachability_location` gives for the handle, from the
    closer stand-off distance that pulling a container needs.

    :param container: The container to be opened or closed.
    :param context: The context in which to create the location.
    :param arm: The arm that works the handle.
    :param reach_fraction: The fraction of the arm's length the robot stands off the
        handle by.
    :returns: A location from which the handle is within reach.
    """
    return reachability_location(
        target_pose=container.handle.root.global_pose,
        context=context,
        arm=arm,
        reach_fraction=reach_fraction,
    )


def visibility_location(target: Union[Pose, Body], context: Context) -> Location:
    """
    Where the robot can stand to see a target with its camera.

    :param target: The pose or body that should be visible.
    :param context: The context in which to create the location.
    :returns: A location from which the target is in view.
    """
    target_pose = target.global_pose if isinstance(target, Body) else target

    camera = context.robot.get_default_camera()
    occupancy_costmap = OccupancyCostmap.default_map(
        context=context, target=target_pose
    )
    visibility_costmap = VisibilityCostmap(
        minimum_height=camera.minimal_height,
        maximum_height=camera.maximal_height,
        world=context.world,
        width=200,
        height=200,
        resolution=0.02,
        origin=target_pose,
        draw=context.candidate_draw,
    )
    return occupancy_costmap & visibility_costmap
