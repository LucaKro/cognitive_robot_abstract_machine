from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from typing_extensions import Iterable, Iterator, List, Optional, Union

from krrood.adapters.json_serializer import list_like_classes
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ReachFraction
from coraplex.locations.backends import GiskardLocationBackend
from coraplex.locations.base import Location
from coraplex.locations.costmaps import OccupancyCostmap, RingCostmap, VisibilityCostmap
from coraplex.locations.pose_validator import (
    AreReachableBy,
    IsObjectReachableBy,
    IsVisibleBy,
)
from coraplex.robot_plans.mixins import HasApproachesGraspPoses
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Drawer,
)
from semantic_digital_twin.robots.robot_parts import Arm, EndEffector
from semantic_digital_twin.semantic_annotations.mixins import GraspPose, HasGraspPoses
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


def occupancy_location(target_pose: Pose, context: Context) -> Location:
    """
    Where the robot can stand around a target without standing in anything.

    Nothing else is asked of a candidate: the poses are offered as the map has them.

    :param target_pose: The pose the standing poses are drawn around.
    :param context: The context in which to create the location.
    :returns: A location of poses clear of the surroundings.
    """
    return Location(
        context=context,
        target_pose=target_pose,
        generator=OccupancyCostmap.default_map(context=context, target=target_pose),
    )


def reachability_location(
    grasp: GraspPose,
    context: Context,
    arm: Arm,
    destination: Optional[Pose] = None,
    approach_clearance: float = HasApproachesGraspPoses.approach_clearance,
    retreat_distance: float = HasApproachesGraspPoses.retreat_distance,
    reach_fraction: float = ReachFraction.GRASPING,
) -> Location:
    """
    Checks one grasp the caller already chose: where can the robot stand to reach it?

    .. note::
        - *Grasp*: given by the caller as ``grasp``; no other grasp is ever tried.
        - *Result*: standing poses only.
        - To have the grasp chosen as well, use :func:`grasping_location`.

    :param grasp: The grasp the gripper takes or holds the object by.
    :param context: The context in which to create the location
    :param arm: The arm with which to reach the object
    :param destination: Where the object is going to be, such as where a carried body
        is placed. ``None`` reaches the object where it is. An object reached at a
        destination is released there, which runs the approach backwards, so the check
        follows it backwards too.
    :param approach_clearance: The gap left between the object and the gripper before
        the final approach.
    :param retreat_distance: How far the gripper rises after closing on the object.
    :param reach_fraction: The fraction of the arm's length the robot stands off the
        target by.
    :returns: Standing poses from which ``grasp`` can be reached, or released at
        ``destination``.
    """
    target_pose = destination or grasp.graspable.root.global_pose
    releases_the_body = destination is not None
    occupancy_costmap = OccupancyCostmap.default_map(
        context=context, target=target_pose
    )
    ring_costmap = RingCostmap.from_arm_reach_distance(
        context=context, arm=arm, origin=target_pose, reach_fraction=reach_fraction
    )
    final_costmap = occupancy_costmap & ring_costmap
    return Location(
        context=context,
        target_pose=target_pose,
        generator=final_costmap,
        validator=AreReachableBy.for_grasp(
            grasp=grasp,
            arm=arm,
            destination=destination,
            context=context,
            reverse=releases_the_body,
            approach_clearance=approach_clearance,
            retreat_distance=retreat_distance,
        ),
    )


def grasping_location(
    graspable: HasGraspPoses,
    context: Context,
    arm: Arm,
    approach_clearance: float = HasApproachesGraspPoses.approach_clearance,
    retreat_distance: float = HasApproachesGraspPoses.retreat_distance,
) -> Location:
    """
    Chooses the grasp as well as the standing pose: only the object is given, and every
    grasp it offers is tried.

    .. note::
        - *Grasp*: chosen here, from the grasps ``graspable`` offers; the one chosen
          for the current pose is on the validator's
          :attr:`~coraplex.locations.pose_validator.IsObjectReachableBy.reachable_grasp`.
        - *Result*: standing poses, each paired with the grasp it was found for.
        - For a grasp the caller already chose, use :func:`reachability_location`.

    :param graspable: The annotation of the object that should be grasped.
    :param context: The context in which to create the location.
    :param arm: The arm with which to grasp the object.
    :param approach_clearance: The gap left between the object and the gripper before
        the final approach.
    :param retreat_distance: How far the gripper rises after closing on the object.
    :returns: Standing poses from which at least one of the object's grasps can be
        reached.
    """
    target_pose = graspable.root.global_pose
    occupancy_costmap = OccupancyCostmap.default_map(
        context=context, target=target_pose
    )
    ring_costmap = RingCostmap.from_arm_reach_distance(
        context=context,
        arm=arm,
        origin=target_pose,
        reach_fraction=ReachFraction.GRASPING,
    )
    final_costmap = occupancy_costmap & ring_costmap
    return Location(
        context=context,
        target_pose=target_pose,
        generator=final_costmap,
        validator=IsObjectReachableBy(
            context=context,
            arm=arm,
            graspable=graspable,
            approach_clearance=approach_clearance,
            retreat_distance=retreat_distance,
        ),
    )


@dataclass
class ReachableGrasps(Iterable[GraspPose]):
    """
    The grasps of an object that some standing pose reaches, worked out when they are
    asked for rather than when the plan is built.

    .. warning::
        :meth:`__iter__` must stay a generator. The domain is wrapped rather than
        consumed by :func:`~krrood.entity_query_language.factories.variable`, so a
        generator is what defers the search to the first ``next``; building the grasps
        eagerly would put the staleness straight back.
    """

    graspable: HasGraspPoses
    """
    The annotation of the object that should be grasped.
    """

    context: Context
    """
    The context the reaching is judged in.
    """

    arm: Arm[EndEffector]
    """
    The arm that should do the grasping.

    Written with its end effector type, since a bound generic is what the ORM maps a
    field of; an unparameterized one is skipped and the arm is then not persisted.
    """

    approach_clearance: float = HasApproachesGraspPoses.approach_clearance
    """
    The gap left between the object and the gripper before the final approach.
    """

    retreat_distance: float = HasApproachesGraspPoses.retreat_distance
    """
    How far the gripper rises after closing on the object.
    """

    def __iter__(self) -> Iterator[GraspPose]:
        """
        :return: The reachable grasps, on :attr:`graspable` in the world of
            :attr:`context` rather than in the copy the reach was judged in.
        """
        location = grasping_location(
            graspable=self.graspable,
            context=self.context,
            arm=self.arm,
            approach_clearance=self.approach_clearance,
            retreat_distance=self.retreat_distance,
        )
        for _ in location:
            yield location.validator.reachable_grasp.copy_for_world(self.context.world)


def accessing_location(
    container: Union[Drawer, Cabinet],
    context: Context,
    arm: Arm,
    reach_fraction: float = ReachFraction.ACCESSING,
) -> Location:
    """
    Where the robot can stand to open or close a container by its handle.

    The same question :func:`reachability_location` answers about the handle, asked from
    the closer stand-off distance that pulling a container needs.

    :param container: The container to be opened or closed.
    :param context: The context in which to create the location.
    :param arm: The arm that works the handle.
    :param reach_fraction: The fraction of the arm's length the robot stands off the
        handle by.
    :returns: A location from which the handle can be reached.
    """
    return reachability_location(
        grasp=GraspPose.from_body_origin(container.handle),
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
    target_pose, target_body = (
        (target.global_pose, target) if isinstance(target, Body) else (target, None)
    )

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
    )
    final_costmap = occupancy_costmap & visibility_costmap
    return Location(
        context=context,
        target_pose=target_pose,
        generator=final_costmap,
        validator=IsVisibleBy(
            context=context,
            target_pose=target_pose,
            target_body=target_body,
        ),
    )


def giskard_reachability_location(
    grasp: GraspPose,
    context: Context,
    arm: Arm,
    destination: Optional[Pose] = None,
    approach_clearance: float = HasApproachesGraspPoses.approach_clearance,
    retreat_distance: float = HasApproachesGraspPoses.retreat_distance,
) -> Location:
    """
    Checks one grasp the caller already chose, like :func:`reachability_location`, but
    finds the standing poses by letting full-body control drive the robot to each
    candidate and offering where it arrived.

    .. note::
        - *Grasp*: given by the caller as ``grasp``; no other grasp is ever tried.
        - *Result*: standing poses only.

    :param grasp: The grasp the gripper takes or holds the object by.
    :param context: Plan context in which to create the location
    :param arm: Arm to use for reachability estimation
    :param destination: Where the object is going to be, such as where a carried body
        is placed. ``None`` reaches the object where it is.
    :param approach_clearance: The gap left between the object and the gripper before
        the final approach.
    :param retreat_distance: How far the gripper rises after closing on the object.
    :returns: Standing poses from which ``grasp`` can be reached, or released at
        ``destination``.
    """
    target_pose = destination or grasp.graspable.root.global_pose
    releases_the_body = destination is not None

    backend = GiskardLocationBackend(
        target_pose=target_pose,
        arm=arm,
        robot=context.robot,
        world=context.world,
        grasp=grasp,
        contact_bodies=[grasp.graspable.root],
        reverse=releases_the_body,
        approach_clearance=approach_clearance,
        retreat_distance=retreat_distance,
    )

    return Location(
        context=context,
        target_pose=target_pose,
        generator=backend,
        validator=AreReachableBy.for_grasp(
            grasp=grasp,
            arm=arm,
            destination=destination,
            context=context,
            reverse=releases_the_body,
            approach_clearance=approach_clearance,
            retreat_distance=retreat_distance,
        ),
    )
