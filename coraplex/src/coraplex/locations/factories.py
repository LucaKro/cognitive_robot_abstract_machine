from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from typing_extensions import Iterable, Iterator, List, Optional, Union

from krrood.adapters.json_serializer import list_like_classes
from coraplex.datastructures.dataclasses import Context
from coraplex.config.action_conf import ActionConfig
from coraplex.locations.backends import GiskardLocationBackend
from coraplex.locations.base import Location
from coraplex.locations.costmaps import OccupancyCostmap, RingCostmap, VisibilityCostmap
from coraplex.locations.pose_validator import (
    AreReachableBy,
    IsObjectReachableBy,
    IsVisibleBy,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Drawer,
)
from semantic_digital_twin.robots.robot_parts import Arm, EndEffector
from semantic_digital_twin.semantic_annotations.mixins import HasGraspPoses
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


def occupancy_location(target_pose: Pose, context: Context) -> Location:
    """
    Factory that creates a Location for robot base poses, does not have any validators.

    :param target_pose: Target pose around which robot base poses should be sampled
    :param context: Context of the plan in which the location should be created
    :returns: The Location for robot base poses
    """
    return Location(
        context=context,
        target_pose=target_pose,
        generator=OccupancyCostmap.default_map(context, target_pose),
        validators=[],
    )


def reachability_location(
    body: Body,
    context: Context,
    arm: Arm,
    grasp_pose: Optional[Pose] = None,
    destination: Optional[Pose] = None,
    approach_clearance: float = ActionConfig.approach_clearance,
    retreat_distance: float = ActionConfig.retreat_distance,
    reach_fraction: float = ActionConfig.reach_fraction,
) -> Location:
    """
    Factory method that creates a Location for robot poses from which one named grasp on
    a body can be reached, where the body is or where it is going to be.

    The grasp is settled on before the pose is: this asks whether that one grasp works
    from a candidate pose. :func:`grasping_location` asks the other way round, for a
    pose from which any of an object's grasps works.

    :param body: The body the gripper grasps or holds.
    :param context: The context in which to create the location
    :param arm: The arm with which to reach the body
    :param grasp_pose: The grasp frame on the body, in the body's own frame. ``None``
        grasps the body at its origin.
    :param destination: Where the body is going to be, such as where a carried body is
        placed. ``None`` reaches the body where it is. A body reached at a destination
        is released there, which runs the approach backwards, so the check follows it
        backwards too.
    :param approach_clearance: The gap left between the object and the gripper before
        the final approach.
    :param retreat_distance: How far the gripper rises after closing on the object.
    :param reach_fraction: The fraction of the arm's length the robot stands off the
        target by.
    :returns: A location from which the grasp can be reached.
    """
    body_T_grasp = grasp_pose or Pose(reference_frame=body)
    target_pose = destination or body.global_pose
    releases_the_body = destination is not None
    occupancy_costmap = OccupancyCostmap.default_map(context, target_pose)
    ring_costmap = RingCostmap.from_arm_reach_distance(
        context, arm, target_pose, reach_fraction=reach_fraction
    )
    final_costmap = occupancy_costmap & ring_costmap
    return Location(
        context=context,
        target_pose=target_pose,
        generator=final_costmap,
        validators=[
            AreReachableBy.for_grasp(
                grasp_pose=target_pose.to_homogeneous_matrix() @ body_T_grasp,
                arm=arm,
                body_T_grasp=body_T_grasp,
                context=context,
                reverse=releases_the_body,
                approach_clearance=approach_clearance,
                retreat_distance=retreat_distance,
            )
        ],
    )


def grasping_location(validator: IsObjectReachableBy) -> Location:
    """
    Factory that creates a Location for robot poses from which the object can be grasped
    somehow, rather than from which one particular grasp can be reached.

    A grasp is only reachable from somewhere, so settling on one before a standing pose
    is known picks it from wherever the robot happens to be. This asks the other way
    round: a pose qualifies when any of the object's grasps can be reached from it, and
    the validator keeps the one that was, in
    :attr:`~coraplex.locations.pose_validator.IsObjectReachableBy.reachable_grasp`.
    :func:`reachability_location` is the question to ask about a grasp already chosen.

    The validator belongs to the caller, so whoever wants the grasp that was found
    already holds the validator it is kept on.

    :param validator: The validator asking whether the object is graspable, which names
        the object, the arm and the context this location is for.
    :returns: A location from which the object can be grasped.
    """
    target_pose = validator.graspable.root.global_pose
    occupancy_costmap = OccupancyCostmap.default_map(validator.context, target_pose)
    ring_costmap = RingCostmap.from_arm_reach_distance(
        validator.context, validator.arm, target_pose
    )
    final_costmap = occupancy_costmap & ring_costmap
    return Location(
        context=validator.context,
        target_pose=target_pose,
        generator=final_costmap,
        validators=[validator],
    )


@dataclass
class ReachableGrasps(Iterable[Pose]):
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

    approach_clearance: float = ActionConfig.approach_clearance
    """
    The gap left between the object and the gripper before the final approach.
    """

    retreat_distance: float = ActionConfig.retreat_distance
    """
    How far the gripper rises after closing on the object.
    """

    def __iter__(self) -> Iterator[Pose]:
        validator = IsObjectReachableBy(
            context=self.context,
            arm=self.arm,
            graspable=self.graspable,
            approach_clearance=self.approach_clearance,
            retreat_distance=self.retreat_distance,
        )
        for _ in grasping_location(validator):
            yield validator.reachable_grasp


def accessing_location(
    container: Union[Drawer, Cabinet], context: Context, arm: Arm
) -> Location:
    """
    Factory that creates a location for robot base poses for opening and closing
    container.

    :param container: The container that should be accessed
    :param context: Plan context in which to create the location
    :param arm: Arm with which to access the container
    :returns: A location that is accessible from the container.
    """
    return reachability_location(
        body=container.handle.root,
        context=context,
        arm=arm,
        reach_fraction=ActionConfig.accessing_reach_fraction,
    )


def visibility_location(target: Union[Pose, Body], context: Context) -> Location:
    """
    Factory that creates a location for robot base poses from which the target is
    visible.

    :param target: Target pose or body that should be visible
    :param context: Plan context in which to create the location
    :returns: A location that is visible from the target pose.
    """
    target_pose, target_body = (
        (target.global_pose, target) if isinstance(target, Body) else (target, None)
    )

    camera = context.robot.get_default_camera()
    costmap = OccupancyCostmap.default_map(context, target_pose) & VisibilityCostmap(
        minimum_height=camera.minimal_height,
        maximum_height=camera.maximal_height,
        world=context.world,
        width=200,
        height=200,
        resolution=0.02,
        origin=target_pose,
    )
    return Location(
        context=context,
        target_pose=target_pose,
        generator=costmap,
        validators=[
            IsVisibleBy(
                context=context,
                target_pose=target_pose,
                target_body=target_body,
            )
        ],
    )


def giskard_reachability_location(
    body: Body,
    context: Context,
    arm: Arm,
    grasp_pose: Optional[Pose] = None,
    destination: Optional[Pose] = None,
    approach_clearance: float = ActionConfig.approach_clearance,
    retreat_distance: float = ActionConfig.retreat_distance,
) -> Location:
    """
    Factory method that creates a location with a Giskard backend, the giskard backend
    uses the Giskard full-body control to find a robot pose.

    :param body: The body the gripper grasps or holds.
    :param context: Plan context in which to create the location
    :param arm: Arm to use for reachability estimation
    :param grasp_pose: The grasp frame on the body, in the body's own frame. ``None``
        grasps the body at its origin.
    :param destination: Where the body is going to be, such as where a carried body is
        placed. ``None`` reaches the body where it is.
    :param approach_clearance: The gap left between the object and the gripper before
        the final approach.
    :param retreat_distance: How far the gripper rises after closing on the object.
    :returns: A location from which the grasp can be reached, using Giskard for
        reachability estimation.
    """
    body_T_grasp = grasp_pose or Pose(reference_frame=body)
    target_pose = destination or body.global_pose
    grasp_frame = target_pose.to_homogeneous_matrix() @ body_T_grasp
    releases_the_body = destination is not None

    backend = GiskardLocationBackend(
        target_pose=target_pose,
        arm=arm,
        grasp_pose=grasp_frame,
        robot=context.robot,
        world=context.world,
        body_T_grasp=body_T_grasp,
        contact_bodies=[body],
        reverse=releases_the_body,
        approach_clearance=approach_clearance,
        retreat_distance=retreat_distance,
    )

    return Location(
        context=context,
        target_pose=target_pose,
        generator=backend,
        validators=[
            AreReachableBy.for_grasp(
                grasp_pose=grasp_frame,
                arm=arm,
                body_T_grasp=body_T_grasp,
                context=context,
                reverse=releases_the_body,
                approach_clearance=approach_clearance,
                retreat_distance=retreat_distance,
            )
        ],
    )
