from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from typing_extensions import Iterable, Iterator, List, Union, Optional

from krrood.adapters.json_serializer import list_like_classes
from coraplex.datastructures.dataclasses import Context
from coraplex.config.action_conf import ActionConfig
from coraplex.datastructures.enums import Arms
from coraplex.locations.backends import GiskardLocationBackend
from coraplex.locations.base import Location
from coraplex.locations.costmaps import OccupancyCostmap, RingCostmap, VisibilityCostmap
from coraplex.locations.pose_validator import (
    AreReachableBy,
    IsObjectReachableBy,
    IsVisibleBy,
)
from coraplex.robot_plans.mixins import HasApproachesGraspPoses
from coraplex.view_manager import ViewManager
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Drawer,
)
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
        context, target_pose, OccupancyCostmap.default_map(context, target_pose), []
    )


def reachability_location(
    target: Union[Pose, Body],
    context: Context,
    arm: Arms,
    grasp_pose: Optional[Pose] = None,
    approach_clearance: float = ActionConfig.approach_clearance,
    retreat_distance: float = ActionConfig.retreat_distance,
) -> Location:
    """
    Factory method that creates a Location for robot poses from which the target can be
    picked up or placed.

    :param target: Target pose or body that should be reached by the robot
    :param context: The context in which to create the location
    :param arm: The arm with which to reach the target
    :param grasp_pose: The grasp frame with which to grasp the target, in the target's
        own frame. ``None`` grasps the target at its origin.
    :param approach_clearance: The gap left between the object and the gripper before
        the final approach.
    :param retreat_distance: How far the gripper rises after closing on the object.
    :returns: A location that is reachable from the target pose.
    """
    target_pose, target_body = (
        (target.global_pose, target) if isinstance(target, Body) else (target, None)
    )
    end_effector = ViewManager.get_end_effector_view(arm, context.robot)
    target_grasp = HasApproachesGraspPoses.resolve_target_grasp_frames(
        target_pose, target_body, grasp_pose, end_effector
    )

    costmap = OccupancyCostmap.default_map(context, target_pose) & RingCostmap(
        resolution=0.02,
        width=200,
        height=200,
        std=15,
        distance=ViewManager.get_arm_view(arm, context.robot).approximate_length()
        * 0.66,  # That needs to be replaced with an estimate of the reachability space of the robot arms
        world=context.world,
        origin=target_pose,
    )
    return Location(
        context,
        target_pose,
        costmap,
        [
            AreReachableBy.for_grasp(
                target_grasp.grasp_frame,
                end_effector,
                body_T_grasp=target_grasp.body_T_grasp,
                context=Context(
                    world=context.world,
                    robot=context.robot,
                    alternative_motion_mappings=context.alternative_motion_mappings,
                ),
                approach_clearance=approach_clearance,
                retreat_distance=retreat_distance,
            )
        ],
    )


def grasping_location(
    graspable: HasGraspPoses,
    context: Context,
    arm: Arms,
    approach_clearance: float = ActionConfig.approach_clearance,
    retreat_distance: float = ActionConfig.retreat_distance,
) -> Location:
    """
    Factory that creates a Location for robot poses from which the object can be grasped
    somehow, rather than from which one particular grasp can be reached.

    A grasp is only reachable from somewhere, so settling on one before a standing pose
    is known picks it from wherever the robot happens to be. This asks the other way
    round: a pose qualifies when any of the object's grasps can be reached from it, and
    the validator keeps the one that was, in
    :attr:`~coraplex.locations.pose_validator.IsObjectReachableBy.reachable_grasp`.

    :param graspable: The annotation of the object that should be grasped.
    :param context: The context in which to create the location.
    :param arm: The arm with which to grasp the object.
    :param approach_clearance: The gap left between the object and the gripper before
        the final approach.
    :param retreat_distance: How far the gripper rises after closing on the object.
    :returns: A location from which the object can be grasped.
    """
    target_pose = graspable.root.global_pose
    costmap = OccupancyCostmap.default_map(context, target_pose) & RingCostmap(
        resolution=0.02,
        width=200,
        height=200,
        std=15,
        distance=ViewManager.get_arm_view(arm, context.robot).approximate_length()
        * 0.66,
        world=context.world,
        origin=target_pose,
    )
    return Location(
        context,
        target_pose,
        costmap,
        [
            IsObjectReachableBy(
                context=Context(
                    world=context.world,
                    robot=context.robot,
                    alternative_motion_mappings=context.alternative_motion_mappings,
                ),
                arm=arm,
                graspable=graspable,
                approach_clearance=approach_clearance,
                retreat_distance=retreat_distance,
            )
        ],
    )


@dataclass
class ReachableGrasps(Iterable[Pose]):
    """
    The grasps of an object that some standing pose reaches, worked out when they are
    asked for rather than when the plan is built.

    Which grasps qualify depends on where the robot may stand and on where everything
    else has got to, so answering while the plan is still being built answers about a
    world the action will not run in. Used as the domain of a ``grasp_pose`` variable,
    this is asked once the underspecified action grounds, during execution.

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

    arm: Arms
    """
    The arm that should do the grasping.
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
        location = grasping_location(
            self.graspable,
            self.context,
            self.arm,
            approach_clearance=self.approach_clearance,
            retreat_distance=self.retreat_distance,
        )
        (validator,) = location.validators
        for _ in location:
            yield validator.reachable_grasp


def accessing_location(
    container: Union[Drawer, Cabinet], context: Context, arm: Arms
) -> Location:
    """
    Factory that creates a location for robot base poses for opening and closing
    container.

    :param container: The container that should be accessed
    :param context: Plan context in which to create the location
    :param arm: Arm with which to access the container
    :returns: A location that is accessible from the container.
    """
    return reachability_location(container.handle.root, context, arm)


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
        min_height=camera.minimal_height,
        max_height=camera.maximal_height,
        world=context.world,
        width=200,
        height=200,
        resolution=0.02,
        origin=target_pose,
    )
    return Location(
        context,
        target_pose,
        costmap,
        [
            IsVisibleBy(
                context=Context(
                    world=context.world,
                    robot=context.robot,
                    alternative_motion_mappings=context.alternative_motion_mappings,
                ),
                target_pose=target_pose,
                target_body=target_body,
            )
        ],
    )


def giskard_reachability_location(
    target: Union[Pose, Body],
    context: Context,
    arm: Arms,
    grasp_pose: Optional[Pose] = None,
    approach_clearance: float = ActionConfig.approach_clearance,
    retreat_distance: float = ActionConfig.retreat_distance,
) -> Location:
    """
    Factory method that creates a location with a Giskard backend, the giskard backend
    uses the Giskard full-body control to find a robot pose.

    :param target: Target pose or body that should be reachable
    :param context: Plan context in which to create the location
    :param arm: Arm to use for reachability estimation
    :param grasp_pose: The grasp frame with which to grasp the target, in the target's
        own frame. ``None`` grasps the target at its origin.
    :param approach_clearance: The gap left between the object and the gripper before
        the final approach.
    :param retreat_distance: How far the gripper rises after closing on the object.
    :returns: A location that is reachable from the target pose, using Giskard for
        reachability estimation.
    """
    target_pose, target_body = (
        (target.global_pose, target) if isinstance(target, Body) else (target, None)
    )

    end_effector = ViewManager.get_end_effector_view(arm, context.robot)
    target_grasp = HasApproachesGraspPoses.resolve_target_grasp_frames(
        target_pose, target_body, grasp_pose, end_effector
    )

    backend = GiskardLocationBackend(
        target,
        arm,
        target_grasp.grasp_frame,
        context.robot,
        context.world,
        approach_clearance=approach_clearance,
        retreat_distance=retreat_distance,
    )

    return Location(
        context,
        target_pose,
        backend,
        [
            AreReachableBy.for_grasp(
                target_grasp.grasp_frame,
                end_effector,
                body_T_grasp=target_grasp.body_T_grasp,
                context=Context(
                    robot=context.robot,
                    world=context.world,
                    alternative_motion_mappings=context.alternative_motion_mappings,
                ),
                approach_clearance=approach_clearance,
                retreat_distance=retreat_distance,
            )
        ],
    )
