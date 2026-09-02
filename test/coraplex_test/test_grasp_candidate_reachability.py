"""
An experiment over the grasps an object offers, rather than over the first of them.

The gripper ranks an object's grasps by :meth:`EndEffector.distance_to_grasp`, which is
a geometric ranking and explicitly not a reachability test. This measures how far apart
the two are, and decomposes each rejection into whether the grasp itself is out of reach
or only the approach onto it is.
"""

import math
from dataclasses import dataclass, fields

import numpy as np
import pytest
import rclpy

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ExecutionType
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.locations.pose_validator import AreReachableBy, IsReachableBy
from coraplex.view_manager import ViewManager
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

from .test_designator.test_multi_robot_action_designator import stand_facing

# %% where the experiment puts the robot and the object

WORLD_P_MILK = (1.0, -2.0, 0.8)
"""
Where the milk is placed, on the apartment's table height.
"""

WORLD_P_STAND = (0.3, -2.4, 0.0)
"""
Where the robot stands, close enough that some grasps are within reach and far enough
that not all of them are.
"""

# %% one grasp's outcome


@dataclass
class GraspProbe:
    """
    What was measured about one of the grasps an object offers.
    """

    rank: int
    """
    The grasp's position in :meth:`EndEffector.grasp_poses_by_distance`, best first.
    """

    distance: float
    """
    The ranking score itself, from :meth:`EndEffector.distance_to_grasp`, in meters.
    """

    approach_yaw: float
    """
    Compass direction the gripper travels in to enter the grasp, in degrees in the world
    frame, so grasps can be told apart by where they are entered from.
    """

    grasp_reachable: bool
    """
    Whether the grasp frame alone can be reached, ignoring the approach onto it.
    """

    sequence_reachable: bool
    """
    Whether the pre-grasp, grasp and retreat poses can all be reached in order, which is
    what a pick-up actually performs.
    """

    @property
    def rejected_for_its_approach(self) -> bool:
        """
        :return: Whether the grasp is within reach but the sequence onto it is not, which
            says the approach corridor is what rejected it rather than the grasp itself.
        """
        return self.grasp_reachable and not self.sequence_reachable


def _approach_yaw(grasp: Pose, world: World) -> float:
    """
    :param grasp: The grasp frame, whose x-axis is the direction the gripper travels.
    :param world: The world the grasp is placed in.
    :return: That direction's yaw in the world frame, in degrees.
    """
    world_T_grasp = world.transform(grasp.to_homogeneous_matrix(), world.root)
    world_V_approach = (world_T_grasp.to_rotation_matrix() @ Vector3.X()).to_np()[:3]
    return math.degrees(math.atan2(world_V_approach[1], world_V_approach[0]))


def _probe(
    rank: int, grasp: Pose, end_effector: EndEffector, context: Context
) -> GraspProbe:
    """
    Measure one grasp, both as a frame to reach and as a sequence to perform.

    :param rank: The grasp's position in the gripper's own ranking.
    :param grasp: The grasp frame, in the grasped body's own frame.
    :param end_effector: The gripper that would take it.
    :param context: The context the checks run in.
    :return: The measurement.
    """
    # A grasp an annotation generated is already written in its root's frame, so it is
    # its own body_T_grasp.
    sequence_validator = AreReachableBy.for_grasp(
        grasp, end_effector, grasp, context=context
    )
    grasp_validator = IsReachableBy(
        context=context,
        pose=end_effector.tool_frame_goal(grasp),
        tip_link=end_effector.tool_frame,
    )
    return GraspProbe(
        rank=rank,
        distance=end_effector.distance_to_grasp(grasp),
        approach_yaw=_approach_yaw(grasp, context.world),
        grasp_reachable=grasp_validator(),
        sequence_reachable=sequence_validator(),
    )


def _report(probes: list[GraspProbe], destination) -> None:
    """
    Print the measurements as a table and write them for plotting.

    :param probes: The measurements, in ranking order.
    :param destination: Directory to write the comma-separated copy into.
    """
    names = [entry.name for entry in fields(GraspProbe)]
    print(f"\n{' | '.join(name.rjust(18) for name in names)}")
    for probe in probes:
        values = [getattr(probe, name) for name in names]
        formatted = [
            f"{value:.3f}" if isinstance(value, float) else str(value)
            for value in values
        ]
        print(" | ".join(cell.rjust(18) for cell in formatted))

    csv_path = destination / "grasp_candidate_reachability.csv"
    rows = [",".join(names)]
    rows.extend(
        ",".join(str(getattr(probe, name)) for name in names) for probe in probes
    )
    csv_path.write_text("\n".join(rows) + "\n")
    print(f"\nwritten for plotting: {csv_path}")


# %% the experiment


def test_grasp_ranking_does_not_predict_reachability(
    immutable_model_world, tmp_path
) -> None:
    """
    Every grasp the milk offers is measured from one standing pose.

    The point of the experiment is the disagreement it records: the gripper's ranking is
    geometric, so the grasp it puts first is not the grasp the arm can perform, and a
    pick-up that takes the first one is choosing without having asked.
    """
    world, robot, context = immutable_model_world
    rclpy.init()
    VizMarkerPublisher(_world=world, node=rclpy.create_node("test"))
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    milk.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        *WORLD_P_MILK, reference_frame=world.root
    )
    robot.root.parent_connection.origin = stand_facing(
        robot, WORLD_P_STAND, milk.root.global_pose.to_position().to_np(), world
    )
    end_effector = ViewManager.get_end_effector_view(Arms.LEFT, robot)

    grasps = end_effector.grasp_poses_by_distance(milk)
    with ExecutionEnvironment(ExecutionType.SIMULATED, collision_avoidance=True):
        probes = [
            _probe(rank, grasp, end_effector, context)
            for rank, grasp in enumerate(grasps)
        ]
    _report(probes, tmp_path)

    assert len(probes) == milk.grasp_pose_count
    reachable = [probe for probe in probes if probe.sequence_reachable]
    assert reachable, "the standing pose must admit at least one grasp"
    assert not probes[0].sequence_reachable, (
        "the experiment is only worth running while the best-ranked grasp is one the "
        "arm cannot perform; if this fails the ranking and reachability now agree here "
        "and the standing pose needs revisiting"
    )
    assert np.diff([probe.distance for probe in probes]).min() >= 0, (
        "grasp_poses_by_distance must return its grasps best first, since the ranks "
        "reported above are read off that order"
    )
