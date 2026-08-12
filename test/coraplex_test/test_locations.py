from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pytest
from typing_extensions import Iterator, List

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.locations.backends import GiskardLocationBackend
from coraplex.locations.base import Location, PoseGeneratorBackend, PoseValidator
from coraplex.view_manager import ViewManager
from giskardpy.motion_statechart.exceptions import CollisionViolatedError
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% test doubles


@dataclass
class FixedPoseGenerator(PoseGeneratorBackend):
    """
    Yields predetermined candidates, so a location's placement can be asserted exactly.
    """

    poses: List[Pose]
    """
    The candidates to yield, in order.
    """

    def __iter__(self) -> Iterator[Pose]:
        return iter(self.poses)


@dataclass
class RecordsEvaluatedRobot(PoseValidator):
    """
    Accepts every candidate and records the robot it was evaluated against.
    """

    evaluated_robots: List[AbstractRobot] = field(default_factory=list)
    """
    The robot annotation each candidate was evaluated against, in evaluation order.
    """

    evaluated_root_poses: List[Pose] = field(default_factory=list)
    """
    Where the evaluated robot's root stood in the world frame, in evaluation order.
    """

    def __call__(self, *args, **kwargs) -> bool:
        self.evaluated_robots.append(self.robot)
        self.evaluated_root_poses.append(self.robot.root.global_pose)
        return True


@dataclass
class MotionlessExecutor:
    """
    Stands in for a Giskard executor and leaves the world exactly as it found it.
    """

    def tick_until_end(self, *args, **kwargs) -> None:
        pass


@dataclass
class UnsolvableMotionExecutor:
    """
    Stands in for a Giskard executor whose motion never reaches its goal.
    """

    outcome: Exception
    """
    The exception the motion ends with instead of reaching its goal.
    """

    def tick_until_end(self, *args, **kwargs) -> None:
        raise self.outcome


# %% specification-built worlds whose odom is displaced

# The drive is an OmniDrive, which represents x, y and yaw only, so the odom offsets stay
# in that plane. The environment holds nothing but the robots, so a candidate is never
# rejected for collision and each test fails only for the behaviour it names.
_FIRST_ODOM = HomogeneousTransformationMatrix.from_xyz_rpy(0.5, 0.5, 0, yaw=np.pi / 2)
_SECOND_ODOM = HomogeneousTransformationMatrix.from_xyz_rpy(
    -2.0, 1.0, 0, yaw=-np.pi / 4
)


def _world_with_robots_behind_displaced_odoms(
    *world_T_odoms: HomogeneousTransformationMatrix,
) -> World:
    """
    A world holding nothing but PR2s, each reached through its own displaced odom.
    """
    specification = WorldSpecification(
        world_parser=None,
        robots=[
            RobotSpecification(semantic_annotation_type=PR2, world_T_odom=world_T_odom)
            for world_T_odom in world_T_odoms
        ],
    )
    try:
        return specification.to_domain_object()
    except ParsingError as error:
        pytest.skip(f"PR2 URDF not available: {error}")


@pytest.fixture(scope="session")
def _single_robot_world_setup() -> World:
    return _world_with_robots_behind_displaced_odoms(_FIRST_ODOM)


@pytest.fixture
def single_robot_world(_single_robot_world_setup):
    world = deepcopy(_single_robot_world_setup)
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    return world, robot, Context(world, robot)


@pytest.fixture(scope="session")
def _two_robot_world_setup() -> World:
    return _world_with_robots_behind_displaced_odoms(_FIRST_ODOM, _SECOND_ODOM)


@pytest.fixture
def two_robot_world(_two_robot_world_setup):
    return deepcopy(_two_robot_world_setup)


def _candidate(world: World) -> Pose:
    return Pose.from_xyz_rpy(1.3, 2.0, 0.0, yaw=0.25, reference_frame=world.root)


_OBSTACLE_SCALE = Scale(0.3, 0.3, 0.4)
"""
The extents of the obstacle the collision tests place in front of the robot.

As tall as the robot's base and no taller, so the arms above it are never what the
candidate is judged by.
"""

_CLEAR_OF_THE_BASE = 0.08
"""
A gap, in meters, that the robot's base is reported close to but not in collision with.

Wider than the margin the PR2's own rules keep around its base, narrower than the
distance at which the collision detector stops reporting the pair at all -- which is what
makes a candidate at this distance tell a proximity check apart from a collision check.
"""


def _box_at(world: World, position: Point3, scale: Scale) -> Body:
    """
    Put a box into the world, standing on its own in the world frame.

    :param world: The world the box is added to.
    :param position: Where the box's center goes.
    :param scale: The extents of the box.
    :return: The box, which stands in either as an obstacle or as something to reach
        for.
    """
    coordinates = np.asarray(position.to_np()).flatten()[:3]
    box = Body(
        name=PrefixedName(f"box_{world.bodies.__len__()}", "coraplex_test"),
        collision=ShapeCollection([Box(scale=scale)]),
    )
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    *coordinates
                ),
            )
        )
    return box


def _box_in_front_of(world: World, robot: AbstractRobot, gap: float) -> Body:
    """
    Put a box in front of the world origin, ``gap`` meters clear of the base a robot
    standing there would occupy.

    :param world: The world the box is added to.
    :param robot: The robot whose base decides how far out the box goes.
    :param gap: The clearance between the base and the box, negative to overlap.
    :return: The box, which stands in either as an obstacle or as something to reach
        for.
    """
    base_box = robot.mobile_base.bounding_box
    return _box_at(
        world,
        Point3.from_iterable(
            [
                base_box.depth / 2 + _OBSTACLE_SCALE.x / 2 + gap,
                0.0,
                _OBSTACLE_SCALE.z / 2,
            ]
        ),
        _OBSTACLE_SCALE,
    )


def _candidate_at_the_origin(world: World) -> Pose:
    """
    :return: A candidate that puts the robot's root on the world origin, facing the
        box :func:`_box_in_front_of` places.
    """
    return Pose.from_xyz_rpy(0.0, 0.0, 0.0, reference_frame=world.root)


# %% a location evaluates candidates where the world frame says they are


def test_location_places_the_robot_at_the_candidate_in_the_world_frame(
    single_robot_world,
):
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    recorder = RecordsEvaluatedRobot()

    list(Location(context, candidate, FixedPoseGenerator([candidate]), [recorder]))

    np.testing.assert_allclose(
        recorder.evaluated_root_poses[0].to_np(), candidate.to_np(), atol=1e-9
    )


def test_location_yields_the_pose_it_evaluated(single_robot_world):
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    recorder = RecordsEvaluatedRobot()

    yielded_poses = list(
        Location(context, candidate, FixedPoseGenerator([candidate]), [recorder])
    )

    assert len(yielded_poses) == 1
    np.testing.assert_allclose(
        yielded_poses[0].to_np(),
        recorder.evaluated_root_poses[0].to_np(),
        atol=1e-9,
    )


# %% a location evaluates the robot of its context


def test_location_evaluates_the_robot_of_its_context(two_robot_world):
    world = two_robot_world
    second_robot = world.get_semantic_annotations_by_type(PR2)[1]
    context = Context(world, second_robot)
    recorder = RecordsEvaluatedRobot()
    candidate = _candidate(world)

    list(Location(context, candidate, FixedPoseGenerator([candidate]), [recorder]))

    assert recorder.evaluated_robots[0].id == second_robot.id


# %% the giskard backend reports the pose it placed the robot at


def test_giskard_backend_yields_the_candidate_it_placed_the_robot_at(
    single_robot_world, monkeypatch
):
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, robot)
    backend = GiskardLocationBackend(
        target=candidate,
        arm=Arms.RIGHT,
        grasp_description=GraspDescription(
            ApproachDirection.FRONT, VerticalAlignment.NoAlignment, end_effector
        ),
        robot=robot,
        world=world,
    )
    monkeypatch.setattr(
        GiskardLocationBackend, "setup_costmap", lambda self, pose: [candidate]
    )
    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_giskard_executor",
        lambda self, *args, **kwargs: MotionlessExecutor(),
    )

    yielded_poses = list(backend)

    assert len(yielded_poses) == 1
    np.testing.assert_allclose(yielded_poses[0].to_np(), candidate.to_np(), atol=1e-9)


# %% a location rejects what execution would cancel on, and nothing else


def test_location_keeps_a_candidate_that_only_stands_close_to_an_obstacle(
    single_robot_world,
):
    world, robot, context = single_robot_world
    _box_in_front_of(world, robot, gap=_CLEAR_OF_THE_BASE)
    candidate = _candidate_at_the_origin(world)

    yielded_poses = list(
        Location(context, candidate, FixedPoseGenerator([candidate]), [])
    )

    assert len(yielded_poses) == 1
    np.testing.assert_allclose(yielded_poses[0].to_np(), candidate.to_np(), atol=1e-9)


def test_location_rejects_a_candidate_that_stands_inside_an_obstacle(
    single_robot_world,
):
    world, robot, context = single_robot_world
    _box_in_front_of(world, robot, gap=-_OBSTACLE_SCALE.x)
    candidate = _candidate_at_the_origin(world)

    yielded_poses = list(
        Location(context, candidate, FixedPoseGenerator([candidate]), [])
    )

    assert yielded_poses == []


# %% a location leaves the world it was built on alone


def _backend_reaching_for(
    target: Pose | Body, robot: AbstractRobot, world: World
) -> GiskardLocationBackend:
    """
    :return: A backend that reaches for ``target`` with the robot's right arm.
    """
    end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, robot)
    return GiskardLocationBackend(
        target=target,
        arm=Arms.RIGHT,
        grasp_description=GraspDescription(
            ApproachDirection.FRONT, VerticalAlignment.NoAlignment, end_effector
        ),
        robot=robot,
        world=world,
    )


def test_location_leaves_the_context_world_untouched(single_robot_world, monkeypatch):
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    backend = _backend_reaching_for(candidate, robot, world)
    monkeypatch.setattr(
        GiskardLocationBackend, "setup_costmap", lambda self, pose: [candidate]
    )
    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_giskard_executor",
        lambda self, *args, **kwargs: MotionlessExecutor(),
    )
    pose_before = robot.root.global_pose.to_np()

    list(Location(context, candidate, backend, []))

    np.testing.assert_allclose(robot.root.global_pose.to_np(), pose_before, atol=1e-9)


# %% the giskard backend only reports poses it solved for


@pytest.mark.parametrize(
    "outcome",
    [TimeoutError(), CollisionViolatedError(violated_collisions=[], thresholds=[])],
    ids=["timeout", "collision"],
)
def test_giskard_backend_skips_a_candidate_whose_motion_does_not_reach_the_target(
    single_robot_world, monkeypatch, outcome
):
    world, robot, context = single_robot_world
    unreachable, reachable = _candidate(world), _candidate_at_the_origin(world)
    backend = _backend_reaching_for(unreachable, robot, world)
    executors = iter([UnsolvableMotionExecutor(outcome), MotionlessExecutor()])
    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_costmap",
        lambda self, pose: [unreachable, reachable],
    )
    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_giskard_executor",
        lambda self, *args, **kwargs: next(executors),
    )

    yielded_poses = list(backend)

    assert len(yielded_poses) == 1
    np.testing.assert_allclose(yielded_poses[0].to_np(), reachable.to_np(), atol=1e-9)


def test_giskard_backend_builds_an_executor_for_every_candidate(
    single_robot_world, monkeypatch
):
    world, robot, context = single_robot_world
    candidates = [_candidate(world), _candidate_at_the_origin(world)]
    backend = _backend_reaching_for(candidates[0], robot, world)
    built_executors = []
    monkeypatch.setattr(
        GiskardLocationBackend, "setup_costmap", lambda self, pose: candidates
    )

    def build(self, *args, **kwargs):
        built_executors.append(MotionlessExecutor())
        return built_executors[-1]

    monkeypatch.setattr(GiskardLocationBackend, "setup_giskard_executor", build)

    list(backend)

    assert len(built_executors) == len(candidates)


def test_giskard_backend_solves_the_reach_the_grasp_performs(
    single_robot_world, monkeypatch
):
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    target = _box_in_front_of(world, robot, gap=0.5)
    backend = _backend_reaching_for(target, robot, world)
    solved_sequences = []
    monkeypatch.setattr(
        GiskardLocationBackend, "setup_costmap", lambda self, pose: [candidate]
    )

    def record(self, pose_sequence, *args, **kwargs):
        solved_sequences.append(pose_sequence)
        return MotionlessExecutor()

    monkeypatch.setattr(GiskardLocationBackend, "setup_giskard_executor", record)

    list(backend)

    pre_pose, grasp_pose, _ = backend.grasp_description.grasp_pose_sequence(target)
    assert [pose.to_np().tolist() for pose in solved_sequences[0]] == [
        pre_pose.to_np().tolist(),
        grasp_pose.to_np().tolist(),
    ]
