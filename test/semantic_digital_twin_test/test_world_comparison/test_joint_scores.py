import math
from copy import deepcopy

import pytest

from semantic_digital_twin.api import (
    FixedConnectionSpecification,
    PrismaticConnectionSpecification,
    RevoluteConnectionSpecification,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_comparison.joint_scores import (
    JointEvaluation,
    JointScore,
    JointScorer,
    RigidGroups,
)
from semantic_digital_twin.world_comparison.matching import BodyMatcher
from semantic_digital_twin.world_comparison.surface_samples import SurfaceSampler
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from .worlds import CabinetScene, position_limits

# %% fixtures


@pytest.fixture
def matcher() -> BodyMatcher:
    return BodyMatcher(
        distance_tolerance=0.02,
        minimum_overlap=0.5,
        minimum_partial_overlap=0.2,
        sampler=SurfaceSampler(spacing=0.01, seed=0),
    )


@pytest.fixture
def identity() -> HomogeneousTransformationMatrix:
    return HomogeneousTransformationMatrix()


def evaluate(
    ground_truth_world: World,
    reconstructed_world: World,
    matcher: BodyMatcher,
    ground_truth_root_T_reconstructed_root: HomogeneousTransformationMatrix,
) -> JointEvaluation:
    correspondence = matcher.match(
        ground_truth_world, reconstructed_world, ground_truth_root_T_reconstructed_root
    )
    return JointScorer().score(correspondence)


def score_of(evaluation: JointEvaluation, moving_body: str) -> JointScore:
    [score] = [
        score
        for score in evaluation.joint_scores
        if score.ground_truth_connection.child.name.name == moving_body
    ]
    return score


def child_names(connections) -> list[str]:
    return [connection.child.name.name for connection in connections]


def assert_perfect(score: JointScore):
    assert score.same_type
    assert score.axis_angle == pytest.approx(0.0, abs=1e-9)
    assert score.travel_error == pytest.approx(0.0, abs=1e-9)


# %% rigid groups


def test_fixed_connections_join_bodies_into_one_group():
    world = CabinetScene().create_world()
    groups = RigidGroups.of_world(world)
    root_group = groups.group_of(world.root)
    door_group = groups.group_of(world.get_body_by_name("door"))

    assert {body.name.name for body in root_group.bodies} == {
        world.root.name.name,
        "door_cabinet",
        "drawer_cabinet",
    }
    assert root_group.parent_connection is None
    assert door_group.parent_connection.child.name.name == "door"


# %% identical joints


def test_identical_joints_score_perfectly(matcher, identity):
    world = CabinetScene().create_world()
    evaluation = evaluate(world, deepcopy(world), matcher, identity)

    assert_perfect(score_of(evaluation, "door"))
    assert_perfect(score_of(evaluation, "drawer"))
    assert score_of(evaluation, "door").axis_distance == pytest.approx(0.0, abs=1e-9)
    assert evaluation.welded_joints == []
    assert evaluation.unmatched_ground_truth_joints == []
    assert evaluation.extra_joints == []


def test_every_joint_of_the_kitchen_matches_its_copy(kitchen_world, identity):
    matcher = BodyMatcher(
        distance_tolerance=0.05,
        minimum_overlap=0.5,
        minimum_partial_overlap=0.2,
        sampler=SurfaceSampler(spacing=0.02, seed=0),
    )
    evaluation = evaluate(kitchen_world, deepcopy(kitchen_world), matcher, identity)

    movable = [
        connection
        for connection in kitchen_world.connections
        if isinstance(connection, ActiveConnection1DOF)
    ]
    assert len(evaluation.joint_scores) == len(movable)
    assert all(score.same_type for score in evaluation.joint_scores)
    scored_by_axis = [
        score for score in evaluation.joint_scores if score.axis_angle is not None
    ]
    for score in scored_by_axis:
        assert_perfect(score)
    without_axis = {
        connection.name for connection in movable if not connection.axis.to_np().any()
    }
    assert {
        score.ground_truth_connection.name
        for score in evaluation.joint_scores
        if score.axis_angle is None
    } == without_axis
    assert evaluation.welded_joints == []


def test_alignment_carries_the_joints_into_the_ground_truth_frame(matcher):
    root_T_scene = HomogeneousTransformationMatrix.from_xyz_rpy(x=5.0, y=-2.0, yaw=0.3)
    moved = CabinetScene(root_T_scene=root_T_scene)

    evaluation = evaluate(
        CabinetScene().create_world(),
        moved.create_world(),
        matcher,
        root_T_scene.inverse(),
    )

    assert_perfect(score_of(evaluation, "door"))
    assert_perfect(score_of(evaluation, "drawer"))
    assert score_of(evaluation, "door").axis_distance == pytest.approx(0.0, abs=1e-9)


# %% the same motion written down differently


def test_flipped_axis_with_negated_limits_is_the_same_joint(matcher, identity):
    flipped = CabinetScene(
        door_connection=RevoluteConnectionSpecification(
            axis=Vector3(0, 0, -1), dof_limits=position_limits(-1.5, 0.0)
        )
    )
    evaluation = evaluate(
        CabinetScene().create_world(), flipped.create_world(), matcher, identity
    )
    assert_perfect(score_of(evaluation, "door"))


def test_negative_multiplier_is_the_same_joint(matcher, identity):
    negated = CabinetScene(
        door_connection=RevoluteConnectionSpecification(
            axis=Vector3(0, 0, 1),
            dof_limits=position_limits(-1.5, 0.0),
            multiplier=-1.0,
        )
    )
    evaluation = evaluate(
        CabinetScene().create_world(), negated.create_world(), matcher, identity
    )
    assert_perfect(score_of(evaluation, "door"))


def test_hinge_moved_along_its_axis_is_the_same_joint(matcher, identity):
    raised_hinge = CabinetScene(
        cabinet_T_hinge=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.32, y=-0.3, z=0.2
        )
    )
    evaluation = evaluate(
        CabinetScene().create_world(), raised_hinge.create_world(), matcher, identity
    )
    assert score_of(evaluation, "door").axis_distance == pytest.approx(0.0, abs=1e-9)


# %% different joints


def test_hinge_moved_sideways_is_measured_at_the_door(matcher, identity):
    shifted_hinge = CabinetScene(
        cabinet_T_hinge=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.32, y=-0.2)
    )
    evaluation = evaluate(
        CabinetScene().create_world(), shifted_hinge.create_world(), matcher, identity
    )
    assert score_of(evaluation, "door").axis_distance == pytest.approx(0.1, abs=1e-9)


def test_tilted_axis_is_measured_as_an_angle(matcher, identity):
    tilt = math.radians(10)
    tilted = CabinetScene(
        door_connection=RevoluteConnectionSpecification(
            axis=Vector3(math.sin(tilt), 0, math.cos(tilt)),
            dof_limits=position_limits(0.0, 1.5),
        )
    )
    evaluation = evaluate(
        CabinetScene().create_world(), tilted.create_world(), matcher, identity
    )
    assert score_of(evaluation, "door").axis_angle == pytest.approx(tilt, abs=1e-9)


def test_shorter_travel_is_measured_in_the_joints_unit(matcher, identity):
    shorter = CabinetScene(
        door_connection=RevoluteConnectionSpecification(
            axis=Vector3(0, 0, 1), dof_limits=position_limits(0.0, 1.0)
        )
    )
    evaluation = evaluate(
        CabinetScene().create_world(), shorter.create_world(), matcher, identity
    )
    assert score_of(evaluation, "door").travel_error == pytest.approx(0.5, abs=1e-9)


def test_joint_of_another_type_is_compared_by_type_only(matcher, identity):
    sliding_door = CabinetScene(
        door_connection=PrismaticConnectionSpecification(
            axis=Vector3(0, 1, 0), dof_limits=position_limits(0.0, 0.5)
        )
    )
    evaluation = evaluate(
        CabinetScene().create_world(), sliding_door.create_world(), matcher, identity
    )
    score = score_of(evaluation, "door")
    assert not score.same_type
    assert (score.axis_angle, score.axis_distance, score.travel_error) == (
        None,
        None,
        None,
    )
    assert evaluation.type_accuracy == 0.5


def test_joint_without_an_axis_is_compared_by_type_only(matcher, identity):
    stuck_drawer = PrismaticConnectionSpecification(
        axis=Vector3(0, 0, 0), dof_limits=position_limits(0.0, 0.4)
    )
    scene = CabinetScene(drawer_connection=stuck_drawer)
    evaluation = evaluate(scene.create_world(), scene.create_world(), matcher, identity)
    score = score_of(evaluation, "drawer")
    assert score.same_type
    assert (score.axis_angle, score.axis_distance, score.travel_error) == (
        None,
        None,
        None,
    )


# %% joints the reconstruction lacks or adds


def test_joint_fixed_in_the_reconstruction_is_welded(matcher, identity):
    welded = CabinetScene(door_connection=FixedConnectionSpecification())
    evaluation = evaluate(
        CabinetScene().create_world(), welded.create_world(), matcher, identity
    )
    assert child_names(evaluation.welded_joints) == ["door"]
    assert child_names(
        score.ground_truth_connection for score in evaluation.joint_scores
    ) == ["drawer"]


def test_joint_only_the_reconstruction_has_is_extra(matcher, identity):
    evaluation = evaluate(
        CabinetScene(door_connection=FixedConnectionSpecification()).create_world(),
        CabinetScene().create_world(),
        matcher,
        identity,
    )
    assert child_names(evaluation.extra_joints) == ["door"]


def test_joint_of_a_missed_part_is_unmatched(matcher, identity):
    evaluation = evaluate(
        CabinetScene().create_world(),
        CabinetScene(has_drawer=False).create_world(),
        matcher,
        identity,
    )
    assert child_names(evaluation.unmatched_ground_truth_joints) == ["drawer"]
    assert evaluation.welded_joints == []
