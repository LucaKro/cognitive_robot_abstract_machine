"""
Coverage for the GARMI annotation: what of the robot can be collided with, where its arms
park, and the collision rules it registers.
"""

import os
from importlib.resources import files
from pathlib import Path
from xml.etree import ElementTree

import pytest

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AvoidSelfCollisions,
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.garmi import Garmi, GarmiLeftArm, GarmiRightArm

COLLISION_CONFIG = os.path.join(
    Path(files("semantic_digital_twin")).parent.parent,
    "resources",
    "collision_configs",
    "garmi.srdf",
)
"""
The collision matrix GARMI loads, read here to compare against what it registered.
"""


@pytest.fixture(scope="module")
def garmi_world():
    """
    A world holding nothing but GARMI, spawned the way a demonstration spawns it.
    """
    try:
        return WorldSpecification(
            robots=[RobotSpecification(semantic_annotation_type=Garmi)]
        ).to_domain_object()
    except ParsingError as error:
        pytest.skip(f"GARMI URDF not available: {error}")


@pytest.fixture(scope="module")
def garmi(garmi_world) -> Garmi:
    """
    The GARMI annotation of that world.
    """
    return garmi_world.get_semantic_annotations_by_type(Garmi)[0]


def park_targets(arm) -> dict:
    """
    What an arm's park state asks of each of its joints, keyed by connection name.

    :param arm: The arm whose park state is read.
    """
    joint_state = arm.get_joint_state_by_type(StaticJointState.PARK)
    return {
        connection.name.name: target
        for connection, target in zip(
            joint_state.connections, joint_state.target_values
        )
    }


def declared_park_position(arm_type, connection_name: str) -> float:
    """
    The position an arm type's park configuration declares for one of its connections.

    :param arm_type: The arm class carrying the configuration.
    :param connection_name: Name of the connection to look up.
    """
    return next(
        position
        for joint_name, position in arm_type.ARM_PARK_CONFIGURATION.items()
        if connection_name.endswith(joint_name)
    )


# %% collision geometry from visuals


def test_every_drawn_link_can_be_collided_with(garmi_world):
    """
    Parts of GARMI's shell are drawn but never described for contact, so without the
    visuals standing in for them the bodies that bound its real width do not collide.
    """
    description = URDFParser.from_file(Garmi.get_ros_file_path())
    drawn_only = [
        link.name
        for link in description.parsed.links
        if link.visuals and not link.collisions
    ]
    assert drawn_only, "the description no longer has a link that is only drawn"

    for link_name in drawn_only:
        assert garmi_world.get_body_by_name(link_name).has_collision()


# %% park configuration


def test_each_arm_parks_where_its_configuration_says(garmi):
    """
    ``ParkArmsAction`` drives the joints to the park state built from
    ``ARM_PARK_CONFIGURATION``, so the two have to agree.
    """
    for arm_type in (GarmiLeftArm, GarmiRightArm):
        arm = garmi._world.get_semantic_annotations_by_type(arm_type)[0]
        targets = park_targets(arm)
        assert targets
        for connection_name, target in targets.items():
            assert target == declared_park_position(arm_type, connection_name)


def test_the_arms_park_turned_away_from_each_other(garmi):
    """
    The elbows are rolled to opposite sides so the parked arms sit clear of the torso and
    of one another.
    """
    left = GarmiLeftArm.ARM_PARK_CONFIGURATION["fr3_joint3"]
    right = GarmiRightArm.ARM_PARK_CONFIGURATION["fr3_joint3"]

    assert left == -right
    assert left != 0.0


# %% collision rules


def shell_rule(world) -> AvoidExternalCollisions:
    """
    The rule covering only part of the robot, which is the one guarding the base shell.

    :param world: The world the robot registered its rules in.
    """
    return next(
        rule
        for rule in world.collision_manager.default_rules
        if isinstance(rule, AvoidExternalCollisions) and rule.body_subset is not None
    )


def whole_robot_rule(world) -> AvoidExternalCollisions:
    """
    The rule covering the whole robot.

    :param world: The world the robot registered its rules in.
    """
    return next(
        rule
        for rule in world.collision_manager.default_rules
        if isinstance(rule, AvoidExternalCollisions) and rule.body_subset is None
    )


def test_the_base_shell_keeps_a_wider_berth_than_the_rest(garmi_world):
    """
    The shell is what bumps into furniture first, so it is held further off than the rest
    of the robot is.
    """
    shell = shell_rule(garmi_world)

    assert (
        shell.buffer_zone_distance > whole_robot_rule(garmi_world).buffer_zone_distance
    )
    assert shell.violated_distance > whole_robot_rule(garmi_world).violated_distance


def test_the_bodies_the_shell_rule_guards_all_collide(garmi_world):
    """
    A rule naming a body that carries no collision geometry guards nothing.
    """
    for body in shell_rule(garmi_world).body_subset:
        assert body.has_collision()


def test_the_collision_matrix_switches_off_the_links_its_file_names(garmi_world):
    """
    The SRDF's ``disable_all_collisions`` links are taken out of collision checking
    altogether, rather than pair by pair.
    """
    switched_off = {
        element.attrib["link"]
        for element in ElementTree.parse(COLLISION_CONFIG).getroot()
        if element.tag == SelfCollisionMatrixRule.SRDF_DISABLE_ALL_COLLISIONS
    }
    assert switched_off, "the collision matrix no longer switches any link off entirely"

    matrix = next(
        rule
        for rule in garmi_world.collision_manager.ignore_collision_rules
        if isinstance(rule, SelfCollisionMatrixRule)
    )

    assert {body.name.name for body in matrix.allowed_collision_bodies} == switched_off


def test_the_shell_never_collides_with_the_robot_it_covers(garmi_world):
    """
    The shell's geometry overlaps what it covers by construction -- the side covers reach
    several centimetres into the chassis -- so any such pair left in collision checking
    puts the robot in permanent violation of its own self-collision rule.
    """
    robot = garmi_world.get_semantic_annotations_by_type(Garmi)[0]
    buffer_zone = next(
        rule.buffer_zone_distance
        for rule in garmi_world.collision_manager.default_rules
        if isinstance(rule, AvoidSelfCollisions)
    )
    disabled = set()
    for rule in garmi_world.collision_manager.ignore_collision_rules:
        rule.update(garmi_world)
        disabled |= {
            frozenset((check.body_a.name.name, check.body_b.name.name))
            for check in rule.allowed_collision_pairs
        }

    shell = shell_rule(garmi_world).body_subset
    detector = garmi_world.collision_manager.collision_detector
    touching = [
        (covering.name.name, covered.name.name)
        for covering in shell
        for covered in robot.bodies_with_collision
        if covered not in shell
        and detector.check_collision_between_bodies(covering, covered, buffer_zone)
    ]

    assert touching, "the shell no longer overlaps anything it covers"
    assert all(frozenset(pair) in disabled for pair in touching)
