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
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.garmi import (
    GARMI_SHELL,
    Garmi,
    GarmiLeftArm,
    GarmiRightArm,
)

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


def test_the_shell_is_left_out_of_collision_checking(garmi_world):
    """
    Bodywork geometry is the drawn skin, not a contact model: it overlaps both the robot
    underneath it and the ground the robot stands on, so a check against it can only ever
    report a collision the robot cannot do anything about.
    """
    shell = {garmi_world.get_body_by_name(body_name) for body_name in GARMI_SHELL}
    garmi_world.collision_manager.update_collision_matrix()

    checks = garmi_world.collision_manager.collision_matrix.collision_checks

    assert checks, "the collision matrix is empty, so it proves nothing"
    assert not [
        check for check in checks if check.body_a in shell or check.body_b in shell
    ]


def test_the_shell_keeps_the_shape_it_is_drawn_with(garmi_world):
    """
    Leaving bodywork out of collision checking must not cost the robot its true shape:
    the covers are what its width is measured from.
    """
    for body_name in GARMI_SHELL:
        assert garmi_world.get_body_by_name(body_name).has_collision()
