"""
The physics choices the demonstration applies to the exported world.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from pathlib import Path

import mujoco

from semantic_digital_twin.world_description.inertial_properties import Inertial

from experiments.robosimstudio_tiago_demo.scene import (
    Attribute,
    BaseJoint,
    ContactSetting,
    Element,
    GeometryType,
    PhysicsScene,
    as_mjcf_vector,
)
from experiments.robosimstudio_tiago_demo.world import Grocery

# %% helpers


def collides(geometry: ElementTree.Element) -> bool:
    """
    :return: Whether ``geometry`` takes part in contacts.
    """
    return geometry.get(Attribute.CONTACT_TYPE) != ContactSetting.NONE


# %% the scene as a whole


def test_the_scene_loads_in_mujoco(physics_scene: PhysicsScene, scene_file: Path):
    model = mujoco.MjModel.from_xml_path(str(scene_file))

    assert model.nu == len(physics_scene.choices.servo_gains)


# %% the robot


def test_the_base_moves_on_its_own_joints(physics_scene: PhysicsScene):
    [robot_root, *_] = physics_scene.robot_bodies

    assert [
        (joint.get(Attribute.NAME), joint.get(Attribute.TYPE))
        for joint in robot_root.findall(Element.JOINT)
    ] == [(joint, joint.joint_type) for joint in BaseJoint]


def test_every_commanded_joint_is_driven_by_a_servo_with_its_gains(
    physics_scene: PhysicsScene,
):
    servos = {
        servo.get(Attribute.JOINT): (
            float(servo.get(Attribute.STIFFNESS)),
            float(servo.get(Attribute.VELOCITY_GAIN)),
        )
        for servo in physics_scene.root.find(Element.ACTUATOR).iter(Element.POSITION)
    }

    assert servos == {
        joint: (gains.stiffness, gains.damping)
        for joint, gains in physics_scene.choices.servo_gains.items()
    }


def test_only_the_finger_pads_of_the_robot_collide_and_grip(
    physics_scene: PhysicsScene,
):
    colliding = [
        (body.get(Attribute.NAME), geometry)
        for body in physics_scene.robot_bodies
        for geometry in body.findall(Element.GEOM)
        if collides(geometry)
    ]

    assert {name for name, _ in colliding} <= set(physics_scene.grip_body_names)
    assert {geometry.get(Attribute.FRICTION) for _, geometry in colliding} == {
        as_mjcf_vector(physics_scene.choices.grip_friction)
    }


def test_no_robot_link_without_geometry_keeps_the_default_mass(
    physics_scene: PhysicsScene,
):
    frame_only_masses = {
        float(body.find(Element.INERTIAL).get(Attribute.MASS))
        for body in physics_scene.robot_bodies
        if not body.findall(Element.GEOM) and body.find(Element.INERTIAL) is not None
    }

    assert Inertial().mass not in frame_only_masses


# %% the apartment and the groceries


def test_the_apartment_does_not_collide_with_itself(physics_scene: PhysicsScene):
    apartment_names = [
        body.get(Attribute.NAME) for body in physics_scene.apartment_bodies
    ]
    excluded = {
        frozenset((exclude.get(Attribute.BODY_1), exclude.get(Attribute.BODY_2)))
        for exclude in physics_scene.root.find(Element.CONTACT).iter(Element.EXCLUDE)
    }

    assert all(
        frozenset((first, second)) in excluded
        for index, first in enumerate(apartment_names)
        for second in apartment_names[index + 1 :]
    )


def test_each_grocery_collides_as_a_box_and_shows_in_its_color(
    physics_scene: PhysicsScene,
):
    for body in physics_scene.grocery_bodies:
        grocery = Grocery(body.get(Attribute.NAME))
        geometries = body.findall(Element.GEOM)

        assert [
            geometry.get(Attribute.TYPE)
            for geometry in geometries
            if collides(geometry)
        ] == [GeometryType.BOX]
        assert [
            geometry.get(Attribute.COLOR)
            for geometry in geometries
            if not collides(geometry)
        ] == [as_mjcf_vector(grocery.color.to_rgba())]


def test_each_grocery_weighs_its_own_mass(physics_scene: PhysicsScene):
    masses = {
        body.get(Attribute.NAME): float(body.find(Element.INERTIAL).get(Attribute.MASS))
        for body in physics_scene.grocery_bodies
    }

    assert masses == {grocery.value: grocery.mass for grocery in Grocery}


# %% single precision


def test_every_inertia_satisfies_the_triangle_inequality_in_single_precision(
    physics_scene: PhysicsScene,
):
    margin = physics_scene.choices.single_precision_margin
    for inertial in physics_scene.root.iter(Element.INERTIAL):
        smallest, middle, largest = sorted(
            float(value) for value in inertial.get(Attribute.DIAGONAL_INERTIA).split()
        )
        assert smallest + middle >= margin * largest
