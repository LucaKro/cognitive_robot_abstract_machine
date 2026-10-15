"""
The world the demonstration plans in: the TIAGo standing in the apartment, with a milk
carton and a cereal box on the counter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path

import coraplex
import numpy as np
import trimesh
from typing_extensions import Type

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cereal,
    Food,
    Milk,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% poses


@dataclass(frozen=True)
class UprightPose:
    """
    A pose turned only about the vertical axis.
    """

    x: float
    """
    The position along the x-axis, in metres.
    """

    y: float
    """
    The position along the y-axis, in metres.
    """

    z: float = 0.0
    """
    The height, in metres.
    """

    yaw: float = 0.0
    """
    The turn about the vertical axis, in radians.
    """

    def to_pose(self, reference_frame: KinematicStructureEntity) -> Pose:
        """
        :param reference_frame: The frame the pose is expressed in.
        :return: This pose in ``reference_frame``.
        """
        return Pose.from_xyz_rpy(
            self.x, self.y, self.z, yaw=self.yaw, reference_frame=reference_frame
        )


# %% resources


class Resource(Enum):
    """
    Files of :mod:`coraplex` the world is built from.
    """

    DIRECTORY = Path(coraplex.__file__).parents[2] / "resources"
    APARTMENT = DIRECTORY / "worlds" / "apartment.urdf"
    OBJECTS = DIRECTORY / "objects"


class Grocery(StrEnum):
    """
    The groceries on the counter, each named after the mesh file it is built from, which
    is also the name of its body.
    """

    MILK = "milk.stl"
    CEREAL = "breakfast_cereal.stl"

    @property
    def mesh_file(self) -> Path:
        """
        The mesh the grocery is built from.
        """
        return Resource.OBJECTS.value / self.value

    @property
    def start_pose(self) -> UprightPose:
        """
        Where the grocery stands on the counter, in the apartment frame.
        """
        return {
            Grocery.MILK: UprightPose(2.37, 2.0, 1.05),
            Grocery.CEREAL: UprightPose(2.37, 2.5, 1.05),
        }[self]

    @property
    def annotation_type(self) -> Type[Food]:
        """
        The semantic annotation the grocery is known by.
        """
        return {Grocery.MILK: Milk, Grocery.CEREAL: Cereal}[self]

    @property
    def mass(self) -> float:
        """
        The grocery's mass, in kilograms: a full one-litre carton and a 500 g box.
        """
        return {Grocery.MILK: 1.0, Grocery.CEREAL: 0.5}[self]

    @property
    def color(self) -> Color:
        """
        The color the grocery is drawn in.
        """
        return {
            Grocery.MILK: Color(0.85, 0.12, 0.10),
            Grocery.CEREAL: Color(0.95, 0.70, 0.10),
        }[self]

    @property
    def mesh_bounds(self) -> np.ndarray:
        """
        The corners of the box bounding the grocery's mesh, lowest first, in its own
        frame.
        """
        return trimesh.load(self.mesh_file).bounds

    @property
    def inertial(self) -> Inertial:
        """
        The inertial of a solid box filling the grocery's bounds, at its mass.
        """
        lower, upper = self.mesh_bounds
        width, depth, height = upper - lower
        return Inertial(
            mass=self.mass,
            center_of_mass=Point3.from_iterable((upper + lower) / 2),
            inertia=InertiaTensor.from_values(
                ixx=self.mass * (depth**2 + height**2) / 12,
                iyy=self.mass * (width**2 + height**2) / 12,
                izz=self.mass * (width**2 + depth**2) / 12,
                ixy=0.0,
                ixz=0.0,
                iyz=0.0,
            ),
        )


# %% world


@dataclass
class TiagoApartment:
    """
    The TIAGo in the apartment, with the groceries free to move on the counter.
    """

    world: World
    """
    The world holding the apartment, the groceries and the robot.
    """

    robot: Tiago
    """
    The robot that carries out the plan.
    """

    @classmethod
    def build(cls, robot_start: UprightPose = UprightPose(1.5, 2.0)) -> TiagoApartment:
        """
        Build the apartment, stand the groceries on the counter and spawn the TIAGo.

        :param robot_start: Where the TIAGo starts in the apartment.
        :return: The world with everything in place.
        """
        world = URDFParser.from_file(str(Resource.APARTMENT.value)).parse()
        for grocery in Grocery:
            world.merge_world_at_pose(
                STLParser(str(grocery.mesh_file)).parse(),
                grocery.start_pose.to_pose(world.root).homogeneous_matrix,
            )
            world.get_body_by_name(grocery).inertial = grocery.inertial
        with world.modify_world():
            world.add_semantic_annotations(
                [
                    grocery.annotation_type(root=world.get_body_by_name(grocery))
                    for grocery in Grocery
                ]
            )
        robot = RobotSpecification(
            Tiago,
            odom_T_robot_start=HomogeneousTransformationMatrix.from_xyz_rpy(
                robot_start.x, robot_start.y, yaw=robot_start.yaw
            ),
        ).spawn(world)
        return cls(world=world, robot=robot)
