"""
A PR2 takes a milk out of the small kitchen's fridge and puts it down on the kitchen
island.

The plan states the transport only. That the milk stands behind a closed fridge door is
not written into it: the container the milk is in is found from the milk, and the steps
that open it and shut it again are wrapped around the transport.

The world is built from specifications: the kitchen comes from a URDF, the robot from a
:class:`~semantic_digital_twin.api.RobotSpecification`, and the shelf and the milk from
body and annotation specifications placed relative to the fridge.

..note:: This demonstration is written for a simulated run. Running it with
    :attr:`~coraplex.datastructures.enums.ExecutionType.REAL` fetches the world from a
    controller, and the world fetcher does not run
    :class:`~semantic_digital_twin.reasoning.world_reasoner.WorldReasoner` over what it
    receives, so the fridge annotation the scene is placed against would be missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from typing_extensions import ClassVar, List, Optional

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    ApproachDirection,
    Arms,
    ExecutionType,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.demonstrations import RobotDemonstration
from coraplex.locations.base import DeferredLocation
from coraplex.locations.factories import giskard_reachability_location
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import ActionLike, PlanNode
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.view_manager import ViewManager
from krrood.entity_query_language.factories import (
    a,
    an,
    contains,
    entity,
    the,
    variable,
)
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.mixins import (
    HasCaseAsRootBody,
    HasHandle,
    HasMechanicalJoint,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Fridge,
    Milk,
    ShelfLayer,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.world_entity import Body

# %% the demonstration


@dataclass
class KitchenFridgeDemonstration(RobotDemonstration):
    """
    A robot fetches a milk out of the kitchen fridge and puts it on the kitchen island.
    """

    ros_node_name: ClassVar[str] = "kitchen_fridge_demo_node"

    kitchen_urdf: Path = field(
        default_factory=lambda: Path(__file__).parents[2]
        / "resources"
        / "worlds"
        / "kitchen-small.urdf"
    )
    """
    The kitchen the demo plays in.

    Its fridge door is the only door in it that opens.
    """

    robot_start_pose: Pose = field(
        default_factory=lambda: Pose.from_xyz_rpy(0.0, 0.0, 0.0)
    )
    """
    Where the robot starts, on the free floor between the fridge and the kitchen island.

    The PR2's root is its ``base_footprint``, so a height of zero puts it on the floor.
    """

    milk_name: str = "milk"
    """
    Name of the transported body.
    """

    shelf_layer_name: str = "fridge_shelf"
    """
    Name of the shelf layer the milk stands on.
    """

    island_surface_name: str = "kitchen_island_surface"
    """
    Name of the kitchen island body the milk is put down on.
    """

    milk_scale: Scale = field(default_factory=lambda: Scale(0.065, 0.065, 0.2))
    """
    The extents of the milk carton.
    """

    shelf_layer_scale: Scale = field(default_factory=lambda: Scale(0.45, 0.5, 0.02))
    """
    The extents of the shelf layer the milk stands on.

    Fits into the fridge cavity, which is about 0.03 meters narrower than the shell on
    every side.
    """

    shelf_layer_color: Color = field(default_factory=lambda: Color(0.9, 0.93, 0.95))
    """
    The colour of the shelf layer, an off-white against the fridge's own shell.
    """

    fridge_T_shelf_layer: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            0.0, 0.02, 0.0, yaw=np.pi
        )
    )
    """
    The shelf layer in the fridge frame, centered in the cavity.

    The kitchen places its fridge turned by half a turn against the room, so the layer
    turns back: everything spawned below it is then aligned with the room, and grasped
    from the front like any other object standing in it.
    """

    shelf_layer_T_milk: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            -0.16, 0.0, 0.11
        )
    )
    """
    The milk on the shelf layer, standing near its front edge.

    Layer x runs towards the fridge opening, and the layer is 0.4 meters deep, so this
    leaves the 0.065 meter carton just clear of the edge. The further forward it stands,
    the further back the robot can stand to take it, and the less it has to lean into
    the swing of the open door.
    """

    place_x_on_island: float = -0.8
    """
    Where on the kitchen island the milk ends up, in x.

    The southern half of the island, clear of the stove and of the drawer handles.
    """

    place_y_on_island: float = 1.2
    """
    Where on the kitchen island the milk ends up, in y.

    The height is measured from the island surface instead of being fixed here.
    """

    island_approach_yaw: float = np.pi
    """
    The heading the robot faces at the kitchen island, which it reaches from positive x.
    """

    milk_arm: Arms = Arms.RIGHT
    """
    The arm carrying the milk.

    Reaching into the fridge means standing to the left of the opening, out of the swing
    of the open door, which leaves the milk on the robot's right.
    """

    milk_approach_direction: ApproachDirection = ApproachDirection.FRONT
    """
    The side of the milk carton the gripper comes from.
    """

    door_arm: Arms = Arms.LEFT
    """
    The arm opening and closing the fridge door.

    The robot stands to the left of the fridge opening, out of the swing of the open
    door, which puts the handle on its left.
    """

    handle_approach_direction: ApproachDirection = ApproachDirection.BACK
    """
    The side of the fridge door handle the gripper comes from.

    The fridge, and with it its handle, is turned by half a turn against the room, so
    the handle's own front points into the fridge and is reached from its back. The
    handle turns with the door, so this holds whether the door is being opened or shut.
    """

    door_speed_limit: float = 1.0
    """
    How fast the fridge door may swing, in radians per second.

    The kitchen's URDF describes the hinge with 10 rad/s, a speed a door does not turn
    at. The controller keeps a braking margin from a joint's own position limit that
    grows with that limit, and at 10 rad/s the margin is wide enough that the door can
    neither be swung fully open nor shut.
    """

    door_opening_angle: float = 1.45
    """
    How far the fridge door is swung open, in radians.

    Wide enough to reach past it into the fridge, short of the hinge's 1.5708 limit so
    the swing does not have to be timed against the door frame.
    """

    door_closing_angle: float = 0.0
    """
    How far the fridge door is asked to stand open after closing, in radians: shut.
    """

    containment_ratio: float = 0.9
    """
    How much of a body's volume has to lie inside another body for it to count as
    standing in it.
    """

    door_is_closed_angle: float = 0.02
    """
    Up to which fridge door angle, in radians, the door counts as closed.

    The closing motion parks the door within about a hundredth of a radian of the
    hinge's limit, half a degree of swing, so this is how exactly the door shuts rather
    than how far ajar it may stand.
    """

    placement_tolerance: float = 0.02
    """
    How far, in meters, the milk may end up from where it was placed.

    Nothing here is subject to gravity or contact forces, so a placement that goes to
    plan lands on the target rather than near it.
    """

    def build_simulated_world(self) -> World:
        """
        Load the kitchen from its URDF, put the robot in it and infer what its bodies
        mean, which is what turns the fridge's parts into a fridge with a door and a
        handle.
        """
        world = WorldSpecification.from_urdf(
            str(self.kitchen_urdf),
            robots=[
                RobotSpecification(
                    semantic_annotation_type=self.used_robot,
                    world_T_odom=self.robot_start_pose.to_homogeneous_matrix(),
                )
            ],
        ).to_domain_object()

        with world.modify_world():
            WorldReasoner(world).reason()

        hinge = self.fridge_door(world).root.parent_connection
        hinge.raw_dof.limits.upper.velocity = self.door_speed_limit
        hinge.raw_dof.limits.lower.velocity = -self.door_speed_limit

        return world

    def is_scene_populated(self, world: World) -> bool:
        milk = variable(Milk, domain=world.semantic_annotations)
        return next(an(entity(milk)).evaluate(), None) is not None

    def populate_scene(self, world: World) -> None:
        """
        Spawn the fridge's shelf layer and the milk standing on it.
        """
        fridge = variable(Fridge, domain=world.semantic_annotations)
        fridge_annotation = the(entity(fridge)).first()
        shelf_layer = ShelfLayer.get_annotation_specification(
            self.shelf_layer_name,
            ShelfLayer.get_default_root_kinematic_structure_entity_specification(
                scale=self.shelf_layer_scale
            ),
        ).spawn(
            world,
            parent=fridge_annotation.root,
            parent_T_self=self.fridge_T_shelf_layer,
        )
        with world.modify_world():
            fridge_annotation.add(shelf_layer)
            for shape in shelf_layer.root.visual.shapes:
                shape.color = self.shelf_layer_color

        Milk.get_annotation_specification(
            self.milk_name,
            Milk.get_default_root_kinematic_structure_entity_specification(
                scale=self.milk_scale
            ),
        ).spawn(world, parent=shelf_layer.root, parent_T_self=self.shelf_layer_T_milk)

    def fridge_door(self, world: World) -> Door:
        """
        :param world: The world holding the kitchen.
        :return: The fridge door, told apart from the kitchen's other doors by belonging
            to the fridge.
        """
        fridge = variable(Fridge, domain=world.semantic_annotations)
        door = variable(Door, domain=world.semantic_annotations)
        return the(entity(door).where(contains(fridge.doors, door))).first()

    def place_pose_on_island(self, world: World) -> Pose:
        """
        :param world: The world holding the kitchen.
        :return: The pose the milk is placed at, standing on the kitchen island surface.
        """
        island_surface = variable(Body, domain=world.bodies)
        surface_height = (
            the(
                entity(island_surface).where(
                    island_surface.name.name == self.island_surface_name
                )
            )
            .first()
            .collision.as_bounding_box_collection_in_frame(world.root)
            .bounding_box()
            .max_z
        )
        return Pose.from_xyz_rpy(
            self.place_x_on_island,
            self.place_y_on_island,
            surface_height + self.milk_scale.z / 2,
            yaw=self.island_approach_yaw,
            reference_frame=world.root,
        )

    def build_context(self, world: World) -> Context:
        """
        Build the plan context around the robot in ``world``.
        """
        robot = variable(self.used_robot, domain=world.semantic_annotations)
        context = Context(
            world=world,
            robot=the(entity(robot)).first(),
            ros_node=self.ros_node,
            evaluate_conditions=False,
            alternative_motion_mappings=self.alternative_motion_mappings,
        )
        context.debug = True
        return context

    # %% reaching into containers

    @staticmethod
    def container_of(openable: HasMechanicalJoint) -> Optional[Body]:
        """
        :param openable: A part that opens, such as a door or a drawer.
        :return: The body holding whatever the part gives access to. A drawer holds it in
            its own case, while a door only covers the body it hangs from.
        """
        if isinstance(openable, HasCaseAsRootBody):
            return openable.root
        return openable.root.parent_kinematic_structure_entity

    def handle_of_enclosing_container(self, body: Body, world: World) -> Optional[Body]:
        """
        Find what has to be pulled open before ``body`` can be taken.

        :param body: The body that may stand inside a container.
        :param world: The world holding both.
        :return: The handle of the container, or ``None`` when the body stands in the
            open or in something that does not open.
        """
        containers = [
            candidate
            for candidate in world.bodies_with_collision
            if candidate is not body
            and InsideOf(body, candidate).compute_containment_ratio()
            > self.containment_ratio
        ]
        openable = variable(HasMechanicalJoint, domain=world.semantic_annotations)
        return next(
            (
                found.handle.root
                for found in an(entity(openable)).evaluate()
                if isinstance(found, HasHandle)
                and found.handle is not None
                and self.container_of(found) in containers
            ),
            None,
        )

    def add_container_opening_and_closing(
        self, actions: List[ActionLike], body: Body, context: Context
    ) -> List[ActionLike]:
        """
        Open the container ``body`` stands in before the given steps, and shut it again
        after them.

        :param actions: The steps that need the container open.
        :param body: The body those steps reach for.
        :param context: The context the standing poses are chosen in.
        :return: The steps with the container steps around them, and ``actions`` itself
            when the body stands in the open.
        """
        handle = self.handle_of_enclosing_container(body, context.world)
        if handle is None:
            return actions

        handle_grasp = GraspDescription(
            self.handle_approach_direction,
            VerticalAlignment.NoAlignment,
            ViewManager.get_end_effector_view(self.door_arm, context.robot),
        )

        return [
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: giskard_reachability_location(
                            handle, context, self.door_arm, handle_grasp
                        )
                    ),
                ),
                keep_joint_states=True,
            ),
            OpenAction(
                handle,
                self.door_arm,
                self.handle_approach_direction,
                self.door_opening_angle,
            ),
            ParkArmsAction(Arms.BOTH),
            *actions,
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: giskard_reachability_location(
                            handle, context, self.door_arm, handle_grasp
                        )
                    ),
                ),
                keep_joint_states=True,
            ),
            CloseAction(
                handle,
                self.door_arm,
                self.handle_approach_direction,
                self.door_closing_angle,
            ),
            ParkArmsAction(Arms.BOTH),
        ]

    # %% the plan

    def build_plan(self, context: Context) -> PlanNode:
        """
        Carry the milk to the kitchen island, opening and shutting whatever it stands
        in.

        Every action of a plan is expanded before its first motion runs, which is too
        early to choose a standing pose: earlier steps still move both the robot and the
        handle riding the swinging door. The navigations are therefore underspecified and
        take a deferred location, so a standing pose is chosen once execution reaches
        them.
        """
        world = context.world
        milk = variable(Milk, domain=world.semantic_annotations)
        milk_body = the(entity(milk)).first().root
        place_pose = self.place_pose_on_island(world)
        milk_grasp = GraspDescription(
            self.milk_approach_direction,
            VerticalAlignment.NoAlignment,
            ViewManager.get_end_effector_view(self.milk_arm, context.robot),
        )

        transport = [
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: giskard_reachability_location(
                            milk_body, context, self.milk_arm, milk_grasp
                        )
                    ),
                ),
                keep_joint_states=True,
            ),
            PickUpAction(milk_body, self.milk_arm, milk_grasp),
            ParkArmsAction(Arms.BOTH),
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: giskard_reachability_location(
                            place_pose, context, self.milk_arm, milk_grasp
                        )
                    ),
                ),
                keep_joint_states=True,
            ),
            PlaceAction(milk_body, place_pose, self.milk_arm),
            ParkArmsAction(Arms.BOTH),
        ]

        return sequential(
            [
                ParkArmsAction(Arms.BOTH),
                MoveTorsoAction(TorsoState.HIGH),
                *self.add_container_opening_and_closing(transport, milk_body, context),
            ],
            context,
        )


# %% running the demo


def main(execution_type: ExecutionType = ExecutionType.SIMULATED) -> World:
    """
    Run the demonstration and check that the milk ended up on the kitchen island with
    the fridge shut behind it.

    :param execution_type: Whether to drive the real robot or simulate it.
    :return: The world the demonstration acted on.
    """
    demonstration = KitchenFridgeDemonstration(
        used_robot=PR2, execution_type=execution_type
    )
    world = demonstration.run()

    milk = variable(Milk, domain=world.semantic_annotations)
    milk_position = the(entity(milk)).first().root.global_pose.to_position()
    expected_position = demonstration.place_pose_on_island(world).to_position()
    door_angle = demonstration.fridge_door(world).root.parent_connection.position
    print(f"milk placed at {np.round(milk_position, 3)}")
    print(f"Expected milk to be placed at {np.round(expected_position, 3)}")
    print(f"fridge door closed to {door_angle:.4f} rad")

    assert np.allclose(
        milk_position, expected_position, atol=demonstration.placement_tolerance
    )
    assert door_angle < demonstration.door_is_closed_angle
    return world


if __name__ == "__main__":
    main()
