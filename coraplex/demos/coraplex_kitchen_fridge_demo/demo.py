"""
A PR2 takes a milk out of the small kitchen's fridge and puts it down on the kitchen
island.

The plan states the transport only. That the milk stands behind a closed fridge door is
not written into it: the container the milk is in is found from the milk, and the steps
that open it and shut it again are wrapped around the transport.

The world is built from specifications: the kitchen comes from a URDF, the robot from a
:class:`~semantic_digital_twin.api.RobotSpecification`, and the shelf and the milk from
body and annotation specifications. Building the world furnishes the kitchen; populating
the scene puts down the milk the plan is about, and nothing else.

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
from semantic_digital_twin.api import (
    BodySpecification,
    RobotSpecification,
    WorldSpecification,
)
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
    CounterTop,
    Door,
    Fridge,
    Milk,
    ShelfLayer,
    Table,
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

    shelf_layer_name: str = "fridge_shelf"
    """
    Name of the shelf layer the milk stands on.
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

    milk_start_pose: Optional[Pose] = None
    """
    Where the milk stands when the demonstration begins, in the world frame.

    ``None`` stands it on the fridge's shelf layer at :attr:`shelf_layer_T_milk`, which
    is what makes the fridge door part of the problem. Standing it somewhere open
    instead leaves the plan with nothing to unpack.
    """

    milk_arm: Arms = Arms.RIGHT
    """
    The arm carrying the milk.

    Reaching into the fridge means standing to the left of the opening, out of the swing
    of the open door, which leaves the milk on the robot's right.
    """

    def build_simulated_world(self) -> World:
        """
        Load the kitchen from its URDF, put the robot in it, infer what its bodies mean
        (which is what turns the fridge's parts into a fridge with a door and a handle),
        stand a shelf layer in the fridge, and put right what the URDF got wrong.
        """
        world = WorldSpecification.from_urdf(
            str(
                Path(__file__).parents[2]
                / "resources"
                / "worlds"
                / "kitchen-small.urdf"
            ),
            robots=[
                RobotSpecification(
                    semantic_annotation_type=self.used_robot,
                    # The PR2's root is its base_footprint, so a height of zero puts it on
                    # the floor, on the free space between the fridge and the island.
                    world_T_odom=HomogeneousTransformationMatrix(),
                )
            ],
        ).to_domain_object()

        with world.modify_world():
            WorldReasoner(world).reason()

        fridge = variable(Fridge, domain=world.semantic_annotations)
        fridge_annotation = the(entity(fridge)).first()
        shelf_layer = ShelfLayer.get_annotation_specification(
            self.shelf_layer_name,
            BodySpecification.box(
                self.shelf_layer_name,
                # Fits into the fridge cavity, which is about 0.03 meters narrower than
                # the shell on every side.
                Scale(0.45, 0.5, 0.02),
                color=Color(0.9, 0.93, 0.95),
            ),
        ).spawn(
            world,
            parent=fridge_annotation.root,
            # The kitchen places its fridge turned by half a turn against the room, so
            # the layer turns back: everything spawned on it is then aligned with the
            # room, and grasped from the front like any other object standing in it.
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                0.0, 0.02, 0.0, yaw=np.pi
            ),
        )
        with world.modify_world():
            fridge_annotation.add(shelf_layer)

        # The URDF describes the hinge with 10 rad/s, a speed a door does not turn at. The
        # controller keeps a braking margin from a joint's own position limit that grows
        # with that limit, and at 10 rad/s the door can neither be swung fully open nor
        # shut.
        door_speed_limit = 1.0
        hinge = self.fridge_door(world).root.parent_connection
        hinge.raw_dof.limits.upper.velocity = door_speed_limit
        hinge.raw_dof.limits.lower.velocity = -door_speed_limit

        self._rest_table_on_the_floor(world)

        return world

    @staticmethod
    def _rest_table_on_the_floor(world: World) -> None:
        """
        Lift the table until its lowest point sits on the floor.

        The kitchen hangs the table's mesh off its own centre, so the URDF leaves two
        thirds of a table standing and the rest buried.

        :param world: The world holding the kitchen.
        """
        table = the(entity(variable(Table, domain=world.semantic_annotations))).first()
        sunk_by = (
            table.root.collision.as_bounding_box_collection_in_frame(world.root)
            .bounding_box()
            .min_z
        )
        connection = table.root.parent_connection
        with world.modify_world():
            connection.parent_T_connection_expression = (
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=-sunk_by, reference_frame=connection.parent
                )
                @ connection.parent_T_connection_expression
            )

    def is_scene_populated(self, world: World) -> bool:
        milk = variable(Milk, domain=world.semantic_annotations)
        return next(an(entity(milk)).evaluate(), None) is not None

    def populate_scene(self, world: World) -> None:
        """
        Stand the milk where the demonstration starts it, which is the fridge's shelf
        layer unless another pose was given.
        """
        specification = Milk.get_annotation_specification(
            "milk",
            Milk.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(0.065, 0.065, 0.2)
            ),
        )
        if self.milk_start_pose is not None:
            specification.spawn(
                world, parent_T_self=self.milk_start_pose.to_homogeneous_matrix()
            )
            return

        shelf_layer = variable(ShelfLayer, domain=world.semantic_annotations)
        specification.spawn(
            world,
            parent=the(
                entity(shelf_layer).where(
                    shelf_layer.name.name == self.shelf_layer_name
                )
            )
            .first()
            .root,
            parent_T_self=self.shelf_layer_T_milk,
        )

    def fridge_door(self, world: World) -> Door:
        """
        :param world: The world holding the kitchen.
        :return: The fridge door, told apart from the kitchen's other doors by belonging
            to the fridge.
        """
        fridge = variable(Fridge, domain=world.semantic_annotations)
        door = variable(Door, domain=world.semantic_annotations)
        return the(entity(door).where(contains(fridge.doors, door))).first()

    @staticmethod
    def island_counter_top(world: World) -> CounterTop:
        """
        :param world: The world holding the kitchen.
        :return: The counter top the milk is put down on, told apart from the sink's by
            the body the kitchen names it after.
        """
        counter_top = variable(CounterTop, domain=world.semantic_annotations)
        return the(
            entity(counter_top).where(
                counter_top.root.name.name == "kitchen_island_surface"
            )
        ).first()

    def place_pose_on_island(self, world: World, milk: Milk) -> Pose:
        """
        Pick a spot on the kitchen island to put the milk down on.

        The counter top works out where its own surface is and how high the milk has to
        stand to rest on it, so the spot is drawn from the surface rather than measured
        off a bounding box. It comes out somewhere different on every run.

        :param world: The world holding the kitchen.
        :param milk: The carton the spot has to be big enough for.
        :return: The pose the milk is placed at.
        """
        point = self.island_counter_top(world).sample_points_from_surface(
            body_to_sample_for=milk, amount=1
        )[0]
        world_P_milk = world.transform(point, world.root)
        return Pose.from_xyz_rpy(
            world_P_milk.x,
            world_P_milk.y,
            world_P_milk.z,
            # The robot reaches the island from positive x, so the carton faces it.
            yaw=np.pi,
            reference_frame=world.root,
        )

    def build_context(self, world: World) -> Context:
        """
        Build the plan context around the robot in ``world``.

        Conditions are evaluated, which is what makes the underspecified arms work: a
        hand that cannot reach the milk, or that is not the one holding it, fails its
        precondition before it moves and the next candidate is tried.
        """
        robot = variable(self.used_robot, domain=world.semantic_annotations)
        context = Context(
            world=world,
            robot=the(entity(robot)).first(),
            ros_node=self.ros_node,
            evaluate_conditions=True,
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
            # How much of the body's volume has to lie inside the candidate for it
            # to count as standing in it.
            and InsideOf(body, candidate).compute_containment_ratio() > 0.9
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

        # The robot stands to the left of the fridge opening, out of the swing of the
        # open door, which puts the handle on its left.
        door_arm = Arms.LEFT
        # The fridge, and with it its handle, is turned by half a turn against the room,
        # so the handle's own front points into the fridge and is reached from its back.
        # The handle turns with the door, so this holds for opening and shutting alike.
        handle_approach_direction = ApproachDirection.BACK
        handle_grasp = GraspDescription(
            handle_approach_direction,
            VerticalAlignment.NoAlignment,
            ViewManager.get_end_effector_view(door_arm, context.robot),
        )

        return [
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: giskard_reachability_location(
                            handle, context, door_arm, handle_grasp
                        )
                    ),
                ),
                keep_joint_states=True,
            ),
            OpenAction(
                handle,
                door_arm,
                handle_approach_direction,
                # Wide enough to reach past the door into the fridge, short of the
                # hinge's 1.5708 limit so the swing is not timed against the door frame.
                goal_joint_state=1.45,
            ),
            ParkArmsAction(Arms.BOTH),
            *actions,
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: giskard_reachability_location(
                            handle, context, door_arm, handle_grasp
                        )
                    ),
                ),
                keep_joint_states=True,
            ),
            CloseAction(
                handle,
                door_arm,
                handle_approach_direction,
                # Shut.
                goal_joint_state=0.0,
            ),
            ParkArmsAction(Arms.BOTH),
        ]

    # %% the plan

    def grasps_for(self, body: Body, context: Context) -> List[GraspDescription]:
        """
        Work out how the body can be taken hold of, from where it stands and how it is
        turned, rather than assuming which of its sides faces the robot.

        The approach directions are expressed in the body's own frame, so they follow it
        wherever it is put.

        ..note:: The candidates are computed with :attr:`milk_arm`'s gripper, while the
            pick-up may settle on the other one. On the PR2 that makes no difference: a
            grasp reads its end effector only for the orientation its front faces, and
            both of its grippers face the same way.

        :param body: The body to be picked up.
        :param context: The context holding the robot that reaches for it.
        :return: The grasps, the most promising first.
        """
        return GraspDescription.calculate_grasp_descriptions(
            ViewManager.get_end_effector_view(self.milk_arm, context.robot),
            body.global_pose,
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Carry the milk to the kitchen island, opening and shutting whatever it stands
        in.

        Every action of a plan is expanded before its first motion runs, which is too
        early to choose a standing pose: earlier steps still move both the robot and the
        handle riding the swinging door. The navigations are therefore underspecified and
        take a deferred location, so a standing pose is chosen once execution reaches
        them.

        The pick-up leaves its grasp open over the grasps the milk's own pose allows, so
        the plan settles that against the world it finds rather than assuming which side
        of the carton faces the robot.
        """
        world = context.world
        milk = the(entity(variable(Milk, domain=world.semantic_annotations))).first()
        milk_body = milk.root
        place_pose = self.place_pose_on_island(world, milk)
        milk_grasps = self.grasps_for(milk_body, context)

        transport = [
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: giskard_reachability_location(
                            milk_body, context, self.milk_arm, milk_grasps[0]
                        )
                    ),
                ),
                keep_joint_states=True,
            ),
            a(PickUpAction)(
                object_designator=milk_body,
                grasp_description=variable(GraspDescription, domain=milk_grasps),
                arm=self.milk_arm,
            ),
            ParkArmsAction(Arms.BOTH),
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: giskard_reachability_location(
                            place_pose, context, self.milk_arm, milk_grasps[0]
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

    milk = the(entity(variable(Milk, domain=world.semantic_annotations))).first()
    milk_box = milk.root.collision.as_bounding_box_collection_in_frame(
        world.root
    ).bounding_box()
    counter_top = demonstration.island_counter_top(world)
    surface_box = (
        counter_top.supporting_surface.area.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_box()
    )
    door_angle = demonstration.fridge_door(world).root.parent_connection.position

    print(f"milk placed at {np.round(milk.root.global_pose.to_position(), 3)}")
    print(
        f"it rests at {milk_box.min_z:.4f}, the island's top is {surface_box.max_z:.4f}"
    )
    print(f"fridge door closed to {door_angle:.4f} rad")

    # Nothing here is subject to gravity or contact forces, so a placement that goes to
    # plan lands on the surface rather than near it.
    placement_tolerance = 0.02
    assert abs(milk_box.min_z - surface_box.max_z) < placement_tolerance
    assert surface_box.min_x <= milk_box.min_x and milk_box.max_x <= surface_box.max_x
    assert surface_box.min_y <= milk_box.min_y and milk_box.max_y <= surface_box.max_y
    # The closing motion parks the door within about a hundredth of a radian of the
    # hinge's limit, so this is how exactly it shuts rather than how far ajar it may be.
    assert door_angle < 0.02
    return world


if __name__ == "__main__":
    main()
