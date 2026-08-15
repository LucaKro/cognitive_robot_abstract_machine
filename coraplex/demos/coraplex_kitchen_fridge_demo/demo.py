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
from typing_extensions import ClassVar, Iterator, List, Optional, Union

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
from coraplex.plans.plan_node import ActionLike, ActionNode, PlanNode
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.view_manager import ViewManager
from krrood.entity_query_language.factories import a, an, entity, the, variable
from krrood.entity_query_language.predicate import symbolic_function
from krrood.entity_query_language.query.match import Match
from semantic_digital_twin.api import (
    BodySpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.mixins import HasDoors, IsStorageSpace
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CounterTop,
    Door,
    Fridge,
    Milk,
    ShelfLayer,
    Table,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

# %% what holds what


@symbolic_function
def contains(
    container: KinematicStructureEntity, body: KinematicStructureEntity
) -> bool:
    """
    :param container: The entity that may hold the body.
    :param body: The entity that may be held.
    :return: Whether nearly all of the body lies inside the container.
    """
    return InsideOf(body, container)() > 0.9


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

    milk_arm: Arms = Arms.RIGHT
    """
    The arm carrying the milk.
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

        self._correct_where_the_table_stands(world)

        return world

    @staticmethod
    def _correct_where_the_table_stands(world: World) -> None:
        """
        Put the table on the floor and give the robot room to get past it.

        The kitchen hangs the table's mesh off its own centre, so the URDF leaves two
        thirds of a table standing and the rest buried. It also stands close enough to
        the kitchen island that the robot's base clips it on the way round.

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
                    # Away from the island, widening the gap the robot drives through.
                    y=-0.5,
                    z=-sunk_by,
                    reference_frame=connection.parent,
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
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                -0.16, 0.0, 0.11
            ),
        )

    def fridge_door(self, world: World) -> Door:
        """
        :param world: The world holding the kitchen.
        :return: The fridge's door, the only one in this kitchen that opens.
        """
        fridge = variable(Fridge, domain=world.semantic_annotations)
        return the(entity(fridge)).first().doors[0]

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

    def place_spot_on_island(self, world: World, milk: Milk) -> Point3:
        """
        Pick a spot on the kitchen island to put the milk down on.

        The counter top works out where its own surface is and how high the milk has to
        stand to rest on it, so the spot is drawn from the surface rather than measured
        off a bounding box. It comes out somewhere different on every run.

        A spot, not a pose: which way the carton ends up turned is nobody's business
        here. The robot walks round the island to wherever the spot is, and the heading
        follows from where it ends up standing.

        :param world: The world holding the kitchen.
        :param milk: The carton the spot has to be big enough for.
        :return: The spot, in the world frame.
        """
        counter_top = self.island_counter_top(world)
        points = counter_top.sample_points_from_surface(
            body_to_sample_for=milk, amount=100
        )
        return world.transform(next(iter(points)), world.root)

    @staticmethod
    def pose_the_robot_faces(spot: Point3, context: Context) -> Pose:
        """
        :param spot: Where the object has to end up.
        :param context: The context holding the robot.
        :return: That spot, turned the way the robot is standing.
        """
        return Pose.from_xyz_rpy(
            spot.x,
            spot.y,
            spot.z,
            yaw=context.robot.root.global_pose.yaw,
            reference_frame=context.world.root,
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
    def handle_of_enclosing_container(body: Body, world: World) -> Optional[Body]:
        """
        Find what has to be pulled open before ``body`` can be taken.

        Asks for the storage spaces whose own body holds the given one and takes the
        handle of the first door among them. In this kitchen the milk is held by both the
        fridge and the rack the fridge stands in, and only the fridge has a door.

        ..note:: Only doors are followed. A drawer holds what it contains in its own case
            and is found just as well, but its handle hangs off the drawer itself rather
            than off a door.

        :param body: The body that may stand inside a container.
        :param world: The world holding both.
        :return: The handle of the container, or ``None`` when the body stands in the
            open.
        """
        storage_space = variable(IsStorageSpace, domain=world.semantic_annotations)
        containers = an(
            entity(storage_space).where(contains(storage_space.root, body))
        ).evaluate()
        return next(
            (
                container.doors[0].handle.root
                for container in containers
                if isinstance(container, HasDoors) and container.doors
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

    def grasp_from_any_side(self, context: Context) -> Match[GraspDescription]:
        """
        Describe a grasp without saying which side of the object the gripper comes from.

        The plan settles that once it is standing in front of the object, rather than
        against the pose the robot happened to start the plan in.

        :param context: The context holding the robot that reaches for it.
        :return: The grasp, still open on its approach direction.
        """
        return a(GraspDescription)(
            approach_direction=...,
            vertical_alignment=VerticalAlignment.NoAlignment,
            rotate_gripper=False,
            end_effector=ViewManager.get_end_effector_view(
                self.milk_arm, context.robot
            ),
        )

    def grasps_from_every_side(self, context: Context) -> List[GraspDescription]:
        """
        :param context: The context holding the robot that reaches.
        :return: One grasp per side the gripper could come from, for a target whose
            approach the plan has not settled yet.
        """
        end_effector = ViewManager.get_end_effector_view(self.milk_arm, context.robot)
        return [
            GraspDescription(
                approach_direction, VerticalAlignment.NoAlignment, end_effector
            )
            for approach_direction in ApproachDirection
        ]

    def grasp_the_milk_is_held_with(self, context: Context) -> GraspDescription:
        """
        Find how the milk ended up in the gripper.

        Putting it down is the pick-up run backwards, so the place has no say in the
        grasp: it works with whichever side the pick-up settled on, and a standing pose
        for the place has to be found for that one rather than for any of the others.

        :param context: The context holding the plan the pick-up ran in.
        :return: The grasp the milk is carried with, and a front grasp while nothing has
            been picked up yet.
        """
        picked_up = [
            node.designator
            for node in context.plan.nodes
            if isinstance(node, ActionNode)
            and isinstance(node.designator, PickUpAction)
        ]
        if picked_up:
            return picked_up[-1].grasp_description
        return self.grasps_from_every_side(context)[0]

    def standing_poses_to_reach(
        self,
        targets: List[Union[Point3, Pose, Body]],
        grasps: List[GraspDescription],
        context: Context,
    ) -> Iterator[Pose]:
        """
        Look for poses to stand in to reach any of the targets with any of the grasps.

        The searches are drawn from one pose at a time and in turn, so an awkward target
        cannot use up the whole search on its own: when a target the robot cannot easily
        stand at is offered next to one it can, the easy one is reached after a single
        unlucky pose rather than after that target's every pose has been tried.

        :param targets: The bodies or poses, any one of which has to be reachable.
        :param grasps: The grasps, any one of which may be used to reach them.
        :param context: The context the standing poses are chosen in.
        :return: The standing poses, targets varying fastest.
        """
        searches = [
            iter(giskard_reachability_location(target, context, self.milk_arm, grasp))
            for grasp in grasps
            for target in targets
        ]
        while searches:
            for search in list(searches):
                standing_pose = next(search, None)
                if standing_pose is None:
                    searches.remove(search)
                    continue
                yield standing_pose

    def build_plan(self, context: Context) -> PlanNode:
        """
        Carry the milk to the kitchen island, opening and shutting whatever it stands
        in.

        Every action of a plan is expanded before its first motion runs, which is too
        early to choose a standing pose: earlier steps still move both the robot and the
        handle riding the swinging door. The navigations are therefore underspecified and
        take a deferred location, so a standing pose is chosen once execution reaches
        them.

        The pick-up leaves the side it grasps from open, and the place the heading it
        puts the carton down at, so both are settled against the world the plan finds
        rather than against the pose the robot started in.
        """
        world = context.world
        milk = the(entity(variable(Milk, domain=world.semantic_annotations))).first()
        milk_body = milk.root
        place_spot = self.place_spot_on_island(world, milk)

        transport = [
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: self.standing_poses_to_reach(
                            [milk_body], self.grasps_from_every_side(context), context
                        )
                    ),
                ),
                keep_joint_states=True,
            ),
            a(PickUpAction)(
                object_designator=milk_body,
                grasp_description=self.grasp_from_any_side(context),
                arm=self.milk_arm,
            ),
            ParkArmsAction(Arms.BOTH),
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: self.standing_poses_to_reach(
                            [place_spot],
                            [self.grasp_the_milk_is_held_with(context)],
                            context,
                        )
                    ),
                ),
                keep_joint_states=True,
            ),
            a(PlaceAction)(
                object_designator=milk_body,
                # The heading is read off the robot once it has driven up to the spot,
                # so the carton is put down the way the robot that carries it stands.
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: [self.pose_the_robot_faces(place_spot, context)]
                    ),
                ),
                arm=self.milk_arm,
            ),
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
