---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(writing-a-robot-plan-exercise)=
# Writing a Robot Plan

In this exercise you write, step by step, the plan behind
`coraplex/demos/coraplex_kitchen_fridge_demo/demo.py`: a PR2 takes a milk out of a closed
fridge and puts it down on the kitchen island.

Every section acts on the world the section before it left behind, so by the end the whole
transport has been carried out once, and the last section wraps it into the demonstration
class the demo ships.

You will:
- Move the robot's body: park its arms, raise its torso, drive its base
- Work out where the robot has to stand to reach something
- Open and shut a fridge door
- Pick the milk up and put it down somewhere else
- Ask the world where something can stand, instead of measuring a spot yourself
- Leave a plan underspecified, so it settles against the world it finds at execution time
- Derive the opening and shutting from where the milk stands, instead of writing them down

## 0. Setup

Execute the cells in this section as they are.

```{code-cell} ipython3
:tags: [remove-input]
import logging
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
from typing_extensions import ClassVar, List, Optional

from coraplex.alternative_motion_mapping import AlternativeMotion
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    ApproachDirection,
    Arms,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.demonstrations import RobotDemonstration
from coraplex.execution_environment import simulated_robot
from coraplex.locations.base import DeferredLocation
from coraplex.locations.factories import giskard_reachability_location
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import (
    ActionLike,
    ActionNode,
    PlanNode,
    UnderspecifiedNode,
)
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
from semantic_digital_twin.exceptions import ExerciseVerificationFailed
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.mixins import HasDoors, IsStorageSpace
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CounterTop,
    Fridge,
    Milk,
    ShelfLayer,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

logging.disable(logging.CRITICAL)

kitchen_urdf = (
    Path(files("coraplex")).parent.parent
    / "resources"
    / "worlds"
    / "kitchen-small.urdf"
)

SHELF_LAYER_NAME = "fridge_shelf"

# How far, in meters, the base may end up from a navigation target. A motion is driven to
# the controller's own tolerance and then wound down, and the base settles a little further
# during that wind-down, so this is looser than the tolerance the navigation itself holds.
ARRIVAL_TOLERANCE = 0.1

# How far, in meters, the milk may end up from where it was placed. Nothing here is subject
# to gravity or contact forces, so a placement that goes to plan lands on the target rather
# than near it.
PLACEMENT_TOLERANCE = 0.02
```

### The kitchen

The kitchen comes from a URDF, the robot from a
{class}`~semantic_digital_twin.api.RobotSpecification`, and the shelf layer the milk stands on
from a body specification. The
{class}`~semantic_digital_twin.reasoning.world_reasoner.WorldReasoner` turns the URDF's bodies
into a *fridge* with a *door* and a *handle*, which is what the plan below reaches for.

```{code-cell} ipython3
def build_kitchen_world() -> World:
    """
    Load the kitchen from its URDF, put the robot in it, infer what its bodies mean

    -- which is what turns the fridge's parts into a fridge with a door and a handle --
    and stand a shelf layer in the fridge.
    """
    world = WorldSpecification.from_urdf(
        str(kitchen_urdf),
        robots=[
            RobotSpecification(
                semantic_annotation_type=PR2,
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
        SHELF_LAYER_NAME,
        BodySpecification.box(
            SHELF_LAYER_NAME,
            # Fits into the fridge cavity, which is about 0.03 meters narrower than the
            # shell on every side.
            Scale(0.45, 0.5, 0.02),
            color=Color(0.9, 0.93, 0.95),
        ),
    ).spawn(
        world,
        parent=fridge_annotation.root,
        # The kitchen places its fridge turned by half a turn against the room, so the
        # layer turns back: everything spawned on it is then aligned with the room, and
        # grasped from the front like any other object standing in it.
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
            0.0, 0.02, 0.0, yaw=np.pi
        ),
    )
    with world.modify_world():
        fridge_annotation.add(shelf_layer)

    return world


def stand_milk_in_fridge(world: World) -> None:
    """
    Stand the milk where the demonstration starts it, which is the fridge's shelf layer.

    :param world: The world holding the fridge.
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
        parent=the(entity(shelf_layer).where(shelf_layer.name.name == SHELF_LAYER_NAME))
        .first()
        .root,
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(-0.16, 0.0, 0.11),
    )


world = build_kitchen_world()
stand_milk_in_fridge(world)
```

### The context

A {class}`~coraplex.datastructures.dataclasses.Context` says which world a plan acts in and
which robot carries it out. Two of its fields matter here:

- `evaluate_conditions` turns each action's pre- and post-conditions into monitors that run
  alongside the motion. With it switched on, an action whose precondition does not hold fails
  *before it moves*. Section 5 depends on that.
- `alternative_motion_mappings` lets a robot substitute its own motion for a generic one.
  Handing over everything coraplex knows is always safe: resolution filters by robot and
  execution type.

```{code-cell} ipython3
robot = variable(PR2, domain=world.semantic_annotations)
context = Context(
    world=world,
    robot=the(entity(robot)).first(),
    evaluate_conditions=True,
    alternative_motion_mappings=AlternativeMotion.discover_all(),
)
```

### What the plan reaches for

The robot opens the door with its **left** arm and takes the milk with its **right** one,
because it stands to the left of the fridge opening, out of the swing of the open door.

A {class}`~coraplex.datastructures.grasp.GraspDescription` says how a body is taken hold of:
which side the gripper approaches from, how it is aligned vertically, and which end effector
does it. The handle is approached from `ApproachDirection.BACK`, because approach directions
are read in the handle's own frame and the fridge — handle included — is turned by half a turn
against the room, so the handle's own front points into the fridge.

The milk gets no grasp here. Which side of the carton the gripper can come at depends on where
the robot ends up standing, which nothing knows yet; section 5 leaves that side open instead.

```{code-cell} ipython3
DOOR_ARM = Arms.LEFT
MILK_ARM = Arms.RIGHT


def get_grasp_for_handle(arm: Arms, context: Context) -> GraspDescription:
    """
    Describe taking hold of a handle.

    The side is named rather than left open, unlike the grasp the milk is picked up with:
    the container actions pull the handle from behind whatever else happens, so a standing
    pose found for another side is one they could not use.

    :param arm: The hand the standing pose is looked for with.
    :param context: The context holding the robot the hand belongs to.
    :return: The grasp, approaching from behind because the handle's own frame is turned
        against the room it is pulled from.
    """
    return GraspDescription(
        ApproachDirection.BACK,
        VerticalAlignment.NoAlignment,
        ViewManager.get_end_effector_view(arm, context.robot),
    )


fridge = the(entity(variable(Fridge, domain=world.semantic_annotations))).first()
milk = the(entity(variable(Milk, domain=world.semantic_annotations))).first()

handle = fridge.doors[0].handle.root
milk_body = milk.root
handle_grasp = get_grasp_for_handle(DOOR_ARM, context)
```

## 1. Your first plan: move the robot's own body

A plan is a tree of steps. {func}`~coraplex.plans.factories.sequential` builds one whose
children run one after another, and `perform()` on the node it returns executes it.

Actions describing the robot's own body are the simplest ones there are — they name a state
rather than a place. {class}`~coraplex.robot_plans.actions.core.robot_body.ParkArmsAction`
folds the arms into the robot's park position, and
{class}`~coraplex.robot_plans.actions.core.robot_body.MoveTorsoAction` sets the torso to one
of the states the robot itself defines.

Execution happens inside an *execution environment*, which decides whether the motions drive
a real robot or a simulated one. `simulated_robot(collision_avoidance=True)` returns a
simulated environment that keeps every motion clear of the furniture.

Your goal:
- Build a sequential plan that parks both arms and raises the torso to `TorsoState.HIGH`,
  store it in a variable named `plan`, and perform it in a simulated environment with
  collision avoidance

```{code-cell} ipython3
:tags: [exercise]
# TODO: park both arms, raise the torso, and perform the plan
# plan = sequential([...], context)
# with simulated_robot(collision_avoidance=True):
#     plan.perform()

plan = ...
```

```{code-cell} ipython3
:tags: [example-solution]
plan = sequential(
    [
        ParkArmsAction(Arms.BOTH),
        MoveTorsoAction(TorsoState.HIGH),
    ],
    context,
)

with simulated_robot(collision_avoidance=True):
    plan.perform()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if plan is ...: raise ExerciseVerificationFailed("Build the plan.")
if not isinstance(plan, PlanNode): raise ExerciseVerificationFailed("`sequential` returns the plan's root node.")
if not context.robot.get_torso().get_joint_state_by_type(TorsoState.HIGH).is_achieved(): raise ExerciseVerificationFailed("The torso should have reached its HIGH state.")
```

## 2. Driving the base

{class}`~coraplex.robot_plans.actions.core.navigation.NavigateAction` moves the base to a
pose. Its `keep_joint_states` flag decides whether the rest of the robot holds its posture
while the base drives; keeping it means the arms you just parked stay parked.

The pose below is a free patch of floor between the kitchen island and the counters. Writing
standing poses out by hand works, but only as long as you know the room — the next section
lets the robot work one out instead.

Your goal:
- Navigate to `Pose.from_xyz_rpy(0.4, -0.5, 0.0, reference_frame=world.root)`, keeping the
  joint states, and perform the plan

```{code-cell} ipython3
:tags: [exercise]
free_spot = Pose.from_xyz_rpy(0.4, -0.5, 0.0, reference_frame=world.root)

# TODO: drive the base to `free_spot` without disturbing the arms
# plan = sequential([NavigateAction(...)], context)
# with simulated_robot(collision_avoidance=True):
#     plan.perform()

plan = ...
```

```{code-cell} ipython3
:tags: [example-solution]
free_spot = Pose.from_xyz_rpy(0.4, -0.5, 0.0, reference_frame=world.root)

plan = sequential(
    [NavigateAction(target_location=free_spot, keep_joint_states=True)],
    context,
)

with simulated_robot(collision_avoidance=True):
    plan.perform()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if plan is ...: raise ExerciseVerificationFailed("Build the navigation plan.")
if not np.allclose(context.robot.root.global_pose.to_position().to_np()[:2], free_spot.to_position().to_np()[:2], atol=ARRIVAL_TOLERANCE): raise ExerciseVerificationFailed("The robot should be standing at `free_spot`.")
if not context.robot.get_torso().get_joint_state_by_type(TorsoState.HIGH).is_achieved(): raise ExerciseVerificationFailed("keep_joint_states should have left the torso where it was.")
```

## 3. Working out where to stand

To reach for something the robot has to stand somewhere it can be reached *from*, and that
place depends on the thing, the arm and the way the gripper comes at it.

{func}`~coraplex.locations.factories.giskard_reachability_location` produces standing poses
from which a given grasp works, and `ground()` takes the first of them. It hands out a pose
only once the robot has driven there and reached the target from it, so a pose that comes out
has already been tried.

The grasp may also be left out, and then every side the gripper could come from is tried at
each pose the search considers. Here the side is known — the handle is pulled from behind
whatever happens — so it is worth saying.

Your goal:
- Ground a `giskard_reachability_location` for `handle`, reached with `DOOR_ARM` and
  `handle_grasp`, into a variable named `handle_standing_pose`
- Navigate there

```{code-cell} ipython3
:tags: [exercise]
# TODO: find a pose the handle can be reached from, then drive to it
# handle_standing_pose = giskard_reachability_location(handle, context, DOOR_ARM, handle_grasp).ground()
# plan = sequential([NavigateAction(target_location=handle_standing_pose, keep_joint_states=True)], context)
# with simulated_robot(collision_avoidance=True):
#     plan.perform()

handle_standing_pose = ...
plan = ...
```

```{code-cell} ipython3
:tags: [example-solution]
handle_standing_pose = giskard_reachability_location(
    handle, context, DOOR_ARM, handle_grasp
).ground()

plan = sequential(
    [NavigateAction(target_location=handle_standing_pose, keep_joint_states=True)],
    context,
)

with simulated_robot(collision_avoidance=True):
    plan.perform()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if handle_standing_pose is ...: raise ExerciseVerificationFailed("Ground a location for the handle.")
if plan is ...: raise ExerciseVerificationFailed("Build the navigation plan.")
if not isinstance(handle_standing_pose, Pose): raise ExerciseVerificationFailed("A location grounds to a Pose.")
if not np.allclose(context.robot.root.global_pose.to_position().to_np()[:2], handle_standing_pose.to_position().to_np()[:2], atol=ARRIVAL_TOLERANCE): raise ExerciseVerificationFailed("The robot should be standing at the pose the location handed out.")
```

## 4. Opening the fridge

{class}`~coraplex.robot_plans.actions.core.container.OpenAction` takes the *handle* of the
thing that opens, the arm to open it with, the side to come at the handle from, and how far
to swing it. Its `goal_joint_state` is in the units of the joint behind the handle — radians
for a hinge, meters for a drawer rail.

The angle below stops short of the hinge's own limit of about 1.5708 rad on purpose: it is
wide enough to reach past the door into the fridge, and short enough that the swing does not
have to be timed against the door frame. A goal *at* a joint limit tends not to be reported
as reached at all.

Park the arms afterwards, so the open door is not something the robot then has to drive
around with an arm stretched out.

Your goal:
- Swing the fridge door open to `DOOR_OPENING_ANGLE` with `DOOR_ARM`, then park both arms

```{code-cell} ipython3
:tags: [exercise]
DOOR_OPENING_ANGLE = 1.45

# TODO: open the fridge door, then park the arms
# plan = sequential([OpenAction(handle, DOOR_ARM, handle_grasp.approach_direction, ...), ...], context)
# with simulated_robot(collision_avoidance=True):
#     plan.perform()

plan = ...
```

```{code-cell} ipython3
:tags: [example-solution]
DOOR_OPENING_ANGLE = 1.45

plan = sequential(
    [
        OpenAction(handle, DOOR_ARM, handle_grasp.approach_direction, DOOR_OPENING_ANGLE),
        ParkArmsAction(Arms.BOTH),
    ],
    context,
)

with simulated_robot(collision_avoidance=True):
    plan.perform()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if plan is ...: raise ExerciseVerificationFailed("Build the opening plan.")
door_angle = fridge.doors[0].root.parent_connection.position
if door_angle < DOOR_OPENING_ANGLE - 0.1: raise ExerciseVerificationFailed(f"The fridge door should stand open at about {DOOR_OPENING_ANGLE} rad, but it is at {door_angle:.3f}.")
```

## 5. Taking the milk, without deciding everything up front

Now the milk. Two things about it cannot honestly be written down when the plan is built.

**Where to stand.** A plan expands every action before its first motion runs, which is far too
early to choose a standing pose: the door has only just swung open and the robot has not moved
yet. {class}`~coraplex.locations.base.DeferredLocation` wraps a factory that is called when the
pose is actually needed, which is once execution reaches that step.

**Which side to grasp from.** That depends on where the robot ends up standing, which is not
known yet either. So the pick-up is given a grasp description with its `approach_direction`
left as `...`, which stands for *any member of that enum*, and it settles on one itself.

Both are expressed by leaving the action *underspecified*: `a(SomeAction)(...)` describes an
action rather than building one, `variable(SomeType, domain=...)` marks an argument as a
choice over a domain, and `...` marks one as a choice over an enum's members. At execution
time the candidates are tried in turn, and one whose precondition fails — a hand that cannot
reach the milk, or is not free — is discarded before it moves and the next one is taken. That
is what `evaluate_conditions=True` is for.

Note the navigation asks for a location without a grasp: which side works is exactly what is
being left open, so it cannot be handed to the search either.

Your goal:
- Write `get_grasp_for_milk(context) -> Match[GraspDescription]`, a grasp with `MILK_ARM`'s
  end effector, no vertical alignment, no gripper rotation, and its approach direction open
- Build a plan that navigates to a deferred `giskard_reachability_location` for the milk,
  picks it up with `MILK_ARM` using that grasp, and parks both arms
- Perform it

```{code-cell} ipython3
:tags: [exercise]
# TODO: leave both the standing pose and the side the gripper comes from open
# def get_grasp_for_milk(context: Context) -> Match[GraspDescription]:
#     return a(GraspDescription)(approach_direction=..., ...)
#
# plan = sequential(
#     [
#         a(NavigateAction)(
#             target_location=variable(
#                 Pose,
#                 domain=DeferredLocation(
#                     lambda: giskard_reachability_location(milk_body, context, MILK_ARM)
#                 ),
#             ),
#             keep_joint_states=True,
#         ),
#         a(PickUpAction)(
#             object_designator=milk_body,
#             grasp_description=get_grasp_for_milk(context),
#             arm=MILK_ARM,
#         ),
#         ParkArmsAction(Arms.BOTH),
#     ],
#     context,
# )

get_grasp_for_milk = ...
plan = ...
```

```{code-cell} ipython3
:tags: [example-solution]
def get_grasp_for_milk(context: Context) -> Match[GraspDescription]:
    """
    Describe a grasp without saying which side of the object the gripper comes from.

    The plan settles that once it is standing in front of the object, rather than against
    the pose the robot happened to start the plan in.

    :param context: The context holding the robot that reaches for it.
    :return: The grasp, still open on its approach direction.
    """
    return a(GraspDescription)(
        approach_direction=...,
        vertical_alignment=VerticalAlignment.NoAlignment,
        rotate_gripper=False,
        end_effector=ViewManager.get_end_effector_view(MILK_ARM, context.robot),
    )


plan = sequential(
    [
        a(NavigateAction)(
            target_location=variable(
                Pose,
                domain=DeferredLocation(
                    lambda: giskard_reachability_location(milk_body, context, MILK_ARM)
                ),
            ),
            keep_joint_states=True,
        ),
        a(PickUpAction)(
            object_designator=milk_body,
            grasp_description=get_grasp_for_milk(context),
            arm=MILK_ARM,
        ),
        ParkArmsAction(Arms.BOTH),
    ],
    context,
)

with simulated_robot(collision_avoidance=True):
    plan.perform()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if get_grasp_for_milk is ...: raise ExerciseVerificationFailed("Write get_grasp_for_milk.")
if plan is ...: raise ExerciseVerificationFailed("Build the pick-up plan.")
pick_up_node = next((node for node in plan.children if isinstance(node, UnderspecifiedNode) and node.underspecified_action.factory is PickUpAction), None)
if pick_up_node is None: raise ExerciseVerificationFailed("The pick-up should be left underspecified with `a(PickUpAction)`.")
if pick_up_node.underspecified_action.kwargs["grasp_description"].kwargs["approach_direction"] is not Ellipsis: raise ExerciseVerificationFailed("The grasp's approach direction should be left open, so the side is settled once the robot stands in front of the milk.")
navigation_node = next((node for node in plan.children if isinstance(node, UnderspecifiedNode) and node.underspecified_action.factory is NavigateAction), None)
if navigation_node is None: raise ExerciseVerificationFailed("The navigation should be left underspecified too, so its standing pose is chosen at execution time.")
tool_frame = ViewManager.get_end_effector_view(MILK_ARM, context.robot).tool_frame
if milk_body not in world.get_kinematic_structure_entities_of_branch(tool_frame): raise ExerciseVerificationFailed("The milk should be attached to the gripper that picked it up.")
```

## 6. Working out where to put it down

The milk has to go somewhere on the kitchen island. You could measure a spot off the island's
bounding box, but the world already knows more than that: the
{class}`~semantic_digital_twin.reasoning.world_reasoner.WorldReasoner` recognised the island's
work surface as a {class}`~semantic_digital_twin.semantic_annotations.semantic_annotations.CounterTop`,
and a counter top can work out where its own surface is and how high something has to stand to
rest on it. `sample_points_from_surface` does that, given the carton it has to be big enough
for, so the spot comes out somewhere different on every run.

What comes back is a *point*, not a pose. Which way the carton ends up turned is nobody's
business here — the robot walks round the island to wherever the spot is, and the heading
follows from where it ends up standing. That is why `pose_the_robot_faces` reads the yaw off
the robot rather than fixing one now.

Both counter tops in this kitchen are counter tops to the reasoner, so the island's is told
apart by the body the kitchen names it after.

Your goal:
- Write `island_counter_top(world) -> CounterTop`, the one whose root body is named
  `kitchen_island_surface`
- Write `place_spot_on_island(world, milk) -> Point3`, a spot on that surface big enough for
  the carton, in the world frame
- Write `pose_the_robot_faces(spot, context) -> Pose`, that spot turned the way the robot
  stands
- Store a spot in `place_spot`

```{code-cell} ipython3
:tags: [exercise]
# TODO: ask the world where the milk could stand
# def island_counter_top(world: World) -> CounterTop:
#     counter_top = variable(CounterTop, domain=world.semantic_annotations)
#     return the(entity(counter_top).where(...)).first()
#
# def place_spot_on_island(world: World, milk: Milk) -> Point3:
#     points = island_counter_top(world).sample_points_from_surface(body_to_sample_for=milk, amount=100)
#     return world.transform(next(iter(points)), world.root)
#
# def pose_the_robot_faces(spot: Point3, context: Context) -> Pose:
#     return Pose.from_xyz_rpy(..., yaw=context.robot.root.global_pose.yaw, reference_frame=context.world.root)

island_counter_top = ...
place_spot_on_island = ...
pose_the_robot_faces = ...
place_spot = ...
```

```{code-cell} ipython3
:tags: [example-solution]
def island_counter_top(world: World) -> CounterTop:
    """
    :param world: The world holding the kitchen.
    :return: The counter top the milk is put down on, told apart from the sink's by the
        body the kitchen names it after.
    """
    counter_top = variable(CounterTop, domain=world.semantic_annotations)
    return the(
        entity(counter_top).where(
            counter_top.root.name.name == "kitchen_island_surface"
        )
    ).first()


def place_spot_on_island(world: World, milk: Milk) -> Point3:
    """
    Pick a spot on the kitchen island to put the milk down on.

    :param world: The world holding the kitchen.
    :param milk: The carton the spot has to be big enough for.
    :return: The spot, in the world frame.
    """
    points = island_counter_top(world).sample_points_from_surface(
        body_to_sample_for=milk, amount=100
    )
    return world.transform(next(iter(points)), world.root)


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


place_spot = place_spot_on_island(world, milk)
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if island_counter_top is ...: raise ExerciseVerificationFailed("Write island_counter_top.")
if place_spot_on_island is ...: raise ExerciseVerificationFailed("Write place_spot_on_island.")
if pose_the_robot_faces is ...: raise ExerciseVerificationFailed("Write pose_the_robot_faces.")
if place_spot is ...: raise ExerciseVerificationFailed("Draw a spot to put the milk down on.")
if island_counter_top(world) not in world.get_semantic_annotations_by_type(CounterTop): raise ExerciseVerificationFailed("The island's work surface is one the reasoner recognises as a counter top.")
island_box = island_counter_top(world).supporting_surface.area.as_bounding_box_collection_in_frame(world.root).bounding_box()
if not (island_box.min_x <= place_spot.x <= island_box.max_x and island_box.min_y <= place_spot.y <= island_box.max_y): raise ExerciseVerificationFailed("The spot should lie on the island's surface.")
milk_box = milk_body.collision.as_bounding_box_collection_in_frame(world.root).bounding_box()
if abs(float(place_spot.z) - (island_box.max_z + (milk_box.max_z - milk_box.min_z) / 2)) > PLACEMENT_TOLERANCE: raise ExerciseVerificationFailed("The spot should stand the carton on the surface, not in it or above it.")
faced = pose_the_robot_faces(place_spot, context)
if not np.allclose(faced.to_position().to_np()[:3], np.array([float(place_spot.x), float(place_spot.y), float(place_spot.z)])): raise ExerciseVerificationFailed("The pose should keep the spot's position.")
if not np.isclose(float(faced.yaw), float(context.robot.root.global_pose.yaw)): raise ExerciseVerificationFailed("The pose should take the robot's own heading.")
```

## 7. Putting the milk down

{class}`~coraplex.robot_plans.actions.core.placing.PlaceAction` takes the body, the pose to
put it at, and the arm holding it.

The standing pose is deferred again, for the same reason as before: the robot is still holding
the milk in front of the fridge, and where it has to stand to reach the island is not settled
until it gets there. The navigation is given the *spot*, and the search works out a heading for
it at each pose it considers.

The placing pose has to be deferred too, and for a reason worth pausing on: it is read off the
robot, and the robot has not driven to the island yet. Written out at plan-build time it would
carry the heading the robot has while standing at the fridge.

Your goal:
- Navigate to a deferred reachability location for `place_spot`, place the milk at a deferred
  `pose_the_robot_faces` for it with `MILK_ARM`, park both arms, and perform it

```{code-cell} ipython3
:tags: [exercise]
# TODO: carry the milk to the island and put it down
# plan = sequential(
#     [
#         a(NavigateAction)(target_location=variable(Pose, domain=DeferredLocation(...)), keep_joint_states=True),
#         a(PlaceAction)(
#             object_designator=milk_body,
#             target_location=variable(Pose, domain=DeferredLocation(lambda: [pose_the_robot_faces(place_spot, context)])),
#             arm=MILK_ARM,
#         ),
#         ParkArmsAction(Arms.BOTH),
#     ],
#     context,
# )

plan = ...
```

```{code-cell} ipython3
:tags: [example-solution]
plan = sequential(
    [
        a(NavigateAction)(
            target_location=variable(
                Pose,
                domain=DeferredLocation(
                    lambda: giskard_reachability_location(place_spot, context, MILK_ARM)
                ),
            ),
            keep_joint_states=True,
        ),
        a(PlaceAction)(
            object_designator=milk_body,
            target_location=variable(
                Pose,
                domain=DeferredLocation(
                    lambda: [pose_the_robot_faces(place_spot, context)]
                ),
            ),
            arm=MILK_ARM,
        ),
        ParkArmsAction(Arms.BOTH),
    ],
    context,
)

with simulated_robot(collision_avoidance=True):
    plan.perform()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if plan is ...: raise ExerciseVerificationFailed("Build the placing plan.")
placed_box = milk_body.collision.as_bounding_box_collection_in_frame(world.root).bounding_box()
surface_box = island_counter_top(world).supporting_surface.area.as_bounding_box_collection_in_frame(world.root).bounding_box()
if abs(placed_box.min_z - surface_box.max_z) > PLACEMENT_TOLERANCE: raise ExerciseVerificationFailed("The milk should be resting on the island's surface.")
if not (surface_box.min_x <= placed_box.min_x and placed_box.max_x <= surface_box.max_x): raise ExerciseVerificationFailed("The milk should stand within the island's surface, not over its edge.")
if not (surface_box.min_y <= placed_box.min_y and placed_box.max_y <= surface_box.max_y): raise ExerciseVerificationFailed("The milk should stand within the island's surface, not over its edge.")
```

## 8. Shutting the fridge again

Closing is opening in reverse, and
{class}`~coraplex.robot_plans.actions.core.container.CloseAction` takes the same four
arguments. The handle rides the swinging door, so it is somewhere else now than it was when
you stood in front of it — which is exactly why the standing pose has to be found again rather
than remembered.

Your goal:
- Navigate to a deferred reachability location for the handle, shut the door to
  `DOOR_CLOSING_ANGLE`, park both arms, and perform it

```{code-cell} ipython3
:tags: [exercise]
DOOR_CLOSING_ANGLE = 0.0

# TODO: drive back to the handle and shut the door
# plan = sequential(
#     [
#         a(NavigateAction)(target_location=variable(Pose, domain=DeferredLocation(...)), keep_joint_states=True),
#         CloseAction(handle, DOOR_ARM, handle_grasp.approach_direction, DOOR_CLOSING_ANGLE),
#         ParkArmsAction(Arms.BOTH),
#     ],
#     context,
# )

plan = ...
```

```{code-cell} ipython3
:tags: [example-solution]
DOOR_CLOSING_ANGLE = 0.0

plan = sequential(
    [
        a(NavigateAction)(
            target_location=variable(
                Pose,
                domain=DeferredLocation(
                    lambda: giskard_reachability_location(
                        handle, context, DOOR_ARM, handle_grasp
                    )
                ),
            ),
            keep_joint_states=True,
        ),
        CloseAction(handle, DOOR_ARM, handle_grasp.approach_direction, DOOR_CLOSING_ANGLE),
        ParkArmsAction(Arms.BOTH),
    ],
    context,
)

with simulated_robot(collision_avoidance=True):
    plan.perform()
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if plan is ...: raise ExerciseVerificationFailed("Build the closing plan.")
DOOR_IS_CLOSED_ANGLE = 0.02
door_angle = fridge.doors[0].root.parent_connection.position
if door_angle >= DOOR_IS_CLOSED_ANGLE: raise ExerciseVerificationFailed(f"The fridge door should be shut, but it stands at {door_angle:.3f} rad.")
```

The transport is done: the milk is on the island and the fridge is shut behind it.

## 9. Deriving the opening and the shutting

Everything so far was written down in the order it had to happen. But the plan's *subject* is
the transport — that the milk happens to stand behind a closed door is a fact about the
world, not about the task. A plan that has the opening written into it only works in a kitchen
whose fridge is shut.

So find the container instead. A body that stands in something is *inside* it, which
{class}`~semantic_digital_twin.reasoning.predicates.InsideOf` measures directly, as the share
of the body that lies within the other. Anything meant to hold things carries an
{class}`~semantic_digital_twin.semantic_annotations.mixins.IsStorageSpace` annotation, and one
that shuts also has {class}`~semantic_digital_twin.semantic_annotations.mixins.HasDoors`.

To ask that inside a query, the measurement has to be something a query can carry rather than
a number computed up front. `@symbolic_function` turns a plain predicate into one usable inside
a `.where(...)`, evaluated per candidate as the query runs.

In this kitchen the milk is held by two storage spaces: the fridge, and the rack the fridge
stands in. Only the fridge has a door, which is what picks it out.

A drawer is found just as well, since it holds what it contains in its own case — but its
handle hangs off the drawer rather than off a door, so only doors are followed here.

The milk is on the island now, in nothing at all, so this runs against a fresh kitchen.

Your goal:
- Write a `contains(container, body)` predicate, decorated with `@symbolic_function`, that is
  true when more than nine tenths of the body lies inside the container
- Write `handle_of_enclosing_container(body, world) -> Optional[Body]` returning the handle of
  the first door among the storage spaces that hold `body`, and `None` when none do

```{code-cell} ipython3
:tags: [remove-input]
fresh_world = build_kitchen_world()
stand_milk_in_fridge(fresh_world)
fresh_fridge = the(entity(variable(Fridge, domain=fresh_world.semantic_annotations))).first()
fresh_milk = the(entity(variable(Milk, domain=fresh_world.semantic_annotations))).first().root
```

```{code-cell} ipython3
:tags: [exercise]
# TODO: find what has to be pulled open before a body can be taken
# @symbolic_function
# def contains(container: KinematicStructureEntity, body: KinematicStructureEntity) -> bool:
#     return InsideOf(body, container)() > 0.9
#
# def handle_of_enclosing_container(body: Body, world: World) -> Optional[Body]:
#     storage_space = variable(IsStorageSpace, domain=world.semantic_annotations)
#     containers = an(entity(storage_space).where(contains(storage_space.root, body))).evaluate()
#     return next((found.doors[0].handle.root for found in containers if ...), None)

contains = ...
handle_of_enclosing_container = ...
```

```{code-cell} ipython3
:tags: [example-solution]
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


def handle_of_enclosing_container(body: Body, world: World) -> Optional[Body]:
    """
    Find what has to be pulled open before ``body`` can be taken.

    :param body: The body that may stand inside a container.
    :param world: The world holding both.
    :return: The handle of the container, or ``None`` when the body stands in the open or
        in something that does not open.
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
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if contains is ...: raise ExerciseVerificationFailed("Write the contains predicate.")
if handle_of_enclosing_container is ...: raise ExerciseVerificationFailed("Write handle_of_enclosing_container.")
if handle_of_enclosing_container(fresh_milk, fresh_world) is not fresh_fridge.doors[0].handle.root: raise ExerciseVerificationFailed("A milk standing in the fridge should lead to the fridge door's handle.")
if handle_of_enclosing_container(island_counter_top(fresh_world).root, fresh_world) is not None: raise ExerciseVerificationFailed("The island surface stands in nothing that opens.")
```

## 10. Wrapping the transport

With the handle in hand, the opening and the shutting become something you put *around* a
list of steps rather than into it. If nothing has to be opened, the steps come back
untouched — and the plan is the transport alone.

Which hand pulls the handle can be left open here, the same way the milk's approach direction
was: a hand that cannot reach it fails its precondition and the other is tried. The
standing-pose search still names one, because a grasp says which gripper comes at the handle
and one of the two has to be asked.

Your goal:
- Write `add_container_opening_and_closing(actions, body, context) -> List[ActionLike]` that
  returns `actions` unchanged when `body` stands in the open, and otherwise puts a navigation,
  an `OpenAction` and a park in front of it, and a navigation, a `CloseAction` and a park
  behind it, both container actions leaving their arm open

```{code-cell} ipython3
:tags: [exercise]
# TODO: put the container steps around the given ones
# def add_container_opening_and_closing(actions, body, context) -> List[ActionLike]:
#     handle = handle_of_enclosing_container(body, context.world)
#     if handle is None:
#         return actions
#     door_arm = DOOR_ARM
#     handle_grasp = get_grasp_for_handle(door_arm, context)
#     return [..., *actions, ...]

add_container_opening_and_closing = ...
```

```{code-cell} ipython3
:tags: [example-solution]
def add_container_opening_and_closing(
    actions: List[ActionLike], body: Body, context: Context
) -> List[ActionLike]:
    """
    Open the container ``body`` stands in before the given steps, and shut it again after
    them.

    Which hand does the pulling is left to the plan: a hand that cannot reach the handle
    fails its precondition and the other one is tried. The standing pose is still looked
    for with a named hand, since a grasp says which gripper comes at the handle and one of
    the two has to be asked.

    :param actions: The steps that need the container open.
    :param body: The body those steps reach for.
    :param context: The context the standing poses are chosen in.
    :return: The steps with the container steps around them, and ``actions`` itself when
        the body stands in the open.
    """
    handle = handle_of_enclosing_container(body, context.world)
    if handle is None:
        return actions

    door_arm = DOOR_ARM
    handle_grasp = get_grasp_for_handle(door_arm, context)

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
        a(OpenAction)(
            object_designator=handle,
            arm=...,
            approach_direction=handle_grasp.approach_direction,
            goal_joint_state=DOOR_OPENING_ANGLE,
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
        a(CloseAction)(
            object_designator=handle,
            arm=...,
            approach_direction=handle_grasp.approach_direction,
            goal_joint_state=DOOR_CLOSING_ANGLE,
        ),
        ParkArmsAction(Arms.BOTH),
    ]
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if add_container_opening_and_closing is ...: raise ExerciseVerificationFailed("Write add_container_opening_and_closing.")
fresh_robot = variable(PR2, domain=fresh_world.semantic_annotations)
fresh_context = Context(world=fresh_world, robot=the(entity(fresh_robot)).first(), evaluate_conditions=False)
transport = [ParkArmsAction(Arms.BOTH)]
wrapped = add_container_opening_and_closing(transport, fresh_milk, fresh_context)
if wrapped[3:-3] != transport: raise ExerciseVerificationFailed("The given steps should come back untouched, in the middle.")
if wrapped[1].factory is not OpenAction or wrapped[1].kwargs["object_designator"] is not fresh_fridge.doors[0].handle.root: raise ExerciseVerificationFailed("The fridge door's handle should be the one that is opened.")
if wrapped[1].kwargs["goal_joint_state"] != DOOR_OPENING_ANGLE: raise ExerciseVerificationFailed("The door should be opened to DOOR_OPENING_ANGLE.")
if wrapped[-2].factory is not CloseAction or wrapped[-2].kwargs["goal_joint_state"] != DOOR_CLOSING_ANGLE: raise ExerciseVerificationFailed("The door should be shut again to DOOR_CLOSING_ANGLE.")
if wrapped[1].kwargs["arm"] is not Ellipsis or wrapped[-2].kwargs["arm"] is not Ellipsis: raise ExerciseVerificationFailed("Both container actions should leave their arm open.")
if wrapped[1].kwargs["approach_direction"] is not get_grasp_for_handle(DOOR_ARM, fresh_context).approach_direction: raise ExerciseVerificationFailed("The door should be pulled from the side the handle's grasp comes at it from.")
if add_container_opening_and_closing(transport, island_counter_top(fresh_world).root, fresh_context) is not transport: raise ExerciseVerificationFailed("A body standing in the open should leave the steps exactly as they were.")
```

## 11. The whole thing, as a demonstration

{class}`~coraplex.demonstrations.RobotDemonstration` is the scaffolding a demo runs on. It
owns the ROS session, decides whether to build a world or fetch one from a running controller,
and wraps execution in the right environment, so a demonstration writes only its own scene and
its own plan, through five methods.

The plan states the transport only. What it has to unpack to get at the milk comes out of
`add_container_opening_and_closing`.

Your goal:
- Write `KitchenFridgeDemonstration(RobotDemonstration)` implementing `build_simulated_world`,
  `is_scene_populated`, `populate_scene`, `build_context` and `build_plan`
- Instantiate it with `used_robot=PR2` into a variable named `demonstration`

```{code-cell} ipython3
:tags: [exercise]
# TODO: assemble the five methods
# @dataclass
# class KitchenFridgeDemonstration(RobotDemonstration):
#     ros_node_name: ClassVar[str] = "kitchen_fridge_demo_node"
#
#     def build_simulated_world(self) -> World: ...
#     def is_scene_populated(self, world: World) -> bool: ...
#     def populate_scene(self, world: World) -> None: ...
#     def build_context(self, world: World) -> Context: ...
#     def build_plan(self, context: Context) -> PlanNode: ...

KitchenFridgeDemonstration = ...
demonstration = ...
```

```{code-cell} ipython3
:tags: [example-solution]
@dataclass
class KitchenFridgeDemonstration(RobotDemonstration):
    """
    A robot fetches a milk out of the kitchen fridge and puts it on the kitchen island.
    """

    ros_node_name: ClassVar[str] = "kitchen_fridge_demo_node"

    def build_simulated_world(self) -> World:
        return build_kitchen_world()

    def is_scene_populated(self, world: World) -> bool:
        milk = variable(Milk, domain=world.semantic_annotations)
        return next(an(entity(milk)).evaluate(), None) is not None

    def populate_scene(self, world: World) -> None:
        stand_milk_in_fridge(world)

    def build_context(self, world: World) -> Context:
        robot = variable(self.used_robot, domain=world.semantic_annotations)
        return Context(
            world=world,
            robot=the(entity(robot)).first(),
            ros_node=self.ros_node,
            evaluate_conditions=True,
            alternative_motion_mappings=self.alternative_motion_mappings,
        )

    def build_plan(self, context: Context) -> PlanNode:
        world = context.world
        milk = the(entity(variable(Milk, domain=world.semantic_annotations))).first()
        milk_body = milk.root
        place_spot = place_spot_on_island(world, milk)

        transport = [
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: giskard_reachability_location(
                            milk_body, context, MILK_ARM
                        )
                    ),
                ),
                keep_joint_states=True,
            ),
            a(PickUpAction)(
                object_designator=milk_body,
                grasp_description=get_grasp_for_milk(context),
                arm=MILK_ARM,
            ),
            ParkArmsAction(Arms.BOTH),
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: giskard_reachability_location(
                            place_spot, context, MILK_ARM
                        )
                    ),
                ),
                keep_joint_states=True,
            ),
            a(PlaceAction)(
                object_designator=milk_body,
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: [pose_the_robot_faces(place_spot, context)]
                    ),
                ),
                arm=MILK_ARM,
            ),
            ParkArmsAction(Arms.BOTH),
        ]

        return sequential(
            [
                ParkArmsAction(Arms.BOTH),
                MoveTorsoAction(TorsoState.HIGH),
                *add_container_opening_and_closing(transport, milk_body, context),
            ],
            context,
        )


demonstration = KitchenFridgeDemonstration(used_robot=PR2)
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
if demonstration is ...: raise ExerciseVerificationFailed("Instantiate the demonstration.")
demo_world = demonstration.build_simulated_world()
if demonstration.is_scene_populated(demo_world): raise ExerciseVerificationFailed("Building the world should not already put the milk into it.")
demonstration.populate_scene(demo_world)
if not demonstration.is_scene_populated(demo_world): raise ExerciseVerificationFailed("Populating the scene should stand the milk in the fridge.")
demo_context = demonstration.build_context(demo_world)
if not demo_context.evaluate_conditions: raise ExerciseVerificationFailed("The demonstration's context should evaluate conditions.")
demo_plan = demonstration.build_plan(demo_context)
steps = [type(node.designator) if isinstance(node, ActionNode) else node.underspecified_action.factory for node in demo_plan.children]
if steps != [ParkArmsAction, MoveTorsoAction, NavigateAction, OpenAction, ParkArmsAction, NavigateAction, PickUpAction, ParkArmsAction, NavigateAction, PlaceAction, ParkArmsAction, NavigateAction, CloseAction, ParkArmsAction]: raise ExerciseVerificationFailed(f"The plan should open the fridge, transport the milk and shut the fridge again, but it reads {steps}.")
demo_pick_up = next(node for node in demo_plan.children if isinstance(node, UnderspecifiedNode) and node.underspecified_action.factory is PickUpAction)
demo_grasp = demo_pick_up.underspecified_action.kwargs["grasp_description"]
if demo_grasp.factory is not GraspDescription or demo_grasp.kwargs["approach_direction"] is not Ellipsis: raise ExerciseVerificationFailed("The pick-up's grasp should leave the side the gripper comes from open.")
demo_place = next(node for node in demo_plan.children if isinstance(node, UnderspecifiedNode) and node.underspecified_action.factory is PlaceAction)
if not isinstance(demo_place.underspecified_action.kwargs["target_location"]._domain_.domain, DeferredLocation): raise ExerciseVerificationFailed("The placing pose reads the robot's heading, so it has to be deferred until the robot has driven to the island.")
```

## Where to go from here

What you just wrote is, near enough, `coraplex/demos/coraplex_kitchen_fridge_demo/demo.py`.
The shipped version differs mainly in carrying the helpers you wrote here as methods on the
demonstration class, with the shelf layer's name and the carrying arm as documented fields.
Run it with:

```bash
python coraplex/demos/coraplex_kitchen_fridge_demo/demo.py
```

Then try changing something and watch the plan follow: turn the carton around on its shelf
and the grasp turns with it, or stand it on the island to begin with and the fridge steps
disappear from the plan entirely.
