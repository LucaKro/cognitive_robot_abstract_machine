"""
Tests for pushing a free-moving body onto a target pose with a single point of contact.
"""

from __future__ import annotations

import math

import numpy
import pytest

from giskardpy.motion_statechart.goals.pushing import (
    PushContact,
    PushOnce,
    PushPhase,
    PushSelector,
    PushToPose,
)
from giskardpy.motion_statechart.exceptions import UncorrectableOrientationError
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.monitors.cartesian_monitors import PoseReached
from giskardpy.motion_statechart.monitors.progress_monitors import ProgressStalled
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPosition,
    CartesianPositionStraight,
)
from giskardpy.executor import Executor
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% a square body to push

HALF_WIDTH = 0.1
"""
Half the edge length of the square body the selector tests push around.
"""

PUSHER_RADIUS = 0.02
"""
How far the pusher's centre sits off the surface it touches.
"""

STANDOFF_DISTANCE = 0.05
"""
How far behind the contact a push starts.
"""

MINIMUM_PUSH_DISTANCE = 0.01
"""
The shortest a single push may travel past the contact.

Kept well below the errors the tests use, so it never decides the geometry they assert.
"""

MAXIMUM_PUSH_DISTANCE = 0.08
"""
The longest a single push may travel past the contact.
"""

PUSH_GAIN = 0.5
"""
What fraction of the remaining error one push aims to correct.
"""

PUSHING_HEIGHT = 0.02
"""
Height above the ground at which contact is made.
"""

ORIENTATION_TOLERANCE = 0.15
"""
Orientation error, in radians, above which turning the body takes priority.
"""


def square_contacts() -> list[PushContact]:
    """
    The four face midpoints of a square body, each pushed towards the body's centre.

    A square is used rather than a T so that every expected choice can be worked out by
    hand: the four contacts are symmetric, so only the pose error can decide between
    them.

    :return: One contact per face, ordered ``+x``, ``-x``, ``+y``, ``-y``.
    """
    return [
        PushContact(
            point=Point3(x=HALF_WIDTH, y=0.0, z=0.0),
            direction=Vector3(x=-1.0, y=0.0, z=0.0),
        ),
        PushContact(
            point=Point3(x=-HALF_WIDTH, y=0.0, z=0.0),
            direction=Vector3(x=1.0, y=0.0, z=0.0),
        ),
        PushContact(
            point=Point3(x=0.0, y=HALF_WIDTH, z=0.0),
            direction=Vector3(x=0.0, y=-1.0, z=0.0),
        ),
        PushContact(
            point=Point3(x=0.0, y=-HALF_WIDTH, z=0.0),
            direction=Vector3(x=0.0, y=1.0, z=0.0),
        ),
    ]


def offset_contacts() -> list[PushContact]:
    """
    Two contacts on the ``+y`` face, either side of the body's centre.

    Both push in the same direction, so only their lever arm about the centre tells them
    apart - which is exactly what the turning score is supposed to weigh.

    :return: The contact at ``+x`` first, the one at ``-x`` second.
    """
    return [
        PushContact(
            point=Point3(x=HALF_WIDTH, y=HALF_WIDTH, z=0.0),
            direction=Vector3(x=0.0, y=-1.0, z=0.0),
        ),
        PushContact(
            point=Point3(x=-HALF_WIDTH, y=HALF_WIDTH, z=0.0),
            direction=Vector3(x=0.0, y=-1.0, z=0.0),
        ),
    ]


def build_selector(contacts: list[PushContact]) -> PushSelector:
    """
    A selector over ``contacts`` with the module's constants.

    :param contacts: The contacts the selector chooses between.
    :return: The new selector.
    """
    return PushSelector(
        contacts=contacts,
        centroid=Point3(),
        pusher_radius=PUSHER_RADIUS,
        standoff_distance=STANDOFF_DISTANCE,
        minimum_push_distance=MINIMUM_PUSH_DISTANCE,
        maximum_push_distance=MAXIMUM_PUSH_DISTANCE,
        push_gain=PUSH_GAIN,
        pushing_height=PUSHING_HEIGHT,
        orientation_tolerance=ORIENTATION_TOLERANCE,
    )


def planar_pose(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> numpy.ndarray:
    """
    :param x: Position along x.
    :param y: Position along y.
    :param yaw: Heading around z, in radians.
    :return: The matching homogeneous transformation matrix.
    """
    return HomogeneousTransformationMatrix.from_xyz_rpy(x=x, y=y, yaw=yaw).to_np()


# %% choosing a push


def test_a_displaced_body_is_pushed_from_the_face_it_should_move_away_from():
    """
    Shoving a body along ``-x`` means standing on its ``+x`` face, since a point contact
    can only push.
    """
    contacts = square_contacts()
    selector = build_selector(contacts)

    selected = selector.select(
        root_T_body=planar_pose(x=0.5), root_T_target=planar_pose()
    )

    assert selected.phase == PushPhase.TRANSLATE
    assert selected.contact is contacts[0]


def test_a_body_displaced_the_other_way_is_pushed_from_the_opposite_face():
    """
    The choice follows the error rather than any preferred face.
    """
    contacts = square_contacts()
    selector = build_selector(contacts)

    selected = selector.select(
        root_T_body=planar_pose(x=-0.5), root_T_target=planar_pose()
    )

    assert selected.contact is contacts[1]


def test_a_turned_body_is_pushed_where_the_push_turns_it_back():
    """
    Two contacts pushing the same way are told apart by their lever arm: turning a body
    clockwise means pushing the side whose torque about the centre is clockwise.
    """
    contacts = offset_contacts()
    selector = build_selector(contacts)

    selected = selector.select(
        root_T_body=planar_pose(yaw=ORIENTATION_TOLERANCE * 2),
        root_T_target=planar_pose(),
    )

    assert selected.phase == PushPhase.ROTATE
    assert selected.contact is contacts[0]


def test_a_body_turned_the_other_way_is_pushed_on_its_other_side():
    """
    The lever arm that turns a body back reverses with the sign of the error.
    """
    contacts = offset_contacts()
    selector = build_selector(contacts)

    selected = selector.select(
        root_T_body=planar_pose(yaw=-ORIENTATION_TOLERANCE * 2),
        root_T_target=planar_pose(),
    )

    assert selected.contact is contacts[1]


def test_turning_the_body_takes_priority_until_it_is_pointing_about_right():
    """
    The phase switches exactly at the tolerance, so a body that is both turned and
    displaced is straightened before it is shoved.
    """
    selector = build_selector(square_contacts())

    just_over = selector.select(
        root_T_body=planar_pose(x=0.5, yaw=ORIENTATION_TOLERANCE * 1.01),
        root_T_target=planar_pose(),
    )
    just_under = selector.select(
        root_T_body=planar_pose(x=0.5, yaw=ORIENTATION_TOLERANCE * 0.99),
        root_T_target=planar_pose(),
    )

    assert just_over.phase == PushPhase.ROTATE
    assert just_under.phase == PushPhase.TRANSLATE


def test_an_orientation_error_is_measured_the_short_way_around():
    """
    A body turned just past half a revolution is barely turned the other way, so it is
    straightened by turning on rather than back.
    """
    contacts = offset_contacts()
    selector = build_selector(contacts)

    just_past_half_turn = selector.select(
        root_T_body=planar_pose(yaw=math.pi + 0.2), root_T_target=planar_pose()
    )
    just_under_half_turn = selector.select(
        root_T_body=planar_pose(yaw=math.pi - 0.2), root_T_target=planar_pose()
    )

    assert just_past_half_turn.contact is not just_under_half_turn.contact


# %% where the push travels


def test_the_push_runs_from_behind_the_contact_to_beyond_it():
    """
    The three points of a push lie on one line through the contact, spaced by the
    standoff behind it and the push distance past it, and the pusher's own radius keeps
    it off the surface.

    The error here is far larger than one push may correct, so the push runs its full
    permitted length.
    """
    contacts = square_contacts()
    selector = build_selector(contacts)

    selected = selector.select(
        root_T_body=planar_pose(x=0.5), root_T_target=planar_pose()
    )

    contact_surface_x = 0.5 + HALF_WIDTH
    numpy.testing.assert_allclose(
        selected.contact_point,
        [contact_surface_x + PUSHER_RADIUS, 0.0, PUSHING_HEIGHT],
        atol=1e-9,
    )
    numpy.testing.assert_allclose(
        selected.standoff,
        [contact_surface_x + PUSHER_RADIUS + STANDOFF_DISTANCE, 0.0, PUSHING_HEIGHT],
        atol=1e-9,
    )
    numpy.testing.assert_allclose(
        selected.follow_through,
        [
            contact_surface_x + PUSHER_RADIUS - MAXIMUM_PUSH_DISTANCE,
            0.0,
            PUSHING_HEIGHT,
        ],
        atol=1e-9,
    )


def test_the_push_follows_the_body_when_it_is_turned():
    """
    The contacts are given in the body's own frame, so a turned body is pushed on the
    face that has turned with it.
    """
    contacts = square_contacts()
    selector = build_selector(contacts)

    selected = selector.select(
        root_T_body=planar_pose(y=0.5, yaw=math.pi / 2), root_T_target=planar_pose()
    )

    # Turned a quarter turn, the body's +x face points along the root's +y, which is the
    # direction the body has to be pushed away from.
    assert selected.contact is contacts[0]
    numpy.testing.assert_allclose(
        selected.contact_point,
        [0.0, 0.5 + HALF_WIDTH + PUSHER_RADIUS, PUSHING_HEIGHT],
        atol=1e-9,
    )


# %% the statechart the goal builds


@pytest.fixture
def pushing_world() -> World:
    """
    A world with a free-moving block, a fixed target marker, and a pusher on three slide
    joints.
    """
    world = World.create_with_root_body("world")
    block = Body(name=PrefixedName("block"))
    target = Body(name=PrefixedName("target"))
    pusher = Body(name=PrefixedName("pusher"))
    links = [Body(name=PrefixedName(f"link_{axis}")) for axis in "xy"]

    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(world=world, parent=world.root, child=block)
        )
        world.add_connection(FixedConnection(parent=world.root, child=target))
        parent = world.root
        axes = [Vector3.X, Vector3.Y, Vector3.Z]
        children = links + [pusher]
        for axis_name, axis_factory, child in zip("xyz", axes, children):
            degree_of_freedom = DegreeOfFreedom(
                name=PrefixedName(f"slide_{axis_name}"),
                limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap(position=-1.0, velocity=-1.0),
                    upper=DerivativeMap(position=1.0, velocity=1.0),
                ),
            )
            world.add_degree_of_freedom(degree_of_freedom)
            connection = PrismaticConnection(
                name=degree_of_freedom.name,
                parent=parent,
                child=child,
                axis=axis_factory(reference_frame=parent),
                raw_dof=degree_of_freedom,
            )
            world.add_connection(connection)
            parent = child
    return world


def push_to_pose(world: World) -> PushToPose:
    """
    :param world: The world holding the block, the target and the pusher.
    :return: A goal pushing the world's block onto its target.
    """
    return PushToPose(
        pushed_body=world.get_kinematic_structure_entity_by_name("block"),
        target_body=world.get_kinematic_structure_entity_by_name("target"),
        pusher=world.get_kinematic_structure_entity_by_name("pusher"),
        selector=build_selector(square_contacts()),
        travel_height=0.06,
        orientation_threshold=ORIENTATION_TOLERANCE * 2,
    )


def compiled_statechart(world: World, goal: PushToPose) -> MotionStatechart:
    """
    Compile ``goal`` into a statechart, so its children exist and are built.

    :param world: The world the goal acts in.
    :param goal: The goal to compile.
    :return: The statechart holding the compiled goal.
    """
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(goal)
    Executor(context=MotionStatechartContext(world=world)).compile(
        motion_statechart=motion_statechart
    )
    return motion_statechart


def test_one_push_lifts_travels_descends_and_pushes_in_that_order(pushing_world):
    """
    The pusher can only reach another face by going over the body, so a push is four
    steps in a fixed order rather than a single straight run.
    """
    goal = push_to_pose(pushing_world)
    compiled_statechart(pushing_world, goal)

    [push_once] = [node for node in goal.nodes if isinstance(node, PushOnce)]
    [sequence] = [node for node in push_once.nodes if isinstance(node, Sequence)]

    assert [type(node) for node in sequence.nodes] == [
        CartesianPosition,
        CartesianPosition,
        CartesianPosition,
        CartesianPositionStraight,
    ]
    assert [node.name for node in sequence.nodes] == [
        "lift",
        "travel",
        "descend",
        "push",
    ]


def test_every_step_of_a_push_moves_the_pusher(pushing_world):
    """
    Only the pusher is commanded; the block moves because it is in the way, never
    because a task asked it to.
    """
    goal = push_to_pose(pushing_world)
    compiled_statechart(pushing_world, goal)

    [push_once] = [node for node in goal.nodes if isinstance(node, PushOnce)]
    [sequence] = [node for node in push_once.nodes if isinstance(node, Sequence)]

    pusher = pushing_world.get_kinematic_structure_entity_by_name("pusher")
    assert {node.tip_link for node in sequence.nodes} == {pusher}
    assert {node.root_link for node in sequence.nodes} == {pushing_world.root}


def test_a_finished_or_stalled_push_starts_another_one(pushing_world):
    """
    One push rarely lands the body on its target, so the goal resets its push and picks
    a new contact against the pose the body has by then.
    """
    goal = push_to_pose(pushing_world)
    compiled_statechart(pushing_world, goal)

    [push_once] = [node for node in goal.nodes if isinstance(node, PushOnce)]
    [stalled] = [node for node in goal.nodes if isinstance(node, ProgressStalled)]

    reset_dependencies = push_once._reset_condition.node_dependencies
    assert set(reset_dependencies) == {push_once, stalled}


def test_pushing_stops_once_the_body_is_on_its_target(pushing_world):
    """
    The goal watches the body rather than the pusher, since the pusher reaching a point
    says nothing about where the body ended up.
    """
    goal = push_to_pose(pushing_world)
    compiled_statechart(pushing_world, goal)

    [on_target] = [node for node in goal.nodes if isinstance(node, PoseReached)]
    [push_once] = [node for node in goal.nodes if isinstance(node, PushOnce)]

    assert on_target.tip_link is pushing_world.get_kinematic_structure_entity_by_name(
        "block"
    )
    assert on_target.goal_pose.reference_frame is (
        pushing_world.get_kinematic_structure_entity_by_name("target")
    )
    assert push_once._start_condition.node_dependencies == [on_target]


def test_a_push_only_travels_as_far_as_the_error_it_corrects():
    """
    A push as long as the body is far from its target overshoots, and an overshoot has
    to be undone from the other side, so a small error gets a short push.
    """
    contacts = square_contacts()
    selector = build_selector(contacts)
    distance_to_go = MAXIMUM_PUSH_DISTANCE / 2

    selected = selector.select(
        root_T_body=planar_pose(x=distance_to_go), root_T_target=planar_pose()
    )

    travelled = float(
        numpy.linalg.norm(selected.follow_through - selected.contact_point)
    )
    assert travelled == pytest.approx(distance_to_go * PUSH_GAIN)


def test_turning_a_body_pushes_along_the_arc_the_contact_has_to_travel():
    """
    How far a turning push has to travel depends on how far the contact sits from the
    centre it turns about, not on how far the body is from its target.
    """
    contacts = offset_contacts()
    selector = build_selector(contacts)
    orientation_error = ORIENTATION_TOLERANCE * 2

    selected = selector.select(
        root_T_body=planar_pose(yaw=orientation_error), root_T_target=planar_pose()
    )

    radius = math.hypot(selected.contact.point.x, selected.contact.point.y)
    travelled = float(
        numpy.linalg.norm(selected.follow_through - selected.contact_point)
    )
    assert travelled == pytest.approx(orientation_error * radius * PUSH_GAIN)


def test_a_goal_that_could_never_be_reached_is_refused(pushing_world):
    """
    Stopping short of the orientation the goal insists on would leave it turning nothing
    and never finishing, so it is rejected while the statechart is built rather than run
    into.
    """
    goal = push_to_pose(pushing_world)
    goal.orientation_threshold = goal.selector.orientation_tolerance / 2

    with pytest.raises(UncorrectableOrientationError) as exception_info:
        compiled_statechart(pushing_world, goal)

    assert exception_info.value.orientation_threshold == goal.orientation_threshold
    assert (
        exception_info.value.orientation_tolerance
        == goal.selector.orientation_tolerance
    )
