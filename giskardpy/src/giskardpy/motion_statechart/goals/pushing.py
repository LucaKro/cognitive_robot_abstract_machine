"""
Push a free-moving body onto a target pose with a single point of contact.

A body that is only touched, never held, cannot be moved along an arbitrary path: a point
contact can push but not pull, so reaching a pose takes a sequence of pushes, each chosen
against the pose the body has by then.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy
from typing_extensions import List, Optional

from krrood.symbolic_math.symbolic_math import trinary_logic_not, trinary_logic_or
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.exceptions import UncorrectableOrientationError
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import Goal, NodeArtifacts
from giskardpy.motion_statechart.monitors.cartesian_monitors import PoseReached
from giskardpy.motion_statechart.monitors.progress_monitors import ProgressStalled
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPosition,
    CartesianPositionStraight,
)
from semantic_digital_twin.datastructures.types import NpMatrix4x4
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% describing where a body can be pushed


@dataclass
class PushContact:
    """
    A place on a body's outline that can be pushed, and the direction a push there
    travels.
    """

    point: Point3
    """
    The point on the body's surface, in the body's own frame.
    """

    direction: Vector3
    """
    The direction a push at :attr:`point` travels, in the body's own frame.

    Unit length, pointing into the body, since a point contact can only push.
    """


class PushPhase(Enum):
    """
    Which part of a body's pose error the next push is meant to correct.
    """

    ROTATE = 1
    """
    The body is pointing the wrong way, so it is pushed off-centre to turn it.
    """

    TRANSLATE = 2
    """
    The body is pointing about right, so it is pushed towards where it belongs.
    """


@dataclass
class SelectedPush:
    """
    One push: which contact it uses, and the three points the pusher travels through.
    """

    contact: PushContact
    """
    The contact this push was chosen to use.
    """

    standoff: numpy.ndarray
    """
    Where the pusher waits before the push, in the root frame.
    """

    contact_point: numpy.ndarray
    """
    Where the pusher's centre sits when it meets the body, in the root frame.
    """

    follow_through: numpy.ndarray
    """
    Where the pusher travels to, past the body, in the root frame.
    """

    phase: PushPhase
    """
    The part of the pose error this push was chosen to correct.
    """


# %% choosing the next push


@dataclass
class PushSelector:
    """
    Chooses which contact to push next from a body's pose error.

    Restricting the choice to a fixed list of contacts, rather than solving for an
    arbitrary push, keeps the decision to a comparison between a handful of candidates
    that can be checked by hand.
    """

    contacts: List[PushContact]
    """
    The places on the body that can be pushed.
    """

    centroid: Point3
    """
    The body's centroid, in the body's own frame.

    A body's frame need not sit on its centroid, and both scores are about how a push
    acts relative to the centroid rather than the frame.
    """

    pusher_radius: float
    """
    How far the pusher's centre sits off the surface it touches, in metres.
    """

    standoff_distance: float
    """
    How far behind the contact a push starts, in metres.
    """

    minimum_push_distance: float
    """
    The shortest a push may be, in metres.

    A push much shorter than this is taken up by friction and the slack in the pusher's
    own servos without the body moving at all, so a small error is still worth a push of
    some size.
    """

    maximum_push_distance: float
    """
    How far past the contact a push may travel, in metres.
    """

    pushing_height: float
    """
    Height above the root frame at which contact is made, in metres.
    """

    orientation_tolerance: float
    """
    Orientation error above which turning the body takes priority, in radians.
    """

    translation_lever_penalty: float = 5.0
    """
    How strongly a push that would also turn the body is passed over when the body only
    needs moving, relative to how well that push is aimed.

    Aim is a fraction and a lever arm is a length, so this also carries the scale between
    the two: at a value near one, a lever arm of a few centimetres would barely register
    against an aim of up to one.
    """

    push_gain: float = 0.5
    """
    What fraction of the remaining error one push aims to correct.

    Aiming for the whole of it overshoots, because a pushed body keeps sliding after the
    pusher has stopped driving it, and an overshoot has to be undone from the other side.
    Aiming short converges instead, at the cost of more attempts.
    """

    def select(
        self, root_T_body: NpMatrix4x4, root_T_target: NpMatrix4x4
    ) -> SelectedPush:
        """
        Choose the push that best corrects the body's pose error.

        :param root_T_body: The body's current pose.
        :param root_T_target: The pose the body should end up at.
        :return: The chosen push.
        """
        orientation_error = self._orientation_error(root_T_body, root_T_target)
        phase = (
            PushPhase.ROTATE
            if abs(orientation_error) > self.orientation_tolerance
            else PushPhase.TRANSLATE
        )
        position_error = root_T_target[:2, 3] - root_T_body[:2, 3]
        contact = max(
            self.contacts,
            key=lambda candidate: self._score(
                candidate, root_T_body, phase, orientation_error, position_error
            ),
        )
        return self._push_through(
            contact,
            root_T_body,
            phase,
            self._push_distance(contact, phase, orientation_error, position_error),
        )

    def _push_distance(
        self,
        contact: PushContact,
        phase: PushPhase,
        orientation_error: float,
        position_error: numpy.ndarray,
    ) -> float:
        """
        How far past the contact this push should travel.

        A push as long as the error is large is what keeps the body from being shoved
        past its target and having to be brought back from the other side.

        :param contact: The contact being pushed.
        :param phase: The part of the error this push corrects.
        :param orientation_error: How far the body has to turn.
        :param position_error: How far the body has to move, in the plane.
        :return: The distance in metres.
        """
        if phase == PushPhase.ROTATE:
            # Turning the body moves the contact along an arc about the centroid, so the
            # push has to be as long as that arc.
            wanted = abs(orientation_error) * self._distance_to_centroid(contact)
        else:
            wanted = float(numpy.linalg.norm(position_error))
        return min(
            max(wanted * self.push_gain, self.minimum_push_distance),
            self.maximum_push_distance,
        )

    def _distance_to_centroid(self, contact: PushContact) -> float:
        """
        :param contact: The contact to measure.
        :return: How far ``contact`` sits from the body's centroid, in the plane.
        """
        return math.hypot(
            contact.point.x - self.centroid.x, contact.point.y - self.centroid.y
        )

    @staticmethod
    def _orientation_error(
        root_T_body: NpMatrix4x4, root_T_target: NpMatrix4x4
    ) -> float:
        """
        The heading the body has to turn through to match the target, taken the short way
        around.

        :param root_T_body: The body's current pose.
        :param root_T_target: The pose the body should end up at.
        :return: The signed error in radians, within half a revolution of zero.
        """
        body_yaw = math.atan2(root_T_body[1, 0], root_T_body[0, 0])
        target_yaw = math.atan2(root_T_target[1, 0], root_T_target[0, 0])
        return math.remainder(target_yaw - body_yaw, math.tau)

    def _score(
        self,
        contact: PushContact,
        root_T_body: NpMatrix4x4,
        phase: PushPhase,
        orientation_error: float,
        position_error: numpy.ndarray,
    ) -> float:
        """
        How well pushing ``contact`` would correct the part of the error ``phase`` names.

        :param contact: The contact being scored.
        :param root_T_body: The body's current pose.
        :param phase: The part of the error to correct.
        :param orientation_error: How far the body has to turn.
        :param position_error: How far the body has to move, in the plane.
        :return: The score, higher being better.
        """
        direction = self._in_root_frame(contact.direction, root_T_body)
        lever_arm = self._lever_arm(contact, root_T_body, direction)
        if phase == PushPhase.ROTATE:
            return math.copysign(1.0, orientation_error) * lever_arm
        distance_to_go = float(numpy.linalg.norm(position_error))
        if distance_to_go == 0.0:
            return -abs(lever_arm) * self.translation_lever_penalty
        aim = float(numpy.dot(direction[:2], position_error / distance_to_go))
        return aim - self.translation_lever_penalty * abs(lever_arm)

    def _lever_arm(
        self,
        contact: PushContact,
        root_T_body: NpMatrix4x4,
        direction: numpy.ndarray,
    ) -> float:
        """
        The turning effect a push at ``contact`` has about the body's centroid.

        :param contact: The contact being pushed.
        :param root_T_body: The body's current pose.
        :param direction: The push direction, in the root frame.
        :return: The signed moment, positive when the push turns the body anticlockwise.
        """
        offset_from_centroid = self._in_root_frame(
            Vector3(
                x=contact.point.x - self.centroid.x,
                y=contact.point.y - self.centroid.y,
                z=contact.point.z - self.centroid.z,
            ),
            root_T_body,
        )
        return float(
            offset_from_centroid[0] * direction[1]
            - offset_from_centroid[1] * direction[0]
        )

    @staticmethod
    def _in_root_frame(direction: Vector3, root_T_body: NpMatrix4x4) -> numpy.ndarray:
        """
        Rotate a direction given in the body's frame into the root frame.

        :param direction: The direction in the body's frame.
        :param root_T_body: The body's current pose.
        :return: The direction in the root frame.
        """
        return root_T_body[:3, :3] @ direction.to_np().flatten()[:3]

    def _push_through(
        self,
        contact: PushContact,
        root_T_body: NpMatrix4x4,
        phase: PushPhase,
        push_distance: float,
    ) -> SelectedPush:
        """
        Lay out the three points the pusher travels through to push ``contact``.

        :param contact: The contact to push.
        :param root_T_body: The body's current pose.
        :param phase: The part of the error this push corrects.
        :param push_distance: How far past the contact the push travels.
        :return: The push, ready to be driven.
        """
        direction = self._in_root_frame(contact.direction, root_T_body)
        surface_point = (
            root_T_body @ numpy.append(contact.point.to_np().flatten()[:3], 1.0)
        )[:3]
        # The pusher touches the surface rather than reaching it, so its centre stops a
        # radius short along the direction it pushes in.
        contact_point = surface_point - direction * self.pusher_radius
        contact_point[2] = self.pushing_height
        return SelectedPush(
            contact=contact,
            standoff=contact_point - direction * self.standoff_distance,
            contact_point=contact_point,
            follow_through=contact_point + direction * push_distance,
            phase=phase,
        )


# %% driving one push


@dataclass(eq=False, repr=False)
class PushOnce(Goal):
    """
    One push: lift clear of the body, travel over it, descend behind the chosen contact,
    then push through.

    The push is chosen when this goal starts and held for the whole attempt. Recomputing
    it every cycle would make the pusher chase a goal that jumps to the far side of the
    body the moment the required push flips, dragging it straight through what it is
    supposed to be pushing.
    """

    pushed_body: KinematicStructureEntity = field(kw_only=True)
    """
    The body being pushed.
    """

    target_body: KinematicStructureEntity = field(kw_only=True)
    """
    The body marking the pose :attr:`pushed_body` should end up at.
    """

    pusher: KinematicStructureEntity = field(kw_only=True)
    """
    The body doing the pushing.
    """

    selector: PushSelector = field(kw_only=True)
    """
    Chooses which contact this push uses.
    """

    travel_height: float = field(kw_only=True)
    """
    Height at which the pusher crosses the body, in metres.
    """

    approach_velocity: float = field(default=0.6, kw_only=True)
    """
    How fast the pusher moves while it is not touching the body, in metres per second.

    Getting into position touches nothing, so it is only worth the time it takes.
    """

    push_velocity: float = field(
        default=CartesianPosition.default_reference_velocity, kw_only=True
    )
    """
    How fast the pusher moves while pushing, in metres per second.

    Slower than the approach: a shove is harder to predict the faster it is, and the body
    keeps sliding once it has been let go.
    """

    _lift_point: Optional[Point3] = field(default=None, init=False, repr=False)
    """
    Where the pusher rises to before crossing the body.
    """

    _travel_point: Optional[Point3] = field(default=None, init=False, repr=False)
    """
    Where the pusher crosses to, above the standoff.
    """

    _standoff_point: Optional[Point3] = field(default=None, init=False, repr=False)
    """
    Where the pusher descends to, behind the contact.
    """

    _follow_through_point: Optional[Point3] = field(
        default=None, init=False, repr=False
    )
    """
    Where the pusher pushes to, past the contact.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        root = context.world.root
        (
            self._lift_point,
            self._travel_point,
            self._standoff_point,
            self._follow_through_point,
        ) = [
            self._create_goal_point(context, name)
            for name in ("lift", "travel", "standoff", "follow_through")
        ]
        self.add_node(
            Sequence(
                nodes=[
                    CartesianPosition(
                        name="lift",
                        root_link=root,
                        tip_link=self.pusher,
                        goal_point=self._lift_point,
                        reference_velocity=self.approach_velocity,
                    ),
                    CartesianPosition(
                        name="travel",
                        root_link=root,
                        tip_link=self.pusher,
                        goal_point=self._travel_point,
                        reference_velocity=self.approach_velocity,
                    ),
                    CartesianPosition(
                        name="descend",
                        root_link=root,
                        tip_link=self.pusher,
                        goal_point=self._standoff_point,
                        reference_velocity=self.approach_velocity,
                    ),
                    CartesianPositionStraight(
                        name="push",
                        root_link=root,
                        tip_link=self.pusher,
                        goal_point=self._follow_through_point,
                        reference_velocity=self.push_velocity,
                    ),
                ]
            )
        )

    def _create_goal_point(self, context: MotionStatechartContext, name: str) -> Point3:
        """
        Create a point whose value is written when this goal starts.

        :param context: The context holding the variables the point is registered with.
        :param name: Name for the point's variables, unique within this goal.
        :return: The registered point, expressed in the world's root frame.
        """
        point = Point3.create_with_variables(f"{self.name}/{name}")
        point.reference_frame = context.world.root
        context.float_variable_data.register_expression(point)
        return point

    def on_start(self, context: MotionStatechartContext) -> None:
        """
        Choose this attempt's push and freeze the four points it travels through.
        """
        world = context.world
        selected = self.selector.select(
            root_T_body=world.compute_forward_kinematics_np(
                world.root, self.pushed_body
            ),
            root_T_target=world.compute_forward_kinematics_np(
                world.root, self.target_body
            ),
        )
        pusher_position = world.compute_forward_kinematics_np(world.root, self.pusher)[
            :3, 3
        ]
        for point, value in (
            (self._lift_point, self._at_travel_height(pusher_position)),
            (self._travel_point, self._at_travel_height(selected.standoff)),
            (self._standoff_point, selected.standoff),
            (self._follow_through_point, selected.follow_through),
        ):
            context.float_variable_data.set_value(point, value)

    def _at_travel_height(self, position: numpy.ndarray) -> numpy.ndarray:
        """
        :param position: A point in the root frame.
        :return: The same point, raised to the height at which the body is crossed.
        """
        return numpy.array([position[0], position[1], self.travel_height])

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        [sequence] = self.nodes
        return NodeArtifacts(observation=sequence.observation_variable)


# %% pushing until the body is there


@dataclass(eq=False, repr=False)
class PushToPose(Goal):
    """
    Push a body onto a target pose, one contact at a time, until it is there.

    One push rarely lands a body on its target, so each attempt is followed by another
    chosen against the pose the body has by then.
    """

    pushed_body: KinematicStructureEntity = field(kw_only=True)
    """
    The body being pushed.
    """

    target_body: KinematicStructureEntity = field(kw_only=True)
    """
    The body marking the pose :attr:`pushed_body` should end up at.
    """

    pusher: KinematicStructureEntity = field(kw_only=True)
    """
    The body doing the pushing.
    """

    selector: PushSelector = field(kw_only=True)
    """
    Chooses which contact each attempt uses.
    """

    travel_height: float = field(kw_only=True)
    """
    Height at which the pusher crosses the body, in metres.
    """

    position_threshold: float = field(default=0.02, kw_only=True)
    """
    How close the body has to be to its target to count as there, in metres.
    """

    orientation_threshold: float = field(default=0.1, kw_only=True)
    """
    How closely the body has to point at its target to count as there, in radians.
    """

    stall_timeout: float = field(default=1.0, kw_only=True)
    """
    Seconds a push may make no progress before the next one is started.

    A push ends when the pusher has travelled its whole line, but a push that has run out
    of effect long before then would otherwise keep shoving a body that is no longer
    moving.
    """

    _on_target: Optional[PoseReached] = field(default=None, init=False, repr=False)
    """
    Watches whether the body has arrived.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        if self.selector.orientation_tolerance >= self.orientation_threshold:
            raise UncorrectableOrientationError(
                node=self,
                orientation_tolerance=self.selector.orientation_tolerance,
                orientation_threshold=self.orientation_threshold,
            )
        self._on_target = PoseReached(
            name="on target",
            root_link=context.world.root,
            tip_link=self.pushed_body,
            goal_pose=HomogeneousTransformationMatrix(reference_frame=self.target_body),
            position_threshold=self.position_threshold,
            orientation_threshold=self.orientation_threshold,
        )
        push_once = PushOnce(
            name="push once",
            pushed_body=self.pushed_body,
            target_body=self.target_body,
            pusher=self.pusher,
            selector=self.selector,
            travel_height=self.travel_height,
        )
        stalled = ProgressStalled(
            name="push stalled", monitored_node=push_once, timeout=self.stall_timeout
        )
        self.add_nodes([self._on_target, push_once, stalled])

        push_once.start_condition = trinary_logic_not(
            self._on_target.observation_variable
        )
        # Resetting the attempt returns its whole subtree to not-started, so the next
        # tick starts it again and a fresh contact is chosen against the pose the body
        # has by then.
        push_once.reset_condition = trinary_logic_or(
            push_once.observation_variable, stalled.observation_variable
        )

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self._on_target.observation_variable)
