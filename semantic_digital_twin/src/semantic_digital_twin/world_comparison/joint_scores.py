"""
How well the joints of a reconstructed world move like those of the ground truth world.

Joints are compared between rigid groups: the sets of bodies that fixed connections join
into one rigid piece. A front split into three fixed pieces is then still one moving
part, and the joint that moves a group is the one connection attaching it to the rest.

A joint is compared by the motion it allows, resolved into the root frame of the ground
truth world: the line it turns about or the direction it slides along, and how far it
can move from where it stands now. How the joint is written down plays no part: not the
origins of its bodies, not the sign of its axis, not its multiplier or offset, and not
where along a hinge its frame happens to sit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import numpy.typing as npt

from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_comparison.matching import (
    BodyCorrespondence,
    pairs_sharing_the_most,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
    RevoluteConnection,
    ScrewConnection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    Connection,
    KinematicStructureEntity,
)

# %% rigid groups


@dataclass
class RigidGroup:
    """
    Entities that fixed connections join into one rigid piece, and the connection that
    moves the piece relative to the rest of its world.
    """

    entities: list[KinematicStructureEntity]
    """
    The entities of the piece.
    """

    parent_connection: Connection | None
    """
    The connection attaching the piece to the rest of its world; ``None`` for the piece
    holding the world's root.
    """

    @property
    def bodies(self) -> list[Body]:
        """
        :return: The bodies among the entities of the piece.
        """
        return [entity for entity in self.entities if isinstance(entity, Body)]


@dataclass
class RigidGroups:
    """
    A world divided into its rigid pieces.
    """

    groups: list[RigidGroup]
    """
    Every rigid piece of the world.
    """

    @classmethod
    def of_world(cls, world: World) -> RigidGroups:
        """
        :param world: The world to divide.
        :return: The world's rigid pieces, joined across its fixed connections.
        """
        fixed_neighbours: dict[int, list[KinematicStructureEntity]] = {
            id(entity): [] for entity in world.kinematic_structure_entities
        }
        moving_connection_of: dict[int, Connection] = {}
        for connection in world.connections:
            if isinstance(connection, FixedConnection):
                fixed_neighbours[id(connection.parent)].append(connection.child)
                fixed_neighbours[id(connection.child)].append(connection.parent)
            else:
                moving_connection_of[id(connection.child)] = connection

        groups, grouped = [], set()
        for entity in world.kinematic_structure_entities:
            if id(entity) in grouped:
                continue
            members = cls._joined_to(entity, fixed_neighbours)
            grouped.update(id(member) for member in members)
            moving_connections = [
                moving_connection_of[id(member)]
                for member in members
                if id(member) in moving_connection_of
            ]
            groups.append(
                RigidGroup(
                    entities=members,
                    parent_connection=(
                        moving_connections[0] if moving_connections else None
                    ),
                )
            )
        return cls(groups)

    @staticmethod
    def _joined_to(
        entity: KinematicStructureEntity,
        fixed_neighbours: dict[int, list[KinematicStructureEntity]],
    ) -> list[KinematicStructureEntity]:
        """
        :return: The entity and every entity fixed connections join it to.
        """
        members, frontier, seen = [], [entity], {id(entity)}
        while frontier:
            member = frontier.pop()
            members.append(member)
            for neighbour in fixed_neighbours[id(member)]:
                if id(neighbour) not in seen:
                    seen.add(id(neighbour))
                    frontier.append(neighbour)
        return members

    @cached_property
    def _group_by_entity(self) -> dict[int, RigidGroup]:
        """
        :return: The groups, keyed by the identity of each of their entities.
        """
        return {id(entity): group for group in self.groups for entity in group.entities}

    def group_of(self, entity: KinematicStructureEntity) -> RigidGroup:
        """
        :return: The rigid piece the entity belongs to.
        """
        return self._group_by_entity[id(entity)]


# %% motion of a joint


@dataclass
class JointMotion:
    """
    The motion a one-degree-of-freedom joint allows, in the root frame of the ground
    truth world.
    """

    connection: ActiveConnection1DOF
    """
    The joint.
    """

    ground_truth_root_V_axis: Vector3
    """
    The unit direction the joint turns about or slides along.
    """

    ground_truth_root_P_axis: Point3
    """
    A point on the line the joint turns about.
    """

    lower_travel: float | None
    """
    How far the joint can move against its axis from where it stands now, as a negative
    number in its unit (radians or metres); ``None`` when unlimited.
    """

    upper_travel: float | None
    """
    How far the joint can move along its axis from where it stands now, in its unit;
    ``None`` when unlimited.
    """

    @classmethod
    def of(
        cls,
        connection: ActiveConnection1DOF,
        world: World,
        ground_truth_root_T_world_root: HomogeneousTransformationMatrix,
    ) -> JointMotion:
        """
        :param connection: The joint.
        :param world: The world the joint belongs to.
        :param ground_truth_root_T_world_root: The pose of that world's root in the
            ground truth world's root frame.
        :return: The motion the joint allows from the world's current state.
        """
        ground_truth_root_T_connection = (
            ground_truth_root_T_world_root
            @ world.compute_forward_kinematics(world.root, connection.parent)
            @ connection.parent_T_connection_expression
        ).to_np()
        direction = ground_truth_root_T_connection[:3, :3] @ connection.axis.to_np()[:3]
        limits, position = connection.dof.limits, connection.position
        frame = ground_truth_root_T_world_root.reference_frame
        return cls(
            connection=connection,
            ground_truth_root_V_axis=Vector3.from_iterable(
                direction / np.linalg.norm(direction), reference_frame=frame
            ),
            ground_truth_root_P_axis=Point3.from_iterable(
                ground_truth_root_T_connection[:3, 3], reference_frame=frame
            ),
            lower_travel=cls._travel_to(limits.lower.position, position),
            upper_travel=cls._travel_to(limits.upper.position, position),
        )

    @staticmethod
    def _travel_to(limit: float | None, position: float) -> float | None:
        """
        :return: How far the joint is from the limit, or ``None`` for no limit.
        """
        if limit is None or not np.isfinite(limit):
            return None
        return limit - position

    @property
    def direction(self) -> npt.NDArray[np.float64]:
        """
        :return: :attr:`ground_truth_root_V_axis` as three numbers.
        """
        return self.ground_truth_root_V_axis.to_np()[:3]

    @property
    def point(self) -> npt.NDArray[np.float64]:
        """
        :return: :attr:`ground_truth_root_P_axis` as three numbers.
        """
        return self.ground_truth_root_P_axis.to_np()[:3]

    def flipped(self) -> JointMotion:
        """
        :return: The same motion, written with the opposite axis: moving along the
            flipped axis is moving against this one.
        """
        return JointMotion(
            connection=self.connection,
            ground_truth_root_V_axis=Vector3.from_iterable(
                -self.direction,
                reference_frame=self.ground_truth_root_V_axis.reference_frame,
            ),
            ground_truth_root_P_axis=self.ground_truth_root_P_axis,
            lower_travel=None if self.upper_travel is None else -self.upper_travel,
            upper_travel=None if self.lower_travel is None else -self.lower_travel,
        )

    def angle_to(self, other: JointMotion) -> float:
        """
        :return: The angle between the two axes, in radians.
        """
        return float(np.arccos(np.clip(self.direction @ other.direction, -1.0, 1.0)))

    def axis_distance_near(
        self, other: JointMotion, ground_truth_root_P_place: Point3
    ) -> float:
        """
        :param other: The joint whose axis line is measured to.
        :param ground_truth_root_P_place: Where along this axis to measure: at the point
            of the axis nearest to this place.
        :return: The distance from that point of this axis line to the other's, in
            metres.
        """
        place = ground_truth_root_P_place.to_np()[:3]
        on_this_axis = self.point + self.direction * (
            (place - self.point) @ self.direction
        )
        offset = on_this_axis - other.point
        return float(
            np.linalg.norm(offset - other.direction * (offset @ other.direction))
        )

    def travel_error_to(self, other: JointMotion) -> float | None:
        """
        :return: The larger of the differences between the two joints' travel limits,
            in their unit; ``None`` when either joint is unlimited.
        """
        travels = [
            self.lower_travel,
            self.upper_travel,
            other.lower_travel,
            other.upper_travel,
        ]
        if any(travel is None for travel in travels):
            return None
        return max(
            abs(self.lower_travel - other.lower_travel),
            abs(self.upper_travel - other.upper_travel),
        )


# %% scores


@dataclass
class GroupMatch:
    """
    A rigid piece of the ground truth world paired with the rigid piece of the
    reconstructed world that stands for it.
    """

    ground_truth_group: RigidGroup
    """
    The piece of the ground truth world.
    """

    reconstructed_group: RigidGroup
    """
    The piece of the reconstructed world.
    """

    shared_samples: int
    """
    How many surface samples of the reconstructed piece lie on bodies of the ground
    truth piece that its bodies were matched to.
    """


@dataclass
class JointScore:
    """
    How well the joint of a reconstructed piece moves like the joint of the ground truth
    piece it stands for.
    """

    ground_truth_connection: Connection
    """
    The joint of the ground truth piece.
    """

    reconstructed_connection: Connection
    """
    The joint of the reconstructed piece.
    """

    same_type: bool
    """
    Whether both joints are of the same kind, such as both revolute.
    """

    axis_angle: float | None
    """
    The angle between the two axes, in radians, after writing both with axes pointing
    the same way; ``None`` unless both are one-degree-of-freedom joints of the same
    type.
    """

    axis_distance: float | None
    """
    How far the reconstructed axis line lies from the ground truth one, in metres,
    measured where the ground truth axis passes nearest the ground truth piece; ``None``
    unless both joints turn.
    """

    travel_error: float | None
    """
    The larger of the differences in how far the joints can move each way from where
    they stand, in their unit; ``None`` unless both are limited joints of the same type.
    """


@dataclass
class JointEvaluation:
    """
    The joint scores of a correspondence, and the joints that could not be paired.
    """

    correspondence: BodyCorrespondence
    """
    The correspondence the rigid pieces were paired from.
    """

    group_matches: list[GroupMatch] = field(default_factory=list)
    """
    The pairs of rigid pieces.
    """

    joint_scores: list[JointScore] = field(default_factory=list)
    """
    One score per pair of pieces that both move.
    """

    welded_joints: list[Connection] = field(default_factory=list)
    """
    The ground truth joints whose moving piece the reconstruction fused with another
    piece, so that it cannot move.
    """

    unmatched_ground_truth_joints: list[Connection] = field(default_factory=list)
    """
    The ground truth joints whose moving piece the reconstruction missed altogether.
    """

    extra_joints: list[Connection] = field(default_factory=list)
    """
    The reconstructed joints that stand for no ground truth joint.
    """

    @property
    def type_accuracy(self) -> float | None:
        """
        :return: The share of scored joint pairs whose types agree, or ``None`` when no
            pair was scored.
        """
        if not self.joint_scores:
            return None
        return float(np.mean([score.same_type for score in self.joint_scores]))


# %% scoring


@dataclass
class JointScorer:
    """
    Pairs the rigid pieces of two worlds through their matched bodies, and scores how
    well the joints of paired pieces agree.

    .. note:: A screw joint is compared like a revolute one; its pitch is not compared.
    """

    def score(self, correspondence: BodyCorrespondence) -> JointEvaluation:
        """
        :param correspondence: The body correspondence of the two worlds.
        :return: The joint scores, and the joints that were welded, missed or added.
        """
        ground_truth_groups = RigidGroups.of_world(correspondence.ground_truth_world)
        reconstructed_groups = RigidGroups.of_world(correspondence.reconstructed_world)
        group_matches = self._group_matches_in(
            correspondence, ground_truth_groups, reconstructed_groups
        )
        evaluation = JointEvaluation(
            correspondence=correspondence, group_matches=group_matches
        )
        for group_match in group_matches:
            self._add_pair(evaluation, group_match)
        self._add_unpaired(evaluation, ground_truth_groups, reconstructed_groups)
        return evaluation

    def _group_matches_in(
        self,
        correspondence: BodyCorrespondence,
        ground_truth_groups: RigidGroups,
        reconstructed_groups: RigidGroups,
    ) -> list[GroupMatch]:
        """
        :return: The one-to-one pairs of pieces that share the most matched surface.
        """
        shared = correspondence.matched_samples_between(
            [group.bodies for group in ground_truth_groups.groups],
            [group.bodies for group in reconstructed_groups.groups],
        )
        return [
            GroupMatch(
                ground_truth_group=ground_truth_groups.groups[column],
                reconstructed_group=reconstructed_groups.groups[row],
                shared_samples=int(round(shared[row, column])),
            )
            for row, column in pairs_sharing_the_most(shared)
        ]

    def _add_pair(self, evaluation: JointEvaluation, group_match: GroupMatch) -> None:
        """
        Add what a pair of pieces says about their joints: a score when both move, a
        welded joint when only the ground truth one does, an extra joint when only the
        reconstructed one does.
        """
        ground_truth_connection = group_match.ground_truth_group.parent_connection
        reconstructed_connection = group_match.reconstructed_group.parent_connection
        if ground_truth_connection is None and reconstructed_connection is None:
            return
        if reconstructed_connection is None:
            evaluation.welded_joints.append(ground_truth_connection)
            return
        if ground_truth_connection is None:
            evaluation.extra_joints.append(reconstructed_connection)
            return
        evaluation.joint_scores.append(
            self._score_of(
                ground_truth_connection,
                reconstructed_connection,
                group_match.ground_truth_group,
                evaluation.correspondence,
            )
        )

    def _add_unpaired(
        self,
        evaluation: JointEvaluation,
        ground_truth_groups: RigidGroups,
        reconstructed_groups: RigidGroups,
    ) -> None:
        """
        Add the joints of moving pieces that were not paired: a ground truth piece whose
        bodies the reconstruction has, merged into another piece, is welded; one whose
        bodies it lacks is missed; an unpaired reconstructed piece is extra.
        """
        paired_ground_truth = {
            id(group_match.ground_truth_group)
            for group_match in evaluation.group_matches
        }
        paired_reconstructed = {
            id(group_match.reconstructed_group)
            for group_match in evaluation.group_matches
        }
        reconstructed_somewhere = self._ground_truth_bodies_reconstructed(
            evaluation.correspondence
        )
        for group in ground_truth_groups.groups:
            if group.parent_connection is None or id(group) in paired_ground_truth:
                continue
            if any(id(body) in reconstructed_somewhere for body in group.bodies):
                evaluation.welded_joints.append(group.parent_connection)
            else:
                evaluation.unmatched_ground_truth_joints.append(group.parent_connection)
        evaluation.extra_joints.extend(
            group.parent_connection
            for group in reconstructed_groups.groups
            if group.parent_connection is not None
            and id(group) not in paired_reconstructed
        )

    @staticmethod
    def _ground_truth_bodies_reconstructed(
        correspondence: BodyCorrespondence,
    ) -> set[int]:
        """
        :return: The identities of the ground truth bodies some reconstructed body
            stands for, whole or as part of a merged body.
        """
        return {id(match.ground_truth_body) for match in correspondence.matches} | {
            id(part.ground_truth_body)
            for merged_body in correspondence.merged_bodies
            for part in merged_body.parts
        }

    def _score_of(
        self,
        ground_truth_connection: Connection,
        reconstructed_connection: Connection,
        ground_truth_group: RigidGroup,
        correspondence: BodyCorrespondence,
    ) -> JointScore:
        """
        :return: How well the two joints of a pair of pieces agree.
        """
        same_type = type(ground_truth_connection) is type(reconstructed_connection)
        if not (
            same_type
            and self._moves_along_an_axis(ground_truth_connection)
            and self._moves_along_an_axis(reconstructed_connection)
        ):
            return JointScore(
                ground_truth_connection=ground_truth_connection,
                reconstructed_connection=reconstructed_connection,
                same_type=same_type,
                axis_angle=None,
                axis_distance=None,
                travel_error=None,
            )
        ground_truth_world = correspondence.ground_truth_world
        ground_truth_motion = JointMotion.of(
            ground_truth_connection,
            ground_truth_world,
            HomogeneousTransformationMatrix(
                reference_frame=ground_truth_world.root,
                child_frame=ground_truth_world.root,
            ),
        )
        reconstructed_motion = JointMotion.of(
            reconstructed_connection,
            correspondence.reconstructed_world,
            correspondence.ground_truth_root_T_reconstructed_root,
        )
        if ground_truth_motion.direction @ reconstructed_motion.direction < 0:
            reconstructed_motion = reconstructed_motion.flipped()
        turns = isinstance(
            ground_truth_connection, (RevoluteConnection, ScrewConnection)
        )
        return JointScore(
            ground_truth_connection=ground_truth_connection,
            reconstructed_connection=reconstructed_connection,
            same_type=True,
            axis_angle=ground_truth_motion.angle_to(reconstructed_motion),
            axis_distance=(
                ground_truth_motion.axis_distance_near(
                    reconstructed_motion,
                    self._centre_of(ground_truth_group, correspondence),
                )
                if turns
                else None
            ),
            travel_error=ground_truth_motion.travel_error_to(reconstructed_motion),
        )

    @staticmethod
    def _moves_along_an_axis(connection: Connection) -> bool:
        """
        :return: Whether the connection is a one-degree-of-freedom joint with an axis to
            move about or along; a joint whose axis is zero cannot move at all.
        """
        return isinstance(connection, ActiveConnection1DOF) and bool(
            np.any(connection.axis.to_np()[:3])
        )

    @staticmethod
    def _centre_of(group: RigidGroup, correspondence: BodyCorrespondence) -> Point3:
        """
        :return: The centre of the surface samples of the ground truth piece.
        """
        surfaces = correspondence.overlap_table.ground_truth_surfaces
        samples = np.vstack(
            [
                surfaces.surface_of(body).ground_truth_root_P_samples
                for body in group.bodies
                if body.visual
            ]
        )
        return Point3.from_iterable(
            samples.mean(axis=0),
            reference_frame=correspondence.ground_truth_world.root,
        )
