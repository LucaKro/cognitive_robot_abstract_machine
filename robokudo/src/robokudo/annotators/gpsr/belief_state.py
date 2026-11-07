"""
Memory of the persons and objects seen, with their latest known properties.
"""

from __future__ import annotations

# %% Imports
import json
import time
from dataclasses import dataclass
from enum import StrEnum

import py_trees
from krrood_spatial_math.spatial_types import Point3, Quaternion
from typing_extensions import TYPE_CHECKING, Any, ClassVar, Iterator

import robokudo.annotators.core
import robokudo.types.human
import robokudo.types.scene
from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    Classification,
    PoseAnnotation,
    PositionAnnotation,
)
from robokudo.utils.annotator_helper import transform_pose_from_camera_to_world

if TYPE_CHECKING:
    from robokudo.cas import CAS

# %% observations


@dataclass
class BeliefObservation:
    """
    Properties observed for an entity at one time.
    """

    classname: str
    """Class assigned to the entity."""
    yolo_label: str
    """
    Object detector label.
    """

    semantic_labels: list[str]
    """Semantic labels observed for the entity."""
    activities: list[str]
    """
    Activities observed for the entity.
    """

    confidence: float
    """Confidence of the detector label."""
    name: str
    """
    Name assigned to the person.
    """

    position: Point3
    """
    Observed position in the world frame, or in the camera frame without a transform.
    """
    orientation: Quaternion
    """
    Observed orientation, in the same frame as :attr:`position`.
    """

    dimensions: list[float]
    """Observed extent along x, y and z, in meters."""


# %% belief entries


class EntityStatus(StrEnum):
    """
    Visibility of an entity in the belief state.
    """

    VISIBLE = "visible"
    """
    The entity was seen recently.
    """
    PROBABLY_THERE = "probably_there"
    """
    The entity was not seen for a while and is still remembered.
    """


class EntityType(StrEnum):
    """
    Kind of entity represented by a belief entry.
    """

    PERSON = "person"
    """
    A detected person.
    """
    OBJECT = "object"
    """
    A detected object.
    """


@dataclass(init=False, eq=False, repr=False)
class BeliefEntry:
    """
    Latest known properties of an entity.
    """

    id: int
    """Identifier of the entry."""
    entity_type: EntityType
    """Kind of entity."""
    classname: str
    """Class assigned to the entity."""
    yolo_label: str
    """Object detector label."""
    semantic_labels: list[str]
    """Semantic labels assigned to the entity."""
    activities: list[str]
    """Activities observed for the entity."""
    position: Point3
    """Most recently observed position, see :attr:`BeliefObservation.position`."""
    orientation: Quaternion
    """Most recently observed orientation, see :attr:`BeliefObservation.orientation`."""
    dimensions: list[float]
    """Most recently observed extent along x, y and z, in meters."""
    last_seen: float
    """Time of the latest observation, in seconds since the epoch."""
    confidence: float
    """Confidence of the detector label."""
    status: EntityStatus
    """Current visibility status."""
    name: str
    """Name assigned to a person."""

    def __init__(self, entry_id: int, entity_type: EntityType, classname: str) -> None:
        """
        Create an entry for the identified entity.

        :param entry_id: Identifier of the entry.
        :param entity_type: Kind of entity.
        :param classname: Class assigned to the entity.
        """
        self.id = entry_id
        self.entity_type = entity_type
        self.classname = classname
        self.yolo_label = ""
        self.semantic_labels = []
        self.activities = []
        self.position = Point3()
        self.orientation = Quaternion()
        self.dimensions = [0.0, 0.0, 0.0]
        self.last_seen = 0.0
        self.confidence = 0.0
        self.status = EntityStatus.VISIBLE
        self.name = "anonymous"

    def to_dict(self) -> dict[str, Any]:
        """
        :return: The entry as a dictionary that can be written as JSON.
        """
        base = {}
        base["id"] = self.id
        base["entity_type"] = self.entity_type.value
        base["position"] = {
            "x": float(self.position.x),
            "y": float(self.position.y),
            "z": float(self.position.z),
        }
        base["orientation"] = {
            "x": float(self.orientation.x),
            "y": float(self.orientation.y),
            "z": float(self.orientation.z),
            "w": float(self.orientation.w),
        }
        base["dimensions"] = {
            "x": self.dimensions[0],
            "y": self.dimensions[1],
            "z": self.dimensions[2],
        }
        base["last_seen"] = time.strftime("%H:%M:%S", time.localtime(self.last_seen))
        base["confidence"] = self.confidence
        base["status"] = self.status.value
        base["activities"] = list(self.activities)

        if self.entity_type == EntityType.PERSON:
            base["name"] = self.name
        else:
            base["classname"] = self.classname
            base["yolo_label"] = self.yolo_label
            base["semantic_labels"] = list(self.semantic_labels)

        return base


@dataclass(init=False, eq=False, repr=False)
class BeliefState:
    """
    Collection of tracked entity beliefs.
    """

    entries: dict[int, BeliefEntry]
    """Entries indexed by their identifiers."""

    def __init__(self) -> None:
        """Create an empty collection of beliefs."""
        self.entries = {}

    def __len__(self) -> int:
        """
        :return: The number of entries.
        """
        return len(self.entries)

    def __iter__(self) -> Iterator[BeliefEntry]:
        """
        :return: An iterator over the entries.
        """
        return iter(self.entries.values())

    def find_by_id(self, entry_id: int) -> BeliefEntry | None:
        """
        :param entry_id: Identifier of the entry.
        :return: The entry with the identifier, if present.
        """
        return self.entries.get(entry_id)

    def find_by_classname(self, classname: str) -> list[BeliefEntry]:
        """
        :param classname: Class assigned to the entries.
        :return: The entries with the given class.
        """
        result = []
        for entry in self.entries.values():
            if entry.classname == classname:
                result.append(entry)
        return result

    def find_by_yolo_label(self, label: str) -> list[BeliefEntry]:
        """
        :param label: Object detector label of the entries.
        :return: The entries with the given detector label.
        """
        result = []
        for entry in self.entries.values():
            if entry.yolo_label == label:
                result.append(entry)
        return result

    def find_by_semantic_label(self, label: str) -> list[BeliefEntry]:
        """
        :param label: Semantic label of the entries.
        :return: The entries carrying the semantic label.
        """
        result = []
        for entry in self.entries.values():
            if label in entry.semantic_labels:
                result.append(entry)
        return result

    def find_persons(self) -> list[BeliefEntry]:
        """
        :return: The entries representing persons.
        """
        result = []
        for entry in self.entries.values():
            if entry.entity_type == EntityType.PERSON:
                result.append(entry)
        return result


# %% belief state annotator
class BeliefStateAnnotator(robokudo.annotators.core.BaseAnnotator):
    """
    Tracks observed entities and ages entries that are no longer visible.
    """

    _state: ClassVar[BeliefState] = BeliefState()
    """
    Belief state shared by annotator instances.
    """

    _next_id: ClassVar[int] = 1
    """
    Identifier assigned to the next new entry, shared by all annotator instances like
    the belief state.
    """

    def __init__(
        self,
        name: str = "BeliefStateAnnotator",
        match_distance: float = 0.3,
        probably_there_after: float = 60.0,
        person_delete_after: float = 120.0,
        object_delete_after: float = 300.0,
        debug_json_path: str = "",
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        :param match_distance: Maximum distance for matching observations, in meters.
        :param probably_there_after: Time before an unseen entry is marked probably there, in seconds.
        :param person_delete_after: Time before an unseen person is removed, in seconds.
        :param object_delete_after: Time before an unseen object is removed, in seconds.
        :param debug_json_path: Destination for the debug output, empty to disable it.
        """
        super().__init__(name=name)
        self.match_distance = match_distance
        """
        Maximum distance for matching observations, in meters.
        """
        self.probably_there_after = probably_there_after
        """
        Time before an unseen entry is marked probably there, in seconds.
        """
        self.person_delete_after = person_delete_after
        """
        Time before an unseen person is removed, in seconds.
        """
        self.object_delete_after = object_delete_after
        """
        Time before an unseen object is removed, in seconds.
        """
        self.debug_json_path = debug_json_path
        """
        Destination for the debug output, empty to disable it.
        """

    def update(self) -> py_trees.common.Status:
        """
        Overrides :meth:`BaseAnnotator.update` to integrate the observed entities and
        publish the belief state in the CAS.

        :return: :attr:`Status.SUCCESS`.
        """
        now = time.time()
        cas = self.get_cas()

        for hypothesis in list(cas.annotations):
            if isinstance(hypothesis, robokudo.types.scene.HumanHypothesis):
                entity_type = EntityType.PERSON
            elif isinstance(hypothesis, robokudo.types.scene.ObjectHypothesis):
                entity_type = EntityType.OBJECT
            else:
                continue

            observation = self._extract_observation(cas, hypothesis, entity_type)
            if observation is None:
                continue

            self._integrate(observation, entity_type, now, hypothesis=hypothesis)

        self._age_entries(now)

        cas.set_ref("belief_state", self._state)

        if self.debug_json_path:
            self._dump_debug_json()

        self.feedback_message = f"Belief state: {len(self._state)} entries"
        return py_trees.common.Status.SUCCESS

    def _integrate(
        self,
        observation: BeliefObservation,
        entity_type: EntityType,
        now: float,
        hypothesis: robokudo.types.scene.HumanHypothesis
        | robokudo.types.scene.ObjectHypothesis
        | None = None,
    ) -> None:
        """
        Add or update an entry from an observation.

        :param observation: Properties observed for the entity.
        :param entity_type: Kind of entity.
        :param now: Time of the observation, in seconds since the epoch.
        :param hypothesis: Hypothesis that receives the entry, if given.
        """
        match = self._find_match(observation, entity_type)

        if match is None:
            entry = BeliefEntry(
                BeliefStateAnnotator._next_id, entity_type, observation.classname
            )
            entry.yolo_label = observation.yolo_label
            entry.semantic_labels = list(observation.semantic_labels)
            entry.activities = list(observation.activities)
            entry.position = observation.position
            entry.orientation = observation.orientation
            entry.dimensions = observation.dimensions
            entry.last_seen = now
            entry.confidence = observation.confidence
            entry.name = observation.name
            self._state.entries[entry.id] = entry
            BeliefStateAnnotator._next_id += 1
        else:
            entry = match
            entry.position = observation.position
            entry.orientation = observation.orientation
            entry.dimensions = observation.dimensions
            entry.last_seen = now
            entry.confidence = observation.confidence
            entry.status = EntityStatus.VISIBLE

            if observation.yolo_label:
                entry.yolo_label = observation.yolo_label

            for label in observation.semantic_labels:
                if label not in entry.semantic_labels:
                    entry.semantic_labels.append(label)

            entry.activities = list(observation.activities)

            if observation.name != "anonymous":
                entry.name = observation.name

        if hypothesis is not None:
            hypothesis.annotations.append(entry)

    def _extract_observation(
        self,
        cas: CAS,
        hypothesis: robokudo.types.scene.HumanHypothesis
        | robokudo.types.scene.ObjectHypothesis,
        entity_type: EntityType,
    ) -> BeliefObservation | None:
        """
        :param cas: The CAS holding the camera pose.
        :param hypothesis: Hypothesis to read the properties of.
        :param entity_type: Kind of entity.
        :return: The observed entity properties if a position is available.
        """
        yolo_label = ""
        confidence = 0.0
        semantic_labels = []
        activities = []
        name = "anonymous"
        position = None
        orientation = Quaternion()
        dimensions = [0.0, 0.0, 0.0]

        for annotation in hypothesis.annotations:
            if isinstance(annotation, robokudo.types.human.ActivityAnnotation):
                if annotation.type and annotation.type not in activities:
                    activities.append(annotation.type)
                continue

            if isinstance(annotation, Classification):
                source = annotation.source or ""
                if annotation.classification_type == "face":
                    name = annotation.classname
                elif "ZeroShot" in source:
                    if (
                        annotation.classname
                        and annotation.classname not in semantic_labels
                    ):
                        semantic_labels.append(annotation.classname)
                else:
                    if not yolo_label:
                        yolo_label = annotation.classname
                        confidence = float(annotation.confidence)

            if isinstance(annotation, PoseAnnotation):
                try:
                    world_pose = transform_pose_from_camera_to_world(cas, annotation)
                    position = Point3.from_iterable(world_pose.translation)
                    orientation = Quaternion.from_iterable(world_pose.rotation)
                except Exception:
                    position = Point3.from_iterable(annotation.translation)
                    orientation = Quaternion.from_iterable(annotation.rotation)
            elif isinstance(annotation, PositionAnnotation) and position is None:
                pose_like = PoseAnnotation()
                pose_like.translation = list(annotation.translation)
                pose_like.rotation = [0.0, 0.0, 0.0, 1.0]
                try:
                    world_pose = transform_pose_from_camera_to_world(cas, pose_like)
                    position = Point3.from_iterable(world_pose.translation)
                except Exception:
                    position = Point3.from_iterable(annotation.translation)

            if isinstance(annotation, BoundingBox3DAnnotation):
                dimensions = [
                    annotation.x_length,
                    annotation.y_length,
                    annotation.z_length,
                ]

        if position is None:
            return None

        if yolo_label:
            classname = yolo_label
        elif entity_type == EntityType.PERSON:
            classname = "human"
        else:
            classname = "unknown"

        return BeliefObservation(
            classname=classname,
            yolo_label=yolo_label,
            semantic_labels=semantic_labels,
            activities=activities,
            confidence=confidence,
            name=name,
            position=position,
            orientation=orientation,
            dimensions=dimensions,
        )

    def _find_match(
        self, observation: BeliefObservation, entity_type: EntityType
    ) -> BeliefEntry | None:
        """
        :param observation: Properties observed for the entity.
        :param entity_type: Kind of entity.
        :return: The nearest matching entry within the configured distance, if any.
        """
        best = None
        best_distance = self.match_distance
        for entry in self._state.entries.values():
            if entry.entity_type != entity_type:
                continue
            if entry.classname != observation.classname:
                continue
            distance = (
                entry.position.euclidean_distance(observation.position).to_np().item()
            )
            if distance <= best_distance:
                best_distance = distance
                best = entry
        return best

    def _age_entries(self, now: float) -> None:
        """
        Update or remove entries according to their last observation time.

        :param now: Current time, in seconds since the epoch.
        """
        entries_to_delete = []
        for entry_id, entry in self._state.entries.items():
            age = now - entry.last_seen
            if (
                entry.entity_type == EntityType.PERSON
                and age > self.person_delete_after
            ):
                entries_to_delete.append(entry_id)
            elif (
                entry.entity_type == EntityType.OBJECT
                and age > self.object_delete_after
            ):
                entries_to_delete.append(entry_id)
            elif age > self.probably_there_after:
                entry.status = EntityStatus.PROBABLY_THERE

        for entry_id in entries_to_delete:
            del self._state.entries[entry_id]

    def _dump_debug_json(self) -> None:
        """
        Write the current entries to the debug file.
        """
        try:
            payload = {}
            for entry in self._state.entries.values():
                payload[str(entry.id)] = entry.to_dict()
            with open(self.debug_json_path, "w") as debug_file:
                json.dump(payload, debug_file, indent=2)
        except Exception as exception:
            self.rk_logger.warning(
                f"Could not write belief state debug JSON: {exception}"
            )
