from __future__ import annotations

# %% imports
import json
import math
import time
from enum import Enum
from pathlib import Path

import py_trees
from typing_extensions import TYPE_CHECKING, Any, Iterator

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

# %% belief entries


class EntityStatus(str, Enum):
    """
    Visibility of an entity in the belief state.
    """

    VISIBLE = "visible"
    PROBABLY_THERE = "probably_there"


class EntityType(str, Enum):
    """
    Kind of entity represented by a belief entry.
    """

    PERSON = "person"
    OBJECT = "object"


class BeliefEntry:
    """
    Latest known properties of an entity.
    """

    def __init__(self, entry_id: int, entity_type: EntityType, classname: str) -> None:
        self.id = entry_id
        """
        Identifier of the entry.
        """
        self.entity_type = entity_type
        """
        Kind of entity.
        """
        self.classname = classname
        """
        Class assigned to the entity.
        """
        self.yolo_label = ""
        """
        Object detector label.
        """
        self.semantic_labels = []
        """
        Semantic labels assigned to the entity.
        """
        self.activities = []
        """
        Activities observed for the entity.
        """
        self.position = [0.0, 0.0, 0.0]
        """
        Most recently observed position.
        """
        self.orientation = [0.0, 0.0, 0.0, 1.0]
        """
        Most recently observed orientation.
        """
        self.dimensions = [0.0, 0.0, 0.0]
        """
        Most recently observed dimensions.
        """
        self.last_seen = 0.0
        """
        Time of the latest observation.
        """
        self.confidence = 0.0
        """
        Confidence of the detector label.
        """
        self.status = EntityStatus.VISIBLE
        """
        Current visibility status.
        """
        self.name = "anonymous"
        """
        Name assigned to a person.
        """

    def to_dict(self) -> dict[str, Any]:
        """
        Return the entry as a dictionary for the debug output.
        """
        base = {}
        base["id"] = self.id
        base["entity_type"] = self.entity_type.value
        base["position"] = {
            "x": self.position[0],
            "y": self.position[1],
            "z": self.position[2],
        }
        base["orientation"] = {
            "x": self.orientation[0],
            "y": self.orientation[1],
            "z": self.orientation[2],
            "w": self.orientation[3],
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


class BeliefState:
    """
    Collection of tracked entity beliefs.
    """

    def __init__(self) -> None:
        self.entries: dict[int, BeliefEntry] = {}
        """
        Entries indexed by their identifiers.
        """

    def __len__(self) -> int:
        """
        Return the number of entries.
        """
        return len(self.entries)

    def __iter__(self) -> Iterator[BeliefEntry]:
        """
        Iterate over the entries.
        """
        return iter(self.entries.values())

    def find_by_id(self, entry_id: int) -> BeliefEntry | None:
        """
        Return the entry with the identifier, if present.
        """
        return self.entries.get(entry_id)

    def find_by_classname(self, classname: str) -> list[BeliefEntry]:
        """
        Return entries with the given class.
        """
        result = []
        for entry in self.entries.values():
            if entry.classname == classname:
                result.append(entry)
        return result

    def find_by_yolo_label(self, label: str) -> list[BeliefEntry]:
        """
        Return entries with the given detector label.
        """
        result = []
        for entry in self.entries.values():
            if entry.yolo_label == label:
                result.append(entry)
        return result

    def find_by_semantic_label(self, label: str) -> list[BeliefEntry]:
        """
        Return entries carrying the semantic label.
        """
        result = []
        for entry in self.entries.values():
            if label in entry.semantic_labels:
                result.append(entry)
        return result

    def persons(self) -> list[BeliefEntry]:
        """
        Return entries representing people.
        """
        result = []
        for entry in self.entries.values():
            if entry.entity_type == EntityType.PERSON:
                result.append(entry)
        return result


def _distance(first_position: list[float], second_position: list[float]) -> float:
    """
    Return the Euclidean distance between positions.
    """
    squared_distance = 0.0
    for index in range(len(first_position)):
        difference = first_position[index] - second_position[index]
        squared_distance += difference * difference
    return math.sqrt(squared_distance)


# %% belief state annotator
class BeliefStateAnnotator(robokudo.annotators.core.BaseAnnotator):
    """
    Track observed entities and age entries that are no longer visible.
    """

    _state = BeliefState()
    """
    Belief state shared by annotator instances.
    """

    _next_id = 1
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
        debug_json_path: str = str(Path.home() / "Desktop" / "belief_state.json"),
    ) -> None:
        """
        :param match_distance: Maximum distance for matching observations.
        :param probably_there_after: Time before an unseen entry is marked probably there.
        :param person_delete_after: Time before an unseen person is removed.
        :param object_delete_after: Time before an unseen object is removed.
        :param debug_json_path: Destination for the debug output, or an empty string to disable it.
        """
        super().__init__(name=name)
        self.match_distance = match_distance
        """
        Maximum distance for matching observations.
        """
        self.probably_there_after = probably_there_after
        """
        Time before an unseen entry is marked probably there.
        """
        self.person_delete_after = person_delete_after
        """
        Time before an unseen person is removed.
        """
        self.object_delete_after = object_delete_after
        """
        Time before an unseen object is removed.
        """
        self.debug_json_path = debug_json_path
        """
        Destination for debug output.
        """

    def update(self) -> py_trees.common.Status:
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
        observation: dict[str, Any],
        entity_type: EntityType,
        now: float,
        hypothesis: robokudo.types.scene.HumanHypothesis
        | robokudo.types.scene.ObjectHypothesis
        | None = None,
    ) -> None:
        """
        Add or update an entry from an observation.
        """
        match = self._find_match(observation, entity_type)

        if match is None:
            entry = BeliefEntry(
                BeliefStateAnnotator._next_id, entity_type, observation["classname"]
            )
            entry.yolo_label = observation["yolo_label"]
            entry.semantic_labels = list(observation["semantic_labels"])
            entry.activities = list(observation["activities"])
            entry.position = observation["position"]
            entry.orientation = observation["orientation"]
            entry.dimensions = observation["dimensions"]
            entry.last_seen = now
            entry.confidence = observation["confidence"]
            entry.name = observation["name"]
            self._state.entries[entry.id] = entry
            BeliefStateAnnotator._next_id += 1
        else:
            entry = match
            entry.position = observation["position"]
            entry.orientation = observation["orientation"]
            entry.dimensions = observation["dimensions"]
            entry.last_seen = now
            entry.confidence = observation["confidence"]
            entry.status = EntityStatus.VISIBLE

            if observation["yolo_label"]:
                entry.yolo_label = observation["yolo_label"]

            for label in observation["semantic_labels"]:
                if label not in entry.semantic_labels:
                    entry.semantic_labels.append(label)

            entry.activities = list(observation["activities"])

            if observation["name"] != "anonymous":
                entry.name = observation["name"]

        if hypothesis is not None:
            hypothesis.annotations.append(entry)

    def _extract_observation(
        self,
        cas: CAS,
        hypothesis: robokudo.types.scene.HumanHypothesis
        | robokudo.types.scene.ObjectHypothesis,
        entity_type: EntityType,
    ) -> dict[str, Any] | None:
        """
        Return the observed entity properties if a position is available.
        """
        yolo_label = ""
        confidence = 0.0
        semantic_labels = []
        activities = []
        name = "anonymous"
        position = None
        orientation = [0.0, 0.0, 0.0, 1.0]
        dimensions = [0.0, 0.0, 0.0]

        for annotation in hypothesis.annotations:
            if isinstance(annotation, robokudo.types.human.ActivityAnnotation):
                if annotation.type and annotation.type not in activities:
                    activities.append(annotation.type)
                continue

            if isinstance(annotation, Classification):
                source = getattr(annotation, "source", "") or ""
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
                    position = list(world_pose.translation)
                    orientation = list(world_pose.rotation)
                except Exception:
                    position = list(annotation.translation)
                    orientation = list(annotation.rotation)
            elif isinstance(annotation, PositionAnnotation) and position is None:
                pose_like = PoseAnnotation()
                pose_like.translation = list(annotation.translation)
                pose_like.rotation = [0.0, 0.0, 0.0, 1.0]
                try:
                    world_pose = transform_pose_from_camera_to_world(cas, pose_like)
                    position = list(world_pose.translation)
                except Exception:
                    position = list(annotation.translation)

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

        observation = {}
        observation["classname"] = classname
        observation["yolo_label"] = yolo_label
        observation["semantic_labels"] = semantic_labels
        observation["activities"] = activities
        observation["confidence"] = confidence
        observation["name"] = name
        observation["position"] = position
        observation["orientation"] = orientation
        observation["dimensions"] = dimensions

        return observation

    def _find_match(
        self, observation: dict[str, Any], entity_type: EntityType
    ) -> BeliefEntry | None:
        """
        Return the nearest matching entry within the configured distance.
        """
        best = None
        best_distance = self.match_distance
        for entry in self._state.entries.values():
            if entry.entity_type != entity_type:
                continue
            if entry.classname != observation["classname"]:
                continue
            distance = _distance(entry.position, observation["position"])
            if distance <= best_distance:
                best_distance = distance
                best = entry
        return best

    def _age_entries(self, now: float) -> None:
        """
        Update or remove entries according to their last observation time.
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
