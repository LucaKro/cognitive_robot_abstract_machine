"""
Tests for remembering the persons and objects the robot has seen, and for forgetting
them once they were not seen for a while.

The clock of the belief state is stopped, so every test decides how much time passes.
"""

from __future__ import annotations

# %% Imports
import json
import time
from dataclasses import dataclass

import pytest
import robokudo.annotators.gpsr.belief_state
from py_trees.common import Status
from robokudo.annotators.gpsr.belief_state import (
    BeliefEntry,
    BeliefState,
    BeliefStateAnnotator,
    EntityStatus,
    EntityType,
)
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.annotators.zero_shot_classification_annotator import (
    ZeroShotClassificationAnnotator,
)
from robokudo.cas import CAS
from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    Classification,
    PoseAnnotation,
    PositionAnnotation,
)
from robokudo.types.human import ActivityAnnotation
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point3
from typing_extensions import TYPE_CHECKING

from ._data_files import GpsrFileName
from ._gpsr_scene import GpsrScene

if TYPE_CHECKING:
    from pathlib import Path

# %% coordinates


def coordinates_of(position: Point3) -> list[float]:
    """
    :param position: The position to read.
    :return: The x, y and z values of the position.
    """
    return position.to_np()[:3].tolist()


# %% stopped clock


@dataclass
class StoppedClock:
    """
    Stand-in for the time module that reports a time the test sets.
    """

    now: float = 1_000.0
    """
    The current time in seconds.
    """

    def time(self) -> float:
        """
        :return: The current time in seconds.
        """
        return self.now

    @staticmethod
    def strftime(time_format: str, moment: time.struct_time) -> str:
        """
        :param time_format: The format of the text.
        :param moment: The moment to format.
        :return: The moment formatted as the time module formats it.
        """
        return time.strftime(time_format, moment)

    @staticmethod
    def localtime(seconds: float) -> time.struct_time:
        """
        :param seconds: The seconds since the epoch.
        :return: The local time of the given seconds since the epoch.
        """
        return time.localtime(seconds)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> StoppedClock:
    """
    Stop the clock the belief state reads.

    :param monkeypatch: Replaces the clock.
    :return: The stopped clock.
    """
    stopped_clock = StoppedClock()
    monkeypatch.setattr(robokudo.annotators.gpsr.belief_state, "time", stopped_clock)
    return stopped_clock


@pytest.fixture
def debug_file(tmp_path: Path) -> Path:
    """
    :param tmp_path: Directory of the debug file.
    :return: Where the belief state annotator under test writes its debug JSON.
    """
    return tmp_path / GpsrFileName.BELIEF_STATE_DEBUG


@pytest.fixture
def remembering(
    scene: GpsrScene, clock: StoppedClock, debug_file: Path
) -> BeliefStateAnnotator:
    """
    :param scene: The scene of the annotator.
    :param clock: The stopped clock.
    :param debug_file: The path of the debug file.
    :return: A belief state annotator placed in the scene's pipeline.
    """
    annotator = BeliefStateAnnotator(debug_json_path=str(debug_file))
    scene.place_in_pipeline(annotator)
    return annotator


def yolo(classname: str, confidence: float = 0.8) -> Classification:
    """
    :param classname: The detected class.
    :param confidence: The confidence of the detector.
    :return: A classification as the object detector reports it.
    """
    return Classification(
        classname=classname, confidence=confidence, source=YoloAnnotator.__name__
    )


# %% belief state lookups


@pytest.fixture
def known_entries(belief_state: BeliefState) -> BeliefState:
    """
    :param belief_state: The empty belief state.
    :return: A belief state that knows a person, a cup and a bowl.
    """
    person = BeliefEntry(1, EntityType.PERSON, "human")
    cup = BeliefEntry(2, EntityType.OBJECT, "cup")
    cup.yolo_label = "cup_yolo"
    cup.semantic_labels = ["Mug"]
    bowl = BeliefEntry(3, EntityType.OBJECT, "bowl")
    for entry in (person, cup, bowl):
        belief_state.entries[entry.id] = entry
    return belief_state


def test_entry_is_found_by_its_id(known_entries: BeliefState) -> None:
    """
    Entry is found by its ID.
    """
    assert known_entries.find_by_id(2) is known_entries.entries[2]


def test_unknown_id_finds_nothing(known_entries: BeliefState) -> None:
    """
    Unknown ID finds nothing.
    """
    assert known_entries.find_by_id(99) is None


def test_entries_are_found_by_class(known_entries: BeliefState) -> None:
    """
    Entries are found by class.
    """
    assert known_entries.find_by_classname("bowl") == [known_entries.entries[3]]


def test_entries_are_found_by_detector_label(known_entries: BeliefState) -> None:
    """
    Entries are found by detector label.
    """
    assert known_entries.find_by_yolo_label("cup_yolo") == [known_entries.entries[2]]


def test_entries_are_found_by_semantic_label(known_entries: BeliefState) -> None:
    """
    Entries are found by semantic label.
    """
    assert known_entries.find_by_semantic_label("Mug") == [known_entries.entries[2]]


def test_persons_are_the_entries_of_type_person(known_entries: BeliefState) -> None:
    """
    Persons are the entries of type person.
    """
    assert known_entries.find_persons() == [known_entries.entries[1]]


def test_belief_state_counts_and_lists_its_entries(known_entries: BeliefState) -> None:
    """
    Belief state counts and lists its entries.
    """
    assert len(known_entries) == 3
    assert list(known_entries) == list(known_entries.entries.values())


# %% entries as dictionaries


def test_entry_reports_its_state_by_field() -> None:
    """
    The dictionary of an entry holds its fields, with plain numbers for coordinates.
    """
    person = BeliefEntry(1, EntityType.PERSON, "human")
    person.position = Point3.from_iterable([1.0, 2.0, 3.0])
    person.activities = ["waving"]

    reported = person.to_dict()

    expected = {
        "id": 1,
        "entity_type": EntityType.PERSON.value,
        "position": {"x": 1.0, "y": 2.0, "z": 3.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "dimensions": {"x": 0.0, "y": 0.0, "z": 0.0},
        "confidence": 0.0,
        "status": EntityStatus.VISIBLE.value,
        "activities": ["waving"],
    }
    assert {key: reported[key] for key in expected} == expected


def test_person_entry_reports_its_name_but_no_classes() -> None:
    """
    Person entry reports its name but no classes.
    """
    person = BeliefEntry(1, EntityType.PERSON, "human")

    reported = person.to_dict()

    assert reported["name"] == person.name
    assert [
        key for key in ("classname", "yolo_label", "semantic_labels") if key in reported
    ] == []


def test_object_entry_reports_its_classes_but_no_name() -> None:
    """
    Object entry reports its classes but no name.
    """
    cup = BeliefEntry(2, EntityType.OBJECT, "cup")
    cup.yolo_label = "cup"
    cup.semantic_labels = ["Mug"]

    reported = cup.to_dict()

    assert "name" not in reported
    assert reported["classname"] == "cup"
    assert reported["yolo_label"] == "cup"
    assert reported["semantic_labels"] == ["Mug"]


# %% new observations


def test_new_person_becomes_the_first_entry(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    New person becomes the first entry.
    """
    scene.add_human(position=[1.0, 0.0, 2.0])

    status = remembering.update()

    assert status == Status.SUCCESS
    entry = belief_state.find_by_id(1)
    assert entry.entity_type == EntityType.PERSON
    assert entry.classname == "human"
    assert coordinates_of(entry.position) == [1.0, 0.0, 2.0]


def test_new_object_is_remembered_by_its_detected_class(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    New object is remembered by its detected class.
    """
    scene.add_object(yolo("cup", confidence=0.8), position=[1.0, 0.0, 2.0])

    remembering.update()

    entry = belief_state.find_by_id(1)
    assert entry.entity_type == EntityType.OBJECT
    assert entry.classname == "cup"
    assert entry.yolo_label == "cup"
    assert entry.confidence == 0.8


def test_object_without_detected_class_is_remembered_as_unknown(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    Object without detected class is remembered as unknown.
    """
    scene.add_object(position=[1.0, 0.0, 2.0])

    remembering.update()

    assert belief_state.find_by_id(1).classname == "unknown"


def test_hypothesis_without_position_is_not_remembered(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    Hypothesis without position is not remembered.
    """
    scene.add_object(yolo("cup"))

    remembering.update()

    assert len(belief_state) == 0


def test_first_detected_class_names_the_entry(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    First detected class names the entry.
    """
    scene.add_object(
        yolo("cup", confidence=0.8),
        yolo("bowl", confidence=0.9),
        position=[1.0, 0.0, 2.0],
    )

    remembering.update()

    entry = belief_state.find_by_id(1)
    assert entry.classname == "cup"
    assert entry.confidence == 0.8


def test_zero_shot_class_becomes_a_semantic_label(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    A class from the zero-shot classifier is stored as a semantic label.
    """
    scene.add_object(
        yolo("cup"),
        Classification(
            classname="Mug", source=ZeroShotClassificationAnnotator.__name__
        ),
        position=[1.0, 0.0, 2.0],
    )

    remembering.update()

    entry = belief_state.find_by_id(1)
    assert entry.yolo_label == "cup"
    assert entry.semantic_labels == ["Mug"]


def test_zero_shot_class_seen_twice_is_one_semantic_label(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    The same zero-shot class seen twice is stored only once.
    """
    scene.add_object(
        yolo("cup"),
        Classification(
            classname="Mug", source=ZeroShotClassificationAnnotator.__name__
        ),
        Classification(
            classname="Mug", source=ZeroShotClassificationAnnotator.__name__
        ),
        position=[1.0, 0.0, 2.0],
    )

    remembering.update()

    assert belief_state.find_by_id(1).semantic_labels == ["Mug"]


def test_face_classification_names_the_person(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    Face classification names the person.
    """
    face = Classification(classname="Anna", classification_type="face")
    scene.add_human(face, position=[1.0, 0.0, 2.0])

    remembering.update()

    assert belief_state.find_by_id(1).name == face.classname


def test_face_classification_is_no_detected_class(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    A face classification names the person and is not taken as the detector label.
    """
    scene.add_human(
        Classification(classname="Anna", classification_type="face"),
        position=[1.0, 0.0, 2.0],
    )

    remembering.update()

    assert belief_state.find_by_id(1).yolo_label == ""


def test_face_seen_later_names_the_known_person(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    Face seen later names the known person.
    """
    scene.add_human(position=[1.0, 0.0, 2.0])
    remembering.update()
    scene.cas.annotations.clear()
    face = Classification(classname="Anna", classification_type="face")
    scene.add_human(face, position=[1.0, 0.0, 2.0])

    remembering.update()

    assert belief_state.find_by_id(1).name == face.classname


def test_person_seen_again_without_face_keeps_the_name(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    Person seen again without face keeps the name.
    """
    face = Classification(classname="Anna", classification_type="face")
    scene.add_human(face, position=[1.0, 0.0, 2.0])
    remembering.update()
    scene.cas.annotations.clear()
    scene.add_human(position=[1.0, 0.0, 2.0])

    remembering.update()

    assert belief_state.find_by_id(1).name == face.classname


def test_activities_are_remembered_once_each(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    Activities are remembered once each.
    """
    scene.add_human(
        ActivityAnnotation(type="waving"),
        ActivityAnnotation(type="waving"),
        ActivityAnnotation(type="sitting"),
        position=[1.0, 0.0, 2.0],
    )

    remembering.update()

    assert belief_state.find_by_id(1).activities == [
        "waving",
        "sitting",
    ]


def test_bounding_box_gives_the_dimensions(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    Bounding box gives the dimensions.
    """
    scene.add_object(
        yolo("cup"),
        BoundingBox3DAnnotation(x_length=0.1, y_length=0.2, z_length=0.3),
        position=[1.0, 0.0, 2.0],
    )

    remembering.update()

    assert belief_state.find_by_id(1).dimensions == [0.1, 0.2, 0.3]


def test_entry_is_attached_to_its_hypothesis(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    Entry is attached to its hypothesis.
    """
    cup = scene.add_object(yolo("cup"), position=[1.0, 0.0, 2.0])

    remembering.update()

    [entry] = CAS.filter_by_type(BeliefEntry, cup.annotations)
    assert entry is belief_state.find_by_id(1)


def test_cas_refers_to_the_belief_state(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    The annotator publishes the belief state in the CAS.
    """
    remembering.update()

    assert scene.cas.get("belief_state") is belief_state


def test_debug_file_holds_every_entry(
    scene: GpsrScene,
    remembering: BeliefStateAnnotator,
    debug_file: Path,
    belief_state: BeliefState,
) -> None:
    """
    Debug file holds every entry.
    """
    scene.add_object(yolo("cup"), position=[1.0, 0.0, 2.0])

    remembering.update()

    assert json.loads(debug_file.read_text()) == {
        "1": belief_state.find_by_id(1).to_dict()
    }


def test_debug_file_is_not_written_by_default(
    scene: GpsrScene, clock: StoppedClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Without a configured path, the annotator writes no debug file.
    """
    written_by = []
    monkeypatch.setattr(BeliefStateAnnotator, "_dump_debug_json", written_by.append)
    annotator = BeliefStateAnnotator()
    scene.place_in_pipeline(annotator)
    scene.add_object(yolo("cup"), position=[1.0, 0.0, 2.0])

    annotator.update()

    assert written_by == []


# %% positions in the world


def test_position_is_taken_from_the_camera_into_the_world(
    clock: StoppedClock, debug_file: Path, belief_state: BeliefState
) -> None:
    """
    Position is taken from the camera into the world.
    """
    scene = GpsrScene(
        world_T_camera=HomogeneousTransformationMatrix.from_xyz_rpy(x=3.0, y=1.0)
    )
    annotator = BeliefStateAnnotator(debug_json_path=str(debug_file))
    scene.place_in_pipeline(annotator)
    scene.add_object(yolo("cup"), position=[1.0, 0.0, 2.0])

    annotator.update()

    assert coordinates_of(belief_state.find_by_id(1).position) == pytest.approx(
        [4.0, 1.0, 2.0]
    )


@pytest.mark.parametrize(
    "pose_first", [True, False], ids=["pose first", "position first"]
)
def test_pose_is_taken_over_a_position(
    scene: GpsrScene,
    remembering: BeliefStateAnnotator,
    belief_state: BeliefState,
    pose_first: bool,
) -> None:
    """
    A pose wins over a position of the same hypothesis, in either order.
    """
    pose = PoseAnnotation(translation=[1.0, 0.0, 2.0], rotation=[0.0, 0.0, 1.0, 0.0])
    position = PositionAnnotation(translation=[5.0, 0.0, 2.0])
    scene.add_object(
        yolo("cup"), *((pose, position) if pose_first else (position, pose))
    )

    remembering.update()

    assert coordinates_of(belief_state.find_by_id(1).position) == pytest.approx(
        pose.translation
    )


def test_pose_gives_position_and_orientation(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    Pose gives position and orientation.
    """
    scene.add_object(
        yolo("cup"),
        PoseAnnotation(translation=[1.0, 0.0, 2.0], rotation=[0.0, 0.0, 1.0, 0.0]),
    )

    remembering.update()

    entry = belief_state.find_by_id(1)
    assert coordinates_of(entry.position) == pytest.approx([1.0, 0.0, 2.0])
    assert entry.orientation.to_np().tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0])


def test_without_camera_pose_the_camera_frame_position_is_kept(
    clock: StoppedClock, debug_file: Path, belief_state: BeliefState
) -> None:
    """
    Without camera pose the camera frame position is kept.
    """
    scene = GpsrScene(world_T_camera=None)
    annotator = BeliefStateAnnotator(debug_json_path=str(debug_file))
    scene.place_in_pipeline(annotator)
    scene.add_object(yolo("cup"), position=[1.0, 0.0, 2.0])

    annotator.update()

    assert coordinates_of(belief_state.find_by_id(1).position) == [1.0, 0.0, 2.0]


# %% observations of known things


def test_same_class_nearby_updates_the_known_entry(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    Same class nearby updates the known entry.
    """
    scene.add_object(yolo("cup"), position=[1.0, 0.0, 2.0])
    remembering.update()
    scene.cas.annotations.clear()
    scene.add_object(yolo("cup"), position=[1.2, 0.0, 2.0])

    remembering.update()

    assert len(belief_state) == 1
    assert coordinates_of(belief_state.find_by_id(1).position) == [1.2, 0.0, 2.0]


def test_same_class_exactly_at_match_distance_updates_the_known_entry(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    Same class exactly at match distance updates the known entry.
    """
    scene.add_object(yolo("cup"), position=[0.0, 0.0, 2.0])
    remembering.update()
    scene.cas.annotations.clear()
    scene.add_object(yolo("cup"), position=[remembering.match_distance, 0.0, 2.0])

    remembering.update()

    assert len(belief_state) == 1


def test_same_class_beyond_match_distance_is_a_new_entry(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    Same class beyond match distance is a new entry.
    """
    scene.add_object(yolo("cup"), position=[0.0, 0.0, 2.0])
    remembering.update()
    scene.cas.annotations.clear()
    scene.add_object(
        yolo("cup"), position=[remembering.match_distance + 0.01, 0.0, 2.0]
    )

    remembering.update()

    assert len(belief_state) == 2


def test_other_class_at_the_same_place_is_a_new_entry(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    Other class at the same place is a new entry.
    """
    scene.add_object(yolo("cup"), position=[1.0, 0.0, 2.0])
    scene.add_object(yolo("bowl"), position=[1.0, 0.0, 2.0])

    remembering.update()

    assert len(belief_state) == 2


def test_person_and_object_at_the_same_place_are_separate_entries(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    Person and object at the same place are separate entries.
    """
    scene.add_human(position=[1.0, 0.0, 2.0])
    scene.add_object(position=[1.0, 0.0, 2.0])

    remembering.update()

    assert len(belief_state) == 2


def test_nearer_of_two_known_entries_takes_the_observation(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    Of two known entries in range, the nearer one takes the observation.
    """
    scene.add_object(yolo("cup"), position=[0.0, 0.0, 2.0])
    scene.add_object(yolo("cup"), position=[0.5, 0.0, 2.0])
    remembering.update()
    scene.cas.annotations.clear()
    cup = scene.add_object(yolo("cup"), position=[0.28, 0.0, 2.0])

    remembering.update()

    [entry] = CAS.filter_by_type(BeliefEntry, cup.annotations)
    assert entry.id == 2
    assert coordinates_of(belief_state.find_by_id(1).position) == [0.0, 0.0, 2.0]


def test_update_adds_new_semantic_labels_and_replaces_activities(
    scene: GpsrScene, remembering: BeliefStateAnnotator, belief_state: BeliefState
) -> None:
    """
    Update adds new semantic labels and replaces activities.
    """
    scene.add_human(
        Classification(
            classname="Child", source=ZeroShotClassificationAnnotator.__name__
        ),
        ActivityAnnotation(type="waving"),
        position=[1.0, 0.0, 2.0],
    )
    remembering.update()
    scene.cas.annotations.clear()
    scene.add_human(
        Classification(
            classname="Guest", source=ZeroShotClassificationAnnotator.__name__
        ),
        ActivityAnnotation(type="sitting"),
        position=[1.0, 0.0, 2.0],
    )

    remembering.update()

    entry = belief_state.find_by_id(1)
    assert entry.semantic_labels == ["Child", "Guest"]
    assert entry.activities == ["sitting"]


# %% forgetting


def test_entry_seen_for_the_probably_there_time_is_still_visible(
    scene: GpsrScene,
    remembering: BeliefStateAnnotator,
    clock: StoppedClock,
    belief_state: BeliefState,
) -> None:
    """
    An entry unseen for exactly the probably-there time stays visible.
    """
    scene.add_object(yolo("cup"), position=[1.0, 0.0, 2.0])
    remembering.update()
    scene.cas.annotations.clear()
    clock.now += remembering.probably_there_after

    remembering.update()

    assert belief_state.find_by_id(1).status == EntityStatus.VISIBLE


def test_entry_not_seen_beyond_the_probably_there_time_is_probably_there(
    scene: GpsrScene,
    remembering: BeliefStateAnnotator,
    clock: StoppedClock,
    belief_state: BeliefState,
) -> None:
    """
    An entry unseen for longer than the probably-there time becomes probably there.
    """
    scene.add_object(yolo("cup"), position=[1.0, 0.0, 2.0])
    remembering.update()
    scene.cas.annotations.clear()
    clock.now += remembering.probably_there_after + 0.1

    remembering.update()

    assert belief_state.find_by_id(1).status == EntityStatus.PROBABLY_THERE


def test_person_not_seen_for_the_delete_time_is_kept(
    scene: GpsrScene,
    remembering: BeliefStateAnnotator,
    clock: StoppedClock,
    belief_state: BeliefState,
) -> None:
    """
    A person unseen for exactly the delete time stays in the belief state.
    """
    scene.add_human(position=[1.0, 0.0, 2.0])
    remembering.update()
    scene.cas.annotations.clear()
    clock.now += remembering.person_delete_after

    remembering.update()

    assert len(belief_state) == 1


def test_person_not_seen_beyond_the_delete_time_is_forgotten(
    scene: GpsrScene,
    remembering: BeliefStateAnnotator,
    clock: StoppedClock,
    belief_state: BeliefState,
) -> None:
    """
    A person unseen for longer than the delete time is removed.
    """
    scene.add_human(position=[1.0, 0.0, 2.0])
    remembering.update()
    scene.cas.annotations.clear()
    clock.now += remembering.person_delete_after + 0.1

    remembering.update()

    assert len(belief_state) == 0


def test_object_not_seen_for_the_delete_time_is_kept(
    scene: GpsrScene,
    remembering: BeliefStateAnnotator,
    clock: StoppedClock,
    belief_state: BeliefState,
) -> None:
    """
    An object unseen for exactly the delete time stays in the belief state.
    """
    scene.add_object(yolo("cup"), position=[1.0, 0.0, 2.0])
    remembering.update()
    scene.cas.annotations.clear()
    clock.now += remembering.object_delete_after

    remembering.update()

    assert len(belief_state) == 1


def test_object_not_seen_beyond_the_delete_time_is_forgotten(
    scene: GpsrScene,
    remembering: BeliefStateAnnotator,
    clock: StoppedClock,
    belief_state: BeliefState,
) -> None:
    """
    An object unseen for longer than the delete time is removed.
    """
    scene.add_object(yolo("cup"), position=[1.0, 0.0, 2.0])
    remembering.update()
    scene.cas.annotations.clear()
    clock.now += remembering.object_delete_after + 0.1

    remembering.update()

    assert len(belief_state) == 0


def test_object_is_kept_longer_than_a_person(
    scene: GpsrScene,
    remembering: BeliefStateAnnotator,
    clock: StoppedClock,
    belief_state: BeliefState,
) -> None:
    """
    Object is kept longer than a person.
    """
    scene.add_human(position=[1.0, 0.0, 2.0])
    scene.add_object(yolo("cup"), position=[3.0, 0.0, 2.0])
    remembering.update()
    scene.cas.annotations.clear()
    clock.now += remembering.person_delete_after + 0.1

    remembering.update()

    assert [entry.classname for entry in belief_state] == ["cup"]


def test_entry_seen_again_is_visible_again(
    scene: GpsrScene,
    remembering: BeliefStateAnnotator,
    clock: StoppedClock,
    belief_state: BeliefState,
) -> None:
    """
    Entry seen again is visible again.
    """
    scene.add_object(yolo("cup"), position=[1.0, 0.0, 2.0])
    remembering.update()
    scene.cas.annotations.clear()
    clock.now += remembering.probably_there_after + 0.1
    remembering.update()
    scene.add_object(yolo("cup"), position=[1.0, 0.0, 2.0])

    remembering.update()

    assert belief_state.find_by_id(1).status == EntityStatus.VISIBLE


def test_ids_of_forgotten_entries_are_not_given_out_again(
    scene: GpsrScene, remembering: BeliefStateAnnotator, clock: StoppedClock
) -> None:
    """
    IDs of removed entries are never given out again.
    """
    scene.add_human(position=[1.0, 0.0, 2.0])
    remembering.update()
    scene.cas.annotations.clear()
    clock.now += remembering.person_delete_after + 0.1
    remembering.update()
    human_hypothesis = scene.add_human(position=[1.0, 0.0, 2.0])

    remembering.update()

    [entry] = CAS.filter_by_type(BeliefEntry, human_hypothesis.annotations)
    assert entry.id == 2


def test_annotators_of_different_routes_give_out_different_ids(
    scene: GpsrScene,
    remembering: BeliefStateAnnotator,
    belief_state: BeliefState,
    debug_file: Path,
) -> None:
    """
    Annotators of different routes share one ID counter, so their entries never collide.
    """
    other_route = BeliefStateAnnotator(debug_json_path=str(debug_file))
    scene.add_human(position=[1.0, 0.0, 2.0])
    remembering.update()
    scene.place_in_pipeline(other_route)
    scene.add_object(yolo("cup"), position=[3.0, 0.0, 2.0])

    other_route.update()

    assert sorted(belief_state.entries) == [1, 2]
