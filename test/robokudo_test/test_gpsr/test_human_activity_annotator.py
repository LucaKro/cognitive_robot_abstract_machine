"""
Tests for annotating every person with the posture and the gesture CLIP scores highest.

CLIP is replaced by a stand-in that reports probabilities the test chooses.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest
from py_trees.common import Status
from robokudo.annotators.gpsr.human_activity_annotator import HumanActivityAnnotator
from robokudo.cas import CAS, CASViews
from robokudo.types.annotation import Encoding
from typing_extensions import TYPE_CHECKING

from ._clip_mock import ScoringModel, ScoringModelAndProcessor

if TYPE_CHECKING:
    from ._gpsr_scene import GpsrScene

# %% annotator under test


@pytest.fixture
def activity_annotator(
    scoring_clip: ScoringModelAndProcessor,
) -> HumanActivityAnnotator:
    """
    An activity annotator with its own default parameters, scoring with the stand-in.
    """
    return HumanActivityAnnotator(descriptor=HumanActivityAnnotator.Descriptor())


# %% posture


def test_highest_scored_posture_is_annotated(
    scene: GpsrScene,
    scoring_clip: ScoringModelAndProcessor,
    activity_annotator: HumanActivityAnnotator,
) -> None:
    scoring_clip.model.posture_probabilities = [0.1, 0.7, 0.1, 0.1]
    scene.place_in_pipeline(activity_annotator)
    person = scene.add_human()

    status = activity_annotator.compute()

    assert status == Status.SUCCESS
    assert (
        scene.activities_of(person)[0]
        == activity_annotator.parameters.posture_labels[1]
    )


# %% gesture


def test_gesture_above_the_threshold_is_annotated(
    scene: GpsrScene,
    scoring_clip: ScoringModelAndProcessor,
    activity_annotator: HumanActivityAnnotator,
) -> None:
    scoring_clip.model.gesture_probabilities = [0.8, 0.1, 0.05, 0.05]
    scene.place_in_pipeline(activity_annotator)
    person = scene.add_human()

    activity_annotator.compute()

    assert scene.activities_of(person)[1:] == [
        activity_annotator.parameters.gesture_labels[0]
    ]


def test_gesture_exactly_at_the_threshold_is_annotated(
    scene: GpsrScene,
    scoring_clip: ScoringModelAndProcessor,
    activity_annotator: HumanActivityAnnotator,
) -> None:
    probabilities = [0.65, 0.15, 0.1, 0.1]
    scoring_clip.model.gesture_probabilities = probabilities
    activity_annotator.parameters.gesture_threshold = float(
        ScoringModel.seen_probabilities(probabilities)[0]
    )
    scene.place_in_pipeline(activity_annotator)
    person = scene.add_human()

    activity_annotator.compute()

    assert scene.activities_of(person)[1:] == [
        activity_annotator.parameters.gesture_labels[0]
    ]


def test_no_gesture_above_the_threshold_is_annotated_as_no_gesture(
    scene: GpsrScene,
    scoring_clip: ScoringModelAndProcessor,
    activity_annotator: HumanActivityAnnotator,
) -> None:
    scoring_clip.model.gesture_probabilities = [0.3, 0.3, 0.2, 0.2]
    scene.place_in_pipeline(activity_annotator)
    person = scene.add_human()

    activity_annotator.compute()

    assert scene.activities_of(person)[1:] == ["No gesture"]


def test_highest_scored_no_gesture_is_annotated_once(
    scene: GpsrScene,
    scoring_clip: ScoringModelAndProcessor,
    activity_annotator: HumanActivityAnnotator,
) -> None:
    scoring_clip.model.gesture_probabilities = [0.05, 0.05, 0.1, 0.8]
    scene.place_in_pipeline(activity_annotator)
    person = scene.add_human()

    activity_annotator.compute()

    assert scene.activities_of(person)[1:] == ["No gesture"]


def test_only_the_best_gesture_is_annotated(
    scene: GpsrScene,
    scoring_clip: ScoringModelAndProcessor,
    activity_annotator: HumanActivityAnnotator,
) -> None:
    scoring_clip.model.gesture_probabilities = [0.35, 0.4, 0.15, 0.1]
    activity_annotator.parameters.gesture_threshold = 0.3
    scene.place_in_pipeline(activity_annotator)
    person = scene.add_human()

    activity_annotator.compute()

    assert scene.activities_of(person)[1:] == [
        activity_annotator.parameters.gesture_labels[1]
    ]


# %% embedding


def test_person_gets_the_normalized_image_embedding(
    scene: GpsrScene,
    scoring_clip: ScoringModelAndProcessor,
    activity_annotator: HumanActivityAnnotator,
) -> None:
    scoring_clip.model.image_embedding = [3.0, 4.0]
    scene.place_in_pipeline(activity_annotator)
    person = scene.add_human()

    activity_annotator.compute()

    encodings = CAS.filter_by_type(Encoding, person.annotations)
    assert len(encodings) == 1
    embedding = np.array(scoring_clip.model.image_embedding)
    np.testing.assert_allclose(
        encodings[0].encoding, embedding / np.linalg.norm(embedding)
    )


# %% persons to score


def test_every_person_is_annotated(
    scene: GpsrScene,
    scoring_clip: ScoringModelAndProcessor,
    activity_annotator: HumanActivityAnnotator,
) -> None:
    scene.place_in_pipeline(activity_annotator)
    first = scene.add_human()
    second = scene.add_human()

    activity_annotator.compute()

    parameters = activity_annotator.parameters
    best_posture = parameters.posture_labels[
        np.argmax(scoring_clip.model.posture_probabilities)
    ]
    best_gesture = parameters.gesture_labels[
        np.argmax(scoring_clip.model.gesture_probabilities)
    ]
    assert scene.activities_of(first) == [best_posture, best_gesture]
    assert scene.activities_of(second) == [best_posture, best_gesture]


def test_without_persons_nothing_is_scored(
    scene: GpsrScene,
    scoring_clip: ScoringModelAndProcessor,
    activity_annotator: HumanActivityAnnotator,
) -> None:
    scene.place_in_pipeline(activity_annotator)

    status = activity_annotator.compute()

    assert status == Status.SUCCESS
    assert scoring_clip.model.call_count == 0


# %% person crops


def test_posture_is_scored_on_the_whole_person(
    scene: GpsrScene,
    scoring_clip: ScoringModelAndProcessor,
    activity_annotator: HumanActivityAnnotator,
) -> None:
    scene.place_in_pipeline(activity_annotator)
    person = scene.add_human()

    activity_annotator.compute()

    whole_person = scoring_clip.processor.received_images[0][0]
    assert whole_person.size == (person.roi.roi.width, person.roi.roi.height)


def test_gesture_is_scored_on_the_widened_upper_body(
    scene: GpsrScene,
    scoring_clip: ScoringModelAndProcessor,
    activity_annotator: HumanActivityAnnotator,
) -> None:
    scene.place_in_pipeline(activity_annotator)
    person = scene.add_human()
    person.roi.roi.pos.x = 100

    activity_annotator.compute()

    upper_body = scoring_clip.processor.received_images[1][0]
    side_padding = int(person.roi.roi.width * 0.2)
    assert upper_body.size == (
        person.roi.roi.width + 2 * side_padding,
        int(person.roi.roi.height * 0.6),
    )


def test_upper_body_crop_ends_at_the_image_border(
    scene: GpsrScene,
    scoring_clip: ScoringModelAndProcessor,
    activity_annotator: HumanActivityAnnotator,
) -> None:
    scene.place_in_pipeline(activity_annotator)
    person = scene.add_human()
    person.roi.roi.pos.x = 0

    activity_annotator.compute()

    upper_body = scoring_clip.processor.received_images[1][0]
    side_padding = int(person.roi.roi.width * 0.2)
    assert upper_body.size[0] == person.roi.roi.width + side_padding


def test_crops_are_handed_to_clip_in_rgb_order(
    scene: GpsrScene,
    scoring_clip: ScoringModelAndProcessor,
    activity_annotator: HumanActivityAnnotator,
) -> None:
    cas = scene.place_in_pipeline(activity_annotator)
    blue_green_red = np.zeros_like(cas.get(CASViews.COLOR_IMAGE))
    blue_green_red[:, :] = [0, 0, 255]
    cas.set(CASViews.COLOR_IMAGE, blue_green_red)
    scene.add_human()

    activity_annotator.compute()

    whole_person = scoring_clip.processor.received_images[0][0]
    assert whole_person.getpixel((0, 0)) == (255, 0, 0)
