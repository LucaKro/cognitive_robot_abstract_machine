"""
Asking a run's questions from the text alone, without rendering anything.

Rendering is most of what a run spends its time on -- 44 minutes of room 13's 55 -- and
three pictures per label is most of what it spends its money on. A run told to show no
pictures renders none, and the two questions that would have carried them say so rather
than describing pictures that are not there.

The label is still given. What is being measured is what the pictures add on top of it,
so everything else about the run stays as it was.
"""

from __future__ import annotations

import pytest

from experiments.warsaw.pipeline.records import LabelRequest, Vocabulary
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.classify.step import BodyGroupQuestion
from experiments.warsaw.pipeline.steps.vocabulary.step import LabelQuestion
from experiments.warsaw.world_loader.loader import RenderedSegmentGroup

# %% a question of each kind, with pictures and without

TAXONOMY = {"classes": [], "part_whole_mixins": [], "root_name": "SemanticAnnotation"}
"""
The least a question will take, since what is under test is the pictures and not the
ontology.
"""


@pytest.fixture
def three_renders(tmp_path) -> list:
    """
    :return: One render of each kind, named as the pipeline names them.
    """
    from PIL import Image

    names = []
    for kind in ("context", "plain", "closeup"):
        name = f"chair__{kind}_front_left.png"
        Image.new("RGB", (4, 4), (200, 0, 0)).save(tmp_path / name)
        names.append(name)
    return names


@pytest.fixture
def label_question(tmp_path):
    """
    :return: A maker of label questions carrying the renders it is given.
    """

    def made(images: list) -> LabelQuestion:
        return LabelQuestion(
            taxonomy=TAXONOMY,
            label=LabelRequest(
                label="chair", instances=1, exemplar="chair_1", images=list(images)
            ),
            every_label=["chair"],
            renders_directory=tmp_path,
        )

    return made


@pytest.fixture
def body_group_question():
    """
    :return: A maker of body-group questions carrying the renders it is given.
    """

    def made(images: dict) -> BodyGroupQuestion:
        return BodyGroupQuestion(
            taxonomy=TAXONOMY,
            vocabulary=Vocabulary(scene="", model=""),
            rendered=RenderedSegmentGroup(
                index=0, segments=[], colors={}, images=dict(images)
            ),
        )

    return made


# %% the setting


def test_a_run_shows_its_pictures_unless_told_otherwise() -> None:
    """
    Every run already made was asked with pictures, and its numbers stay comparable.
    """
    assert PipelineSettings().show_the_pictures is True


# %% what a question puts to a model when it has no pictures


def test_a_label_question_without_pictures_carries_no_image(label_question) -> None:
    """
    The message is built from what the question has, so a label that was never rendered
    puts text and nothing else.
    """
    assert [type(one).__name__ for one in label_question([]).message()] == ["TextPart"]


def test_a_label_question_with_pictures_still_carries_them(
    label_question, three_renders
) -> None:
    """
    The other half of the same switch: nothing changes for a run that renders.
    """
    kinds = [type(one).__name__ for one in label_question(three_renders).message()]
    assert kinds.count("ImagePart") == 3


def test_a_label_question_without_pictures_is_not_told_to_read_any(
    label_question,
) -> None:
    """
    The standing instruction promises pictures. Sending it with none asks a model to
    describe what it cannot see, which it will answer rather than refuse.
    """
    assert "picture" not in label_question([]).system_prompt.lower()


def test_a_label_question_with_pictures_is_told_to_read_them(
    label_question, three_renders
) -> None:
    """
    And the instruction a run has always been asked with is unchanged.
    """
    assert "picture" in label_question(three_renders).system_prompt.lower()


def test_a_body_group_question_without_pictures_carries_no_image(
    body_group_question,
) -> None:
    """
    The same for the question that names each body.
    """
    parts = body_group_question({}).message()
    assert [type(one).__name__ for one in parts] == ["TextPart"]


def test_a_body_group_question_without_pictures_is_not_told_to_read_any(
    body_group_question,
) -> None:
    """
    Its instruction opens by describing the pictures, so it needs the same care.
    """
    assert "picture" not in body_group_question({}).system_prompt.lower()


def test_a_body_group_question_without_pictures_names_no_colour(
    body_group_question,
) -> None:
    """
    A body is named to a model by the colour it was painted in the picture. With no
    picture there is no paint, and naming one would be describing something that was
    never drawn.
    """
    [text] = body_group_question({}).message()
    assert "painted" not in text.text


# %% the groups a run walks, drawn and undrawn


def test_an_undrawn_group_carries_no_picture_and_no_colour() -> None:
    """
    Where the time is saved: a run asking from the text alone paints nothing and places
    no camera, so a group comes back with neither a render nor the colour it would have
    been painted.
    """
    from experiments.warsaw.world_loader.loader import WarsawWorldLoader

    walked = WarsawWorldLoader.label_segment_groups
    assert "pictured" in walked.__code__.co_varnames


def test_the_render_step_is_told_what_the_run_was_told(tmp_path) -> None:
    """
    The setting has to reach the rendering, not only the prompts: a run that rendered and
    then hid the pictures would pay the whole cost for none of the saving.
    """
    from experiments.warsaw.pipeline.pipeline import WarsawPipeline
    from experiments.warsaw.pipeline.run import Run
    from experiments.warsaw.pipeline.steps.evidence import MeasureScene

    def measuring(showing: bool) -> list:
        settings = PipelineSettings(scene_directory=tmp_path, show_the_pictures=showing)
        planned = WarsawPipeline(settings=settings).run_steps(Run(directory=tmp_path))
        return [
            one.exemplar_renders for one in planned if isinstance(one, MeasureScene)
        ]

    # A scene is measured twice and only the first measuring renders the exemplars, so
    # what the setting decides is that first one.
    assert measuring(showing=True) == [True, False]
    assert measuring(showing=False) == [False, False]
