"""
What a run is made of, and which of its steps it can do without.

Everything a run can be told is a field of its settings, so what the run then does is
decided here rather than assembled from arguments. The scene is measured twice on
purpose: once knowing no classes, which is what the vocabulary question is built from,
and once knowing them, which is what says how the labels may hold one another.
"""

from __future__ import annotations

from pathlib import Path


from experiments.warsaw.pipeline.pipeline import WarsawPipeline
from experiments.warsaw.pipeline.run import Run
from experiments.warsaw.pipeline.settings import Model, PipelineSettings
from experiments.warsaw.pipeline.steps.adjudicate import AdjudicateOverlaps
from experiments.warsaw.pipeline.steps.amend import AmendTaxonomy, RevertAmendments
from experiments.warsaw.pipeline.steps.annotate import AnnotateAndMount
from experiments.warsaw.pipeline.steps.classify import ClassifyBodies
from experiments.warsaw.pipeline.steps.evidence import MeasureScene
from experiments.warsaw.pipeline.steps.split import SplitScene
from experiments.warsaw.pipeline.steps.vocabulary import MapLabelVocabulary
from experiments.warsaw.world_loader import Viewpoint, ViewpointChoice


def planned(settings: PipelineSettings, tmp_path: Path):
    """
    :param settings: What the run is told.
    :param tmp_path: A directory to plan against.
    :return: The steps that run would carry out.
    """
    return WarsawPipeline(settings=settings).run_steps(Run(directory=tmp_path))


# %% the run as it stands


def test_a_run_measures_asks_measures_again_and_then_builds(tmp_path):
    """
    The order is what makes the second measurement worth making: it is the first one
    read again in the light of what the labels turned out to mean.
    """
    steps = planned(PipelineSettings(), tmp_path)
    assert [type(one) for one in steps] == [
        MeasureScene,
        MapLabelVocabulary,
        MeasureScene,
        AdjudicateOverlaps,
        SplitScene,
        ClassifyBodies,
        AnnotateAndMount,
    ]


def test_the_first_measurement_renders_the_labels_and_asks_nothing_of_them(tmp_path):
    """
    The exemplars are what the vocabulary question is put with, and there are no classes
    yet to say what any pair may be.
    """
    first = planned(PipelineSettings(), tmp_path)[0]
    assert first.exemplar_renders
    assert not first.knowing_the_vocabulary
    assert first.question_renders == 0


def test_the_second_measurement_reads_the_vocabulary_and_renders_the_questions(
    tmp_path,
):
    """
    It writes over what the first pass wrote, which is what an answered vocabulary makes
    of the same measurement.
    """
    second = planned(PipelineSettings(), tmp_path)[2]
    assert second.knowing_the_vocabulary
    assert second.overwrite
    assert second.question_renders


# %% what a run can be told to leave out


def test_a_run_that_persists_nothing_stops_at_the_split(tmp_path):
    """
    Everything after the split reads a world back, so without one there is nothing to
    do.
    """
    steps = planned(PipelineSettings(persist=False), tmp_path)
    assert type(steps[-1]) is SplitScene


def test_asking_about_the_ontology_is_a_step_of_its_own(tmp_path):
    """
    It proposes changes to the ontology every later scene would inherit, so it is off
    unless a run asks for it.
    """
    assert not any(
        isinstance(one, AmendTaxonomy) for one in planned(PipelineSettings(), tmp_path)
    )
    steps = planned(PipelineSettings(ask_about_the_ontology=True), tmp_path)
    assert any(isinstance(one, AmendTaxonomy) for one in steps)


def test_an_amended_ontology_is_put_back_when_the_run_ends(tmp_path):
    """
    An amendment is applied for the length of a run, not kept: the next scene should
    start from the ontology as it was written by hand.
    """
    steps = planned(
        PipelineSettings(ask_about_the_ontology=True, amend_the_ontology=True), tmp_path
    )
    assert type(steps[-1]) is RevertAmendments


def test_the_steps_a_run_can_do_without_are_the_ones_about_the_ontology(tmp_path):
    """
    A failure in any other step leaves the run unable to go on.
    """
    steps = planned(
        PipelineSettings(ask_about_the_ontology=True, amend_the_ontology=True), tmp_path
    )
    assert {type(one).__name__ for one in steps if one.is_optional} == {
        "AmendTaxonomy",
        "RevertAmendments",
    }


# %% what every step is told


def test_every_step_is_told_the_same_thing(tmp_path):
    """
    A step reads its settings rather than being handed arguments, so two steps cannot
    disagree about what the run was told.
    """
    settings = PipelineSettings(model=Model.GPT_5_6_LUNA, group_size=4)
    for step in planned(settings, tmp_path):
        assert step.settings is settings
        assert step.run.directory == tmp_path


def test_choosing_a_viewpoint_renders_them_all_to_decide_between_them(tmp_path):
    """
    Choosing costs the renders it then discards, so a run that chooses nothing keeps a
    fixed pair instead.
    """
    choosing = planned(
        PipelineSettings(viewpoint_choice=ViewpointChoice.ALONE), tmp_path
    )[0]
    assert choosing.viewpoints is None
    assert choosing.chooses_viewpoint is ViewpointChoice.ALONE


def test_keeping_every_viewpoint_keeps_the_two_the_run_names(tmp_path):
    """
    Four pictures of the same object from four sides is three of them saying what the
    first already said.
    """
    keeping = planned(PipelineSettings(viewpoint_choice=ViewpointChoice.ALL), tmp_path)[
        0
    ]
    assert keeping.chooses_viewpoint is None
    assert keeping.viewpoints == [Viewpoint.FRONT_LEFT, Viewpoint.BACK_RIGHT]
