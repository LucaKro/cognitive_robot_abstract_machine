"""
Everything a run of the Warsaw pipeline can be told.

A run is started by constructing :class:`PipelineSettings` and handing it to
:class:`experiments.warsaw.pipeline.pipeline.WarsawPipeline`. Change a default here and
run the pipeline again; nothing is passed on a command line, because a setting spelled on
a command line is a setting nobody can find again when they want to know what a run was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from typing_extensions import Optional, Tuple

from experiments.warsaw.world_loader.viewpoints import (
    RenderSizes,
    Viewpoint,
    ViewpointChoice,
)

# %% the models a run can be put to


class Model(StrEnum):
    """
    A model to put the pipeline's questions to.

    Every question comes with pictures, so every model here reads images; a text-only
    model would fail on the first call rather than answer worse. They are listed cheapest
    first, with what a million prompt tokens costs, because a run is about a hundred calls
    and the difference between the ends of this list is the difference between two cents
    and a dollar.

    The value is the identifier OpenRouter knows the model by.
    """

    QWEN3_VL_32B = "qwen/qwen3-vl-32b-instruct"
    """
    $0.10 per million prompt tokens. Dense 32B; the cheapest of these.
    """

    QWEN3_VL_30B = "qwen/qwen3-vl-30b-a3b-instruct"
    """
    $0.15. Mixture-of-experts, 3B active. What every run so far used, and what the
    reported numbers come from.
    """

    GPT_5_6_LUNA = "openai/gpt-5.6-luna"
    """
    $0.20.
    """

    GEMINI_2_5_FLASH = "google/gemini-2.5-flash"
    """
    $0.30.
    """

    CLAUDE_HAIKU_4_5 = "anthropic/claude-haiku-4.5"
    """
    $1.00.
    """

    GEMINI_2_5_PRO = "google/gemini-2.5-pro"
    """
    $1.25.
    """

    CLAUDE_SONNET_4_5 = "anthropic/claude-sonnet-4.5"
    """
    $3.00. Worth trying on the steps that were unstable: the ownership answers vary
    between runs on about six of the thirty-three patterns, and the vocabulary step
    composes a class differently from one run to the next.
    """


# %% what a run is told to do


@dataclass
class PipelineSettings:
    """
    What a run is told, in full.
    """

    scene_directory: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
        / "dataset"
        / "kitchen2_meshes_out_20260910"
        # / "kitchen2_meshes_out_20260908_better_handles"
        # / "kitchenlab_new_mesh_agreement_dataset"
    )
    """
    The directory holding the scene's labelled mesh.
    """

    model: Model = Model.QWEN3_VL_30B
    """
    Which model every question goes to.
    """

    render_sizes: RenderSizes = field(default_factory=RenderSizes)
    """
    How large the renders are drawn.
    """

    viewpoint_choice: ViewpointChoice = ViewpointChoice.ALONE
    """
    How the one viewpoint a question is shown from is picked.
    """

    kept_viewpoints: Tuple[Viewpoint, ...] = (
        Viewpoint.FRONT_LEFT,
        Viewpoint.BACK_RIGHT,
    )
    """
    Which viewpoints a render keeps when nothing is choosing between them.

    Two opposite corners rather than all four: without a choice every render is kept, and
    four pictures of the same object from four sides is three of them saying what the
    first already said.
    """

    group_size: int = 8
    """
    How many bodies are painted and named at once in the classification step.
    """

    nearest: int = 5
    """
    How many nearest neighbours each object's evidence reaches for.
    """

    corrections: int = 1
    """
    How often an unusable answer is put back to the model with what was wrong with it.
    """

    headless: bool = False
    """
    Whether to render without opening a window. False shows the renders as they are made.
    """

    persist: bool = True
    """
    Whether to write the worlds to the database. Without it the run stops at the split's
    report, since everything after it reads a world back.
    """

    ask_about_the_ontology: bool = False
    """
    Whether to ask whether the taxonomy itself is missing a relation -- that a countertop
    can have drawers, say. Off by default: it proposes changes to the ontology every later
    scene would inherit.
    """

    amend_the_ontology: bool = False
    """
    Whether to carry those out for the length of this run. The edits are put back when the
    run ends; they are never committed. Needs :attr:`ask_about_the_ontology`.
    """

    ignore_amendments: bool = False
    """
    Whether to start even though the ontology's own files are left amended, which is what
    a run deliberately made against an amended ontology needs.
    """

    describe_the_classes: bool = False
    """
    Whether a class reaches the model with the first sentence of its docstring beside its
    name, rather than with its name and its bases alone.

    Off, because the runs already made were not asked that way. It is worth asking only
    once nearly every class carries a sentence: a taxonomy where some are described and
    some are not tells a model more about the described ones for no reason but that
    somebody wrote about them.
    """

    skip_classes_a_body_cannot_make: bool = False
    """
    Whether to leave a body alone when the class it was answered as cannot be made from
    a body, instead of letting the failure out of the step.

    Off, because a scan answers as it always has and its numbers stay comparable with
    the ones already reported. On, a body whose class needs something a body is not --
    a room is its floor, an aperture is rooted on a region -- costs that one body and is
    reported, rather than costing the run its world, its report and its evaluation graph.
    """

    make_a_region_where_a_class_needs_one: bool = False
    """
    Whether a class rooted on a region is given one built from the body it was answered
    about, instead of being left alone for having nowhere to stand.

    Off, because it changes what a run asserts rather than only what it reports: the
    annotation is real, and a mount into a field the ontology says removes the part's
    volume cuts the whole. On, a window becomes an aperture over a region the size and
    pose of the measured body, and the window-in-wall relation the measurement already
    finds is carried through to the world instead of being dropped at the last step.

    ..note:: What this does to a scan has not been measured. The scans label windows and
        ceilings too, so turning it on for them would assert apertures that were never
        asserted before and cut walls that were never cut; that is worth evaluating on
        its own rather than folding into a comparison of something else.
    """

    show_the_pictures: bool = True
    """
    Whether a run renders its objects and puts the pictures to the model, or asks from the
    text alone.

    On, which is what every run so far was asked with. Off, nothing is rendered and the two
    questions that would have carried pictures are asked without them -- which measures
    what the pictures add, since the label is still given either way. Rendering is most of
    a run's time, 44 minutes of room 13's 55, and three pictures a label is most of its
    bill.
    """

    show_the_contested_faces: bool = True
    """
    Whether the faces two labels both claim are rendered and put to the model, which is a
    different question from whether the objects are.

    Kept apart from :attr:`show_the_pictures` because the two are worth different amounts.
    The other renders ask what an object *is*, and a label already says that; this one
    asks whose a surface is, where the measurement genuinely cannot say and the label
    cannot either. A scan contests faces constantly -- a drawer front is labelled both as
    the drawer and as the cabinet holding it -- and a converted HM3D room never does,
    since those files give every face to exactly one object.
    """

    settle_the_superclass: bool = False
    """
    Whether what a proposed class is a kind of is asked as a question of its own, once per
    class, after its name is settled and before it is written.

    Off, the step that names a class also chooses what to build it from, in the same
    answer. The names that come back are stable and the superclasses are not, and saying
    so in the prompt did not help -- a run composed ``Tap(Aperture)`` from a prompt that
    named that exact pair as wrong. This asks the subsumption on its own instead, which
    costs one text-only call per class a run wants.
    """

    skip_classes_that_name_a_category: bool = False
    """
    Whether a class the ontology declares a category -- ``Furniture``, ``Decor``,
    ``ElectricalDevice`` -- is refused as an answer about an object.

    Off, which is how every run before 2026-09-14 behaved, so the scans stay comparable
    with what was already reported. On, the taxonomy marks those classes and the annotate
    step refuses them, which is what makes a run compose ``Stool`` and ``Ornament``
    instead of answering the category. It is not free: the same three rooms lost between
    20 and 39 bodies to answers the guard then refused.

    A class Python itself calls abstract, carrying an unimplemented method, is refused
    either way. That one is not a choice: it cannot be instantiated at all.
    """

    reuse_answers: bool = False
    """
    Whether to read back the responses a run already kept instead of asking again, which
    re-reads a run without spending anything on it.
    """

    runs_directory: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "pipeline_runs"
    )
    """
    Where the run's directory is made. The runs live beside the scenes they read, and
    neither is committed.
    """
