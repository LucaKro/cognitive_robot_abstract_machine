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

from experiments.warsaw.world_loader import Viewpoint, ViewpointChoice


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


@dataclass
class PipelineSettings:
    """
    What a run is told, in full.
    """

    scene: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
        / "dataset"
        / "kitchenlab_new_mesh_agreement_dataset"
    )
    """
    The directory holding the scene's labelled mesh.
    """

    model: Model = Model.QWEN3_VL_30B
    """
    Which model every question goes to.
    """

    render_resolution: Tuple[int, int] = (1024, 768)
    """
    How large the pictures a model is shown are drawn.
    """

    deciding_resolution: Optional[Tuple[int, int]] = None
    """
    How large to draw the renders made only to choose a viewpoint and then thrown away.
    ``(256, 192)`` is a sixteenth of the pixels; None draws them full size.
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

    headless: bool = True
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
