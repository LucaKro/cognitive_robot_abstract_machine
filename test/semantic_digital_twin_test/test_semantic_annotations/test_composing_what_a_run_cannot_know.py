"""
Composing a class under a base that demands something a run cannot supply.

``Tool`` requires ``tool_alignment``, the normal pairs that must stay aligned while the
tool acts. A whisk answers with one pair of vectors and a cutting knife with two: it is
motion-planning geometry, and nothing about a class name or a list of bases says which
way a tap points.

A run that places a tap under ``Tool`` is right to -- a tap is a kind of tool -- and the
composed class was then refused by the annotate step for being abstract, so the correct
answer cost every body that carried it. The class is written with the method stubbed
instead, raising where it is called, so the semantic claim is kept and only acting on
the object fails, at the point where the missing knowledge is what is missing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pytest

from semantic_digital_twin.exceptions import ComposedClassCannotAct
from semantic_digital_twin.semantic_annotations.in_memory_builder import (
    SemanticAnnotationClassBuilder,
)
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import Tool

TEMPLATE = "dataclass_template.py.jinja"
"""
What a generated class is written from, which the builder takes even where only the
class itself is wanted.
"""

# %% a base demanding what a name cannot say


@dataclass(eq=False)
class DemandsWhatANameCannotSay(HasRootBody, ABC):
    """
    A base whose subclasses must answer something no run can work out for itself.
    """

    @abstractmethod
    def how_to_act_on(self, target: object) -> list:
        """
        :param target: What is acted on.
        :return: How to act on it.
        """


def composed(name: str = "Tap"):
    """
    :param name: What to call the composed class.
    :return: That class, built under the demanding base the way a run builds one.
    """
    builder = SemanticAnnotationClassBuilder(name, template_name=TEMPLATE)
    builder.add_base(DemandsWhatANameCannotSay)
    return builder.build()


# %% the class a run gets


def test_a_composed_class_can_be_made() -> None:
    """
    The defect this fixes: the class was abstract, so every body answered as one was
    left unannotated and the run lost the correct answer.
    """
    made = composed()
    assert not getattr(made, "__abstractmethods__", frozenset())


def test_what_the_run_could_not_know_raises_when_it_is_asked_for() -> None:
    """
    Answering something plausible instead would be worse: an empty list reads as "no
    constraints", which is an answer rather than an absence.
    """
    with pytest.raises(ComposedClassCannotAct):
        composed().how_to_act_on(object())


def test_the_exception_names_the_class_and_what_is_missing() -> None:
    """
    So the message tells the next reader what to write and why a run did not write it.
    """
    with pytest.raises(ComposedClassCannotAct) as raised:
        composed("Tap").how_to_act_on(object())
    assert "Tap" in str(raised.value)
    assert "how_to_act_on" in str(raised.value)


def test_a_class_under_a_base_demanding_nothing_is_left_alone() -> None:
    """
    Nothing is stubbed where nothing is missing, so the classes runs have always
    composed are written exactly as they were.
    """
    builder = SemanticAnnotationClassBuilder("Ornament", template_name=TEMPLATE)
    builder.add_base(HasRootBody)
    assert builder.stubbed_methods == []


def test_the_run_can_see_which_classes_carry_a_stub() -> None:
    """
    A stubbed class looks complete to everything that only asks whether it can be made,
    so what was stubbed is reported rather than left to be discovered by a robot.
    """
    builder = SemanticAnnotationClassBuilder("Tap", template_name=TEMPLATE)
    builder.add_base(DemandsWhatANameCannotSay)
    assert builder.stubbed_methods == ["how_to_act_on"]


# %% the class a run writes to disk


def test_the_written_class_carries_the_stub(tmp_path) -> None:
    """
    A run writes its classes to a file and imports them in a second interpreter, so the
    source has to carry what the in-memory class carries -- otherwise the class that is
    actually used is the abstract one.
    """
    builder = SemanticAnnotationClassBuilder("Tap", template_name=TEMPLATE)
    builder.add_base(DemandsWhatANameCannotSay)
    written = tmp_path / "generated_classes.py"
    SemanticAnnotationClassBuilder.write_classes_to_file([builder], written)

    source = written.read_text()
    assert "def how_to_act_on" in source
    assert "ComposedClassCannotAct" in source


def test_the_written_class_can_be_imported_and_made(tmp_path) -> None:
    """
    The end the pipeline actually reaches, and the case that produced this: a run placed
    a tap under Tool, which is right, and every body carrying it was then left alone for
    the class being abstract.
    """
    import importlib.util

    builder = SemanticAnnotationClassBuilder("Tap", template_name=TEMPLATE)
    builder.add_base(Tool)
    written = tmp_path / "generated_classes.py"
    SemanticAnnotationClassBuilder.write_classes_to_file([builder], written)

    specification = importlib.util.spec_from_file_location("generated", written)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    assert not module.Tap.__abstractmethods__
