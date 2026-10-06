"""
Telling a category apart from something that stands in a room.

The taxonomy a model is shown marks a class ``abstract`` and tells it that such a class
cannot be given to an object, name one of its subclasses instead. That marker is derived
from :func:`inspect.isabstract`, which is true only of a class carrying an abstract
method -- and an ontology's categories carry none. ``Furniture`` declares ``ABC`` and
Python instantiates it all the same, so the marker never appeared, the model was never
told, and eight stools were annotated ``Furniture``.

Declaring ``ABC`` is how this ontology says a class is a category. Reading that is what
makes the marker mean what the taxonomy already says it means.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Decor,
    ElectricalDevice,
    Furniture,
    Tool,
    WallDecor,
)
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    names_a_category,
)

# %% classes standing in for the two shapes a category comes in


@dataclass(eq=False)
class CategoryDeclaringItself(HasRootBody, ABC):
    """
    A category that says so with ``ABC`` and carries no abstract method, which is how
    every category in this ontology is written.
    """


@dataclass(eq=False)
class ThingUnderThatCategory(CategoryDeclaringItself):
    """
    Something that stands in a room, inheriting from one.
    """


# %% what counts as a category


def test_a_class_declaring_abc_is_a_category() -> None:
    """
    Even carrying no abstract method, which Python would not call abstract.
    """
    assert names_a_category(CategoryDeclaringItself)


def test_inheriting_from_a_category_does_not_make_a_category() -> None:
    """
    Otherwise every subclass of Furniture would be unanswerable too, and the rule would
    empty the ontology rather than sharpen it.
    """
    assert not names_a_category(ThingUnderThatCategory)


def test_a_class_python_calls_abstract_is_a_category() -> None:
    """
    The older kind, carrying a real abstract method, still counts.
    """
    assert names_a_category(Tool)


# %% the ontology's own categories


def test_the_ontologys_categories_are_categories() -> None:
    """
    The three a run has actually been answered with instead of a subclass.
    """
    assert [
        one.__name__
        for one in (Furniture, Decor, ElectricalDevice)
        if not names_a_category(one)
    ] == []


def test_the_things_under_them_are_not() -> None:
    """
    A cabinet and a wall decoration are things a room holds, and stay answerable.
    """
    assert not names_a_category(Cabinet)
    assert not names_a_category(WallDecor)
