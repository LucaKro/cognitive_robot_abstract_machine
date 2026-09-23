"""
Reading the annotation taxonomy back out of the classes the interpreter holds.

What a model is shown about the ontology is built here, so what matters is that the
order it puts bases in is one Python can actually combine, and that a class's reported
bases are the ones it really derives from.
"""

from __future__ import annotations

from semantic_digital_twin.semantic_annotations.mixins import (
    HasDoors,
    HasDrawers,
    HasRootBody,
    IsStorageSpace,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Furniture
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
    compose_class,
    in_base_order,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

# %% ordering the bases a proposal names


def test_a_base_named_before_its_own_subclass_is_put_after_it():
    """
    Python refuses a class naming a base in front of something deriving from that base.
    """
    assert issubclass(IsStorageSpace, HasRootBody)

    assert in_base_order([HasRootBody, IsStorageSpace]) == (
        IsStorageSpace,
        HasRootBody,
    )


def test_bases_already_in_a_usable_order_are_left_as_they_are():
    """
    Reordering what is already right would change a class's method resolution for
    nothing.
    """
    assert in_base_order([IsStorageSpace, HasRootBody]) == (
        IsStorageSpace,
        HasRootBody,
    )


def test_a_base_named_twice_is_kept_once():
    """
    Python refuses a class naming the same base twice.
    """
    assert in_base_order([HasDoors, HasDrawers, HasDoors]) == (HasDoors, HasDrawers)


def test_the_order_it_returns_can_be_combined_into_a_class():
    """
    The point of the ordering: the classes go together afterwards.
    """
    composed = compose_class(
        "StorageWithRootBody", Furniture, in_base_order([HasRootBody, IsStorageSpace])
    )

    assert issubclass(composed, IsStorageSpace)
    assert issubclass(composed, HasRootBody)


# %% which classes there are


def test_every_annotation_class_is_reported_under_its_own_name():
    """
    A name read back from a file or a model is looked up in exactly this mapping.
    """
    classes = annotation_classes(SemanticAnnotation)

    assert classes["Furniture"] is Furniture
    assert classes["HasDrawers"] is HasDrawers
    assert all(name == found.__name__ for name, found in classes.items())
