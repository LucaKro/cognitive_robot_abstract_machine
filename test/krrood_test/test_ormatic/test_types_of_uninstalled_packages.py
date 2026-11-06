"""
Tests that ORMatic leaves out a field whose type comes from a package that is not
installed, instead of failing to resolve the type.
"""

import logging

import pytest

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.class_diagrams.exceptions import CouldNotResolveType
from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.ormatic.field_storage import FieldStorage, FieldToStore, NotStoredRule
from krrood.ormatic.ormatic import ORMatic
from ..dataset.field_typed_from_uninstalled_package import (
    OwnerOfHandleFromUninstalledPackage,
    OwnerOfNameImportedFromNowhere,
)

# %% helpers


@pytest.fixture
def ormatic() -> ORMatic:
    """
    :return: An ORMatic over the owner of a field typed from an uninstalled package.
    """
    return ORMatic(ClassDiagram([OwnerOfHandleFromUninstalledPackage]))


def field_named(ormatic: ORMatic, name: str) -> WrappedField:
    """
    :return: The field of :class:`OwnerOfHandleFromUninstalledPackage` with the given
        name.
    """
    wrapped_class = ormatic.class_dependency_graph.get_wrapped_class(
        OwnerOfHandleFromUninstalledPackage
    )
    return next(f for f in wrapped_class.fields if f.field.name == name)


# %% a type from a package that is not installed


def test_field_typed_from_an_uninstalled_package_is_not_stored(ormatic):
    """
    A field whose type comes from a package that is not installed is left out.
    """
    handle = field_named(ormatic, "handle")

    assert ormatic.field_classifier.classify(handle) is FieldStorage.NOT_STORED


def test_other_fields_of_the_owner_are_still_stored(ormatic):
    """
    Leaving out such a field leaves the owner's other fields as they are stored.
    """
    label = field_named(ormatic, "label")

    assert ormatic.field_classifier.classify(label) is FieldStorage.BUILTIN


def test_leaving_out_a_field_of_an_uninstalled_package_is_logged(ormatic, caplog):
    """
    The log names the type whose package is missing, so the field left out is visible.
    """
    handle = field_named(ormatic, "handle")

    with caplog.at_level(logging.INFO, logger="krrood.ormatic.field_storage"):
        ormatic.field_classifier.classify(handle)

    assert NotStoredRule.describe(FieldToStore(handle, ormatic)) in caplog.messages


# %% a name imported from nowhere


def test_a_name_imported_from_nowhere_still_cannot_be_resolved():
    """
    A name no import binds is still an error, not a stand-in.
    """
    get_type_hints_of_object.cache_clear()

    with pytest.raises(CouldNotResolveType):
        get_type_hints_of_object(OwnerOfNameImportedFromNowhere)
