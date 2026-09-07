"""
Giving a written-down class another base by editing the file it is declared in.

This is the only part of the taxonomy that writes to source files, so what matters is
that an amendment is worked out from what the file actually says, that applying it and
undoing it leave the file exactly as it was, and that a class it cannot be worked out
for is refused rather than guessed at.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from semantic_digital_twin.semantic_annotations.mixins import HasDoors, HasDrawers
from semantic_digital_twin.semantic_annotations.taxonomy_amendment import (
    ClassNotWrittenDown,
    DeclarationMovedSinceRead,
    MixinNotImportedWhereDeclared,
    SourceAmendment,
    amend_class_source,
)

from .dataset.amendable_annotations import Pedestal, Sideboard

# %% working the edit out


def test_an_amendment_adds_the_mixin_after_the_last_base():
    """
    A class keeps its identity in front of what it can hold, so the mixin goes last.
    """
    amendment = amend_class_source(Sideboard, HasDrawers)

    assert amendment.before == "class Sideboard(Furniture, HasDoors):"
    assert amendment.after == "class Sideboard(Furniture, HasDoors, HasDrawers):"
    assert amendment.annotation_class is Sideboard
    assert amendment.mixin is HasDrawers


def test_the_line_number_points_at_the_declaration_it_read():
    """
    The edit is applied by line number, so that number has to name the line it read.
    """
    amendment = amend_class_source(Sideboard, HasDrawers)

    written = amendment.path.read_text(encoding="utf-8").splitlines()
    assert written[amendment.line_number - 1] == amendment.before


def test_a_class_that_already_has_the_mixin_needs_no_amendment():
    """
    Nothing to do is not a failure, and adding the base twice would not compile.
    """
    assert amend_class_source(Pedestal, HasDrawers) is None


def test_a_class_built_rather_than_written_is_refused():
    """
    A class made at run time has no source to edit.
    """
    built = type("BuiltSideboard", (Sideboard,), {})
    with pytest.raises(ClassNotWrittenDown):
        amend_class_source(built, HasDrawers)


def test_a_mixin_the_declaring_module_does_not_import_is_refused():
    """
    A declaration naming something the module never imported would not resolve.
    """

    class NotImportedHere:
        pass

    with pytest.raises(MixinNotImportedWhereDeclared):
        amend_class_source(Sideboard, NotImportedHere)


# %% applying it and putting it back


@pytest.fixture
def copied_module(tmp_path) -> Path:
    """
    :return: A copy of the fixture module, so a test edits it rather than the original.
    """
    original = Path(amend_class_source(Sideboard, HasDrawers).path)
    copy = tmp_path / original.name
    shutil.copy(original, copy)
    return copy


def amendment_on(copy: Path) -> SourceAmendment:
    """
    :param copy: The copied module to edit.
    :return: The same amendment, pointed at the copy.
    """
    worked_out = amend_class_source(Sideboard, HasDrawers)
    return SourceAmendment(
        annotation_class=worked_out.annotation_class,
        mixin=worked_out.mixin,
        path=copy,
        line_number=worked_out.line_number,
        before=worked_out.before,
        after=worked_out.after,
    )


def test_applying_an_amendment_writes_the_amended_declaration(copied_module):
    """
    The one line changes and the rest of the file is left alone.
    """
    before = copied_module.read_text(encoding="utf-8").splitlines()
    amendment = amendment_on(copied_module)

    amendment.apply()

    after = copied_module.read_text(encoding="utf-8").splitlines()
    assert after[amendment.line_number - 1] == amendment.after
    changed = [index for index, line in enumerate(after) if line != before[index]]
    assert changed == [amendment.line_number - 1]


def test_undoing_an_amendment_leaves_the_file_as_it_was(copied_module):
    """
    A run puts the ontology back when it ends, so the undo has to be exact.
    """
    original = copied_module.read_text(encoding="utf-8")
    amendment = amendment_on(copied_module)

    amendment.apply()
    amendment.reverted().apply()

    assert copied_module.read_text(encoding="utf-8") == original


def test_an_amendment_whose_line_moved_is_refused(copied_module):
    """
    A line number read from one version of a file says nothing about another.
    """
    amendment = amendment_on(copied_module)
    copied_module.write_text(
        "# a line that was not there when the declaration was read\n"
        + copied_module.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(DeclarationMovedSinceRead):
        amendment.apply()
