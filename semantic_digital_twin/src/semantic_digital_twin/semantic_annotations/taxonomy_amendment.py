"""
Amend the taxonomy itself: give a class a mixin it turns out to be missing.

A scanned room can show that the taxonomy is wrong rather than that the room is odd. If
objects of one class are measured to hold objects of another and no field admits it, one
of two things is true: the objects want a class of their own, or the class is missing a
mixin. This module carries out the second, which means editing the source the class is
written in, because a dataclass collects its fields when ``@dataclass`` runs and the ORM
is generated from those fields -- assigning to ``__bases__`` afterwards is accepted by
Python and changes nothing that anything reads.

That makes an amendment permanent and global: every room, every world, every row already
in the database. Nothing here decides that a class is missing something; it reports what
*would* grant a relation and carries out a decision made elsewhere.
"""

from __future__ import annotations

import ast
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Type

from semantic_digital_twin.semantic_annotations.part_whole import admissible_relations
from semantic_digital_twin.semantic_annotations.taxonomy_export import compose_class


class CannotAmendClass(Exception):
    """
    Raised when a class cannot be given a mixin by editing the source it is written in.
    """


@dataclass
class SourceAmendment:
    """
    One edit giving a class another base, as it would be written.
    """

    annotation_class: Type
    """
    The class being amended.
    """

    mixin: Type
    """
    The mixin it gains.
    """

    path: Path
    """
    The file its declaration is written in.
    """

    line_number: int
    """
    Which line of that file declares it, counting from one.
    """

    before: str
    """
    The declaration as it reads now.
    """

    after: str
    """
    The declaration as it would read.
    """

    def apply(self) -> None:
        """
        Write the amendment to the file the class is declared in.

        :raises CannotAmendClass: If the declaration is no longer what it was read as,
            which means the file changed underneath and the line number is stale.
        """
        lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
        index = self.line_number - 1
        if lines[index].rstrip("\n") != self.before:
            raise CannotAmendClass(
                f"{self.path}:{self.line_number} no longer reads as it was read"
            )
        lines[index] = self.after + "\n"
        self.path.write_text("".join(lines), encoding="utf-8")

    def reverted(self) -> "SourceAmendment":
        """
        :return: The amendment that undoes this one.
        """
        return SourceAmendment(
            annotation_class=self.annotation_class,
            mixin=self.mixin,
            path=self.path,
            line_number=self.line_number,
            before=self.after,
            after=self.before,
        )


def granting_mixins(
    whole: Type, part: Type, mixins: Sequence[Type]
) -> List[Type]:
    """
    Report which mixins would let one class hold another as a structural part.

    Answered by composing the class with each mixin and asking the same question the
    mount does, so a mixin is reported only if it really grants the relation. A mixin
    counts as granting it only if it brings a field the class did not already have for
    that part: once a cabinet declares ``drawers``, every mixin composed onto it leaves
    a drawer admissible, and reporting them all would say every mixin grants everything.

    :param whole: The class that would hold the part.
    :param part: The class that would be held.
    :param mixins: The mixins to try.
    :return: Those that make a part-whole relation admissible, in the order given, empty
        when the class already admits it.
    """
    already = _fields_holding(whole, part)
    if already:
        return []

    granting = []
    for mixin in mixins:
        if issubclass(whole, mixin):
            continue
        composed = compose_class(f"{whole.__name__}With{mixin.__name__}", whole, [mixin])
        if _fields_holding(composed, part) - already:
            granting.append(mixin)
    return granting


def _fields_holding(whole: Type, part: Type) -> set:
    """
    :param whole: The class that would hold the part.
    :param part: The class that would be held.
    :return: The names of that class's own fields a part of that type could be mounted
        into, ignoring what the part could hold in turn.
    """
    return {
        relation.field_name
        for relation in admissible_relations(whole, part)
        if relation.whole is whole
    }


def _declaration_of(annotation_class: Type) -> tuple:
    """
    Find where a class is declared.

    :param annotation_class: The class to look for.
    :return: The file it is written in, the syntax tree of its declaration, and the
        source of that file.
    :raises CannotAmendClass: If it has no source to edit, which is what a class built
        at run time rather than written down looks like.
    """
    try:
        path = Path(inspect.getsourcefile(annotation_class))
    except TypeError as failure:
        raise CannotAmendClass(f"{annotation_class.__name__} has no source") from failure

    source = path.read_text(encoding="utf-8")
    declaration = next(
        (
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ClassDef)
            and node.name == annotation_class.__name__
        ),
        None,
    )
    if declaration is None:
        raise CannotAmendClass(
            f"{annotation_class.__name__} is not declared in {path}, so it was built "
            f"rather than written"
        )
    return path, declaration, source


def amend_class_source(
    annotation_class: Type, mixin: Type
) -> Optional[SourceAmendment]:
    """
    Work out the edit that gives a class another base.

    The mixin is added last, where the taxonomy already puts them -- a cabinet is
    ``Cabinet(Furniture, HasCaseAsRootBody, HasDoors, HasDrawers)`` -- which keeps a
    class's identity in front of what it can hold.

    :param annotation_class: The class to amend.
    :param mixin: The mixin to give it.
    :return: The edit, or None when the class already has that mixin.
    :raises CannotAmendClass: If the class cannot be amended this way: it has no source,
        it is declared without bases, or the mixin is not imported where it is declared.
    """
    if issubclass(annotation_class, mixin):
        return None

    path, declaration, source = _declaration_of(annotation_class)
    if not declaration.bases:
        raise CannotAmendClass(
            f"{annotation_class.__name__} is declared without bases, so there is no "
            f"base list to add {mixin.__name__} to"
        )

    module = sys.modules.get(annotation_class.__module__)
    if getattr(module, mixin.__name__, None) is not mixin:
        raise CannotAmendClass(
            f"{mixin.__name__} is not imported in {annotation_class.__module__}, so a "
            f"declaration naming it would not resolve"
        )

    last_base = declaration.bases[-1]
    lines = source.splitlines()
    line = lines[last_base.end_lineno - 1]
    amended = (
        line[: last_base.end_col_offset]
        + f", {mixin.__name__}"
        + line[last_base.end_col_offset :]
    )
    return SourceAmendment(
        annotation_class=annotation_class,
        mixin=mixin,
        path=path,
        line_number=last_base.end_lineno,
        before=line,
        after=amended,
    )
