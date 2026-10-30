"""
The table an HM3D scene writes its annotations in.

HM3D-Semantics names every annotated object in one text file beside the mesh: an id, the
colour that object's faces are painted in the semantic mesh, what an annotator called it,
and which room it was assigned to. That is the whole schema, and it is the only place the
labels live.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from typing_extensions import Dict, List, Tuple

from experiments.warsaw.exceptions import HabitatAnnotationsUnreadableError

HEADER = "HM3D Semantic Annotations"
"""
What the first line of one of these tables says, and nothing else does.
"""

UNASSIGNED_ROOM = 0
"""
The room an object is put in when it was assigned to none.

Not a place: across the four annotated minival scenes it holds containers, bags, luggage
and armchairs from all over the building, beside a handful of walls.
"""


class Column(IntEnum):
    """
    Where each field of an annotation stands in a row of the table.
    """

    OBJECT_ID = 0
    """
    The id identifying the object among the scene's objects.
    """

    COLOR = 1
    """
    The colour its faces carry, as six hexadecimal digits.
    """

    LABEL = 2
    """
    What it was called, in quotes.
    """

    ROOM_ID = 3
    """
    The room it was assigned to.
    """


# %% one annotated object


@dataclass(frozen=True)
class AnnotatedObject:
    """
    One object an HM3D scene annotates.
    """

    object_id: int
    """
    Which object of the scene this is.
    """

    color: Tuple[int, int, int]
    """
    The colour the semantic mesh paints its faces in.
    """

    label: str
    """
    What an annotator called it, in their own words and possibly with spaces in it.
    """

    room_id: int
    """
    The room it was assigned to, or :data:`UNASSIGNED_ROOM` for none.
    """

    @classmethod
    def read(cls, row: str, table: Path) -> AnnotatedObject:
        """
        ..note:: The label is quoted but the row is split on commas all the same, since
            no label in the four annotated minival scenes holds one. A release whose
            labels did would be refused here rather than read wrongly.

        :param row: One row of the table.
        :param table: The file it came from, for the error message.
        :return: The object it describes.
        :raises HabitatAnnotationsUnreadableError: If the row is not one of these rows.
        """
        fields = row.split(",")
        if len(fields) != len(Column):
            raise HabitatAnnotationsUnreadableError(
                table=table,
                complaint=f"a row has {len(fields)} fields rather than {len(Column)}",
                row=row,
            )
        hexadecimal = fields[Column.COLOR].strip()
        if len(hexadecimal) != 6:
            raise HabitatAnnotationsUnreadableError(
                table=table,
                complaint=f"'{hexadecimal}' is not six hexadecimal digits",
                row=row,
            )
        return cls(
            object_id=int(fields[Column.OBJECT_ID]),
            color=tuple(int(hexadecimal[digit : digit + 2], 16) for digit in (0, 2, 4)),
            label=fields[Column.LABEL].strip().strip('"'),
            room_id=int(fields[Column.ROOM_ID]),
        )


# %% the whole table


@dataclass
class SceneAnnotations:
    """
    Every object one HM3D scene annotates, in the order the table names them.
    """

    objects: List[AnnotatedObject]
    """
    The objects themselves.
    """

    @classmethod
    def from_file(cls, table: Path) -> SceneAnnotations:
        """
        :param table: The scene's ``.semantic.txt``.
        :return: Everything it annotates.
        :raises HabitatAnnotationsUnreadableError: If the file is not one of these
            tables.
        """
        table = Path(table)
        lines = table.read_text().splitlines()
        if not lines or lines[0].strip() != HEADER:
            raise HabitatAnnotationsUnreadableError(
                table=table,
                complaint=f"the first line does not say '{HEADER}'",
                row=lines[0] if lines else "",
            )
        return cls(
            objects=[
                AnnotatedObject.read(row, table) for row in lines[1:] if row.strip()
            ]
        )

    def by_color(self) -> Dict[Tuple[int, int, int], AnnotatedObject]:
        """
        :return: The objects by the colour that marks their faces, which is how the mesh
            says which object a face belongs to.
        """
        return {one.color: one for one in self.objects}

    @property
    def room_ids(self) -> List[int]:
        """
        :return: Every room the table assigns an object to, in order.
        """
        return sorted({one.room_id for one in self.objects})

    def in_room(self, room_id: int) -> List[AnnotatedObject]:
        """
        :param room_id: The room to gather.
        :return: The objects assigned to it, in the order the table names them.
        """
        return [one for one in self.objects if one.room_id == room_id]

    def labels_in_room(self, room_id: int) -> List[str]:
        """
        :param room_id: The room to gather.
        :return: The labels its objects carry, each named once.
        """
        return sorted({one.label for one in self.in_room(room_id)})
