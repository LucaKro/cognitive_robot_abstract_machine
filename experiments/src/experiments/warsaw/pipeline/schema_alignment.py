"""
Give a database that already holds worlds the columns the ORM has grown.

``create_all`` creates tables that are absent and leaves alone every table that is already
there, however far it has drifted from the class it is meant to hold. The pipeline
regenerates the ORM from the ontology as it is committed, so the day the ontology gains an
attribute -- a colour becoming a class with subclasses, a modification learning to point at
the annotation it made -- every world written after that names a column the database has
never heard of, and the write fails on the first insert::

    column "polymorphic_type" of relation "ColorDAO" does not exist

The drift is additive: the ORM asks for columns that are missing, never for the removal of
one that is there. So it can be closed by adding them, which costs no stored world. Rows
written before the column existed get the value they would have been written with, which
for a class discriminator is the class that was the only one at the time.

A run does not need this -- every run builds its tables in a schema of its own, so it never
meets a table another run left standing. What needs it is the standing schema holding the
worlds written before that was true.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import Column, MetaData, Table, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapper
from typing_extensions import Dict, Iterable, List, Optional

from experiments.warsaw.exceptions import ColumnsCannotBeAddedError


@dataclass
class AlignmentReport:
    """
    What aligning a database did, and what it could not do.
    """

    added: List[str] = field(default_factory=list)
    """
    One sentence per column that was added.
    """

    refused: List[str] = field(default_factory=list)
    """
    One sentence per column that cannot be added without a decision about the rows that
    are already stored.
    """

    @property
    def is_aligned(self) -> bool:
        """
        :return: Whether the database now has every column the ORM asks for.
        """
        return not self.refused


@dataclass
class SchemaAlignment:
    """
    The columns one database is missing, and the adding of them.
    """

    engine: Engine
    """
    The database to look at and change.
    """

    base: type
    """
    The ORM's declarative base, which holds both the tables and the mappers that say
    what a stored row already is.
    """

    def drift(self) -> Dict[Table, List[Column]]:
        """
        Find the columns the ORM declares and the database does not have.

        :return: Per table, the columns that are missing from it. Tables the database
            does not have at all are left out, since creating those is what
            ``create_all`` does.
        """
        reader = inspect(self.engine)
        present = set(reader.get_table_names())
        missing: Dict[Table, List[Column]] = {}
        for table in self.metadata.sorted_tables:
            if table.name not in present:
                continue
            columns = {column["name"] for column in reader.get_columns(table.name)}
            absent = [column for column in table.columns if column.name not in columns]
            if absent:
                missing[table] = absent
        return missing

    @property
    def metadata(self) -> MetaData:
        """
        :return: The tables the ORM declares.
        """
        return self.base.metadata

    @property
    def mappers(self) -> Iterable[Mapper]:
        """
        :return: The ORM's mappers, which say what a stored row already is.
        """
        return list(self.base.registry.mappers)

    def value_for_rows_written_before(
        self, table: Table, column: Column
    ) -> Optional[str]:
        """
        Say what a row written before this column existed should hold in it.

        A class discriminator is the one case where that is knowable: rows in the table
        predate the subclasses, so they are instances of the class that maps the table
        itself, and its identity is the value the ORM would have written.

        :param table: The table the column belongs to.
        :param column: The column being added.
        :return: The value to fill in, or None if there is nothing it can be.
        """
        for mapper in self.mappers:
            if mapper.local_table is not table or mapper.base_mapper is not mapper:
                continue
            discriminator = mapper.polymorphic_on
            if discriminator is not None and discriminator.name == column.name:
                return mapper.polymorphic_identity
        return None

    def align(self) -> AlignmentReport:
        """
        Add the missing columns to the database.

        :return: What was added, and what could not be added without a decision about
            the rows that are already stored.
        """
        report = AlignmentReport()
        for table, columns in self.drift().items():
            quoted = f'"{table.name}"'
            with self.engine.begin() as connection:
                stored = connection.execute(
                    text(f"select count(*) from {quoted}")
                ).scalar()
                for column in columns:
                    fill = self.value_for_rows_written_before(table, column)
                    if stored and not column.nullable and fill is None:
                        report.refused.append(
                            f"{table.name}.{column.name} is required, holds no value a "
                            f"stored row can be given, and {stored} rows are stored"
                        )
                        continue
                    kind = column.type.compile(self.engine.dialect)
                    self._add_column(connection, table, column, kind, fill)
                    filled = f", filled with {fill!r}" if fill is not None else ""
                    report.added.append(f"{table.name}.{column.name} {kind}{filled}")
        return report

    @staticmethod
    def _add_column(connection, table: Table, column: Column, kind: str, fill) -> None:
        """
        Write one column, its value for the rows that predate it, and its constraints.

        :param connection: The transaction to write in.
        :param table: The table to change.
        :param column: The column to add.
        :param kind: Its type, as the database spells it.
        :param fill: What rows written before it should hold in it, or None.
        """
        quoted = f'"{table.name}"'
        connection.execute(
            text(f'alter table {quoted} add column "{column.name}" {kind}')
        )
        if fill is not None:
            connection.execute(
                text(f'update {quoted} set "{column.name}" = :fill'), {"fill": fill}
            )
        if not column.nullable:
            connection.execute(
                text(
                    f"alter table {quoted} alter column "
                    f'"{column.name}" set not null'
                )
            )
        for reference in column.foreign_keys:
            target = reference.column
            connection.execute(
                text(
                    f'alter table {quoted} add foreign key ("{column.name}") '
                    f'references "{target.table.name}" ("{target.name}")'
                )
            )

    def align_or_raise(self) -> AlignmentReport:
        """
        Add the missing columns, refusing to report success where a gap is left.

        :return: What was added.
        :raises ColumnsCannotBeAddedError: If a column cannot be added without a
            decision about the rows that are stored.
        """
        report = self.align()
        if not report.is_aligned:
            raise ColumnsCannotBeAddedError(refusals=report.refused)
        return report
