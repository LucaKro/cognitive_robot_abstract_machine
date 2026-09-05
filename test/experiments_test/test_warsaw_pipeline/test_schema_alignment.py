"""
Adding the columns a regenerated ORM asks for to a database that already holds worlds.

``create_all`` creates tables that are absent and leaves alone every table that is
already there, however far it has drifted from the class it is meant to hold. The day
the ontology gains an attribute, every world written after that names a column the
database has never heard of, and the write fails on the first insert.

The drift is additive, so it can be closed by adding columns, which costs no stored
world. What cannot be guessed at is what a row written before a required column existed
should hold in it.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Column, Integer, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase

from experiments.warsaw.exceptions import ColumnsCannotBeAddedError
from experiments.warsaw.pipeline.schema_alignment import SchemaAlignment


class StoredThing(DeclarativeBase):
    """
    A declarative base standing for the one the ORM generates.
    """


class ColouredShape(StoredThing):
    """
    A stored class that has since grown a discriminator and an optional attribute.

    Named for the shape it exercises: a table written before its class had subclasses,
    which the ORM now maps polymorphically.
    """

    __tablename__ = "ColouredShape"

    database_id = Column(Integer, primary_key=True)
    """
    What the row is addressed by.
    """

    polymorphic_type = Column(String)
    """
    Which subclass the row is, which rows written before the subclasses existed do not
    carry.
    """

    named_after = Column(String, nullable=True)
    """
    An attribute the class gained, which a stored row may simply not have.
    """

    __mapper_args__ = {
        "polymorphic_on": polymorphic_type,
        "polymorphic_identity": "ColouredShape",
    }


@pytest.fixture
def standing_database(tmp_path):
    """
    :return: A database holding one row written before the ORM grew its later columns.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'standing.db'}")
    with engine.begin() as connection:
        connection.execute(
            text('create table "ColouredShape" (database_id integer primary key)')
        )
        connection.execute(text('insert into "ColouredShape" (database_id) values (1)'))
    return engine


# %% what the ORM asks for and the database does not have


def test_the_columns_the_orm_has_grown_are_found(standing_database):
    """
    A column the ORM declares and the table lacks is what the next write fails on.
    """
    alignment = SchemaAlignment(engine=standing_database, base=StoredThing)
    missing = {
        column.name for columns in alignment.drift().values() for column in columns
    }
    assert missing == {"polymorphic_type", "named_after"}


def test_a_table_the_database_does_not_have_at_all_is_left_alone(standing_database):
    """
    Creating those is what ``create_all`` does, and doing it here would do it twice.
    """
    alignment = SchemaAlignment(engine=standing_database, base=StoredThing)
    assert {table.name for table in alignment.drift()} == {"ColouredShape"}


# %% what a row written before a column existed holds in it


def test_a_discriminator_is_filled_with_the_class_that_was_the_only_one(
    standing_database,
):
    """
    Rows in the table predate the subclasses, so they are instances of the class that
    maps the table itself, and its identity is the value the ORM would have written.
    """
    alignment = SchemaAlignment(engine=standing_database, base=StoredThing)
    table = ColouredShape.__table__
    assert (
        alignment.value_for_rows_written_before(
            table, table.columns["polymorphic_type"]
        )
        == "ColouredShape"
    )


def test_an_ordinary_column_holds_nothing_a_stored_row_can_be_given(standing_database):
    """
    What a world written last year should say about an attribute invented since is a
    question about that world, not something a migration answers.
    """
    alignment = SchemaAlignment(engine=standing_database, base=StoredThing)
    table = ColouredShape.__table__
    assert (
        alignment.value_for_rows_written_before(table, table.columns["named_after"])
        is None
    )


# %% adding them


def test_adding_the_columns_leaves_the_stored_rows_where_they_were(standing_database):
    """
    The whole point of closing the drift by adding is that it costs no stored world.
    """
    report = SchemaAlignment(engine=standing_database, base=StoredThing).align()
    assert report.is_aligned
    with standing_database.begin() as connection:
        stored = connection.execute(
            text(
                'select database_id, polymorphic_type, named_after from "ColouredShape"'
            )
        ).all()
    assert stored == [(1, "ColouredShape", None)]


def test_a_database_that_already_has_every_column_is_left_alone(standing_database):
    """
    Aligning twice adds nothing the second time.
    """
    alignment = SchemaAlignment(engine=standing_database, base=StoredThing)
    alignment.align()
    assert alignment.align().added == []


def test_a_column_that_cannot_be_added_is_reported_rather_than_guessed_at(
    standing_database,
):
    """
    A required column with no value a stored row could have been written with is a
    decision about those worlds rather than a migration.
    """

    class RequiredSince(StoredThing):
        """
        A stored class that has since gained a required attribute.
        """

        __tablename__ = "RequiredSince"

        database_id = Column(Integer, primary_key=True)
        """
        What the row is addressed by.
        """

        must_be_said = Column(String, nullable=False)
        """
        An attribute that was made required after rows were already stored.
        """

    with standing_database.begin() as connection:
        connection.execute(
            text('create table "RequiredSince" (database_id integer primary key)')
        )
        connection.execute(text('insert into "RequiredSince" (database_id) values (1)'))

    alignment = SchemaAlignment(engine=standing_database, base=StoredThing)
    report = alignment.align()
    assert not report.is_aligned
    assert any("RequiredSince.must_be_said" in one for one in report.refused)

    with pytest.raises(ColumnsCannotBeAddedError):
        SchemaAlignment(engine=standing_database, base=StoredThing).align_or_raise()
