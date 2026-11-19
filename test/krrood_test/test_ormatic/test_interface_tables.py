"""
An interface that fields are typed with and no alternative mapping covers gets a table
of its own without columns, and every implementation links its rows to a row of it.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, inspect, select

from krrood.ormatic.data_access_objects.dao import InterfaceDataAccessObject
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.exceptions import InterfaceNotLinkedError
from ..dataset.interface_tables import (
    FrameChain,
    FramedJoint,
    FramedOffset,
    FramedPart,
    LandmarkFrame,
    StationFrame,
    TaggedRoute,
    TrackedPart,
)

# %% generated schema


def test_interface_table_holds_only_its_key_and_the_implementer_table():
    """
    The interface's table has no column for anything the interface declares.
    """
    from ..dataset.ormatic_interface import ReferenceFrameDAO

    table = ReferenceFrameDAO.__table__

    assert issubclass(ReferenceFrameDAO, InterfaceDataAccessObject)
    assert set(table.columns) == {
        table.c.database_id,
        table.c.implementer_table_name,
    }


def test_interface_typed_fields_relate_to_the_interface_table():
    """
    A field typed with the interface, alone or in a collection, refers to the
    interface's table.
    """
    from ..dataset.ormatic_interface import (
        FrameChainDAO,
        FramedOffsetDAO,
        ReferenceFrameDAO,
    )

    offset_relationships = inspect(FramedOffsetDAO).relationships
    frames_association = inspect(FrameChainDAO).relationships["frames"].mapper

    assert offset_relationships["reference_frame"].mapper.class_ is ReferenceFrameDAO
    assert frames_association.relationships["target"].mapper.class_ is ReferenceFrameDAO


def test_implementations_keep_their_own_parent_tables():
    """
    An implementation inherits from the table of its stored ancestor, never from the
    interface's table.
    """
    from ..dataset.ormatic_interface import (
        FramedJointDAO,
        FramedPartDAO,
        LandmarkFrameDAO,
        ReferenceFrameDAO,
        TrackedPartDAO,
    )

    assert issubclass(FramedPartDAO, TrackedPartDAO)
    assert issubclass(FramedJointDAO, FramedPartDAO)
    assert not issubclass(LandmarkFrameDAO, ReferenceFrameDAO)
    assert not issubclass(FramedPartDAO, ReferenceFrameDAO)


def test_the_link_lives_on_the_first_implementing_table():
    """
    The table of the first implementing class holds the link, and the tables of its
    subclasses inherit it.
    """
    from ..dataset.ormatic_interface import (
        FramedJointDAO,
        FramedPartDAO,
        ReferenceFrameDAO,
        TrackedPartDAO,
    )

    def tables_referencing_the_interface(dao_class: type) -> set[str]:
        """
        :param dao_class: A DAO class.
        :return: The names of the tables of the class that refer to the interface's.
        """
        return {
            foreign_key.parent.table.name
            for column in inspect(dao_class).columns
            for foreign_key in column.foreign_keys
            if foreign_key.column.table is ReferenceFrameDAO.__table__
        }

    assert tables_referencing_the_interface(TrackedPartDAO) == set()
    assert tables_referencing_the_interface(FramedPartDAO) == {
        FramedPartDAO.__tablename__
    }
    assert tables_referencing_the_interface(FramedJointDAO) == {
        FramedPartDAO.__tablename__
    }


def test_a_generic_interface_gets_an_interface_table():
    """
    An interface with a generic base gets a table without columns of its own, although
    its generic base makes it look like a dataclass.
    """
    from ..dataset.ormatic_interface import StationFrameDAO, TaggedFrameDAO

    assert issubclass(TaggedFrameDAO, InterfaceDataAccessObject)
    assert not issubclass(StationFrameDAO, TaggedFrameDAO)


# %% round trip


def test_a_field_typed_with_the_interface_round_trips(session, database):
    """
    An implementation without stored ancestors comes back through a field typed with the
    interface.
    """
    from ..dataset.ormatic_interface import FramedOffsetDAO

    offset = FramedOffset(
        distance=2.5, reference_frame=LandmarkFrame(landmark_name="door")
    )

    session.add(to_dao(offset))
    session.commit()
    session.expunge_all()

    reconstructed = session.scalars(select(FramedOffsetDAO)).one().from_dao()

    assert reconstructed == offset


def test_an_implementation_with_its_own_parents_round_trips(session, database):
    """
    An implementation that keeps a parent chain of its own comes back as itself through
    a field typed with the interface.
    """
    from ..dataset.ormatic_interface import FramedOffsetDAO

    offset = FramedOffset(
        distance=0.1,
        reference_frame=FramedJoint(part_number="J-7", stiffness=40.0),
    )

    session.add(to_dao(offset))
    session.commit()
    session.expunge_all()

    reconstructed = session.scalars(select(FramedOffsetDAO)).one().from_dao()

    assert reconstructed == offset


def test_an_empty_field_typed_with_the_interface_round_trips(session, database):
    """
    A field typed with the interface that holds nothing comes back empty.
    """
    from ..dataset.ormatic_interface import FramedOffsetDAO

    offset = FramedOffset(distance=1.0)

    session.add(to_dao(offset))
    session.commit()
    session.expunge_all()

    reconstructed = session.scalars(select(FramedOffsetDAO)).one().from_dao()

    assert reconstructed == offset


def test_a_list_typed_with_the_interface_keeps_shared_objects(session, database):
    """
    A list typed with the interface comes back with its implementations in order, and an
    object listed twice comes back as one object.
    """
    from ..dataset.ormatic_interface import FrameChainDAO

    landmark = LandmarkFrame(landmark_name="window")
    chain = FrameChain(
        frames=[landmark, FramedPart(part_number="P-1"), landmark],
    )

    session.add(to_dao(chain))
    session.commit()
    session.expunge_all()

    reconstructed = session.scalars(select(FrameChainDAO)).one().from_dao()

    assert reconstructed == chain
    assert reconstructed.frames[0] is reconstructed.frames[2]


# %% interface rows


def test_each_implementation_gets_one_interface_row(session, database):
    """
    Storing implementations gives each object exactly one row of the interface's table,
    however often fields refer to it.
    """
    from ..dataset.ormatic_interface import ReferenceFrameDAO

    landmark = LandmarkFrame(landmark_name="window")
    part = FramedPart(part_number="P-1")
    chain = FrameChain(frames=[landmark, part, landmark])

    session.add(to_dao(chain))
    session.commit()

    assert session.scalar(select(func.count()).select_from(ReferenceFrameDAO)) == len(
        {id(frame) for frame in chain.frames}
    )


def test_an_interface_row_loads_its_implementation(session, database):
    """
    An implementation stored on its own gets an interface row, and loading that row
    gives back the implementation.
    """
    from ..dataset.ormatic_interface import ReferenceFrameDAO

    joint = FramedJoint(part_number="J-2", stiffness=12.0)

    session.add(to_dao(joint))
    session.commit()
    session.expunge_all()

    reconstructed = session.scalars(select(ReferenceFrameDAO)).one().from_dao()

    assert reconstructed == joint


def test_storing_an_object_that_does_not_implement_the_interface_fails():
    """
    A mapped object in a field typed with an interface its class does not implement has
    no row in the interface's table to refer to.
    """
    offset = FramedOffset(distance=1.0, reference_frame=TrackedPart(part_number="T-3"))

    with pytest.raises(InterfaceNotLinkedError):
        to_dao(offset)


def test_a_field_typed_with_a_generic_interface_round_trips(session, database):
    """
    An implementation of a generic interface comes back through a field typed with it.
    """
    from ..dataset.ormatic_interface import TaggedRouteDAO

    route = TaggedRoute(start=StationFrame(station_code="HB"))

    session.add(to_dao(route))
    session.commit()
    session.expunge_all()

    reconstructed = session.scalars(select(TaggedRouteDAO)).one().from_dao()

    assert reconstructed == route
