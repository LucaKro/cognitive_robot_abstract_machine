"""
Give each run its own corner of the database, the way it has its own directory.

A run writes two worlds and, for the classes it had to invent, tables of its own. Those
tables outlive the run: ``create_all`` creates what is absent and leaves alone whatever is
already standing, however little it now resembles the class it is meant to hold. So the
run after this one inherits them. One run generated ``Ceiling(HasRootRegion)`` and the
next ``Ceiling(HasRootBody)``, and the second failed on the first one's table::

    insert or update on table "CeilingDAO" violates foreign key constraint
    DETAIL: Key (database_id)=(7046) is not present in table "HasRootRegionDAO".

A run reads nothing another run concluded, and this is the one place that was not true of.
Each run gets a schema named for it, everything it writes goes there, and the tables are
built by the same ORM that is about to write to them -- so there is nothing for a run to
inherit and nothing to drift. Throwing a run away is :meth:`RunSchema.drop`, beside
deleting its directory.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from typing_extensions import Optional

from experiments.warsaw.exceptions import DatabaseNotConfiguredError


@dataclass
class RunSchema:
    """
    The schema of the database one run writes into.
    """

    name: str
    """
    What the schema is called.
    """

    variable: str = "SEMANTIC_DIGITAL_TWIN_DATABASE_URI"
    """
    The environment variable the semantic digital twin takes its database from.

    A step is pointed at its run's schema by setting this before it starts, which is why
    every step reaches the right one without being told: they are all children of the run.
    """

    uri: Optional[str] = None
    """
    The database to reach, or None to read it from the environment when it is needed.
    """

    @classmethod
    def for_run(
        cls,
        directory: Path,
        prefix: str = "run_",
        unusable: str = r"[^0-9a-zA-Z]",
        variable: str = "SEMANTIC_DIGITAL_TWIN_DATABASE_URI",
    ) -> RunSchema:
        """
        Name the schema a run writes into.

        Derived from the run's directory so the two can be read off one another without
        anything recording the pairing.

        :param directory: The run's directory.
        :param prefix: What a run's schema name begins with.
        :param unusable: What a schema name may not hold, which is replaced.
        :param variable: The environment variable the database is taken from.
        :return: The schema, not yet made.
        """
        return cls(
            name=prefix + re.sub(unusable, "_", Path(directory).name), variable=variable
        )

    @property
    def base_uri(self) -> str:
        """
        :return: The database to reach, as it was given or as the environment sets it.
        :raises DatabaseNotConfiguredError: If neither says.
        """
        if self.uri is not None:
            return self.uri
        configured = os.environ.get(self.variable)
        if configured is None:
            raise DatabaseNotConfiguredError(variable=self.variable)
        return configured

    @property
    def schema_uri(self) -> str:
        """
        Point the database URI at this schema and nothing else.

        The search path holds the run's schema alone, without ``public`` behind it. With
        ``public`` in the path the ORM would find the tables standing there and build
        none of its own, which is the inheritance this is here to stop.

        :return: The URI, asking for that search path.
        """
        url = make_url(self.base_uri)
        query = dict(url.query)
        query["options"] = f"-csearch_path={self.name}"
        return url.set(query=query).render_as_string(hide_password=False)

    def create(self) -> None:
        """
        Make the schema, if it is not already there.
        """
        engine = create_engine(self.base_uri)
        with engine.begin() as connection:
            connection.execute(text(f'create schema if not exists "{self.name}"'))
        engine.dispose()

    def drop(self, at_a_time: int = 100) -> int:
        """
        Throw away everything the run wrote to the database.

        Not ``DROP SCHEMA ... CASCADE``: a run's schema holds the whole ORM, and taking a
        lock on eleven hundred tables in one transaction runs the server out of the shared
        memory it keeps locks in::

            psycopg.errors.OutOfMemory: out of shared memory
            HINT:  You might need to increase max_locks_per_transaction.

        So the tables go in batches, each its own transaction, and the empty schema last.

        :param at_a_time: How many tables to drop per transaction.
        :return: How many tables were dropped.
        """
        engine = create_engine(self.base_uri)
        listing = text(
            "select tablename from pg_tables where schemaname = :schema "
            "order by tablename"
        )
        dropped = 0
        try:
            while True:
                with engine.begin() as connection:
                    names = [
                        row[0]
                        for row in connection.execute(listing, {"schema": self.name})
                    ][:at_a_time]
                    if not names:
                        break
                    targets = ", ".join(f'"{self.name}"."{name}"' for name in names)
                    connection.execute(text(f"drop table {targets} cascade"))
                    dropped += len(names)
            with engine.begin() as connection:
                connection.execute(text(f'drop schema if exists "{self.name}" cascade'))
        finally:
            engine.dispose()
        return dropped

    def use(self) -> None:
        """
        Point this process, and everything it starts, at the schema.

        Set before anything asks the semantic digital twin for a session, which is what
        reads it.
        """
        os.environ[self.variable] = self.schema_uri

    def environment(self) -> dict:
        """
        :return: The environment a new interpreter needs to write into this schema.
        """
        return {**os.environ, self.variable: self.schema_uri}
