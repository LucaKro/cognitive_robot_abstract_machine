"""
Keeping one run's conclusions out of every other run's.

Two runs of the same scene may disagree, and a later one silently inheriting half of an
earlier one's answers would be neither of them. Three things enforce that: a directory
of its own, a database schema of its own, and generated classes that belong to the run
that proposed them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from experiments.warsaw.exceptions import (
    DatabaseNotConfiguredError,
    GeneratedClassesAlreadyImportedError,
    RunClassTakenOverByTheOntologyError,
    RunOutputAlreadyWrittenError,
)
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.run_classes import GeneratedClasses
from experiments.warsaw.pipeline.database.run_schema import RunSchema

# %% the directory a run writes into


def test_a_run_is_named_for_when_it_started(tmp_path):
    """
    Runs sort by age and none can be mistaken for another.
    """
    run = Run.create(tmp_path, name_format="%Y-%m-%d")
    assert run.directory.parent == tmp_path
    assert run.directory.exists()


def test_two_runs_do_not_share_a_directory(tmp_path):
    """
    A second run made in the same moment would write over the first one's answers.
    """
    Run.create(tmp_path, name_format="fixed")
    with pytest.raises(FileExistsError):
        Run.create(tmp_path, name_format="fixed")


def test_every_file_a_run_writes_is_inside_it(tmp_path):
    """
    A run reads nothing outside its own directory, the ontology it read out included.
    """
    run = Run.create(tmp_path)
    for run_file in RunFile:
        assert run.path(run_file).parent == run.directory


def test_writing_over_a_step_s_own_output_is_refused(tmp_path):
    """
    Writing beside it is how a run comes to be part one run and part another: the files
    a step does not happen to rewrite stay as an earlier pass left them, and nothing
    says so.
    """
    run = Run.create(tmp_path)
    run.write_json(RunFile.SPLIT, {"scene": "somewhere"})
    with pytest.raises(RunOutputAlreadyWrittenError) as raised:
        run.refuse_to_write_over(RunFile.SPLIT)
    assert raised.value.written == [RunFile.SPLIT.value]


def test_writing_beside_what_an_earlier_step_wrote_is_not_refused(tmp_path):
    """
    A run's directory holds the ontology it read out before its first step has written
    anything, and every step after that writes beside what the ones before it wrote.
    """
    run = Run.create(tmp_path)
    run.write_json(RunFile.TAXONOMY, {"classes": []})
    run.refuse_to_write_over(RunFile.RELATIONS, RunFile.QUESTIONS)


def test_writing_over_output_is_allowed_when_it_is_asked_for(tmp_path):
    """
    A step run twice in one run, the second time knowing something the first did not,
    writes what it wrote again rather than adding to it.
    """
    run = Run.create(tmp_path)
    run.write_json(RunFile.SPLIT, {"scene": "somewhere"})
    run.refuse_to_write_over(RunFile.SPLIT, overwrite=True)


def test_a_file_a_run_never_wrote_reads_as_nothing(tmp_path):
    """
    A run that stopped early is still worth reporting on.
    """
    run = Run.create(tmp_path)
    assert not run.holds(RunFile.SPLIT)
    assert run.read_json_if_written(RunFile.SPLIT) == {}


# %% the schema a run writes into


def test_a_schema_is_named_for_the_run(tmp_path):
    """
    The schema and the directory can be read off one another without anything recording
    the pairing.
    """
    schema = RunSchema.for_run(Path("/anywhere/2026-09-05_112539"))
    assert schema.name == "run_2026_09_05_112539"


def test_a_connection_looks_in_the_run_s_schema_alone():
    """
    With ``public`` in the search path the ORM would find the tables standing there and
    build none of its own, which is the inheritance the schema exists to stop.

    One run generated ``Ceiling(HasRootRegion)`` and the next ``Ceiling(HasRootBody)``,
    and the second failed on the first one's table.
    """
    schema = RunSchema(name="run_x", uri="postgresql+psycopg://user:secret@host/twin")
    assert "options=-csearch_path%3Drun_x" in schema.schema_uri
    assert "public" not in schema.schema_uri


def test_the_rest_of_the_connection_is_left_as_it_was():
    """
    Only the search path is the run's business; the database it reaches is not.
    """
    schema = RunSchema(name="run_x", uri="postgresql+psycopg://user:secret@host/twin")
    assert schema.schema_uri.startswith("postgresql+psycopg://user:secret@host/twin?")


def test_a_step_started_by_a_run_is_pointed_at_the_run_s_schema(monkeypatch):
    """
    Every step is this process or is started by it, so pointing this one at the schema
    points all of them at it and none has to be told which.
    """
    schema = RunSchema(name="run_x", uri="postgresql+psycopg://host/twin")
    monkeypatch.delenv(schema.variable, raising=False)
    assert schema.environment()[schema.variable] == schema.schema_uri


# %% the classes a run generates


def test_a_run_that_generated_no_classes_points_at_none(tmp_path):
    """
    Most scenes need a class the ontology lacks; a scene that does not needs nothing
    done.
    """
    assert GeneratedClasses(directory=tmp_path).use() is None


def test_a_run_s_classes_are_looked_for_inside_it(tmp_path):
    """
    Written into the ontology's own package they would be one careless commit away from
    becoming part of the shared ontology.
    """
    generated = GeneratedClasses(directory=tmp_path)
    assert generated.path.parent == generated.searched_directory
    assert generated.searched_directory.parent == tmp_path.resolve()
    assert not generated.were_generated
    generated.searched_directory.mkdir()
    generated.path.write_text("")
    assert generated.were_generated


def test_only_the_generated_classes_are_put_on_the_annotations_path(tmp_path):
    """
    What is put on that path becomes part of the annotations package, so a run's own.

    directory would offer the inspector script it leaves behind as an annotation module
    -- which the ORM generator then tries to map, and a run cannot be annotated twice.
    """
    generated = GeneratedClasses(directory=tmp_path)
    generated.searched_directory.mkdir()
    generated.path.write_text("")
    # Only its presence matters: the search must not reach outside its own directory.
    (tmp_path / "inspect_world.py").write_text("not read by the search\n")

    assert generated.searched_directory != tmp_path.resolve()
    assert [one.name for one in generated.searched_directory.iterdir()] == [
        generated.file_name
    ]


def test_classes_already_imported_from_elsewhere_are_refused(tmp_path, monkeypatch):
    """
    By then everything holding one of those classes holds the wrong one, so pointing at
    the run's file now would leave two versions of the same class in one interpreter.
    """
    generated = GeneratedClasses(directory=tmp_path)
    generated.searched_directory.mkdir()
    generated.path.write_text("")

    class ImportedFromSomewhereElse:
        __file__ = str(tmp_path / "somewhere_else.py")

    monkeypatch.setitem(sys.modules, generated.module_name, ImportedFromSomewhereElse)
    with pytest.raises(GeneratedClassesAlreadyImportedError) as raised:
        generated.use()
    assert raised.value.wanted == generated.path


def test_pointing_twice_at_the_same_run_is_not_a_conflict(tmp_path, monkeypatch):
    """
    Several steps of one run point at its classes, and the second must not undo the
    first.
    """
    generated = GeneratedClasses(directory=tmp_path)
    generated.searched_directory.mkdir()
    generated.path.write_text("")

    class AlreadyThisRun:
        __file__ = str(generated.path)

    monkeypatch.setitem(sys.modules, generated.module_name, AlreadyThisRun)
    assert generated.use() is AlreadyThisRun


# %% reaching the worlds a run writes


def test_a_world_can_be_converted_only_once_the_mappings_are_loaded():
    """
    Which class stores a world is registered by importing the generated interface, and a
    step that converts a world without having imported it is told the world is unmapped::

        NoDAOFoundError: Class <class 'semantic_digital_twin.world.World'> does not
        have a DAO.

    The interface is named nowhere else, because the pipeline rewrites the ORM twice
    during a run and a module holding the version from before that hands out the wrong
    classes.
    """
    from krrood.ormatic.data_access_objects.helper import get_dao_class
    from semantic_digital_twin.world import World

    from experiments.warsaw.pipeline.database.world_store import WorldStore

    WorldStore().mappings()
    assert get_dao_class(World) is not None


def test_a_schema_with_no_uri_and_no_environment_is_reported(monkeypatch):
    """
    Neither the run nor the environment saying where the database is is a configuration
    error, not a schema pointing nowhere.
    """
    schema = RunSchema(name="run_x")
    monkeypatch.delenv(schema.variable, raising=False)
    with pytest.raises(DatabaseNotConfiguredError):
        schema.base_uri


def test_the_script_that_rebuilds_the_orm_is_where_a_run_looks_for_it(tmp_path):
    """
    The rebuild happens in a new interpreter, so a moved or renamed generator would fail
    there rather than here, in output nobody reads until a run is already expensive.
    """
    assert GeneratedClasses(directory=tmp_path).orm_generator.is_file()


def test_a_run_that_generated_no_classes_still_names_the_generator(tmp_path):
    """
    Where the generator lives does not depend on what a run happened to generate.
    """
    generated = GeneratedClasses(directory=tmp_path)

    assert not generated.were_generated
    assert generated.orm_generator.name == "generate_orm.py"


def test_the_classes_a_run_generated_are_read_without_importing_them(tmp_path):
    """
    Asking what a run generated must not put those classes into this interpreter, which
    is what deciding whether they can be used has to happen before.
    """
    generated = GeneratedClasses(directory=tmp_path)
    generated.searched_directory.mkdir(parents=True)
    generated.path.write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(eq=False)\nclass Ceiling: ...\n\n\n"
        "@dataclass(eq=False)\nclass Faucet: ...\n"
    )

    assert generated.class_names == ["Ceiling", "Faucet"]
    assert sys.modules.get(generated.module_name) is None


def test_a_run_that_generated_nothing_names_no_classes(tmp_path):
    """
    A scene needing nothing the ontology lacks generates nothing to be asked about.
    """
    assert GeneratedClasses(directory=tmp_path).class_names == []


def test_a_class_the_ontology_has_since_gained_is_reported(tmp_path):
    """
    Two classes of one name are two tables of one name, and an ORM holding both cannot
    be imported at all -- so rebuilding for such a run would leave every other run
    unable to read anything.
    """
    generated = GeneratedClasses(directory=tmp_path)
    generated.searched_directory.mkdir(parents=True)
    generated.path.write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(eq=False)\nclass KitchenIsland: ...\n\n\n"
        "@dataclass(eq=False)\nclass NothingIsCalledThis: ...\n"
    )

    assert generated.taken_over_by_the_ontology() == ["KitchenIsland"]


def test_rebuilding_for_a_taken_over_class_is_refused_before_anything_is_touched(
    tmp_path,
):
    """
    Refused rather than attempted, because the damage is to the one interface every run
    shares and undoing it means rebuilding from a repository nobody has broken yet.
    """
    generated = GeneratedClasses(directory=tmp_path)
    generated.searched_directory.mkdir(parents=True)
    generated.path.write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(eq=False)\nclass KitchenIsland: ...\n"
    )
    written_before = generated.interface.read_text()

    with pytest.raises(RunClassTakenOverByTheOntologyError) as raised:
        generated.rebuild_orm()

    assert "KitchenIsland" in str(raised.value)
    assert generated.interface.read_text() == written_before
