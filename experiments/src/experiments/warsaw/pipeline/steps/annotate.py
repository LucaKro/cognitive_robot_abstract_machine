"""
Annotate a split scene's bodies and mount the parts into their wholes.

This is where everything the pipeline decided becomes a world: each body gets an
annotation of the class it was named as, and every pairing the adjudication settled is
carried out with the method the ontology says mounts it.

The mount is done with ``add()`` rather than by filling constructor fields, because a
whole holds *several* drawers and a constructor slot takes one value: ``add`` routes a
part to the field its type matches and appends where the field holds many.

The classes a scene needs that the ontology does not have are generated first, into the
run's own directory, and the ORM is rebuilt so the database knows them. That has to
happen before anything imports the classes, so the world is annotated in a second
interpreter.

Nothing here decides anything. Which body is which came from the split, what each one is
came from the classification, and which whole each part belongs to came from the
adjudication; if any of those is wrong, this writes it faithfully into the world.
"""

from __future__ import annotations

import inspect
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

import semantic_digital_twin
from semantic_digital_twin.exceptions import UsageError
from semantic_digital_twin.semantic_annotations.in_memory_builder import (
    SemanticAnnotationClassBuilder,
)
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    MountKind,
    annotation_classes,
    in_base_order,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from typing_extensions import Dict, List, Type

from experiments.warsaw.exceptions import NoWorldRecordedError
from experiments.warsaw.pipeline.records import (
    Classifications,
    SplitRecord,
    Vocabulary,
)
from experiments.warsaw.pipeline.report import RunReport
from experiments.warsaw.pipeline.reporting import Reporting
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.run_classes import GeneratedClasses
from experiments.warsaw.pipeline.run_database import RunSchema
from experiments.warsaw.pipeline.steps.step import PipelineStep
from experiments.warsaw.pipeline.world_store import WorldStore
from experiments.warsaw.scene_split import Pairing


@dataclass
class RefusedMount:
    """
    One pairing the world would not carry out, and why.
    """

    pairing: Pairing
    """
    The mount that was refused.
    """

    reason: str
    """
    What the world said about it.
    """


@dataclass
class Mounted:
    """
    What carrying out a run's pairings produced.
    """

    carried_out: int = 0
    """
    How many mounts the world accepted.
    """

    refused: List[RefusedMount] = field(default_factory=list)
    """
    The ones it would not carry out, and why.
    """

    @property
    def attempted(self) -> int:
        """
        :return: How many mounts were tried.
        """
        return self.carried_out + len(self.refused)


@dataclass
class MountAnnotations(Reporting):
    """
    The annotating and mounting half, done in an interpreter that knows the run's
    classes.

    Constructed with a run directory and nothing else: which body is which, what each one
    is and which whole each belongs to are all in the run's own files, and the world to
    annotate is the one the split recorded.
    """

    directory: Path
    """
    The run's directory.
    """

    templates: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "templates"
    )
    """
    Where the script a run leaves behind to open its world is written from.
    """

    inspector_template: str = "inspect_world.py.template"
    """
    That script, with the world ids left to fill in.
    """

    @property
    def run(self) -> Run:
        """
        :return: The run, as a directory of files.
        """
        return Run(directory=self.directory)

    def carry_out(self) -> None:
        """
        Read the split world back, annotate it, mount its parts, and write it afresh.
        """
        # Before the ORM is reached, so that the classes it names are this run's.
        GeneratedClasses(directory=self.directory).use()
        RunSchema.for_run(self.directory).use()

        split = SplitRecord.from_json(self.run.read_json(RunFile.SPLIT))
        classifications = Classifications.from_json(
            self.run.read_json(RunFile.CLASSIFICATIONS)
        )
        if split.world_id is None:
            raise NoWorldRecordedError(step="split")

        store = WorldStore()
        # The ORM was regenerated with the classes this scene needed, but a generated
        # class has no table until one is made for it, and a world holding one cannot be
        # written.
        store.create_tables()
        world = store.read(split.world_id)
        self.logger.info(
            "world %s read back, %s bodies", split.world_id, len(world.bodies)
        )

        annotations = self.annotate(world, classifications)
        mounted = self.mount(world, annotations, split.pairings)

        self.logger.info(
            "%s of %s pairings mounted", mounted.carried_out, mounted.attempted
        )
        for refusal in mounted.refused:
            self.logger.warning(
                "  %s <- %s: %s",
                refusal.pairing.whole,
                refusal.pairing.part,
                refusal.reason,
            )

        # Written as a world of its own rather than merged onto the one it was read from.
        # Merging serialises a world afresh and grafts it onto the stored graph, and what
        # came back could not be read at all: replaying its modifications reached an entity
        # that was not there. Two rows also keep the split world as it was, which is what
        # the classification and the pairings were decided against.
        split.annotated_world_id = store.write(world)
        self.logger.info(
            "written as world %s, annotated, beside the split world %s",
            split.annotated_world_id,
            split.world_id,
        )
        self.run.write_json(RunFile.SPLIT, split.to_json())

        report = RunReport(run=self.run)
        report.write_inspector(
            Template((self.templates / self.inspector_template).read_text())
        )
        report.write()
        self.logger.info(
            "written to %s and %s",
            self.run.path(RunFile.REPORT),
            self.run.path(RunFile.INSPECTOR),
        )

    def annotate(
        self, world, classifications: Classifications
    ) -> Dict[str, SemanticAnnotation]:
        """
        Give every body an annotation of the class it was named as.

        :param world: The world the bodies are in.
        :param classifications: What each body was answered to be.
        :return: The annotation made for each body, by name.
        """
        # Every subclass this interpreter holds, and deliberately: it composed nothing, so
        # nothing here is another step's invention, and the classes this run generated are
        # exactly what it has to be able to find.
        known = annotation_classes(SemanticAnnotation)
        bodies = {str(body.name.name): body for body in world.bodies}

        annotations: Dict[str, SemanticAnnotation] = {}
        left_alone: Counter = Counter()
        with world.modify_world():
            for name, answer in classifications.bodies.items():
                body = bodies.get(name)
                if body is None:
                    left_alone["no such body"] += 1
                    continue
                if answer.class_name not in known:
                    left_alone[answer.class_name] += 1
                    continue
                # An abstract class cannot be instantiated, and one answer naming one
                # should cost that one body rather than every body after it.
                if inspect.isabstract(known[answer.class_name]):
                    left_alone[f"{answer.class_name} (abstract)"] += 1
                    continue
                annotation = known[answer.class_name](root=body, _world=world)
                # Registered as it is made, before anything is mounted into it. A mount
                # records an attribute update against the annotation it changes, and a
                # world replaying its modifications has to have that annotation already:
                # registering them all afterwards puts every update before its own subject
                # and the world cannot be read back at all.
                if annotation not in world.semantic_annotations:
                    world.add_semantic_annotation(annotation)
                annotations[name] = annotation

        self.logger.info("%s bodies annotated", len(annotations))
        for missing, count in left_alone.most_common():
            self.logger.info("  %s left alone: %s", count, missing)
        return annotations

    def mount(
        self,
        world,
        annotations: Dict[str, SemanticAnnotation],
        pairings: List[Pairing],
    ) -> Mounted:
        """
        Carry out every pairing, through the channel the ontology says mounts it.

        :param world: The world the annotations are in.
        :param annotations: The annotation made for each body, by name.
        :param pairings: The mounts the adjudication settled.
        :return: What was mounted, and what was refused.
        """
        mounted = Mounted()
        # A mount moves the part's branch under the whole, so it modifies the world model
        # and has to be told so.
        with world.modify_world():
            for pairing in pairings:
                whole = annotations.get(pairing.whole)
                part = annotations.get(pairing.part)
                if whole is None or part is None:
                    mounted.refused.append(
                        RefusedMount(
                            pairing=pairing, reason="one end has no annotation"
                        )
                    )
                    continue
                try:
                    self.mount_one(whole, part, pairing)
                except UsageError as refusal:
                    # The world refusing a mount is an answer about this pairing, not a
                    # state the run should not have reached: the class a body was given
                    # may simply not admit the part another step said it holds.
                    mounted.refused.append(
                        RefusedMount(
                            pairing=pairing,
                            reason=f"{type(refusal).__name__}: "
                            f"{str(refusal).splitlines()[0]}",
                        )
                    )
                else:
                    mounted.carried_out += 1
        return mounted

    @staticmethod
    def mount_one(
        whole: SemanticAnnotation, part: SemanticAnnotation, pairing: Pairing
    ) -> None:
        """
        Mount one part into one whole, through the channel its kind names.

        :param whole: The annotation that holds.
        :param part: The annotation it holds.
        :param pairing: The mount to carry out.
        :raises UsageError: If the world will not hold the part that way.
        """
        if pairing.kind is MountKind.CONTAINS:
            whole.add_object(part)
        elif pairing.kind is MountKind.SUPPORTS:
            whole.add_supporting_surface(part)
        else:
            whole.add(part, field_name=pairing.field_name)


@dataclass
class AnnotateAndMount(PipelineStep):
    """
    The classes a scene needed, the ORM that knows them, and the world they annotate.
    """

    @property
    def name(self) -> str:
        return "annotate the bodies and mount the parts"

    @property
    def class_template(self) -> str:
        """
        :return: What a generated class is written from.
        """
        return "dataclass_template.py.jinja"

    @property
    def orm_generator(self) -> Path:
        """
        :return: The script that rebuilds the ORM.
        """
        root = Path(semantic_digital_twin.__file__).resolve().parent
        return root.parent.parent / "scripts" / "generate_orm.py"

    def carry_out(self) -> None:
        """
        Generate what the ontology lacks, rebuild the ORM, then annotate in a new one.
        """
        classifications = Classifications.from_json(
            self.run.read_json(RunFile.CLASSIFICATIONS)
        )
        vocabulary = Vocabulary.from_json(self.run.read_json(RunFile.VOCABULARY))
        wanted = self.wanted_classes(classifications, vocabulary)
        known = self.ontology_classes()
        self.logger.info(
            "%s classes over %s bodies", len(wanted), len(classifications.bodies)
        )

        generated = self.generate_missing(wanted, known)
        if generated:
            self.logger.info("generated %s: %s", len(generated), ", ".join(generated))
            self.logger.info("regenerating the ORM ...")
            self.regenerate_orm()
        else:
            self.logger.info("every class the scene needs is already in the ontology")

        # The classes were written after this process read the ontology, so the world is
        # annotated in an interpreter that starts after they exist.
        self.logger.info("annotating in a new interpreter ...")
        printed = self.in_new_interpreter(
            "import sys\n"
            "from pathlib import Path\n"
            "import logging\n"
            "logging.basicConfig(level=logging.INFO, format='%(message)s', "
            "stream=sys.stdout)\n"
            "from experiments.warsaw.pipeline.steps.annotate import MountAnnotations\n"
            "MountAnnotations(directory=Path(sys.argv[1])).carry_out()\n",
            [str(self.run.directory)],
            what="annotating the world and mounting its parts",
        )
        for line in printed.splitlines():
            self.logger.info(line)

    @staticmethod
    def wanted_classes(
        classifications: Classifications, vocabulary: Vocabulary
    ) -> Dict[str, List[str]]:
        """
        Say what each class a body was given should be built from.

        Two steps proposed compositions and only one of them was asked to. The
        vocabulary step answers what a *label* means and names a superclass and the
        mixins to compose it from, having been shown what objects of that label were
        measured to meet; the classification step answers which class each *object* is,
        from a picture, and its schema carries a superclass as well. Read from the
        classification alone, a Faucet that the vocabulary had composed with HasHandle
        comes out with no way to hold a handle at all, and the pairing measured for it
        cannot be mounted.

        So the bases come from the vocabulary where it composed that class, and from the
        classification only where it did not.

        :param classifications: What the classification step wrote.
        :param vocabulary: What the vocabulary step wrote.
        :return: Per class name, the names of the classes to derive it from.
        """
        composed = {
            answer.class_name: [answer.superclass] + list(answer.mixins)
            for answer in vocabulary.labels.values()
            if answer.class_name and answer.is_new_class and answer.superclass
        }

        wanted: Dict[str, List[str]] = {}
        for answer in classifications.bodies.values():
            if not answer.class_name or answer.class_name in wanted:
                continue
            wanted[answer.class_name] = composed.get(
                answer.class_name, [answer.superclass or SemanticAnnotation.__name__]
            )
        return wanted

    def generate_missing(
        self, wanted: Dict[str, List[str]], known: Dict[str, Type]
    ) -> List[str]:
        """
        Write the classes a scene needs that the ontology does not have.

        The file is written whole rather than appended to: a class one scene needed is not
        a class the next one starts with, and nothing outside this run should ever import
        it.

        :param wanted: Per class name, the names of the classes to derive it from.
        :param known: The ontology's classes by name.
        :return: The names that were generated, with what each was built from.
        """
        builders, generated = [], []
        for name, base_names in sorted(wanted.items()):
            if name in known:
                continue
            bases = [known[one] for one in base_names if one in known]
            for missing in [one for one in base_names if one not in known]:
                self.logger.warning(
                    "  %s: %s is not in the ontology, leaving it out", name, missing
                )
            if not bases:
                bases = [SemanticAnnotation]

            # Ordered as a class must declare them, since a proposal naming HasRootBody
            # beside IsStorageSpace -- which derives from it -- names them the wrong way
            # round for Python.
            ordered = in_base_order(bases)
            builder = SemanticAnnotationClassBuilder(
                name, template_name=self.class_template
            )
            for base in ordered:
                builder.add_base(base)
            builders.append(builder)
            generated.append(f"{name}({', '.join(base.__name__ for base in ordered)})")

        if builders:
            generated_classes = GeneratedClasses(directory=self.run.directory)
            generated_classes.searched_directory.mkdir(parents=True, exist_ok=True)
            SemanticAnnotationClassBuilder.write_classes_to_file(
                builders, generated_classes.path
            )
        return generated

    def regenerate_orm(self) -> None:
        """
        Rebuild the ORM so the database knows the classes this run generated.

        Run in a new interpreter with the run's directory at the front of the
        annotations package's search path: the generator finds classes by walking that
        path, so the run's file is the one it maps, without the generator being told
        anything and without the classes ever being written into the ontology's own
        package.

        :raises SubprocessStepFailedError: If the rebuild fails.
        """
        self.in_new_interpreter(
            "import importlib.util, sys\n"
            "from pathlib import Path\n"
            "from experiments.warsaw.pipeline.run_classes import GeneratedClasses\n"
            "GeneratedClasses(directory=Path(sys.argv[1])).use()\n"
            "specification = importlib.util.spec_from_file_location("
            "'generate_orm', sys.argv[2])\n"
            "generator = importlib.util.module_from_spec(specification)\n"
            "specification.loader.exec_module(generator)\n"
            "generator.generate_orm()\n",
            [str(self.run.directory.resolve()), str(self.orm_generator)],
            what="rebuilding the ORM with the run's generated classes",
        )
