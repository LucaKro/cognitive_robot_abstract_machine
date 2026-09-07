"""
Say what a run made, from what its steps wrote.

A run's numbers are spread over six files and a terminal that has scrolled away, and the
question asked of a run afterwards is usually how much of it went through rather than
what any one step said.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from typing_extensions import Any, Dict, List

from experiments.warsaw.pipeline.records import (
    Adjudications,
    Classifications,
    Relations,
    SplitRecord,
    Vocabulary,
)
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.templates import PipelineTemplates

# %% what the report counts


@dataclass
class ClassCount:
    """
    How many of a run's bodies were given one class.
    """

    class_name: str
    """
    The class they were given.
    """

    count: int
    """
    How many bodies were given it.
    """


# %% the report itself


@dataclass
class RunReport:
    """
    Everything a run wrote, gathered into one page.
    """

    run: Run
    """
    The run to read.
    """

    unknown_model: str = "?"
    """
    What to say where a step recorded no model.
    """

    unknown_scene: str = "unknown"
    """
    What to say where no step recorded the scene.
    """

    templates: PipelineTemplates = field(default_factory=PipelineTemplates)
    """
    Where the report and the inspector are written from.
    """

    report_template: str = "report.md.jinja"
    """
    The report, with the run's numbers left to fill in.
    """

    inspector_template: str = "inspect_world.py.jinja"
    """
    The script that opens the run's world, with the world ids left to fill in.
    """

    # %% what each step wrote

    @property
    def relations(self) -> Relations:
        """
        :return: How the scene's objects were measured to meet.
        """
        return self.run.read_record_if_written(RunFile.RELATIONS, Relations(scene=""))

    @property
    def vocabulary(self) -> Vocabulary:
        """
        :return: What every label was answered to mean.
        """
        return self.run.read_record_if_written(
            RunFile.VOCABULARY, Vocabulary(model="", scene="")
        )

    @property
    def adjudications(self) -> Adjudications:
        """
        :return: What every open question was answered with.
        """
        return self.run.read_record_if_written(
            RunFile.ADJUDICATIONS, Adjudications(model="", scene="")
        )

    @property
    def split(self) -> SplitRecord:
        """
        :return: What the split built and what it cost.
        """
        return self.run.read_record_if_written(RunFile.SPLIT, SplitRecord(scene=""))

    @property
    def classifications(self) -> Classifications:
        """
        :return: What every body was answered to be.
        """
        return self.run.read_record_if_written(
            RunFile.CLASSIFICATIONS, Classifications(model="", scene="")
        )

    # %% writing it out

    def write(self) -> str:
        """
        Write the report into the run.

        :return: The report, as Markdown.
        """
        report = self.markdown()
        self.run.path(RunFile.REPORT).write_text(report)
        return report

    def write_inspector(self) -> None:
        """
        Leave behind the script that opens the run's world without knowing anything.
        """
        split = self.split
        self.run.path(RunFile.INSPECTOR).write_text(
            self.templates.render(
                self.inspector_template,
                annotated=split.annotated_world_id,
                split=split.world_id,
            )
        )

    def markdown(self) -> str:
        """
        :return: The report, as Markdown.
        """
        return self.templates.render(self.report_template, **self.context())

    # %% counting it up

    def context(self) -> Dict[str, Any]:
        """
        Count everything the report says, so the template only lays it out.

        :return: Every number and name the report is written from.
        """
        relations = self.relations
        vocabulary, adjudications = self.vocabulary, self.adjudications
        split, classifications = self.split, self.classifications
        answered = adjudications.answered
        bodies = classifications.bodies
        return {
            "run_name": self.run.directory.name,
            "scene": relations.scene or self.unknown_scene,
            "annotated_world": split.annotated_world_id,
            "split_world": split.world_id,
            "vocabulary_model": vocabulary.model or self.unknown_model,
            "adjudication_model": adjudications.model or self.unknown_model,
            "classification_model": classifications.model or self.unknown_model,
            "segments": len(relations.segments),
            "pairs": len(relations.pairs),
            "overlapping": sum(
                1 for pair in relations.pairs if pair.evidence.shared_faces
            ),
            "settled": len(relations.settled),
            "forced": len(relations.forced),
            "labels": len(vocabulary.labels),
            "labels_mapped": sum(1 for one in vocabulary.labels if one.class_name),
            "labels_new": sum(1 for one in vocabulary.labels if one.is_new_class),
            "patterns_adjudicated": len(adjudications.ownership),
            "memberships_adjudicated": len(adjudications.membership),
            "answers_with_problems": sum(1 for one in answered if one.problems),
            "bodies_named": len(bodies),
            "distinct_classes": len(
                {one.class_name for one in bodies if one.class_name}
            ),
            "bodies": len(split.bodies),
            "faces": sum(one.faces for one in split.bodies),
            "still_contested": split.still_contested,
            "pairings": len(split.pairings),
            "emptied": sorted(split.emptied, key=lambda one: one.name),
            "refused": sorted(split.refused, key=lambda one: one.pairing.part),
            "mounted": len(split.pairings) - len(split.refused),
            "classes_given": self.classes_given(classifications),
            "inspector": RunFile.INSPECTOR.value,
        }

    def classes_given(self, classifications: Classifications) -> List[ClassCount]:
        """
        :param classifications: What each body was answered to be.
        :return: How many bodies each class was given to, the most given first.
        """
        counted = Counter(
            one.class_name for one in classifications.bodies if one.class_name
        )
        return [
            ClassCount(class_name=class_name, count=count)
            for class_name, count in counted.most_common()
        ]
