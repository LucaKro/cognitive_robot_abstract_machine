"""
Say what a run made, from what its steps wrote.

A run's numbers are spread over six files and a terminal that has scrolled away, and the
question asked of a run afterwards is usually how much of it went through rather than
what any one step said.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from string import Template

from typing_extensions import List

from experiments.warsaw.pipeline.records import (
    Adjudications,
    Classifications,
    OpenQuestions,
    Relations,
    SplitRecord,
    Vocabulary,
)
from experiments.warsaw.pipeline.run import Run, RunFile


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

    @property
    def relations(self) -> Relations:
        """
        :return: How the scene's objects were measured to meet.
        """
        return Relations.from_json(
            self.run.read_json_if_written(RunFile.RELATIONS) or {"scene": ""}
        )

    @property
    def questions(self) -> OpenQuestions:
        """
        :return: What the measurements and the ontology left open.
        """
        return OpenQuestions.from_json(self.run.read_json_if_written(RunFile.QUESTIONS))

    @property
    def vocabulary(self) -> Vocabulary:
        """
        :return: What every label was answered to mean.
        """
        return Vocabulary.from_json(self.run.read_json_if_written(RunFile.VOCABULARY))

    @property
    def adjudications(self) -> Adjudications:
        """
        :return: What every open question was answered with.
        """
        return Adjudications.from_json(
            self.run.read_json_if_written(RunFile.ADJUDICATIONS)
        )

    @property
    def split(self) -> SplitRecord:
        """
        :return: What the split built and what it cost.
        """
        return SplitRecord.from_json(self.run.read_json_if_written(RunFile.SPLIT))

    @property
    def classifications(self) -> Classifications:
        """
        :return: What every body was answered to be.
        """
        return Classifications.from_json(
            self.run.read_json_if_written(RunFile.CLASSIFICATIONS)
        )

    def write(self) -> str:
        """
        Write the report into the run.

        :return: The report, as Markdown.
        """
        report = self.markdown()
        self.run.path(RunFile.REPORT).write_text(report)
        return report

    def write_inspector(self, template: Template) -> None:
        """
        Leave behind the script that opens the run's world without knowing anything.

        :param template: The script, with the world ids left to fill in.
        """
        split = self.split
        self.run.path(RunFile.INSPECTOR).write_text(
            template.substitute(
                annotated=split.annotated_world_id, split=split.world_id
            )
        )

    def markdown(self) -> str:
        """
        :return: The report, as Markdown.
        """
        return "\n".join(
            self.heading()
            + self.what_was_measured()
            + self.what_was_asked()
            + self.what_was_built()
            + self.what_was_lost()
            + self.classes_given()
            + self.how_to_look()
        )

    def heading(self) -> List[str]:
        """
        :return: What the run was, and which worlds it wrote.
        """
        split = self.split
        return [
            f"# {self.run.name}",
            "",
            f"- scene: `{self.relations.scene or self.unknown_scene}`",
            f"- worlds: **{split.annotated_world_id}** annotated, "
            f"{split.world_id} as it was split",
            f"- models: {self.vocabulary.model or self.unknown_model} (vocabulary), "
            f"{self.adjudications.model or self.unknown_model} (adjudication), "
            f"{self.classifications.model or self.unknown_model} (classification)",
            "",
        ]

    def what_was_measured(self) -> List[str]:
        """
        :return: What the scan and the ontology said before anything was asked.
        """
        relations, questions = self.relations, self.questions
        overlapping = [pair for pair in relations.pairs if pair.evidence.shared_faces]
        return [
            "## What was measured",
            "",
            f"- {len(relations.segments)} labelled objects over {len(relations.pairs)} "
            f"measurable pairs, {len(overlapping)} of them sharing faces",
            f"- {len(questions.settled)} sets of contested faces the ontology settled, "
            f"{len(questions.forced)} memberships with only one candidate",
            "",
        ]

    def what_was_asked(self) -> List[str]:
        """
        :return: What was put to a model, and how much of it came back usable.
        """
        vocabulary, adjudications = self.vocabulary, self.adjudications
        bodies = self.classifications.bodies
        answered = adjudications.ownership + adjudications.membership
        return [
            "## What was asked",
            "",
            f"- {len(vocabulary.labels)} labels, "
            f"{sum(1 for one in vocabulary.labels.values() if one.class_name)} mapped to "
            f"a class, "
            f"{sum(1 for one in vocabulary.labels.values() if one.is_new_class)} of them "
            f"new",
            f"- {len(adjudications.ownership)} class patterns and "
            f"{len(adjudications.membership)} memberships adjudicated, "
            f"{sum(1 for one in answered if one.problems)} with problems",
            f"- {len(bodies)} bodies named, "
            f"{len({one.class_name for one in bodies.values() if one.class_name})} "
            f"distinct classes",
            "",
        ]

    def what_was_built(self) -> List[str]:
        """
        :return: What came out of the split.
        """
        split = self.split
        return [
            "## What was built",
            "",
            f"- {len(split.bodies)} bodies, "
            f"{sum(one.faces for one in split.bodies.values())} faces between them, "
            f"{split.still_contested} faces still claimed twice",
            f"- {len(split.pairings)} pairings carried past the split",
        ]

    def what_was_lost(self) -> List[str]:
        """
        :return: The objects that ended up with nothing, and what took their faces.
        """
        emptied = self.split.emptied
        if not emptied:
            return []
        lines = ["", f"### {len(emptied)} objects lost every face", ""]
        for name, took in sorted(emptied.items()):
            whom = ", ".join(f"{who} ({count})" for who, count in took.items())
            lines.append(f"- `{name}` -> {whom}")
        return lines

    def classes_given(self) -> List[str]:
        """
        :return: How many bodies each class was given to.
        """
        bodies = self.classifications.bodies
        if not bodies:
            return []
        lines = ["", "### Classes given", ""]
        for name, count in Counter(
            one.class_name for one in bodies.values() if one.class_name
        ).most_common():
            lines.append(f"- {count} x `{name}`")
        return lines

    def how_to_look(self) -> List[str]:
        """
        :return: How to open what the run built.
        """
        return [
            "",
            "## Looking at it",
            "",
            "```",
            f"python {RunFile.INSPECTOR.value}",
            "```",
            "",
        ]
