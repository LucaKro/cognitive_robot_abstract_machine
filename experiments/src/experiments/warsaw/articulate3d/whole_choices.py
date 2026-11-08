"""
Choosing the whole each part of a scene hangs from without asking a model.

By geometry alone, a part hangs from the larger object it touches most. By the
pipeline's own rule, a part is offered only the wholes the ontology allows for the
classes read, and where the rule would ask a model, the first candidate it lists is
taken instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

from experiments.warsaw.articulate3d.measured_scene import MeasuredScene
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.evidence import MeasureScene, Memberships
from experiments.warsaw.segment_relations import PairEvidence

# %% a choice of wholes


class WholeChoice(ABC):
    """
    A way of choosing the whole each part of a scene hangs from.
    """

    @abstractmethod
    def wholes_of(
        self, scene: MeasuredScene, classes: dict[str, str | None]
    ) -> dict[str, str]:
        """
        :param scene: The measured scene.
        :param classes: Per label, the class it was read as.
        :return: Per segment hung from a whole, that whole's segment.
        """


# %% by geometry alone


@dataclass
class LargerObjectTouchedMost(WholeChoice):
    """
    Each part hangs from the larger object it touches along the most edges, then shares
    the most faces with, then lies nearest.

    With none meeting it, the nearest larger object within the gap is taken. Labels,
    classes and the ontology are not looked at.
    """

    settings: PipelineSettings = field(default_factory=PipelineSettings)
    """
    The settings saying what counts as touching and how wide a gap may be.
    """

    def wholes_of(
        self, scene: MeasuredScene, classes: dict[str, str | None]
    ) -> dict[str, str]:
        faces = {one.name: one.faces for one in scene.segments}
        meeting: dict[str, list[PairEvidence]] = defaultdict(list)
        nearby: dict[str, list[PairEvidence]] = defaultdict(list)
        for pair in scene.pairs:
            if faces[pair.one] == faces[pair.other]:
                continue
            part = min((pair.one, pair.other), key=faces.__getitem__)
            if pair.meets(self.settings.largest_gap_counted_as_touching):
                meeting[part].append(pair)
            elif pair.distance <= self.settings.largest_gap_to_a_whole:
                nearby[part].append(pair)
        return {
            part: self.whole_in(
                min(
                    meeting.get(part) or nearby[part],
                    key=lambda pair: (
                        -pair.touching_edges,
                        -pair.shared_faces,
                        pair.distance,
                        self.whole_in(pair, part),
                    ),
                ),
                part,
            )
            for part in set(meeting) | set(nearby)
        }

    @staticmethod
    def whole_in(pair: PairEvidence, part: str) -> str:
        """
        :param pair: A pair the part is one of.
        :param part: The part.
        :return: The other segment of the pair.
        """
        return pair.other if pair.one == part else pair.one


# %% by the pipeline's rule


@dataclass
class FirstCandidateOfTheRule(WholeChoice):
    """
    The pipeline's own candidate rule, with every question it would ask a model answered
    by the first candidate it lists: the one touching the part along the most edges.
    """

    step: MeasureScene
    """
    The pipeline's measuring step, whose settings and rule work out the candidates.
    """

    ontology: dict[str, type]
    """
    The ontology's classes, by name.
    """

    def memberships(
        self, scene: MeasuredScene, classes: dict[str, str | None]
    ) -> Memberships:
        """
        :param scene: The measured scene.
        :param classes: Per label, the class it was read as.
        :return: The parts the rule gives a whole without asking, and the questions it
            would ask about the others.
        """
        typed = {
            label: None if name is None else self.ontology[name]
            for label, name in classes.items()
        }
        relations = self.step.relations_of(
            scene.converted.scene_id, scene.measured, typed, {}
        )
        return self.step.memberships(
            relations, relations.labels, typed, scene.measured, {}
        )

    def wholes_of(
        self, scene: MeasuredScene, classes: dict[str, str | None]
    ) -> dict[str, str]:
        memberships = self.memberships(scene, classes)
        return {
            **{one.part: one.whole for one in memberships.forced},
            **{
                question.part: question.candidates[0].name
                for question in memberships.to_ask_about
            },
        }
