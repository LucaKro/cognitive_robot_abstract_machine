"""
Answer what the measurements and the ontology leave open about a scene's overlaps.

Two things stand between a labelled mesh and a world of bodies, and neither follows from
the geometry:

- **who owns a face two labels both claim**, without which the mesh cannot be split, since
  a face belongs to exactly one body;
- **which whole a part belongs to**, where a door meets more than one cabinet.

Both are asked as few times as they are actually open. Ownership is asked once per set of
claimants rather than once per pair of them, never where the ontology already settled every
relation inside the set, and once per *pattern* of classes rather than once per occurrence
-- a door and a window sharing a pane is one question however many glazed doors the room
has. Membership is asked only where a part meets more than one candidate.

This decides nothing itself.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from semantic_digital_twin.adapters.vision_language_model.client import ModelResponse
from semantic_digital_twin.adapters.vision_language_model.exceptions import (
    ModelRefusedError,
)
from semantic_digital_twin.adapters.vision_language_model.message import (
    ImagePart,
    MessagePart,
    TextPart,
)
from typing_extensions import Any, Dict, Generic, List, Optional, TypeVar

from experiments.warsaw.pipeline.asking import AnswerType, Question
from experiments.warsaw.pipeline.prompts import Prompt
from experiments.warsaw.pipeline.records import (
    Adjudications,
    MembershipAnswer,
    MembershipQuestion,
    OntologySlice,
    OpenQuestions,
    OwnershipAnswer,
    OwnershipQuestion,
    PictureKind,
    Relations,
)
from experiments.warsaw.pipeline.run import RunFile
from experiments.warsaw.pipeline.steps.step import PipelineStep

AskedType = TypeVar("AskedType")
"""
The question record one kind of adjudication is put from.
"""


class OverlapCaption(StrEnum):
    """
    What each render of an open question shows, in the order they are shown in.
    """

    CLOSEUP = "Picture 1 -- the objects alone, with nothing in front of them."
    """
    The objects by themselves.
    """

    CONTEXT = "Picture 2 -- where they are in the room, painted the same way."
    """
    The objects in the room around them.
    """

    PLAIN = "Picture 3 -- the same objects, in the colors they were scanned in."
    """
    The objects as they were scanned.
    """

    @property
    def shows(self) -> PictureKind:
        """
        :return: The kind of render this captions.
        """
        return PictureKind(self.name.lower())


@dataclass
class OverlapQuestion(Question[AnswerType], Generic[AnswerType, AskedType]):
    """
    What the pictures of a set of overlapping objects are shown to settle.
    """

    asked: AskedType
    """
    The question as the measurement wrote it.
    """

    labels: Dict[str, str]
    """
    Per segment, the label it carries.
    """

    images: Path
    """
    The directory holding its renders.
    """

    @property
    def key(self) -> str:
        return f"{self.kind_name}__{self.asked.name}"

    @property
    @abstractmethod
    def kind_name(self) -> str:
        """
        :return: What kind of question this is, which its kept reply is filed under.
        """

    def pictures(self) -> List[MessagePart]:
        """
        :return: The renders as parts of a message, captioned.
        """
        named = {
            PictureKind.of_render(filename): filename for filename in self.asked.images
        }
        content: List[MessagePart] = []
        for caption in OverlapCaption:
            filename = named.get(caption.shows)
            if filename is None:
                continue
            content.append(TextPart(caption.value))
            content.append(ImagePart.from_file(self.images / filename))
        return content

    def painted(self) -> str:
        """
        :return: What each color in the pictures stands for.
        """
        lines = [
            f'{name} (labelled "{self.labels[name]}") is {self.asked.legend[name]}'
            for name in self.asked.shown
            if name in self.asked.legend
        ]
        if "contested" in self.asked.legend:
            lines.append(
                f"the faces all of them claim are {self.asked.legend['contested']}"
            )
        return "\n".join(lines)

    def ontology(self) -> str:
        """
        :return: What the ontology holds about the objects in it, as a model reads it.
        """
        known: OntologySlice = self.asked.ontology
        if not known.read_as:
            return ""
        read_as = "\n".join(
            f"{name} was read as {class_name or 'no class'}"
            for name, class_name in known.read_as.items()
        )
        lines = [f"## What the ontology says\n{read_as}"]
        if known.classes:
            lines.append("\n".join(known.classes))
        lines.append(
            "\n".join(known.admits)
            if known.admits
            else "Between these classes it admits no mount at all."
        )
        return "\n\n".join(lines)

    def measured(self) -> str:
        """
        :return: What was measured of each object on its own.
        """
        return "\n".join(
            f"{name}: {one.faces} faces, {one.area} m2, "
            f"middle {one.height} m up, {one.pieces} piece(s)"
            for name, one in self.asked.measured.items()
        )


@dataclass
class OwnershipDecision(OverlapQuestion[OwnershipAnswer, OwnershipQuestion]):
    """
    Whose the faces a set of labels all claim are.
    """

    @property
    def kind_name(self) -> str:
        return "ownership"

    @property
    def system_prompt(self) -> str:
        return Prompt.OWNERSHIP.read()

    def message(self) -> List[MessagePart]:
        # What the picture cannot say. One claimant is often many times the size of the
        # others -- an island label covers the whole block including its drawers -- and
        # then the contested faces read as a patch of detail on the big object rather than
        # as the whole of the small one. The shares say which it is.
        shares = "\n".join(
            f"of {name} the contested {self.asked.exemplar_faces} faces are "
            f"{share.contested_share:.0%}"
            for name, share in self.asked.shares.items()
        )
        return [
            TextPart(
                f"## The labels\n{', '.join(self.asked.pattern)}\n\n"
                f"{self.ontology()}\n\n"
                f"## The picture\n{self.painted()}\n\n"
                f"## What was measured\n{self.measured()}\n{shares}\n\n"
                f"## How often this happens\n"
                f"Objects with these labels are labelled over the same faces "
                f"{len(self.asked.covers)} time(s) in this room, "
                f"{self.asked.contested_faces} faces in all. The pictures show the "
                f"largest of them."
            )
        ] + self.pictures()

    def read(self, response: ModelResponse) -> OwnershipAnswer:
        answered: Dict[str, Any] = response.parse_json()
        return OwnershipAnswer(
            name=self.asked.name,
            pattern=list(self.asked.pattern),
            owner=answered.get("owner"),
            covers=list(self.asked.covers),
            confidence=answered.get("confidence"),
            reason=answered.get("reason"),
        )

    def refusal(self, refused: ModelRefusedError) -> OwnershipAnswer:
        return OwnershipAnswer(
            name=self.asked.name,
            pattern=list(self.asked.pattern),
            covers=list(self.asked.covers),
            problems=[str(refused)],
        )

    def problems_with(self, answer: OwnershipAnswer) -> List[str]:
        if answer.owner in self.asked.pattern:
            return []
        return [
            f"{answer.owner!r} is not one of the owners to choose from: "
            f"{', '.join(self.asked.pattern)}"
        ]


@dataclass
class MembershipDecision(OverlapQuestion[MembershipAnswer, MembershipQuestion]):
    """
    Which of several candidates a part belongs to.
    """

    @property
    def kind_name(self) -> str:
        return "membership"

    @property
    def system_prompt(self) -> str:
        return Prompt.MEMBERSHIP.read()

    def message(self) -> List[MessagePart]:
        candidates = "\n".join(
            f"{name}: shares {how.shared_faces} faces with it, touches it along "
            f"{how.touching_edges} edges, {how.distance} m between their surfaces, "
            f"and would hold it in its {how.field_name}"
            for name, how in self.asked.candidates.items()
        )
        return [
            TextPart(
                f"## The part\n{self.asked.part}, labelled "
                f'"{self.labels[self.asked.part]}"\n\n'
                f"{self.ontology()}\n\n"
                f"## What was measured\n{self.measured()}\n\n"
                f"## The candidates\n{candidates}\n\n"
                f"## The picture\n{self.painted()}"
            )
        ] + self.pictures()

    def read(self, response: ModelResponse) -> MembershipAnswer:
        answered: Dict[str, Any] = response.parse_json()
        return MembershipAnswer(
            name=self.asked.name,
            part=self.asked.part,
            whole=answered.get("whole"),
            confidence=answered.get("confidence"),
            reason=answered.get("reason"),
        )

    def refusal(self, refused: ModelRefusedError) -> MembershipAnswer:
        return MembershipAnswer(
            name=self.asked.name, part=self.asked.part, problems=[str(refused)]
        )

    def problems_with(self, answer: MembershipAnswer) -> List[str]:
        if answer.whole in self.asked.candidates:
            return []
        return [
            f"{answer.whole!r} is not one of the wholes to choose from: "
            f"{', '.join(self.asked.candidates)}"
        ]


@dataclass
class RenderedQuestions:
    """
    The open questions sorted by whether there are pictures to put them with.
    """

    with_pictures: List[OverlapQuestion] = field(default_factory=list)
    """
    The ones that can be asked.
    """

    without: List[OverlapQuestion] = field(default_factory=list)
    """
    The ones that cannot: a question is what the pictures settle.
    """

    @property
    def total(self) -> int:
        """
        :return: How many questions there are in all.
        """
        return len(self.with_pictures) + len(self.without)


@dataclass
class AdjudicateOverlaps(PipelineStep):
    """
    The answers to everything the measurements and the ontology leave open.
    """

    limit: Optional[int] = None
    """
    Ask only the first so many questions, rather than all of them.
    """

    @property
    def name(self) -> str:
        return "adjudicate what is left open"

    def carry_out(self) -> None:
        """
        Put every rendered question to the model and write the answers.
        """
        questions = OpenQuestions.from_json(self.run.read_json(RunFile.QUESTIONS))
        relations = Relations.from_json(self.run.read_json(RunFile.RELATIONS))
        labels = relations.labels
        images = self.run.path(RunFile.QUESTION_RENDERS)
        questioner = self.questioner(RunFile.QUESTION_ANSWERS)

        ownership = [
            OwnershipDecision(asked=one, labels=labels, images=images)
            for one in questions.ownership
        ]
        membership = [
            MembershipDecision(asked=one, labels=labels, images=images)
            for one in questions.membership
        ]
        rendered = self.rendered(ownership + membership)
        if rendered.without:
            self.logger.warning(
                "%s of %s questions have no renders yet, so they are not asked",
                len(rendered.without),
                rendered.total,
            )
        asked = (
            rendered.with_pictures[: self.limit]
            if self.limit
            else rendered.with_pictures
        )

        self.logger.info(
            "asking %s about %s questions ...", self.settings.model.value, len(asked)
        )
        adjudications = Adjudications(
            model=self.settings.model.value,
            scene=relations.scene,
            settled=list(questions.settled),
            forced=list(questions.forced),
        )
        for question in asked:
            answered = questioner.answer(question)
            answer = answered.answer
            answer.problems = answered.problems
            if isinstance(answer, OwnershipAnswer):
                adjudications.ownership.append(answer)
                self.logger.info("  %-44s -> %s", answer.name, answer.owner)
            else:
                adjudications.membership.append(answer)
                self.logger.info("  %-44s in %s", answer.name, answer.whole)
            for problem in answer.problems:
                self.logger.warning("      ! %s", problem)

        self.run.write_json(RunFile.ADJUDICATIONS, adjudications.to_json())
        self.report(adjudications)

    @staticmethod
    def rendered(questions: List[OverlapQuestion]) -> RenderedQuestions:
        """
        :param questions: Every open question.
        :return: Them, sorted by whether there are pictures to put them with.
        """
        return RenderedQuestions(
            with_pictures=[one for one in questions if one.asked.images],
            without=[one for one in questions if not one.asked.images],
        )

    def report(self, adjudications: Adjudications) -> None:
        """
        :param adjudications: What was answered.
        """
        answered = adjudications.ownership + adjudications.membership
        troubled = [one for one in answered if one.problems]
        self.logger.info(
            "%s patterns and %s memberships answered, %s with problems",
            len(adjudications.ownership),
            len(adjudications.membership),
            len(troubled),
        )
        self.logger.info("written to %s", self.run.path(RunFile.ADJUDICATIONS))
