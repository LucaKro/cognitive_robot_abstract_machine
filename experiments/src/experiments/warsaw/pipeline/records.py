"""
What one step of the pipeline hands to the next.

Every step reads what the step before it wrote, so each field name is written in one
module and read in another. Mirroring each file in a dataclass writes those names once:
the reader and the writer are the same declaration, and a field that moves moves for
both.

The shapes here are the pipeline's own, so they carry no type name into the files. A
run's artefacts are read by people comparing one run against another, and stamping a
class name into every node would make them harder to read for nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Any, Dict, List, Optional, Tuple

from experiments.warsaw.scene_split import Pairing
from experiments.warsaw.segment_relations import PairEvidence, SegmentDescriptor

# %% named alternatives


class QuestionKind(StrEnum):
    """
    What an open question is about.
    """

    OWNERSHIP = "ownership"
    """
    Whose surface a face several labels claim is.
    """

    MEMBERSHIP = "membership"
    """
    Which whole a part belongs to.
    """


class RelationStatus(StrEnum):
    """
    What the ontology makes of two labels measured to meet.
    """

    CLASS_UNKNOWN = "class-unknown"
    """
    Neither label has been mapped to a class yet, so the ontology has nothing to say.
    """

    NO_LEGAL_RELATION = "no-legal-relation"
    """
    The classes cannot hold one another as a part, so an overlap is something else.
    """

    RELATION_KNOWN = "relation-known"
    """
    Exactly one part-whole relation is admissible, so only the pair itself is in
    question.
    """

    RELATION_AMBIGUOUS = "relation-ambiguous"
    """
    Several relations are admissible, so which field a mount would use is in question
    too.
    """


class PictureKind(StrEnum):
    """
    What one of a question's renders shows.

    A render is written as ``<subject>__<kind>_<viewpoint>.png``, so the kind is read
    back out of the filename when the question is put to a model.
    """

    CONTEXT = "context"
    """
    Where in the room the subject is.
    """

    PLAIN = "plain"
    """
    The subject alone, in the colors it was scanned in.
    """

    CLOSEUP = "closeup"
    """
    The subject alone, painted, which is exactly the faces in question.
    """

    @classmethod
    def of_render(cls, filename: str) -> Optional[PictureKind]:
        """
        :param filename: A render's name.
        :return: What it shows, or None if its name does not say.
        """
        tail = filename.rsplit("__", 1)[-1].split("_", 1)[0]
        return cls(tail) if tail in set(cls) else None


# %% what the ontology admits between two classes


@dataclass
class AdmissibleRelation:
    """
    One part-whole relation the ontology allows between two classes.
    """

    whole: str
    """
    The class that would hold.
    """

    part: str
    """
    The class it would hold.
    """

    field_name: str
    """
    The field it would be held in.
    """

    holds_many: bool
    """
    Whether that field holds several parts rather than one.
    """

    removes_geometry: bool
    """
    Whether mounting takes the part's geometry out of the whole's.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> AdmissibleRelation:
        return cls(
            whole=payload["whole"],
            part=payload["part"],
            field_name=payload["field"],
            holds_many=payload["many"],
            removes_geometry=payload["removes_geometry"],
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "whole": self.whole,
            "part": self.part,
            "field": self.field_name,
            "many": self.holds_many,
            "removes_geometry": self.removes_geometry,
        }


@dataclass
class AdmissibleMount:
    """
    One mount the ontology allows between two classes that is not a structural part.
    """

    kind: str
    """
    The channel that mounts it.
    """

    whole: str
    """
    The class that would hold.
    """

    field_name: str
    """
    The field it would be held in.
    """

    target: str
    """
    What that field accepts.
    """

    mounted_by: str
    """
    The method that carries the mount out.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> AdmissibleMount:
        return cls(
            kind=payload["kind"],
            whole=payload["whole"],
            field_name=payload["field"],
            target=payload["target"],
            mounted_by=payload["mounted_by"],
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "whole": self.whole,
            "field": self.field_name,
            "target": self.target,
            "mounted_by": self.mounted_by,
        }


@dataclass
class OntologyView:
    """
    What the ontology admits between the classes of two segments.

    ..note:: This is what is *admissible*, never what is the case: whether this cabinet
        holds this drawer is a question about the two objects, which no amount of reading
        the taxonomy answers.
    """

    status: RelationStatus
    """
    What that leaves open.
    """

    admissible: List[AdmissibleRelation] = field(default_factory=list)
    """
    The part-whole relations allowed between them.
    """

    other_mounts: List[AdmissibleMount] = field(default_factory=list)
    """
    The mounts allowed between them that are not structural parts.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> OntologyView:
        return cls(
            status=RelationStatus(payload["status"]),
            admissible=[
                AdmissibleRelation.from_json(one) for one in payload["admissible"]
            ],
            other_mounts=[
                AdmissibleMount.from_json(one) for one in payload["other_mounts"]
            ],
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "admissible": [one.to_json() for one in self.admissible],
            "other_mounts": [one.to_json() for one in self.other_mounts],
        }


# %% relations.json


@dataclass
class PairRecord:
    """
    Two segments measured to meet, and what the ontology makes of them.
    """

    evidence: PairEvidence
    """
    How they were measured to meet.
    """

    classes: Dict[str, Optional[str]] = field(default_factory=dict)
    """
    Per label the two carry, the class it was read as.
    """

    view: Optional[OntologyView] = None
    """
    What the ontology admits between those classes.
    """

    prompt_block: str = ""
    """
    The measurements as the text a model reads them in.
    """

    @property
    def one(self) -> str:
        """
        :return: The name of the first segment.
        """
        return self.evidence.one

    @property
    def other(self) -> str:
        """
        :return: The name of the second segment.
        """
        return self.evidence.other

    @property
    def status(self) -> RelationStatus:
        """
        :return: What the ontology leaves open about the pair.
        """
        return self.view.status if self.view else RelationStatus.CLASS_UNKNOWN

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> PairRecord:
        return cls(
            evidence=PairEvidence.from_json(payload),
            classes=dict(payload.get("classes") or {}),
            view=OntologyView.from_json(payload) if "status" in payload else None,
            prompt_block=payload.get("prompt_block") or "",
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            **self.evidence.to_json(),
            "classes": self.classes,
            **(self.view.to_json() if self.view else {}),
            "prompt_block": self.prompt_block,
        }


@dataclass
class Relations:
    """
    How a scene's labelled objects were measured to meet.
    """

    scene: str
    """
    The mesh they were measured on.
    """

    segments: List[SegmentDescriptor] = field(default_factory=list)
    """
    What each labelled object is, measured rather than judged.
    """

    pairs: List[PairRecord] = field(default_factory=list)
    """
    Every pair that shares faces, touches, or is among a segment's nearest.
    """

    @property
    def descriptors(self) -> Dict[str, SegmentDescriptor]:
        """
        :return: The segments by name.
        """
        return {descriptor.name: descriptor for descriptor in self.segments}

    @property
    def labels(self) -> Dict[str, str]:
        """
        :return: Per segment, the label it carries.
        """
        return {descriptor.name: descriptor.class_name for descriptor in self.segments}

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> Relations:
        return cls(
            scene=payload["scene"],
            segments=[
                SegmentDescriptor.from_json(one)
                for one in payload.get("segments") or []
            ],
            pairs=[PairRecord.from_json(one) for one in payload.get("pairs") or []],
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "scene": self.scene,
            "segments": [one.to_json() for one in self.segments],
            "pairs": [one.to_json() for one in self.pairs],
        }


# %% vocabulary_request.json


@dataclass
class LabelRequest:
    """
    One label of a scene, and the object standing for it.
    """

    label: str
    """
    The word the scene's annotator chose.
    """

    instances: int
    """
    How many objects carry it.
    """

    exemplar: str
    """
    The one standing for it, chosen as the one whose surface is least claimed by others.
    """

    exemplar_faces: int = 0
    """
    How many faces that one is made of.
    """

    exemplar_exclusive_share: float = 0.0
    """
    The share of those no other segment claims.
    """

    exemplar_exclusive_area: float = 0.0
    """
    How much of its area no other segment claims.
    """

    images: List[str] = field(default_factory=list)
    """
    The renders of it, by filename.
    """

    color: Optional[str] = None
    """
    What it was painted in those renders.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> LabelRequest:
        return cls(
            label=payload["label"],
            instances=payload["instances"],
            exemplar=payload["exemplar"],
            exemplar_faces=payload.get("exemplar_faces") or 0,
            exemplar_exclusive_share=payload.get("exemplar_exclusive_share") or 0.0,
            exemplar_exclusive_area=payload.get("exemplar_exclusive_area") or 0.0,
            images=list(payload.get("images") or []),
            color=payload.get("color"),
        )

    def to_json(self) -> Dict[str, Any]:
        written = {
            "label": self.label,
            "instances": self.instances,
            "exemplar": self.exemplar,
            "exemplar_faces": self.exemplar_faces,
            "exemplar_exclusive_share": self.exemplar_exclusive_share,
            "exemplar_exclusive_area": self.exemplar_exclusive_area,
            "images": self.images,
        }
        if self.color is not None:
            written["color"] = self.color
        return written


@dataclass
class VocabularyRequest:
    """
    The question asking which class each of a scene's labels means.
    """

    scene: str
    """
    The mesh the labels were read from.
    """

    question: str
    """
    What is being asked.
    """

    labels: List[LabelRequest] = field(default_factory=list)
    """
    One entry per label.
    """

    @property
    def label_names(self) -> List[str]:
        """
        :return: Every label of the scene, which is what says what each was left to mean:
            a room that labels handles separately does not mean them by ``drawer``.
        """
        return [entry.label for entry in self.labels]

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> VocabularyRequest:
        return cls(
            scene=payload["scene"],
            question=payload["question"],
            labels=[LabelRequest.from_json(one) for one in payload.get("labels") or []],
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "scene": self.scene,
            "question": self.question,
            "labels": [one.to_json() for one in self.labels],
        }


# %% vocabulary.json


@dataclass
class LabelAnswer:
    """
    What was answered about one label.
    """

    class_name: Optional[str] = None
    """
    The class the label means, or None where the ontology should hold nothing for it.
    """

    is_new_class: bool = False
    """
    Whether that class is proposed rather than found in the ontology.
    """

    superclass: Optional[str] = None
    """
    What a proposed class derives from.
    """

    mixins: List[str] = field(default_factory=list)
    """
    What a proposed class is composed with, which decides what it can hold.
    """

    confidence: Optional[float] = None
    """
    How sure the model said it was.
    """

    reason: Optional[str] = None
    """
    Why, in one sentence.
    """

    problems: List[str] = field(default_factory=list)
    """
    What makes the answer unusable, empty when nothing does.
    """

    exemplar: Optional[str] = None
    """
    The object that was pictured when it was asked.
    """

    @property
    def is_usable(self) -> bool:
        """
        :return: Whether the answer names a class and nothing is wrong with it.
        """
        return bool(self.class_name) and not self.problems

    @classmethod
    def of(cls, payload: Any) -> LabelAnswer:
        """
        Read an answer however it was written.

        A mapping written by hand to try something out names the class and nothing else,
        so a bare name and a null are read as answers too.

        :param payload: The answer, as a mapping or as the class name alone.
        :return: It, as an answer.
        """
        if payload is None or isinstance(payload, str):
            return cls(class_name=payload or None)
        return cls.from_json(payload)

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> LabelAnswer:
        return cls(
            class_name=payload.get("class"),
            is_new_class=bool(payload.get("is_new_class")),
            superclass=payload.get("superclass"),
            mixins=list(payload.get("mixins") or []),
            confidence=payload.get("confidence"),
            reason=payload.get("reason"),
            problems=list(payload.get("problems") or []),
            exemplar=payload.get("exemplar"),
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "class": self.class_name,
            "is_new_class": self.is_new_class,
            "superclass": self.superclass,
            "mixins": self.mixins,
            "confidence": self.confidence,
            "reason": self.reason,
            "problems": self.problems,
            "exemplar": self.exemplar,
        }


@dataclass
class Vocabulary:
    """
    What each of a scene's labels was answered to mean.
    """

    model: str
    """
    Which model was asked.
    """

    scene: str
    """
    The mesh the labels were read from.
    """

    labels: Dict[str, LabelAnswer] = field(default_factory=dict)
    """
    The answer per label.
    """

    def answer_for(self, label: str) -> LabelAnswer:
        """
        :param label: The label to look up.
        :return: What was answered about it, blank where nothing was.
        """
        return self.labels.get(label, LabelAnswer())

    @property
    def proposals(self) -> Dict[str, LabelAnswer]:
        """
        :return: Per label, the answer that proposed a class rather than naming one.
        """
        return {
            label: answer
            for label, answer in self.labels.items()
            if answer.is_new_class and answer.class_name
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> Vocabulary:
        return cls(
            model=payload.get("model") or "",
            scene=payload.get("scene") or "",
            labels={
                label: LabelAnswer.of(one)
                for label, one in (payload.get("labels") or {}).items()
            },
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "scene": self.scene,
            "labels": {label: one.to_json() for label, one in self.labels.items()},
        }


# %% questions.json


@dataclass
class ClaimantSet:
    """
    A set of faces claimed by exactly the same segments, counted rather than listed.
    """

    claimants: Tuple[str, ...]
    """
    The segments claiming them.
    """

    faces: int
    """
    How many faces they all claim.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> ClaimantSet:
        return cls(claimants=tuple(payload["claimants"]), faces=payload["faces"])

    def to_json(self) -> Dict[str, Any]:
        return {"claimants": list(self.claimants), "faces": self.faces}


@dataclass
class OntologySlice:
    """
    What the taxonomy holds about the handful of objects one question is about.

    The slice rather than the whole taxonomy: a question about three objects is not helped
    by a hundred and thirty-nine classes.
    """

    read_as: Dict[str, Optional[str]] = field(default_factory=dict)
    """
    Per segment, the class it was read as.
    """

    classes: List[str] = field(default_factory=list)
    """
    Each of those classes, written out.
    """

    admits: List[str] = field(default_factory=list)
    """
    What the ontology admits between them.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> OntologySlice:
        return cls(
            read_as=dict(payload.get("read_as") or {}),
            classes=list(payload.get("classes") or []),
            admits=list(payload.get("admits") or []),
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "read_as": self.read_as,
            "classes": self.classes,
            "admits": self.admits,
        }


@dataclass
class MeasuredSegment:
    """
    What was measured of one object on its own.
    """

    faces: int
    """
    How many of the scene's faces it is made of.
    """

    area: float
    """
    Its surface area, in square metres.
    """

    height: float
    """
    How high its middle sits above the lowest point of the scene.
    """

    pieces: int
    """
    How many connected pieces it falls into.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> MeasuredSegment:
        return cls(
            faces=payload["faces"],
            area=payload["area"],
            height=payload["height"],
            pieces=payload["pieces"],
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "faces": self.faces,
            "area": self.area,
            "height": self.height,
            "pieces": self.pieces,
        }


@dataclass
class ContestedShare:
    """
    How much of one claimant the contested faces are.

    Without it the picture is all a reader has, and a picture cannot be read when one
    claimant is twenty times the size of the others: an island label covers the whole block
    including its drawers, so a drawer front reads as a patch of detail on the island
    rather than as the drawer.
    """

    faces: int
    """
    How many faces the claimant has in all.
    """

    contested_share: float
    """
    What share of them are contested.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> ContestedShare:
        return cls(faces=payload["faces"], contested_share=payload["contested_share"])

    def to_json(self) -> Dict[str, Any]:
        return {"faces": self.faces, "contested_share": self.contested_share}


@dataclass
class OwnershipQuestion:
    """
    Whose surface a set of faces several labels claim is.

    Asked once per *pattern* of classes rather than once per occurrence: a door and a
    window sharing a pane is one question however many glazed doors the room has.
    """

    name: str
    """
    What the question is called, which is its pattern joined up.
    """

    pattern: List[str] = field(default_factory=list)
    """
    The labels that meet like this.
    """

    shown: List[str] = field(default_factory=list)
    """
    The objects in the pictures.
    """

    covers: List[ClaimantSet] = field(default_factory=list)
    """
    Every set of faces this one answer decides.
    """

    contested_faces: int = 0
    """
    How many faces those sets hold between them.
    """

    exemplar_faces: int = 0
    """
    How many the pictured set holds.
    """

    shares: Dict[str, ContestedShare] = field(default_factory=dict)
    """
    Per claimant, how much of it the contested faces are.
    """

    ontology: OntologySlice = field(default_factory=OntologySlice)
    """
    What the taxonomy holds about the objects shown.
    """

    measured: Dict[str, MeasuredSegment] = field(default_factory=dict)
    """
    What was measured of each of them.
    """

    images: List[str] = field(default_factory=list)
    """
    The renders of the question, by filename.
    """

    legend: Dict[str, str] = field(default_factory=dict)
    """
    What each color in those renders stands for.
    """

    kind: QuestionKind = QuestionKind.OWNERSHIP
    """
    What the question is about.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> OwnershipQuestion:
        return cls(
            name=payload["name"],
            pattern=list(payload.get("pattern") or []),
            shown=list(payload.get("shown") or []),
            covers=[ClaimantSet.from_json(one) for one in payload.get("covers") or []],
            contested_faces=payload.get("contested_faces") or 0,
            exemplar_faces=payload.get("exemplar_faces") or 0,
            shares={
                name: ContestedShare.from_json(one)
                for name, one in (payload.get("shares") or {}).items()
            },
            ontology=OntologySlice.from_json(payload.get("ontology") or {}),
            measured={
                name: MeasuredSegment.from_json(one)
                for name, one in (payload.get("measured") or {}).items()
            },
            images=list(payload.get("images") or []),
            legend=dict(payload.get("legend") or {}),
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "pattern": self.pattern,
            "shown": self.shown,
            "covers": [one.to_json() for one in self.covers],
            "contested_faces": self.contested_faces,
            "shares": {name: one.to_json() for name, one in self.shares.items()},
            "exemplar_faces": self.exemplar_faces,
            "ontology": self.ontology.to_json(),
            "measured": {name: one.to_json() for name, one in self.measured.items()},
            "images": self.images,
            "legend": self.legend,
        }


@dataclass
class MembershipCandidate:
    """
    One object a part could belong to, and how it was measured to meet it.
    """

    field_name: str
    """
    The field it would be held in.
    """

    shared_faces: int
    """
    How many faces the two share.
    """

    touching_edges: int
    """
    How many edges they touch along.
    """

    distance: float
    """
    How far apart their surfaces are, in metres.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> MembershipCandidate:
        return cls(
            field_name=payload["field"],
            shared_faces=payload["shared_faces"],
            touching_edges=payload["touching_edges"],
            distance=payload["distance"],
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "field": self.field_name,
            "shared_faces": self.shared_faces,
            "touching_edges": self.touching_edges,
            "distance": self.distance,
        }


@dataclass
class MembershipQuestion:
    """
    Which whole a part belongs to, asked only where it meets more than one candidate.
    """

    name: str
    """
    What the question is called, which is the part.
    """

    part: str
    """
    The object that belongs to one of the candidates.
    """

    shown: List[str] = field(default_factory=list)
    """
    The objects in the pictures.
    """

    candidates: Dict[str, MembershipCandidate] = field(default_factory=dict)
    """
    The wholes it could belong to.
    """

    ontology: OntologySlice = field(default_factory=OntologySlice)
    """
    What the taxonomy holds about them.
    """

    measured: Dict[str, MeasuredSegment] = field(default_factory=dict)
    """
    What was measured of each of them.
    """

    images: List[str] = field(default_factory=list)
    """
    The renders of the question, by filename.
    """

    legend: Dict[str, str] = field(default_factory=dict)
    """
    What each color in those renders stands for.
    """

    kind: QuestionKind = QuestionKind.MEMBERSHIP
    """
    What the question is about.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> MembershipQuestion:
        return cls(
            name=payload["name"],
            part=payload["part"],
            shown=list(payload.get("shown") or []),
            candidates={
                name: MembershipCandidate.from_json(one)
                for name, one in (payload.get("candidates") or {}).items()
            },
            ontology=OntologySlice.from_json(payload.get("ontology") or {}),
            measured={
                name: MeasuredSegment.from_json(one)
                for name, one in (payload.get("measured") or {}).items()
            },
            images=list(payload.get("images") or []),
            legend=dict(payload.get("legend") or {}),
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "part": self.part,
            "shown": self.shown,
            "candidates": {
                name: one.to_json() for name, one in self.candidates.items()
            },
            "ontology": self.ontology.to_json(),
            "measured": {name: one.to_json() for name, one in self.measured.items()},
            "images": self.images,
            "legend": self.legend,
        }


@dataclass
class ForcedMembership:
    """
    A part that meets exactly one candidate, so there is nothing to choose between.
    """

    part: str
    """
    The object that belongs to it.
    """

    whole: str
    """
    The one object it could belong to.
    """

    field_name: str
    """
    The field it would be held in.
    """

    shared_faces: int = 0
    """
    How many faces the two share.
    """

    touching_edges: int = 0
    """
    How many edges they touch along.
    """

    distance: float = 0.0
    """
    How far apart their surfaces are, in metres.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> ForcedMembership:
        return cls(
            part=payload["part"],
            whole=payload["whole"],
            field_name=payload["field"],
            shared_faces=payload.get("shared_faces") or 0,
            touching_edges=payload.get("touching_edges") or 0,
            distance=payload.get("distance") or 0.0,
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "part": self.part,
            "whole": self.whole,
            "field": self.field_name,
            "shared_faces": self.shared_faces,
            "touching_edges": self.touching_edges,
            "distance": self.distance,
        }


@dataclass
class OpenQuestions:
    """
    What the measurements and the ontology leave open about a scene's overlaps.
    """

    scene: str
    """
    The mesh they were measured on.
    """

    ownership: List[OwnershipQuestion] = field(default_factory=list)
    """
    Whose the contested faces are, once per class pattern.
    """

    membership: List[MembershipQuestion] = field(default_factory=list)
    """
    Which whole each part belongs to, where more than one is possible.
    """

    settled: List[ClaimantSet] = field(default_factory=list)
    """
    The sets the ontology already decides, which are not questions at all.
    """

    forced: List[ForcedMembership] = field(default_factory=list)
    """
    The memberships with only one candidate.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> OpenQuestions:
        return cls(
            scene=payload.get("scene") or "",
            ownership=[
                OwnershipQuestion.from_json(one)
                for one in payload.get("ownership") or []
            ],
            membership=[
                MembershipQuestion.from_json(one)
                for one in payload.get("membership") or []
            ],
            settled=[
                ClaimantSet.from_json(one) for one in payload.get("settled") or []
            ],
            forced=[
                ForcedMembership.from_json(one) for one in payload.get("forced") or []
            ],
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "scene": self.scene,
            "ownership": [one.to_json() for one in self.ownership],
            "membership": [one.to_json() for one in self.membership],
            "settled": [one.to_json() for one in self.settled],
            "forced": [one.to_json() for one in self.forced],
        }


# %% adjudications.json


@dataclass
class OwnershipAnswer:
    """
    Whose the contested faces of one class pattern are.
    """

    name: str
    """
    The question it answers.
    """

    pattern: List[str] = field(default_factory=list)
    """
    The labels that meet like this.
    """

    owner: Optional[str] = None
    """
    The label whose surface those faces are.
    """

    covers: List[ClaimantSet] = field(default_factory=list)
    """
    Every set of faces this answer decides.
    """

    confidence: Optional[float] = None
    """
    How sure the model said it was.
    """

    reason: Optional[str] = None
    """
    Why, in one sentence.
    """

    problems: List[str] = field(default_factory=list)
    """
    What makes the answer unusable, empty when nothing does.
    """

    kind: QuestionKind = QuestionKind.OWNERSHIP
    """
    What the question was about.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> OwnershipAnswer:
        return cls(
            name=payload["name"],
            pattern=list(payload.get("pattern") or []),
            owner=payload.get("owner"),
            covers=[ClaimantSet.from_json(one) for one in payload.get("covers") or []],
            confidence=payload.get("confidence"),
            reason=payload.get("reason"),
            problems=list(payload.get("problems") or []),
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "problems": self.problems,
            "confidence": self.confidence,
            "reason": self.reason,
            "pattern": self.pattern,
            "owner": self.owner,
            "covers": [one.to_json() for one in self.covers],
        }


@dataclass
class MembershipAnswer:
    """
    Which whole one part belongs to.
    """

    name: str
    """
    The question it answers.
    """

    part: str
    """
    The object that belongs somewhere.
    """

    whole: Optional[str] = None
    """
    The object it belongs to.
    """

    confidence: Optional[float] = None
    """
    How sure the model said it was.
    """

    reason: Optional[str] = None
    """
    Why, in one sentence.
    """

    problems: List[str] = field(default_factory=list)
    """
    What makes the answer unusable, empty when nothing does.
    """

    kind: QuestionKind = QuestionKind.MEMBERSHIP
    """
    What the question was about.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> MembershipAnswer:
        return cls(
            name=payload["name"],
            part=payload["part"],
            whole=payload.get("whole"),
            confidence=payload.get("confidence"),
            reason=payload.get("reason"),
            problems=list(payload.get("problems") or []),
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "problems": self.problems,
            "confidence": self.confidence,
            "reason": self.reason,
            "part": self.part,
            "whole": self.whole,
        }


@dataclass
class Adjudications:
    """
    What was answered about everything the measurements and the ontology left open.
    """

    model: str
    """
    Which model was asked.
    """

    scene: str
    """
    The mesh the questions were measured on.
    """

    ownership: List[OwnershipAnswer] = field(default_factory=list)
    """
    Whose the contested faces of each class pattern are.
    """

    membership: List[MembershipAnswer] = field(default_factory=list)
    """
    Which whole each part belongs to.
    """

    settled: List[ClaimantSet] = field(default_factory=list)
    """
    The sets the ontology decides, carried through so the split need read one file.
    """

    forced: List[ForcedMembership] = field(default_factory=list)
    """
    The memberships with only one candidate, carried through for the same reason.
    """

    @property
    def owner_by_pattern(self) -> Dict[Tuple[str, ...], str]:
        """
        :return: Per class pattern, the label the faces belong to.
        """
        return {
            tuple(answer.pattern): answer.owner
            for answer in self.ownership
            if answer.owner
        }

    @property
    def settled_claimants(self) -> set:
        """
        :return: The claimant sets the ontology decides, as their names alone.
        """
        return {one.claimants for one in self.settled}

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> Adjudications:
        answered = payload.get("answered") or []
        return cls(
            model=payload.get("model") or "",
            scene=payload.get("scene") or "",
            ownership=[
                OwnershipAnswer.from_json(one)
                for one in answered
                if one["kind"] == QuestionKind.OWNERSHIP
            ],
            membership=[
                MembershipAnswer.from_json(one)
                for one in answered
                if one["kind"] == QuestionKind.MEMBERSHIP
            ],
            settled=[
                ClaimantSet.from_json(one) for one in payload.get("settled") or []
            ],
            forced=[
                ForcedMembership.from_json(one) for one in payload.get("forced") or []
            ],
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "scene": self.scene,
            "answered": [one.to_json() for one in self.ownership]
            + [one.to_json() for one in self.membership],
            "settled": [one.to_json() for one in self.settled],
            "forced": [one.to_json() for one in self.forced],
        }


# %% split.json


@dataclass
class SplitBody:
    """
    One body the split built.
    """

    faces: int
    """
    How many of the scene's faces are its alone.
    """

    label: str
    """
    The label the scene gave it.
    """

    body_id: Optional[str] = None
    """
    The id the world addresses it by.

    The name is what everything else addresses a body by, and the world addresses it by
    an id of its own; a step reading the world back needs both to say the same thing.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> SplitBody:
        return cls(
            faces=payload["faces"], label=payload["label"], body_id=payload.get("id")
        )

    def to_json(self) -> Dict[str, Any]:
        return {"faces": self.faces, "label": self.label, "id": self.body_id}


@dataclass
class SplitRecord:
    """
    What the split built, what it cost, and the mounts carried past it.
    """

    scene: str
    """
    The mesh it was cut from.
    """

    bodies: Dict[str, SplitBody] = field(default_factory=dict)
    """
    Every body, by the name it carries everywhere else.
    """

    emptied: Dict[str, Dict[str, int]] = field(default_factory=dict)
    """
    Per segment left with no faces, how many each owner took from it.
    """

    still_contested: int = 0
    """
    How many faces are still claimed twice, which should be none.
    """

    pairings: List[Pairing] = field(default_factory=list)
    """
    The mounts that still have both ends.
    """

    world_id: Optional[int] = None
    """
    The world the split was written to.
    """

    annotated_world_id: Optional[int] = None
    """
    The world the annotations were written to, once the last step has run.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> SplitRecord:
        return cls(
            scene=payload.get("scene") or "",
            bodies={
                name: SplitBody.from_json(one)
                for name, one in (payload.get("bodies") or {}).items()
            },
            emptied={
                name: dict(took)
                for name, took in (payload.get("emptied") or {}).items()
            },
            still_contested=payload.get("still_contested") or 0,
            pairings=[Pairing.from_json(one) for one in payload.get("pairings") or []],
            world_id=payload.get("world_db_id"),
            annotated_world_id=payload.get("annotated_world_db_id"),
        )

    def to_json(self) -> Dict[str, Any]:
        written = {
            "scene": self.scene,
            "world_db_id": self.world_id,
            "bodies": {
                name: one.to_json() for name, one in sorted(self.bodies.items())
            },
            "emptied": self.emptied,
            "still_contested": self.still_contested,
            "pairings": [one.to_json() for one in self.pairings],
        }
        if self.annotated_world_id is not None:
            written["annotated_world_db_id"] = self.annotated_world_id
        return written


# %% classifications.json


@dataclass
class BodyAnswer:
    """
    What one body was answered to be.
    """

    class_name: Optional[str] = None
    """
    The class it is.
    """

    is_new_class: bool = False
    """
    Whether that class is proposed rather than found in the ontology.
    """

    superclass: Optional[str] = None
    """
    What a proposed class derives from.
    """

    confidence: Optional[float] = None
    """
    How sure the model said it was.
    """

    reason: Optional[str] = None
    """
    Why, in one sentence.
    """

    label: Optional[str] = None
    """
    The label the scene gave the body, which the answer may disagree with.
    """

    faces: int = 0
    """
    How many faces the body is made of.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> BodyAnswer:
        return cls(
            class_name=payload.get("class"),
            is_new_class=bool(payload.get("is_new_class")),
            superclass=payload.get("superclass"),
            confidence=payload.get("confidence"),
            reason=payload.get("reason"),
            label=payload.get("label"),
            faces=payload.get("faces") or 0,
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "class": self.class_name,
            "is_new_class": self.is_new_class,
            "superclass": self.superclass,
            "confidence": self.confidence,
            "reason": self.reason,
            "label": self.label,
            "faces": self.faces,
        }


@dataclass
class Classifications:
    """
    What each body of a split scene was answered to be.
    """

    model: str
    """
    Which model was asked.
    """

    scene: str
    """
    The mesh the bodies were cut from.
    """

    bodies: Dict[str, BodyAnswer] = field(default_factory=dict)
    """
    The answer per body.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> Classifications:
        return cls(
            model=payload.get("model") or "",
            scene=payload.get("scene") or "",
            bodies={
                name: BodyAnswer.from_json(one)
                for name, one in (payload.get("bodies") or {}).items()
            },
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "scene": self.scene,
            "bodies": {name: one.to_json() for name, one in self.bodies.items()},
        }


# %% taxonomy_amendments.json


@dataclass
class SourceEdit:
    """
    One line of the ontology's own source as it was and as it became.
    """

    file: str
    """
    The file that holds it.
    """

    line: int
    """
    Which line.
    """

    before: str
    """
    What stood there.
    """

    after: str
    """
    What was written instead.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> SourceEdit:
        return cls(
            file=payload["file"],
            line=payload["line"],
            before=payload["before"],
            after=payload["after"],
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "before": self.before,
            "after": self.after,
        }


@dataclass
class AmendmentRecord:
    """
    One mixin a class could be given, what raised it, and what became of the proposal.
    """

    whole: str
    """
    The class that would hold the part.
    """

    mixin: str
    """
    The mixin that would let it.
    """

    part: str
    """
    The class that would be held.
    """

    whole_labels: List[str] = field(default_factory=list)
    """
    The scene's labels that were read as the holding class.
    """

    part_labels: List[str] = field(default_factory=list)
    """
    The scene's labels that were read as the part.
    """

    measured_pairs: int = 0
    """
    How many measured pairs of overlapping objects raised it.
    """

    shared_faces: int = 0
    """
    How many faces those pairs share in total.
    """

    amend: bool = False
    """
    Whether the model said the class should have the mixin.
    """

    confidence: Optional[float] = None
    """
    How sure it said it was.
    """

    reason: Optional[str] = None
    """
    Why, in one sentence.
    """

    blocked: Optional[str] = None
    """
    Why the accepted amendment could not be written, when it could not.
    """

    edit: Optional[SourceEdit] = None
    """
    The line it would change.
    """

    applied: bool = False
    """
    Whether it is in force right now.
    """

    reverted: bool = False
    """
    Whether it was put back after having been in force.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> AmendmentRecord:
        return cls(
            whole=payload["whole"],
            mixin=payload["mixin"],
            part=payload["part"],
            whole_labels=list(payload.get("whole_labels") or []),
            part_labels=list(payload.get("part_labels") or []),
            measured_pairs=payload.get("measured_pairs") or 0,
            shared_faces=payload.get("shared_faces") or 0,
            amend=bool(payload.get("amend")),
            confidence=payload.get("confidence"),
            reason=payload.get("reason"),
            blocked=payload.get("blocked"),
            edit=SourceEdit.from_json(payload["edit"]) if payload.get("edit") else None,
            applied=bool(payload.get("applied")),
            reverted=bool(payload.get("reverted")),
        )

    def to_json(self) -> Dict[str, Any]:
        written = {
            "whole": self.whole,
            "mixin": self.mixin,
            "part": self.part,
            "whole_labels": sorted(self.whole_labels),
            "part_labels": sorted(self.part_labels),
            "measured_pairs": self.measured_pairs,
            "shared_faces": self.shared_faces,
            "amend": self.amend,
            "confidence": self.confidence,
            "reason": self.reason,
            "applied": self.applied,
            "reverted": self.reverted,
        }
        if self.blocked is not None:
            written["blocked"] = self.blocked
        if self.edit is not None:
            written["edit"] = self.edit.to_json()
        return written
