"""
Measure how a scene's labelled objects meet, and say what the ontology makes of it.

Splitting a scene into bodies means deciding what to do where its labels overlap: the same
faces can be a cabinet and its door, a drawer and the handle on it. This gathers what those
decisions need and decides nothing itself:

- the geometry, measured (shared faces, edges touched along, distance apart),
- the ontology, in the form that shows what each class can be composed of,
- per pair, which part-whole relations the ontology admits between their classes.

Labels are not classes -- a scene labelling something ``kitchen_island`` says nothing about
which class that is, and some labels have no class at all. That mapping is a question for a
model, and the vocabulary request is what it is asked with. Run again once the answer is
back, and the same measurement says what the ontology admits for every pair.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
from semantic_digital_twin.semantic_annotations.part_whole import admissible_relations
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    admissible_mounts,
    describe_class,
)
from semantic_digital_twin.world_description.geometry import Color
from typing_extensions import Dict, List, Optional, Sequence, Tuple, Type

from experiments.warsaw.pipeline.label_classes import VocabularyClasses
from experiments.warsaw.pipeline.records import (
    AdmissibleMount,
    AdmissibleRelation,
    ClaimantSet,
    ContestedShare,
    ForcedMembership,
    MeasuredSegment,
    MembershipCandidate,
    MembershipQuestion,
    OntologySlice,
    OntologyView,
    OpenQuestions,
    OwnershipQuestion,
    PairRecord,
    Relations,
    RelationStatus,
    Vocabulary,
    VocabularyRequest,
    LabelRequest,
)
from experiments.warsaw.pipeline.run import RunFile
from experiments.warsaw.pipeline.steps.step import PipelineStep
from experiments.warsaw.segment_relations import (
    SegmentRelations,
    claimant_groups,
    segment_evidence,
)
from experiments.warsaw.world_loader import (
    LabelSegment,
    ViewpointChoice,
    WarsawWorldLoader,
)


@dataclass
class WhatIsOpen:
    """
    The questions a scene's overlaps raise, and the faces each is rendered around.
    """

    questions: OpenQuestions
    """
    What is left to decide.
    """

    contested: Dict[str, np.ndarray] = field(default_factory=dict)
    """
    Per ownership question, the faces every one of its objects claims.

    Kept apart from the question itself because it is what the render is painted with
    rather than anything the run writes down: a picture that does not mark the contested
    faces shows them as belonging to whichever object was painted over them, which is the
    very thing in question.
    """


@dataclass
class Memberships:
    """
    The parts of a scene that have a whole to choose, and those that have only one.
    """

    to_ask_about: List[MembershipQuestion] = field(default_factory=list)
    """
    The parts that meet more than one candidate.
    """

    forced: List[ForcedMembership] = field(default_factory=list)
    """
    The parts that meet exactly one, where there is nothing to choose between.
    """


@dataclass
class MeasureScene(PipelineStep):
    """
    The scene as it was measured, and what the ontology makes of the measurements.

    Run twice in one pipeline: once knowing no classes, to ask what the labels mean, and
    once knowing them, to say what the ontology admits and what is left open.
    """

    exemplar_renders: bool = False
    """
    Whether to render one exemplar per label, which the vocabulary question needs.
    """

    question_renders: int = 0
    """
    How many of each kind of open question to render.

    One picture per class pattern and per contested membership, not one per overlapping
    pair: there are two hundred of those and a fifth as many questions in them.
    """

    knowing_the_vocabulary: bool = False
    """
    Whether to read the vocabulary the run has since answered, without which the
    ontology has nothing to say about any pair.
    """

    overwrite: bool = False
    """
    Whether to write over what an earlier pass wrote, which is what the second pass
    needs.
    """

    question_prompt: str = (
        "Each label below names objects in a scanned room. Say which class of the "
        "ontology each label is, or propose a new class by naming a superclass and any "
        "mixins it should be composed of. The ontology is in taxonomy.json; its "
        "part_whole_mixins list what a new class can be given."
    )
    """
    What the vocabulary request asks.
    """

    @property
    def name(self) -> str:
        return (
            "measure again, knowing the classes"
            if self.knowing_the_vocabulary
            else "measure the scene"
        )

    @property
    def viewpoints(self) -> Optional[List[str]]:
        """
        :return: Which viewpoints to render, or None to render all of them so that one can
            be chosen between.
        """
        if self.settings.viewpoint_choice is ViewpointChoice.ALL:
            return list(self.settings.kept_viewpoints)
        return None

    @property
    def chooses_viewpoint(self) -> Optional[ViewpointChoice]:
        """
        :return: How to choose the one viewpoint to keep, or None to keep them all.
        """
        choice = self.settings.viewpoint_choice
        return None if choice is ViewpointChoice.ALL else choice

    def loader(self) -> WarsawWorldLoader:
        """
        :return: The scene, loaded and ready to be measured and rendered.
        """
        return WarsawWorldLoader(
            input_directory=self.settings.scene,
            render_resolution=self.settings.render_resolution,
            deciding_resolution=self.settings.deciding_resolution,
        )

    def vocabulary_classes(self) -> VocabularyClasses:
        """
        :return: What each label stands for, empty on the pass that has not asked yet.
        """
        vocabulary = (
            Vocabulary.from_json(self.run.read_json(RunFile.VOCABULARY))
            if self.knowing_the_vocabulary
            else Vocabulary(model="", scene="")
        )
        return VocabularyClasses(vocabulary=vocabulary, known=self.ontology_classes())

    def carry_out(self) -> None:
        """
        Measure the scene, write what was measured, and say what is left open.
        """
        self.run.refuse_to_write_over(
            RunFile.RELATIONS,
            RunFile.VOCABULARY_REQUEST,
            RunFile.QUESTIONS,
            overwrite=self.overwrite,
        )

        loader = self.loader()
        segments = {str(segment.name): segment for segment in loader.label_segments}
        self.logger.info("%s segments; measuring how they meet ...", len(segments))
        measured = segment_evidence(loader, nearest=self.settings.nearest)
        self.logger.info(
            "%s pairs stand in some measurable relation", len(measured.pairs)
        )

        classes = self.vocabulary_classes().by_label()
        relations = self.relations_of(loader, measured, classes)
        self.run.write_json(RunFile.RELATIONS, relations.to_json())

        request = self.vocabulary_request(loader, measured)
        if self.exemplar_renders:
            self.render_exemplars(loader, measured, segments, request)
        self.run.write_json(RunFile.VOCABULARY_REQUEST, request.to_json())

        open_now = self.open_questions(loader, measured, relations, classes)
        self.logger.info(
            "%s class patterns and %s memberships are open; %s groups the ontology "
            "settles",
            len(open_now.questions.ownership),
            len(open_now.questions.membership),
            len(open_now.questions.settled),
        )
        if self.question_renders:
            self.render_questions(
                loader, measured, segments, open_now.questions, open_now.contested
            )
        self.run.write_json(RunFile.QUESTIONS, open_now.questions.to_json())

        self.report(relations)

    # %% what the ontology makes of a pair

    def ontology_view(
        self, one_class: Optional[Type], other_class: Optional[Type]
    ) -> OntologyView:
        """
        Ask the ontology what two classes may be to one another.

        The status is decided by the part-whole channel alone, because that is the one that
        discriminates: ``contains`` is admissible between almost any two annotations, since
        ``IsStorageSpace.objects`` accepts anything with a root body, so letting it decide
        would report a relation for every pair in the room. The other channels are reported
        beside it instead, since a mug on a counter and a jar in a box do stand in one, and
        an adjudication told only about parts would force "part" onto them.

        :param one_class: The class of one segment, or None when its label maps to none.
        :param other_class: The class of the other segment.
        :return: The admissible relations and what that leaves open.
        """
        if one_class is None or other_class is None:
            return OntologyView(status=RelationStatus.CLASS_UNKNOWN)

        admissible = [
            AdmissibleRelation(
                whole=relation.whole.__name__,
                part=relation.part.__name__,
                field_name=relation.field_name,
                holds_many=relation.holds_many,
                removes_geometry=relation.removes_part_geometry_from_whole,
            )
            for relation in admissible_relations(one_class, other_class)
        ]
        if not admissible:
            status = RelationStatus.NO_LEGAL_RELATION
        elif len(admissible) == 1:
            status = RelationStatus.RELATION_KNOWN
        else:
            status = RelationStatus.RELATION_AMBIGUOUS

        return OntologyView(
            status=status,
            admissible=admissible,
            other_mounts=[
                AdmissibleMount(
                    kind=relation.kind,
                    whole=whole.__name__,
                    field_name=relation.field_name,
                    target=relation.target,
                    mounted_by=relation.mounted_by,
                )
                for whole, relation in admissible_mounts(one_class, other_class)
                if relation.kind != "part"
            ],
        )

    def relations_of(
        self,
        loader: WarsawWorldLoader,
        measured: SegmentRelations,
        classes: Dict[str, Optional[Type]],
    ) -> Relations:
        """
        :param loader: The loaded scene.
        :param measured: The measured scene.
        :param classes: Per label, the class it was read as.
        :return: Every measured pair, with what the ontology makes of it.
        """
        pairs = []
        for pair in measured.pairs:
            one = measured.descriptors[pair.one]
            other = measured.descriptors[pair.other]
            pairs.append(
                PairRecord(
                    evidence=pair,
                    classes={
                        one.class_name: self.class_name_of(classes.get(one.class_name)),
                        other.class_name: self.class_name_of(
                            classes.get(other.class_name)
                        ),
                    },
                    view=self.ontology_view(
                        classes.get(one.class_name), classes.get(other.class_name)
                    ),
                    prompt_block=pair.as_prompt_block(measured.descriptors),
                )
            )
        return Relations(
            scene=str(loader.scene.mesh_path),
            segments=list(measured.descriptors.values()),
            pairs=pairs,
        )

    @staticmethod
    def class_name_of(annotation_class: Optional[Type]) -> Optional[str]:
        """
        :param annotation_class: A class, or None where a label maps to none.
        :return: Its name, or None.
        """
        return None if annotation_class is None else annotation_class.__name__

    # %% the question asking what the labels mean

    @staticmethod
    def exemplars(measured: SegmentRelations) -> Dict[str, str]:
        """
        Pick the instance of each label that shows the label best.

        The one with the most surface no other segment claims is the one a viewer can
        judge without judging an overlap at the same time. Ranking by the *share*
        instead picks slivers, which are wholly unclaimed precisely because they are
        fragments: on one scene it offered a strip of 545 faces as the example of a
        cabinet while a whole cabinet front of 960 stood beside it, equally unclaimed.

        :param measured: The measured scene.
        :return: Per label, the name of the segment standing for it.
        """
        best: Dict[str, str] = {}
        for descriptor in measured.descriptors.values():
            standing = best.get(descriptor.class_name)
            if (
                standing is None
                or descriptor.exclusive_area
                > measured.descriptors[standing].exclusive_area
            ):
                best[descriptor.class_name] = descriptor.name
        return best

    def vocabulary_request(
        self, loader: WarsawWorldLoader, measured: SegmentRelations
    ) -> VocabularyRequest:
        """
        :param loader: The loaded scene.
        :param measured: The measured scene.
        :return: The question asking which class each label means.
        """
        instances = Counter(
            descriptor.class_name for descriptor in measured.descriptors.values()
        )
        return VocabularyRequest(
            scene=str(loader.scene.mesh_path),
            question=self.question_prompt,
            labels=[
                LabelRequest(
                    label=label,
                    instances=instances[label],
                    exemplar=name,
                    exemplar_faces=measured.descriptors[name].faces,
                    exemplar_exclusive_share=round(
                        measured.descriptors[name].exclusive_share, 4
                    ),
                    exemplar_exclusive_area=round(
                        measured.descriptors[name].exclusive_area, 4
                    ),
                )
                for label, name in sorted(self.exemplars(measured).items())
            ],
        )

    # %% rendering

    @staticmethod
    def neighbourhood(
        measured: SegmentRelations,
        segments: Dict[str, LabelSegment],
        names: Sequence[str],
    ) -> List[LabelSegment]:
        """
        Gather what a context view has to show for its subject to be recognisable.

        A picture of the whole room is useless for anything small -- a mug covers none
        of it. Framed on the mug together with the segments measured to stand nearest
        it, the same mug covers a tenth of the picture, and what surrounds it is what
        says which mug it is. The neighbours come from the same measurement the rest of
        the evidence does, so the neighbourhood is not a radius anyone chose.

        :param measured: The measured scene.
        :param segments: The scene's segments by name.
        :param names: The segments the view is about.
        :return: Those segments together with the ones measured to stand near them.
        """
        wanted = set(names)
        for name in names:
            for pair in measured.pairs_of(name):
                wanted.update((pair.one, pair.other))
        return [segments[name] for name in sorted(wanted)]

    @staticmethod
    def group_highlights(
        segments: Sequence[LabelSegment], contested: np.ndarray
    ) -> Tuple[List[Tuple[Color, np.ndarray]], Dict[str, Color]]:
        """
        Color a set of objects so that what they disagree about can be seen.

        Each gets a color of its own, and the faces all of them claim get one more,
        painted last. A picture without that last color shows the contested faces as
        belonging to whichever object was painted over them, which is the very thing in
        question.

        :param segments: The objects to color.
        :param contested: The faces they all claim.
        :return: What to paint, and what each color stands for.
        """
        colors = Color.distinct_colors(len(segments) + 1)
        highlights = [
            (color, segment.faces) for color, segment in zip(colors, segments)
        ]
        highlights.append((colors[-1], contested))
        legend = {str(segment.name): color for color, segment in zip(colors, segments)}
        legend["contested"] = colors[-1]
        return highlights, legend

    @staticmethod
    def write_images(
        images: Dict[str, bytes], directory: Path, prefix: str
    ) -> List[str]:
        """
        :param images: The renders to write, by viewpoint.
        :param directory: Where to write them.
        :param prefix: What to name them after.
        :return: The names they were written as.
        """
        directory.mkdir(parents=True, exist_ok=True)
        written = []
        for name, image in images.items():
            filename = f"{prefix}__{name}.png"
            (directory / filename).write_bytes(image)
            written.append(filename)
        return sorted(written)

    def render_exemplars(
        self,
        loader: WarsawWorldLoader,
        measured: SegmentRelations,
        segments: Dict[str, LabelSegment],
        request: VocabularyRequest,
    ) -> None:
        """
        Render one object per label, and record what each was painted.

        :param loader: The loaded scene.
        :param measured: The measured scene.
        :param segments: The scene's segments by name.
        :param request: The question to fill the renders into.
        """
        self.logger.info("rendering %s exemplars ...", len(request.labels))
        for entry in request.labels:
            segment = segments[entry.exemplar]
            color = Color.distinct_colors(1)[0]
            entry.images = self.write_images(
                loader.render_region(
                    [segment],
                    [(color, segment.faces)],
                    viewpoints=self.viewpoints,
                    headless=self.settings.headless,
                    choose_viewpoint=self.chooses_viewpoint,
                    context_segments=self.neighbourhood(
                        measured, segments, [entry.exemplar]
                    ),
                ),
                self.run.path(RunFile.EXEMPLARS),
                entry.label,
            )
            entry.color = color.closest_css3_name()
            self.logger.info("  %s: %s", entry.label, entry.exemplar)

    def render_questions(
        self,
        loader: WarsawWorldLoader,
        measured: SegmentRelations,
        segments: Dict[str, LabelSegment],
        questions: OpenQuestions,
        contested: Dict[str, np.ndarray],
    ) -> None:
        """
        Render each open question, and record what each color in it stands for.

        :param loader: The loaded scene.
        :param measured: The measured scene.
        :param segments: The scene's segments by name.
        :param questions: The questions to fill the renders into.
        :param contested: Per question, the faces all its objects claim.
        """
        asked = (
            questions.ownership[: self.question_renders]
            + questions.membership[: self.question_renders]
        )
        self.logger.info("rendering %s questions ...", len(asked))
        for question in asked:
            shown = [segments[name] for name in question.shown]
            highlights, legend = self.group_highlights(
                shown, contested.get(question.name, np.array([], dtype=np.int64))
            )
            question.images = self.write_images(
                loader.render_region(
                    shown,
                    highlights,
                    viewpoints=self.viewpoints,
                    headless=self.settings.headless,
                    choose_viewpoint=self.chooses_viewpoint,
                    context_segments=self.neighbourhood(
                        measured, segments, question.shown
                    ),
                ),
                self.run.path(RunFile.QUESTION_RENDERS),
                question.name,
            )
            question.legend = {
                name: color.closest_css3_name() for name, color in legend.items()
            }
            self.logger.info("  %s", question.name)

    # %% what is left to decide

    @staticmethod
    def ontology_slice(
        names: Sequence[str],
        labels: Dict[str, str],
        classes: Dict[str, Optional[Type]],
    ) -> OntologySlice:
        """
        Say what the ontology holds about a handful of objects.

        The questions about a set of objects were asked with the pictures and the
        measurements alone, and the ontology already knows things that bear on them:
        that a cabinet can hold a drawer says the drawer is the finer of the two, which
        is exactly what "whose surface is this" turns on.

        :param names: The segments in question.
        :param labels: Per segment, the label it carries.
        :param classes: Per label, the class it was read as.
        :return: Per segment its class, each class written out, and what the ontology
            admits between each pair of them.
        """
        read_as = {name: classes.get(labels[name]) for name in names}
        admits = []
        for one, other in combinations(names, 2):
            if read_as[one] is None or read_as[other] is None:
                continue
            for whole, relation in admissible_mounts(read_as[one], read_as[other]):
                admits.append(
                    f"{whole.__name__}.{relation.field_name} may hold a "
                    f"{relation.target} ({relation.kind}, mounted with "
                    f"{relation.mounted_by}())"
                )
        return OntologySlice(
            read_as={
                name: None if held is None else held.__name__
                for name, held in read_as.items()
            },
            classes=[
                describe_class(held)
                for held in dict.fromkeys(
                    one for one in read_as.values() if one is not None
                )
            ],
            # The same mount turns up once per pair that could use it, and almost every
            # class has objects -> HasRootBody, so without this the list reads as though
            # containment were being urged six times over.
            admits=list(dict.fromkeys(admits)),
        )

    @staticmethod
    def measured_of(
        names: Sequence[str], measured: SegmentRelations
    ) -> Dict[str, MeasuredSegment]:
        """
        :param names: The segments in question.
        :param measured: The measured scene.
        :return: Per segment, what was measured of it on its own.
        """
        return {
            name: MeasuredSegment(
                faces=int(measured.descriptors[name].faces),
                area=round(measured.descriptors[name].area, 3),
                height=round(measured.descriptors[name].height, 2),
                pieces=int(measured.descriptors[name].components),
            )
            for name in names
        }

    def open_questions(
        self,
        loader: WarsawWorldLoader,
        measured: SegmentRelations,
        relations: Relations,
        classes: Dict[str, Optional[Type]],
    ) -> WhatIsOpen:
        """
        Work out what is actually left to decide, and how few questions it takes.

        Three things shrink the pile. A face belongs to one object, so the question is
        asked once per set of claimants rather than once per pair of them. A group whose
        every internal pair the ontology settled as a part-whole relation is not a question
        at all, since the part keeps the surface it is made of. And what is left repeats: a
        door and a window sharing a pane is one question however many glazed doors the room
        has, so the groups are gathered by the classes in them and asked once per pattern.

        Which whole a part belongs to is a separate question, and only where a part
        overlaps more than one candidate: a drawer that meets exactly one cabinet has
        nothing to choose between.

        :param loader: The loaded scene.
        :param measured: The measured scene.
        :param relations: The pairs, as they were written out.
        :param classes: Per label, the class it was read as.
        :return: What is left to decide, and what each question is rendered around.
        """
        segments = loader.label_segments
        groups = claimant_groups(
            [segment.faces for segment in segments],
            [str(segment.name) for segment in segments],
            len(loader.scene_mesh.faces),
        )
        status = {
            tuple(sorted((pair.one, pair.other))): pair.status
            for pair in relations.pairs
        }
        labels = relations.labels

        settled, patterns = [], defaultdict(list)
        for group in groups:
            inside = [
                status.get(tuple(sorted(pair))) for pair in combinations(group.names, 2)
            ]
            if all(answer is RelationStatus.RELATION_KNOWN for answer in inside):
                settled.append(ClaimantSet.from_json(group.to_json()))
            else:
                patterns[tuple(sorted(labels[name] for name in group.names))].append(
                    group
                )

        ownership, contested = [], {}
        for pattern, members in sorted(patterns.items(), key=lambda one: -len(one[1])):
            exemplar = max(members, key=lambda group: len(group.faces))
            question = OwnershipQuestion(
                name="__".join(pattern),
                pattern=list(pattern),
                shown=list(exemplar.names),
                covers=[ClaimantSet.from_json(group.to_json()) for group in members],
                contested_faces=sum(len(group.faces) for group in members),
                exemplar_faces=int(len(exemplar.faces)),
                shares={
                    name: ContestedShare(
                        faces=int(measured.descriptors[name].faces),
                        contested_share=round(
                            len(exemplar.faces) / measured.descriptors[name].faces, 4
                        ),
                    )
                    for name in exemplar.names
                },
                ontology=self.ontology_slice(exemplar.names, labels, classes),
                measured=self.measured_of(exemplar.names, measured),
            )
            ownership.append(question)
            contested[question.name] = exemplar.faces

        memberships = self.memberships(relations, labels, classes, measured)
        return WhatIsOpen(
            questions=OpenQuestions(
                scene=relations.scene,
                ownership=ownership,
                membership=memberships.to_ask_about,
                settled=settled,
                forced=memberships.forced,
            ),
            contested=contested,
        )

    def memberships(
        self,
        relations: Relations,
        labels: Dict[str, str],
        classes: Dict[str, Optional[Type]],
        measured: SegmentRelations,
    ) -> Memberships:
        """
        Work out which parts have a whole to choose and which have only one.

        A part is attached to the whole it belongs to, so a candidate has to share faces
        with it or touch it along an edge. Everything else the measurement reached is
        merely nearby, and offering it as an alternative is offering a wrong answer: it
        trebles the questions and none of what it adds could be right.

        :param relations: The pairs, as they were written out.
        :param labels: Per segment, the label it carries.
        :param classes: Per label, the class it was read as.
        :param measured: The measured scene.
        :return: The parts with a whole to choose, and the parts with only one.
        """
        candidates: Dict[str, Dict[str, MembershipCandidate]] = defaultdict(dict)
        for pair in relations.pairs:
            if pair.status is not RelationStatus.RELATION_KNOWN:
                continue
            if not pair.evidence.shared_faces and not pair.evidence.touching_edges:
                continue
            admitted = pair.view.admissible[0]
            whole, part = pair.one, pair.other
            if pair.classes.get(labels[whole]) != admitted.whole:
                whole, part = part, whole
            candidates[part][whole] = MembershipCandidate(
                field_name=admitted.field_name,
                shared_faces=pair.evidence.shared_faces,
                touching_edges=pair.evidence.touching_edges,
                distance=round(pair.evidence.distance, 4),
            )

        to_ask_about = [
            MembershipQuestion(
                name=part,
                part=part,
                shown=[part] + sorted(wholes),
                candidates=dict(sorted(wholes.items())),
                ontology=self.ontology_slice([part] + sorted(wholes), labels, classes),
                measured=self.measured_of([part] + sorted(wholes), measured),
            )
            for part, wholes in sorted(candidates.items())
            if len(wholes) > 1
        ]
        forced = [
            ForcedMembership(
                part=part,
                whole=next(iter(wholes)),
                field_name=next(iter(wholes.values())).field_name,
                shared_faces=next(iter(wholes.values())).shared_faces,
                touching_edges=next(iter(wholes.values())).touching_edges,
                distance=next(iter(wholes.values())).distance,
            )
            for part, wholes in sorted(candidates.items())
            if len(wholes) == 1
        ]
        return Memberships(to_ask_about=to_ask_about, forced=forced)

    def report(self, relations: Relations) -> None:
        """
        Say what the ontology made of the pairs.

        :param relations: The pairs, as they were written out.
        """
        taxonomy = self.run.read_json(RunFile.TAXONOMY)
        self.logger.info(
            "ontology: %s classes, %s mixins to compose new ones from",
            len(taxonomy["classes"]),
            len(taxonomy["part_whole_mixins"]),
        )
        counted = Counter(pair.status for pair in relations.pairs)
        self.logger.info("what the ontology makes of the pairs:")
        for status, count in counted.most_common():
            self.logger.info("  %-20s %s", status.value, count)
        self.logger.info("written to %s", self.run.directory)
