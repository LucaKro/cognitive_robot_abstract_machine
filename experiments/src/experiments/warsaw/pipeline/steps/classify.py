"""
Ask what each of a split scene's bodies is.

The vocabulary step answered what the *label* ``cabinet`` means in the ontology. This
asks the other question that was hiding under it: whether this particular object really
is one. It could not be asked before, because until the mesh was split, colouring
``cabinet_4`` painted exactly ``door_10``'s faces and a model would rightly have
answered "a door".

Bodies are addressed by the name they carry everywhere else -- ``drawer_19``, the same
key in the split record, in the pairings, and in the world the split built -- and an
answer is kept only if it names bodies that were actually in the picture. Nothing else
in the pipeline can mount a part into a whole if the two ends stop meaning the same
objects, so the identity is checked rather than assumed.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

import numpy as np
from semantic_digital_twin.adapters.vision_language_model.client import ModelResponse
from semantic_digital_twin.adapters.vision_language_model.exceptions import (
    ModelRefusedError,
)
from semantic_digital_twin.adapters.vision_language_model.message import (
    ImagePart,
    MessagePart,
    TextPart,
)
from typing_extensions import Any, Dict, List, Sequence

from experiments.warsaw.pipeline.asking import Question
from experiments.warsaw.pipeline.label_classes import VocabularyClasses
from experiments.warsaw.pipeline.records import BodyAnswer, Classifications, Vocabulary
from experiments.warsaw.pipeline.run import RunFile
from experiments.warsaw.pipeline.steps.step import PipelineStep
from experiments.warsaw.world_loader.loader import (
    RenderedSegmentGroup,
    WarsawWorldLoader,
)
from experiments.warsaw.world_loader.scene import LabelSegment

# %% what a handful of painted bodies are


@dataclass(kw_only=True)
class BodyGroupQuestion(Question[Dict[str, BodyAnswer]]):
    """
    What a handful of painted bodies are, put to a model with the room around them.
    """

    rendered: RenderedSegmentGroup
    """
    The painted bodies and the renders they were painted into.
    """

    taxonomy: Dict[str, Any]
    """
    The ontology as a model reads it, widened with the classes this run proposed.
    """

    vocabulary: Vocabulary
    """
    What the vocabulary step answered per label.
    """

    prompt: str = "classification"
    """
    The prompt this question is put with, both halves of it.
    """

    @property
    def key(self) -> str:
        return f"group{self.rendered.index}"

    def listed(self) -> List[Dict[str, Any]]:
        """
        :return: Per body in the pictures, what a model is told about it: the colour it
            was painted, the label the scan gave it, and what that label was read as.
        """
        return [
            {
                "name": f"{segment.name}",
                "color": self.rendered.colors[segment.name].closest_css3_name(),
                "label": segment.class_name,
                "read_as": self.vocabulary.answer_for(segment.class_name).class_name
                or "no class",
            }
            for segment in self.rendered.segments
        ]

    def message(self) -> List[MessagePart]:
        content: List[MessagePart] = [
            TextPart(
                self.templates.render_document(
                    self.message_template,
                    taxonomy=json.dumps(self.taxonomy),
                    bodies=self.listed(),
                )
            )
        ]
        for viewpoint, image in sorted(self.rendered.images.items()):
            content.append(TextPart(f"The room from the {viewpoint}."))
            content.append(ImagePart(image=image))
        return content

    def read(self, response: ModelResponse) -> Dict[str, BodyAnswer]:
        answered = response.parse_json()
        # Asked for {"objects": [...]}, a model will sometimes answer with the array alone.
        objects = (
            answered if isinstance(answered, list) else answered.get("objects", [])
        )
        return {
            str(one["name"]): BodyAnswer.spoken(one)
            for one in objects
            if one.get("name")
        }

    def refusal(self, refused: ModelRefusedError) -> Dict[str, BodyAnswer]:
        return {}

    def problems_with(self, answer: Dict[str, BodyAnswer]) -> List[str]:
        """
        Say what is wrong with an answer about a group, if anything.

        :param answer: What the model said, by body name.
        :return: One sentence per problem, empty when there are none.
        """
        wanted = {str(segment.name) for segment in self.rendered.segments}
        problems = []
        missing = sorted(wanted - set(answer))
        unknown = sorted(set(answer) - wanted)
        if missing:
            problems.append(f"nothing was said about {', '.join(missing)}")
        if unknown:
            problems.append(
                f"{', '.join(unknown)} were named but are not in the picture"
            )
        for name, one in answer.items():
            if name in wanted and not one.class_name:
                problems.append(f"{name} was given no class")
        return problems


# %% naming every body of the scene


@dataclass
class ClassifyBodies(PipelineStep):
    """
    What each body of a split scene is, asked a group at a time.
    """

    @property
    def name(self) -> str:
        return "name each body"

    def carry_out(self) -> None:
        """
        Paint every body, ask what it is, and write the answers.
        """
        vocabulary = self.run.read_record(RunFile.VOCABULARY, Vocabulary)
        taxonomy = VocabularyClasses(
            vocabulary=vocabulary, known=self.ontology_classes()
        ).widened(self.run.read_json(RunFile.TAXONOMY))

        loader = WarsawWorldLoader(input_directory=self.settings.scene_directory)
        bodies = self.split_segments(loader)
        self.logger.info(
            "%s bodies to name, %s at a time", len(bodies), self.settings.group_size
        )

        renders = self.run.directory_for(RunFile.CLASSIFICATION_RENDERS)
        questioner = self.questioner(RunFile.CLASSIFICATION_ANSWERS)
        named: List[BodyAnswer] = []

        for rendered in loader.render_label_segment_groups(
            group_size=self.settings.group_size,
            segments=bodies,
            headless=self.settings.headless,
        ):
            for viewpoint, image in rendered.images.items():
                (renders / f"group{rendered.index}__{viewpoint}.png").write_bytes(image)

            answered = questioner.answer(
                BodyGroupQuestion(
                    rendered=rendered,
                    taxonomy=taxonomy,
                    vocabulary=vocabulary,
                )
            )
            for segment in rendered.segments:
                name = str(segment.name)
                answer = answered.answer.get(name, BodyAnswer())
                answer.name = name
                answer.label = segment.class_name
                answer.faces = int(len(segment))
                named.append(answer)
                new = " [new]" if answer.is_new_class else ""
                self.logger.info("  %-22s -> %s%s", name, answer.class_name, new)
            for problem in answered.problems:
                self.logger.warning("      ! %s", problem)

        classifications = Classifications(
            model=self.settings.model.value,
            scene=str(loader.scene.mesh_path),
            bodies=named,
        )
        self.run.write_record(RunFile.CLASSIFICATIONS, classifications)
        self.report(classifications, vocabulary)

    def split_segments(self, loader: WarsawWorldLoader) -> List[LabelSegment]:
        """
        Rebuild the scene's segments from the faces the split left them.

        The name a segment carries is made of its label and instance, so a segment
        rebuilt with the split's faces answers to exactly the name its body has -- which
        is what lets an answer about ``drawer_19`` reach the body called ``drawer_19``.

        :param loader: The loaded scene.
        :return: One segment per body, carrying only the faces that body kept.
        """
        kept = np.load(self.run.path(RunFile.SPLIT_FACES))
        by_name = {str(segment.name): segment for segment in loader.label_segments}
        return [
            LabelSegment(
                class_name=by_name[name].class_name,
                instance=by_name[name].instance,
                face_indices=kept[name],
            )
            for name in kept.files
        ]

    def report(self, classifications: Classifications, vocabulary: Vocabulary) -> None:
        """
        :param classifications: What every body was answered to be.
        :param vocabulary: What every label was answered to mean.
        """
        counted = Counter(
            one.class_name for one in classifications.bodies if one.class_name
        )
        agreed = sum(
            1
            for one in classifications.bodies
            if one.class_name
            and one.class_name == vocabulary.answer_for(one.label).class_name
        )
        self.logger.info(
            "%s distinct classes over %s bodies; %s agree with what their label was "
            "mapped to",
            len(counted),
            len(classifications.bodies),
            agreed,
        )
        self.logger.info("written to %s", self.run.path(RunFile.CLASSIFICATIONS))
