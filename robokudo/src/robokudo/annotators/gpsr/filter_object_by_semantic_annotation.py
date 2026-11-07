"""
Removal of everything but the objects of the queried semantic type.
"""

from __future__ import annotations

# %% Imports
import cv2
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.gpsr.type_mapping_annotator import (
    SemanticTypeMappingAnnotator,
)
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.error_handling import catch_and_raise_to_blackboard

if TYPE_CHECKING:
    from robokudo.cas import CAS

# %% semantic type filter


class FilterObjectBySemanticAnnotation(SemanticTypeMappingAnnotator):
    """
    Keeps only the object hypotheses with a classification of the queried semantic type.

    The type is the query type, or the first description when the query type is not in
    the mapping file. All other annotations are removed from the CAS, which stays
    unchanged when the query names no type of the mapping.
    """

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        """
        Overrides :meth:`BaseAnnotator.update` to remove everything but the object
        hypotheses of the queried semantic type.

        :return: :attr:`Status.SUCCESS`.
        """
        query = self.get_cas().get(CASViews.QUERY)
        # Receptionist sends the type itself, the GPSR dispatcher sends 'object' with
        # the type in the description.
        query_type = query.obj.type
        if query_type not in self._type_mapping.values() and query.obj.description:
            query_type = query.obj.description[0]

        if query_type not in self._type_mapping.values():
            self.feedback_message = (
                f"No semantic category to filter by ('{query_type}'), skipping"
            )
            return Status.SUCCESS

        cas = self.get_cas()
        object_hypotheses = [
            annotation
            for annotation in cas.annotations
            if isinstance(annotation, ObjectHypothesis)
        ]
        matching_object_hypotheses = [
            object_hypothesis
            for object_hypothesis in object_hypotheses
            if self._matches(object_hypothesis, query_type)
        ]

        try:
            self._visualize(cas, object_hypotheses, query_type)
        except Exception as error:
            self.rk_logger.warning(f"Could not render filter visualization: {error}")

        cas.annotations = matching_object_hypotheses
        self.feedback_message = f"Kept {len(matching_object_hypotheses)} object(s) with annotation '{query_type}'"
        return Status.SUCCESS

    def _matches(self, object_hypothesis: ObjectHypothesis, query_type: str) -> bool:
        """
        :param object_hypothesis: The hypothesis to check.
        :param query_type: The queried semantic type.
        :return: Whether any classification of the hypothesis names the queried semantic
            type.
        """
        for annotation in object_hypothesis.annotations:
            if (
                isinstance(annotation, Classification)
                and annotation.classname == query_type
            ):
                return True
        return False

    def _create_label(self, object_hypothesis: ObjectHypothesis) -> str:
        """
        :param object_hypothesis: The hypothesis to label.
        :return: The first detector class of the hypothesis that the mapping knows and
            its semantic type, as ``class -> type``. Without such a class, the first
            classification. Without any classification, ``object``.
        """
        classnames = [
            annotation.classname
            for annotation in object_hypothesis.annotations
            if isinstance(annotation, Classification)
        ]
        detector_class = next(
            (classname for classname in classnames if classname in self._type_mapping),
            None,
        )
        if detector_class is not None:
            return f"{detector_class} -> {self._type_mapping[detector_class]}"
        return classnames[0] if classnames else "object"

    def _visualize(
        self,
        cas: CAS,
        object_hypotheses: list[ObjectHypothesis],
        query_type: str,
    ) -> None:
        """
        Draw the box and the label of every object hypothesis into the output image,
        green for a matching hypothesis and red for a removed one.

        :param cas: The CAS whose color image is drawn on.
        :param object_hypotheses: The hypotheses to draw.
        :param query_type: The semantic type the hypotheses are matched against.
        """
        image = cas.get_copy(CASViews.COLOR_IMAGE)
        font = cv2.FONT_HERSHEY_COMPLEX
        for object_hypothesis in object_hypotheses:
            color = (
                (0, 255, 0)
                if self._matches(object_hypothesis, query_type)
                else (0, 0, 255)
            )
            left = int(object_hypothesis.roi.roi.pos.x)
            top = int(object_hypothesis.roi.roi.pos.y)
            right = left + int(object_hypothesis.roi.roi.width)
            bottom = top + int(object_hypothesis.roi.roi.height)
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            cv2.putText(
                image,
                self._create_label(object_hypothesis),
                (left, max(top - 5, 12)),
                font,
                0.5,
                color,
                1,
                2,
            )
        self.get_annotator_output_struct().set_image(image)
