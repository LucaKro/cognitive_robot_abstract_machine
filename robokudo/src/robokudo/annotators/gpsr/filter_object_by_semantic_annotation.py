from __future__ import annotations

# %% Imports
import json

import cv2
from ament_index_python.packages import get_package_share_directory
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.error_handling import catch_and_raise_to_blackboard

if TYPE_CHECKING:
    from robokudo.cas import CAS

# %% semantic type filter


class FilterObjectBySemanticAnnotation(BaseAnnotator):
    """
    Keeps only the object hypotheses classified as the semantic type the query asks for.

    The semantic type is the query type or, when the query type is no semantic type of
    the mapping file, the first description of the query. A hypothesis matches when any
    of its classifications names that type. When the query names no semantic type of the
    mapping, the CAS is left unchanged; otherwise every annotation other than the
    matching hypotheses is removed from it.
    """

    def __init__(
        self,
        name: str = "FilterObjectBySemanticAnnotation",
        mapping_filename: str = "mapping_perception_semantic-annotations.json",
    ) -> None:
        """
        :param mapping_filename: Name of the mapping file in ``data/json`` of the
            ``robokudo_suturo`` package.
        """
        super().__init__(name=name)
        self._type_map: dict[str, str] = {}
        """
        The semantic type of each detector class, empty if the mapping file could not be
        read.
        """
        try:
            package_share = get_package_share_directory("robokudo_suturo")
            mapping_path = f"{package_share}/data/json/{mapping_filename}"
            with open(mapping_path, "r") as mapping_file:
                self._type_map = json.load(mapping_file)
        except Exception as error:
            self.rk_logger.warning(
                f"Could not load mapping file '{mapping_filename}': {error}"
            )

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        query = self.get_cas().get(CASViews.QUERY)
        # the queried category is normally in the type field (e.g. Receptionist);
        # the GPSR dispatcher puts 'object' there for routing and carries the
        # category in the description instead, so fall back to it.
        query_type = query.obj.type
        if query_type not in self._type_map.values() and query.obj.description:
            query_type = query.obj.description[0]

        if query_type not in self._type_map.values():
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
        matching_objects = [
            object_hypothesis
            for object_hypothesis in object_hypotheses
            if self._matches(object_hypothesis, query_type)
        ]

        try:
            self._visualize(cas, object_hypotheses, query_type)
        except Exception as error:
            self.rk_logger.warning(f"Could not render filter visualization: {error}")

        cas.annotations = matching_objects
        self.feedback_message = (
            f"Kept {len(matching_objects)} object(s) with annotation '{query_type}'"
        )
        return Status.SUCCESS

    def _matches(self, object_hypothesis: ObjectHypothesis, query_type: str) -> bool:
        """
        :return: Whether any classification of the hypothesis names the queried
            semantic type.
        """
        for annotation in object_hypothesis.annotations:
            if (
                isinstance(annotation, Classification)
                and annotation.classname == query_type
            ):
                return True
        return False

    def _label(self, object_hypothesis: ObjectHypothesis) -> str:
        """
        :return: The first detector class of the hypothesis that the mapping knows,
            with the semantic type it maps to, as ``class -> type``; without one the
            first classification, and ``object`` for a hypothesis without
            classification.
        """
        classnames = [
            annotation.classname
            for annotation in object_hypothesis.annotations
            if isinstance(annotation, Classification)
        ]
        detector_class = next(
            (classname for classname in classnames if classname in self._type_map),
            None,
        )
        if detector_class is not None:
            return f"{detector_class} -> {self._type_map[detector_class]}"
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
                self._label(object_hypothesis),
                (left, max(top - 5, 12)),
                font,
                0.5,
                color,
                1,
                2,
            )
        self.get_annotator_output_struct().set_image(image)
