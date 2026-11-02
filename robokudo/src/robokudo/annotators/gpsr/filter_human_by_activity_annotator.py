from __future__ import annotations

# %% Imports
import cv2
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.gpsr.lying import HumanLyingAnnotator
from robokudo.cas import CASViews
from robokudo.types.human import ActivityAnnotation
from robokudo.types.scene import HumanHypothesis
from robokudo.utils.error_handling import catch_and_raise_to_blackboard

if TYPE_CHECKING:
    from robokudo.cas import CAS

# %% activity filter


class FilterHumanByActivityAnnotator(BaseAnnotator):
    """
    Keeps only the persons that hold the posture and make the gesture the query asks
    for.

    The first attribute of the query is the posture, the second the gesture; an empty or
    missing attribute does not filter. Both are compared to the activities of a person
    without regard to letter case or surrounding spaces, and the query word for lying
    stands for the lying posture. When the query filters, every annotation other than
    the matching persons is removed from the CAS.
    """

    def __init__(self, name: str = "FilterHumanByActivityAnnotator") -> None:
        super().__init__(name=name)

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        query = self.get_cas().get(CASViews.QUERY)
        attributes = query.obj.attribute
        cas = self.get_cas()
        matching_humans = []

        if len(attributes) > 0:
            posture_filter = attributes[0].strip().lower()
        else:
            posture_filter = ""
        if posture_filter == HumanLyingAnnotator.query_word:
            posture_filter = HumanLyingAnnotator.posture

        if len(attributes) > 1:
            gesture_filter = attributes[1].strip().lower()
        else:
            gesture_filter = ""

        if not posture_filter and not gesture_filter:
            self.feedback_message = "No activity filter requested, skipping"
            return Status.SUCCESS

        for human in cas.annotations:
            if not isinstance(human, HumanHypothesis):
                continue

            activity_types = {
                annotation.type.lower()
                for annotation in human.annotations
                if isinstance(annotation, ActivityAnnotation)
            }

            if posture_filter == "":
                holds_posture = True
            else:
                holds_posture = posture_filter in activity_types

            if gesture_filter == "":
                makes_gesture = True
            else:
                makes_gesture = gesture_filter in activity_types

            if holds_posture and makes_gesture:
                matching_humans.append(human)

        self._draw_filter_overlay(cas, matching_humans)
        cas.annotations = matching_humans
        return Status.SUCCESS

    def _draw_filter_overlay(
        self, cas: CAS, matching_humans: list[HumanHypothesis]
    ) -> None:
        """
        Draw every person of the CAS into the output image: a green frame for a matching
        person, a red frame over a red veil for a removed one.

        :param cas: The CAS whose color image and persons are drawn.
        :param matching_humans: The persons the filter keeps.
        """
        image = cas.get(CASViews.COLOR_IMAGE)
        if image is None:
            return
        image = image.copy()
        for human in cas.annotations:
            if not isinstance(human, HumanHypothesis):
                continue
            region = human.roi.roi
            x, y = int(region.pos.x), int(region.pos.y)
            width, height = int(region.width), int(region.height)
            kept = human in matching_humans
            color = (0, 200, 0) if kept else (0, 0, 255)
            if not kept:
                overlay = image.copy()
                cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
                cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 3)
            cv2.putText(
                image,
                "match" if kept else "filtered",
                (x, max(y - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
        self.get_annotator_output_struct().set_image(image)
