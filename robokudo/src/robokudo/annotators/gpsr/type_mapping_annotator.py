from __future__ import annotations

# %% Imports
import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis

# %% type mapping


class TypeMappingAnnotator(BaseAnnotator):
    """
    Gives each object hypothesis the semantic type its first classification maps to, as
    a new first classification.

    A hypothesis that already carries the semantic type, or whose first classification
    has no mapping, is left unchanged. When a classification is added, the other
    annotations of the hypothesis follow its classifications.
    """

    def __init__(
        self,
        name: str = "TypeMappingAnnotator",
        mapping_filename: str = "mapping_perception_semantic-annotations.json",
        mapping_directory: Path | None = None,
    ) -> None:
        """
        :param mapping_filename: Name of the mapping file.
        :param mapping_directory: Directory containing the mapping, if configured directly.
        """
        super().__init__(name=name)
        self._type_map: dict[str, str] = {}
        """
        The semantic type of each detector class, empty if the mapping file could not be
        read.
        """
        try:
            if mapping_directory is None:
                package_share = get_package_share_directory("robokudo_suturo")
                mapping_path = Path(package_share) / "data" / "json" / mapping_filename
            else:
                mapping_path = mapping_directory / mapping_filename
            with open(mapping_path, "r") as mapping_file:
                self._type_map = json.load(mapping_file)
        except Exception as error:
            self.rk_logger.warning(
                f"Could not load mapping file '{mapping_filename}': {error}"
            )

    def update(self) -> Status:
        cas = self.get_cas()
        for object_hypothesis in cas.annotations:
            if not isinstance(object_hypothesis, ObjectHypothesis):
                continue

            classifications = []
            other_annotations = []
            for annotation in object_hypothesis.annotations:
                if isinstance(annotation, Classification):
                    classifications.append(annotation)
                else:
                    other_annotations.append(annotation)
            if not classifications:
                continue

            detected_class = classifications[0].classname
            semantic_type = self._type_map.get(detected_class)
            if not semantic_type:
                continue

            for classification in classifications:
                if classification.classname == semantic_type:
                    break
            else:
                semantic_classification = Classification()
                semantic_classification.classname = semantic_type
                semantic_classification.source = self.name
                object_hypothesis.annotations = (
                    [semantic_classification] + classifications + other_annotations
                )

        return Status.SUCCESS
