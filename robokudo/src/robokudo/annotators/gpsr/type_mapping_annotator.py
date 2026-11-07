"""
Mapping of detector classes to semantic types.
"""

from __future__ import annotations

# %% Imports
import json

from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.file_loader import FileLoader


# %% semantic type mapping


class SemanticTypeMappingAnnotator(BaseAnnotator):
    """
    Annotator that knows the semantic type of each detector class.

    The mapping is read from a JSON file of the ``robokudo_suturo`` package.
    """

    def __init__(
        self,
        name: str | None = None,
        mapping_file_name: str = "mapping_perception_semantic-annotations.json",
    ) -> None:
        """
        :param name: Name of this annotator in the tree, the class name if not given.
        :param mapping_file_name: Name of the mapping file.
        """
        super().__init__(name=self.get_class_name() if name is None else name)
        self._type_mapping: dict[str, str] = {}
        """
        The semantic type of each detector class, empty if the mapping file could not be
        read.
        """
        try:
            mapping_path = FileLoader.get_path_to_file_in_ros_package(
                ros_pkg_name="robokudo_suturo",
                relative_path=f"data/json/{mapping_file_name}",
            )
            with open(mapping_path, "r") as mapping_file:
                self._type_mapping = json.load(mapping_file)
        except Exception as error:
            self.rk_logger.warning(
                f"Could not load mapping file '{mapping_file_name}': {error}"
            )


# %% type mapping


class TypeMappingAnnotator(SemanticTypeMappingAnnotator):
    """
    Gives each object hypothesis the semantic type of its first classification.

    The type becomes a new first classification, followed by the other classifications
    and then the other annotations. A hypothesis that already carries the type, or whose
    first classification has no mapping, is left unchanged.
    """

    def update(self) -> Status:
        """
        Overrides :meth:`BaseAnnotator.update` to add the semantic type to each object
        hypothesis with a mapped classification.

        :return: :attr:`Status.SUCCESS`.
        """
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
            semantic_type = self._type_mapping.get(detected_class)
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
