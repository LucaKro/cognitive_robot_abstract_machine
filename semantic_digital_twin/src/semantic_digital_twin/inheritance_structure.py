"""
Export a class hierarchy as structured JSON.
"""

from __future__ import annotations

import inspect
import json
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import Any, Dict, Iterator, List, Type, get_args, get_origin

# %% the name a missing annotation is written under

UNANNOTATED_TYPE_NAME = "Any"
"""
What a required field carries as its type when it has no annotation.
"""

# %% exporting the hierarchy


@dataclass
class InheritanceStructureExporter:
    """
    Introspects a Python class hierarchy and exports a structured JSON representation of
    all subclasses derived from a given root class.

    The exporter builds a recursive tree where each node describes:
      - the class name
      - whether the class directly inherits from `ABC` (marking it as abstract)
      - the list of superclasses that are *outside* the main hierarchy
      - the list of subclasses (for non-superclass nodes)

    Superclasses are represented using the same node structure as regular classes
    but omit the `subclasses` field, preventing upward recursion.

    For each class, the exporter collects all public fields from the class's __init__ method.

    This exporter only includes:
      - direct and indirect subclasses of `root_class`
      - superclasses that are not `object`, not `ABC`, and not part of the root hierarchy
    """

    root_class: Type
    """
    The root class from which to start the hierarchy exploration.
    """

    output_path: Path = field(default=Path("class_hierarchy.json"))
    """
    The file path where the JSON output will be written.
    """

    def export(self) -> None:
        """
        Build the hierarchy and write it to `self.output_path` as JSON.
        """
        data = self._build_inheritance_structure()
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _build_inheritance_structure(self) -> Dict[str, Any]:
        """
        Build the full hierarchy as a Python dict.

        Root is represented by its name + its immediate subclasses.
        """
        return {
            "root_name": self.root_class.__name__,
            "subclasses": [
                self._build_node(sub, include_subclasses=True)
                for sub in self._walk_related_classes(
                    self.root_class, relation="subclasses"
                )
            ],
        }

    def _build_node(self, clazz: Type, *, include_subclasses: bool) -> Dict[str, Any]:
        """
        Recursively build a node representing `clazz` and its related classes.

        If `include_subclasses` is True, also include subclasses in the node.
        """
        node: Dict[str, Any] = {
            "name": clazz.__name__,
            "is_abstract": self._is_inheriting_from_abc(clazz),
            "other_superclasses": [
                self._build_node(base, include_subclasses=False)
                for base in self._walk_related_classes(clazz, relation="bases")
            ],
            "fields": self.collect_required_public_fields(clazz),
        }

        if include_subclasses:
            node["subclasses"] = [
                self._build_node(sub, include_subclasses=True)
                for sub in self._walk_related_classes(clazz, relation="subclasses")
            ]

        return node

    # %% reading a class's required fields

    def collect_required_public_fields(self, clazz: type) -> List[Dict[str, Any]]:
        """
        Collects all required public fields from the class's __init__ method.

        :param clazz: The class to inspect.
        :return: A list of dictionaries, each containing 'name' and 'type' of a required
            public field.
        """
        # The __init__ of a class that declares none is object's, which carries no
        # annotations at all, so both lookups go through inspect rather than the
        # attribute.
        init = clazz.__init__
        signature = inspect.signature(init)
        init_annotations = inspect.get_annotations(init)
        class_annotations = inspect.get_annotations(clazz)

        return [
            required_public_field
            for name, param in signature.parameters.items()
            if (
                required_public_field := self._get_only_required_public_fields(
                    name, param, init_annotations, class_annotations
                )
            )
        ]

    def _get_only_required_public_fields(
        self,
        name: str,
        param: inspect.Parameter,
        init_ann: Dict[str, Any],
        class_ann: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Returns a dictionary with field info if the parameter is a required public
        field, otherwise returns an empty dictionary.

        :param name: The name of the parameter.
        :param param: The inspect.Parameter object.
        :param init_ann: Annotations from the __init__ method.
        :param class_ann: Annotations from the class.
        :return: A dictionary with field info or an empty dictionary.
        """
        if name == "self":
            return {}

        # Skip private / "hidden" params
        if name.startswith("_"):
            return {}

        # Only normal args / keyword-only; no *args, **kwargs
        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return {}

        # Only required parameters: nothing with a default (even default=None)
        if param.default is not inspect._empty:
            return {}

        # Resolve an annotation if we have one
        ann = init_ann.get(name, class_ann.get(name, None))

        return {
            "name": name,
            "type": self._type_to_string(ann),
        }

    # %% naming a type

    def _type_to_string(self, tp: Any) -> str:
        """
        Convert a type annotation to a string representation.
        """
        # Missing annotation
        if tp is None:
            return UNANNOTATED_TYPE_NAME

        # Forward refs like "World"
        if isinstance(tp, str):
            return tp

        origin = get_origin(tp)
        args = get_args(tp)

        if origin is not None:
            # A typing special form such as Union is not a class but still reads back
            # under its own name.
            origin_name = (
                origin.__name__
                if inspect.isclass(origin)
                else str(origin).removeprefix("typing.")
            )
            if args:
                arg_str = ", ".join(self._type_to_string(a) for a in args)
                return f"{origin_name}[{arg_str}]"
            return origin_name

        if inspect.isclass(tp):
            return tp.__name__

        return str(tp)

    # %% walking the hierarchy

    @staticmethod
    def _is_inheriting_from_abc(clazz: Type) -> bool:
        """
        Returns True if `clazz` directly inherits from ABC (excluding ABC itself).
        """
        return any(base is ABC for base in clazz.__bases__)

    def _walk_related_classes(self, clazz: Type, relation: str) -> Iterator[Type]:
        """
        Yields related classes based on the specified relation.
        """
        match relation:
            case "subclasses":
                yield from clazz.__subclasses__()

            case "bases":
                bases_of_interest = [
                    base
                    for base in clazz.__bases__
                    if base not in (object, ABC)
                    and not issubclass(base, self.root_class)
                ]
                for base in bases_of_interest:
                    yield base

            case _:
                raise ValueError(f"Unsupported relation: {relation}")
