"""
Export the semantic annotation taxonomy in the form a model has to read it.

A model asked to classify an object into this taxonomy needs to know two things about
every class: what it *is* (its place in the hierarchy) and what it can be *composed of*.
The second is carried by fields that all have defaults, so an export built from required
constructor parameters loses it entirely.

The relations reported here are the ones the world can actually realize, each named by
the method that realizes it, because a model shown only one of them will force that one
onto everything: a mug resting on a countertop is mounted with
:meth:`IsStorageSpace.add_object`, not with :meth:`PartWholeRelationship.add`, and a
mount through the wrong channel raises rather than building the wrong world quietly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Type

from krrood.class_diagrams.class_diagram import WrappedClass

from semantic_digital_twin.semantic_annotations.part_whole import (
    IsPartWholeRelationship,
    part_whole_fields,
)

MIXIN_MODULE_SUFFIX = "semantic_annotations.mixins"
"""
Where the classes that exist to contribute relations live.
"""

INFRASTRUCTURE_FIELDS = frozenset(
    {"root", "name", "id", "simulator_additional_properties"}
)
"""
Fields every annotation carries, which say nothing about what it is.

``root`` in particular is declared by every annotation, so reporting it tells a model
nothing that distinguishes one class from another.
"""


@dataclass(frozen=True)
class MountChannel:
    """
    One way a world mounts something into an annotation.
    """

    kind: str
    """
    What the relation means, as the export names it.
    """

    field_name: str
    """
    The field holding it.
    """

    owner: str
    """
    The name of the class declaring that field.
    """

    mounted_by: str
    """
    The method realizing the relation.
    """


OCCUPANCY_CHANNELS: List[MountChannel] = [
    MountChannel(
        kind="contains",
        field_name="objects",
        owner="IsStorageSpace",
        mounted_by="add_object",
    ),
    MountChannel(
        kind="supports",
        field_name="supporting_surface",
        owner="HasSupportingSurface",
        mounted_by="add_supporting_surface",
    ),
]
"""
The relations that are not part-whole but are still mounted into the world.

Part-whole relations are found through :class:`IsPartWholeRelationship` metadata rather
than listed here, since they declare themselves.
"""


@dataclass
class SemanticRelation:
    """
    One relation an annotation class can stand in, as a model is told about it.
    """

    kind: str
    """
    ``part``, ``contains`` or ``supports``.
    """

    field_name: str
    """
    The field holding the related annotation.
    """

    target: str
    """
    The name of the type the field accepts.
    """

    holds_many: bool
    """
    Whether the field holds several of them rather than one.
    """

    mounted_by: str
    """
    The method that realizes the relation in the world.
    """

    removes_geometry: bool = False
    """
    Whether mounting here cuts the part's volume out of the whole's geometry.
    """

    target_class: Optional[Type] = None
    """
    The type the field accepts, where it is one, so that what may be mounted into it can
    be asked rather than compared by name. Left out of the exported form, which names it.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The relation as JSON-ready data.
        """
        rendered = {
            "kind": self.kind,
            "field": self.field_name,
            "target": self.target,
            "many": self.holds_many,
            "mounted_by": self.mounted_by,
        }
        if self.removes_geometry:
            rendered["removes_geometry"] = True
        return rendered


def _own_field_names(annotation_class: Type) -> Set[str]:
    """
    :param annotation_class: The class to inspect.
    :return: The names of the fields that class declares itself, rather than inherits.
    """
    if not is_dataclass(annotation_class):
        return set()
    inherited = {
        base_field.name
        for base in annotation_class.__mro__[1:]
        if is_dataclass(base)
        for base_field in dataclass_fields(base)
    }
    return {
        own_field.name for own_field in dataclass_fields(annotation_class)
    } - inherited


def relations_of(annotation_class: Type) -> List[SemanticRelation]:
    """
    Report every relation an annotation class can stand in.

    :param annotation_class: The class to inspect.
    :return: Its part, containment and support relations, part-whole first.
    """
    relations = [
        SemanticRelation(
            kind="part",
            field_name=part_whole_relationship_field.field_name,
            target=part_whole_relationship_field.part.__name__,
            holds_many=part_whole_relationship_field.holds_many,
            mounted_by="add",
            removes_geometry=part_whole_relationship_field.removes_part_geometry_from_whole,
            target_class=part_whole_relationship_field.part,
        )
        for part_whole_relationship_field in part_whole_fields(annotation_class)
    ]

    wrapped_fields = {
        wrapped_field.field.name: wrapped_field
        for wrapped_field in WrappedClass(annotation_class).fields
    }
    for channel in OCCUPANCY_CHANNELS:
        wrapped_field = wrapped_fields.get(channel.field_name)
        if wrapped_field is None:
            continue
        target = wrapped_field.type_endpoint
        relations.append(
            SemanticRelation(
                kind=channel.kind,
                field_name=channel.field_name,
                target=getattr(target, "__name__", str(target)),
                holds_many=wrapped_field.is_many_to_many_relationship,
                mounted_by=channel.mounted_by,
                target_class=target if isinstance(target, type) else None,
            )
        )
    return relations


def _summary_of(annotation_class: Type) -> Optional[str]:
    """
    A dataclass without a docstring is given one listing its signature, which says
    nothing a model needs and costs more than everything else in the node, so it counts
    as having none.

    :param annotation_class: The class to describe.
    :return: The first sentence of its docstring, or None when it has none of its own.
    """
    documentation = annotation_class.__dict__.get("__doc__")
    if not documentation or documentation.startswith(f"{annotation_class.__name__}("):
        return None
    return " ".join(documentation.split()).split(". ")[0].rstrip(".") or None


def load_annotation_modules() -> None:
    """
    Import the modules declaring the taxonomy.

    The hierarchy is discovered through ``__subclasses__``, which only ever reports
    classes Python has already imported, so a taxonomy exported before these modules are
    loaded comes back empty rather than wrong -- which is worse, because it looks like a
    taxonomy.

    Classes generated at run time are picked up as well, once whoever generated them has
    imported them.
    """
    import semantic_digital_twin.semantic_annotations.mixins  # noqa: F401
    import semantic_digital_twin.semantic_annotations.semantic_annotations  # noqa: F401


def _walk_subclasses(root_class: Type) -> Iterator[Type]:
    """
    :param root_class: The class to walk below.
    :return: Every class deriving from it, each once, in breadth-first order.
    """
    seen = {root_class}
    frontier = [root_class]
    while frontier:
        current = frontier.pop(0)
        for subclass in current.__subclasses__():
            if subclass in seen:
                continue
            seen.add(subclass)
            frontier.append(subclass)
            yield subclass


def annotation_classes(root_class: Type) -> Dict[str, Type]:
    """
    :param root_class: The root of the hierarchy, normally ``SemanticAnnotation``.
    :return: Every class below it, by name, so a name read back from a file or a model
        can be turned into the class it stands for.
    """
    load_annotation_modules()
    return {
        annotation_class.__name__: annotation_class
        for annotation_class in _walk_subclasses(root_class)
    }


def build_taxonomy(
    root_class: Type, include_summaries: bool = False
) -> Dict[str, Any]:
    """
    Describe the annotation taxonomy below a root class.

    Classes are reported flat, each naming its own bases, because the hierarchy is not a
    tree: a cabinet is furniture *and* a thing with doors *and* a thing with drawers, and
    it is the second and third of those that say what it can hold.

    :param root_class: The root of the hierarchy, normally ``SemanticAnnotation``.
    :param include_summaries: Whether to describe each class by its docstring. Off by
        default: only some classes carry one, so including them would tell a model more
        about the documented classes than about the others for no reason but that
        someone happened to write a sentence.
    :return: The taxonomy as JSON-ready data.
    """
    load_annotation_modules()
    annotation_classes = list(_walk_subclasses(root_class))
    known = {annotation_class.__name__ for annotation_class in annotation_classes} | {
        root_class.__name__
    }

    classes = []
    mixins = []
    for annotation_class in annotation_classes:
        relations = relations_of(annotation_class)
        node: Dict[str, Any] = {
            "name": annotation_class.__name__,
            "bases": [
                base.__name__
                for base in annotation_class.__bases__
                if base.__name__ in known
            ],
        }
        if annotation_class.__module__.endswith(MIXIN_MODULE_SUFFIX):
            # A mixin is a base to build with, not a thing standing in a room. Saying so
            # is the difference between a model deriving a ceiling from HasRootBody and
            # answering that a ceiling *is* one.
            node["mixin"] = True
        summary = _summary_of(annotation_class) if include_summaries else None
        if summary:
            node["summary"] = summary
        if relations:
            node["relations"] = [relation.to_json() for relation in relations]
        classes.append(node)

        # A mixin is worth offering as a building block only for the relations it
        # introduces; the ones it inherits come with its own bases anyway.
        own_fields = _own_field_names(annotation_class)
        introduced = [
            relation for relation in relations if relation.field_name in own_fields
        ]
        if introduced and annotation_class.__module__.endswith(MIXIN_MODULE_SUFFIX):
            mixins.append(
                {
                    "name": annotation_class.__name__,
                    "introduces": [relation.to_json() for relation in introduced],
                }
            )

    return {
        "root_name": root_class.__name__,
        "note": (
            "Relations say what a class can hold. 'part' is mounted with add(), "
            "'contains' with add_object() for something merely inside or on it, "
            "'supports' with add_supporting_surface(). A class marked 'mixin' exists "
            "to be built with rather than to name something in a room. A new class is "
            "composed by naming a superclass and any of the mixins below."
        ),
        "classes": classes,
        "part_whole_mixins": mixins,
    }


def export_taxonomy(
    root_class: Type, output_path: Path, include_summaries: bool = False
) -> Dict[str, Any]:
    """
    Write the taxonomy of a hierarchy to a JSON file.

    :param root_class: The root of the hierarchy.
    :param output_path: Where to write it.
    :param include_summaries: See :func:`build_taxonomy`.
    :return: The taxonomy that was written.
    """
    taxonomy = build_taxonomy(root_class, include_summaries=include_summaries)
    Path(output_path).write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    return taxonomy


def compose_class(name: str, superclass: Type, mixins: Sequence[Type] = ()) -> Type:
    """
    Build the class a proposal names, so the ontology can be asked about it.

    A label with no class in the taxonomy is answered by naming a superclass and the
    mixins to compose it from, and it is those that decide what the class admits: a
    ``KitchenIsland(Furniture)`` can hold nothing, while a
    ``KitchenIsland(Furniture, HasDrawers, HasDoors)`` can hold both drawers and doors.
    Composing the class here answers that before anything is generated or persisted.

    ..note:: The result is a real subclass of its bases, so ``__subclasses__`` reports it
        and :func:`build_taxonomy` includes it from here on. Export the taxonomy before
        composing anything that is only a proposal.

    :param name: The name the proposal gives the class.
    :param superclass: The class it derives from.
    :param mixins: The mixins contributing relations to it.
    :return: The composed dataclass.
    :raises TypeError: If the bases cannot be combined into a class, which is itself an
        answer about the proposal.
    """
    composed = type(name, _in_base_order([superclass, *mixins]), {"__annotations__": {}})
    return dataclass(composed)


def _in_base_order(bases: Sequence[Type]) -> tuple:
    """
    Order bases so that Python can combine them.

    A class may not name a base before one of that base's own subclasses, and a proposal
    naming ``HasRootBody`` with ``IsStorageSpace`` -- which already derives from it --
    names them the wrong way round. Which of two classes derives from the other is not
    something the proposal decides, so it is settled here rather than sent back as a
    problem with the answer.

    :param bases: The classes to derive from, as they were named.
    :return: The same classes, each once, every one before any class it derives from.
    """
    remaining = list(dict.fromkeys(bases))
    ordered = []
    while remaining:
        most_derived = next(
            candidate
            for candidate in remaining
            if not any(
                other is not candidate and issubclass(other, candidate)
                for other in remaining
            )
        )
        remaining.remove(most_derived)
        ordered.append(most_derived)
    return tuple(ordered)


def admissible_mounts(one_class: Type, other_class: Type) -> List[Tuple[Type, SemanticRelation]]:
    """
    Report every way two classes could be mounted into one another, in either direction.

    A part-whole relation is one of three channels, and it is the narrow one: a mug on a
    counter and a jar in a box are mounted with add_object, not with add, so
    asking only about parts reports "no relation" for pairs that have a perfectly good
    one. What comes back is what the world would accept, not what is the case.

    ..note:: contains is admissible very widely -- IsStorageSpace.objects accepts
        anything with a root body -- so its presence says much less about a pair than a
        part-whole relation does.

    :param one_class: One of the annotation classes.
    :param other_class: The other annotation class.
    :return: Each admissible mount, as the class that would hold and the relation it
        would be held in.
    """
    return [
        (whole, relation)
        for whole, part in ((one_class, other_class), (other_class, one_class))
        for relation in relations_of(whole)
        if isinstance(part, type)
        and isinstance(relation.target_class, type)
        and issubclass(part, relation.target_class)
    ]
