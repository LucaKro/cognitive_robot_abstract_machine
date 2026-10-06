"""
What of two graphs is compared, and what is left out of the comparison.

A reconstruction and a modelled world do not describe the same things. One is a working
room with objects and a robot in it; the other is an empty apartment modelled for
planning. Scoring everything against everything punishes a run for finding what the
ground truth never claimed to hold, and credits ground truth for things no scan could
have seen.

So what counts is declared once, in a file read beside both graphs, and applied to each
of them identically. Applying it to only one side would move the score without either
graph having changed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from typing_extensions import Iterable, List, Optional, Protocol, TypeVar

from experiments.warsaw.bases import JsonRecord

# %% what a comparison can place


class ComparableNode(Protocol):
    """
    Something a comparison can place: one object, named, standing as some classes.

    Both graphs answer this, which is what lets one scope apply to the two of them.
    """

    name: str
    """
    What identifies the object within its own graph.
    """

    faces: int
    """
    How many faces its own geometry has, zero where it has none.
    """

    @property
    def classes(self) -> List[str]:
        """
        :return: The classes it stands as in a comparison.
        """


class ComparableEdge(Protocol):
    """
    A relation a comparison can place, named by the objects at its two ends.
    """

    whole: str
    """
    The object at the holding end.
    """

    part: str
    """
    The object at the held end.
    """


NodeType = TypeVar("NodeType", bound=ComparableNode)
EdgeType = TypeVar("EdgeType", bound=ComparableEdge)


# %% what an object does in a comparison


class ComparisonRole(StrEnum):
    """
    What an object does in a comparison.
    """

    INSTANCE = "instance"
    """
    Matched one to one against the other graph and scored as an object.
    """

    AREA = "area"
    """
    Counted as covered area rather than as objects, because the two graphs divide it
    differently: one merged body against many segments cannot be matched one to one.
    """

    EXCLUDED = "excluded"
    """
    Kept out of the comparison, because the other graph has nothing that could
    correspond to it.
    """


@dataclass(frozen=True)
class ClassDecision(JsonRecord):
    """
    What everything of one class does in a comparison.
    """

    semantic_class: str
    """
    The class being placed.
    """

    role: ComparisonRole
    """
    What objects of that class do.
    """

    reason: str
    """
    Why, so the decision can be reviewed rather than trusted.
    """


@dataclass(frozen=True)
class EntityDecision(JsonRecord):
    """
    What one named object does in a comparison, whatever its class.

    Needed where a class cannot say it: a body the world leaves unnamed carries no class
    to decide by.
    """

    entity: str
    """
    The object being placed, by the name its own graph gives it.
    """

    role: ComparisonRole
    """
    What that object does.
    """

    reason: str
    """
    Why, so the decision can be reviewed rather than trusted.
    """


# %% the declared scope of one comparison


@dataclass(frozen=True)
class ComparisonScope(JsonRecord):
    """
    Everything declared about what one comparison covers.

    An object nothing here mentions is an instance, so the file records departures from
    comparing everything rather than restating the whole vocabulary.
    """

    scene: str
    """
    The comparison these decisions were written for.
    """

    classes: List[ClassDecision] = field(default_factory=list)
    """
    The classes placed other than as instances.
    """

    entities: List[EntityDecision] = field(default_factory=list)
    """
    The individual objects placed other than as their class would place them.
    """

    @classmethod
    def read(cls, path: Path) -> ComparisonScope:
        """
        :param path: The declared scope to read.
        :return: What it holds.
        """
        return cls.from_json(json.loads(Path(path).read_text()))

    def role_of(self, node: ComparableNode) -> ComparisonRole:
        """
        Say what one object does in the comparison.

        A decision naming the object itself wins over one naming its class, since it was
        written about that object knowing what class it carries.

        An object with no geometry is left out whatever its class, because no
        reconstruction could have found it: world models carry joints and hierarchy on
        bodies with no shapes, and counting those as ground truth would make every one of
        them a miss.

        :param node: The object to place.
        :return: What it does.
        """
        named = self._decision_for_entity(node.name)
        if named is not None:
            return named.role
        if node.faces == 0:
            return ComparisonRole.EXCLUDED
        for semantic_class in node.classes:
            by_class = self._decision_for_class(semantic_class)
            if by_class is not None:
                return by_class.role
        return ComparisonRole.INSTANCE

    def _decision_for_entity(self, name: str) -> Optional[EntityDecision]:
        """
        :param name: The object's name in its own graph.
        :return: The decision written about that object, where one was.
        """
        return next(
            (decision for decision in self.entities if decision.entity == name), None
        )

    def _decision_for_class(self, semantic_class: str) -> Optional[ClassDecision]:
        """
        :param semantic_class: The class to look up.
        :return: The decision written about that class, where one was.
        """
        return next(
            (
                decision
                for decision in self.classes
                if decision.semantic_class == semantic_class
            ),
            None,
        )

    def nodes_playing(
        self, role: ComparisonRole, nodes: Iterable[NodeType]
    ) -> List[NodeType]:
        """
        :param role: The role to collect.
        :param nodes: The objects of one graph.
        :return: Those that do that in the comparison.
        """
        return [node for node in nodes if self.role_of(node) is role]

    def edges_between_instances(
        self, edges: Iterable[EdgeType], nodes: Iterable[NodeType]
    ) -> List[EdgeType]:
        """
        Keep only the relations both of whose ends are compared as objects.

        A relation reaching something left out of the comparison cannot be right or
        wrong in it, and scoring it would charge a graph for the scope rather than for
        what it built.

        :param edges: The relations of one graph.
        :param nodes: The objects of that same graph.
        :return: The relations that can be compared.
        """
        instances = {
            node.name for node in self.nodes_playing(ComparisonRole.INSTANCE, nodes)
        }
        return [
            edge for edge in edges if edge.whole in instances and edge.part in instances
        ]
