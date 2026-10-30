"""
Comparing what two graphs are made of, without deciding which object is which.

Asking whether a run got the semantics right does not always mean asking about any
particular object. "Does it build cabinets holding drawers, and drawers holding handles,
through the fields the ontology says so" is a question about the two graphs as a whole,
and it can be answered by counting: how many relations of each kind each graph holds,
and how many objects of each class.

Counting this way needs no correspondence between the graphs, and therefore nothing that
would relate their coordinate frames. What it cannot see is where a relation sits: a run
that puts the right number of drawers into the wrong cabinets counts the same as one
that puts them all in the right places. That limit is the price of not needing to know
which object is which, and it is why these results are reported as what they are rather
than as detection.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from typing_extensions import Dict, Iterable, List, Protocol

from experiments.warsaw.bases import JsonRecord

# %% what a graph is counted from


class ClassifiedObject(Protocol):
    """
    An object of either graph, named and standing as some classes.
    """

    name: str
    """
    What identifies it within its own graph.
    """

    @property
    def classes(self) -> List[str]:
        """
        :return: The classes it stands as.
        """


class TypedRelation(Protocol):
    """
    A relation of either graph, named by the objects at its ends.
    """

    whole: str
    """
    The object at the holding end.
    """

    part: str
    """
    The object at the held end.
    """

    relation: str
    """
    What the relation means: part, contains, or supports.
    """

    field_name: str
    """
    The ontology field it is held in.
    """


UNCLASSIFIED = "?"
"""
Stands for an object its graph never named, so it is counted rather than dropped.
"""

UNNAMED_FIELD = "(unnamed)"
"""
Stands for a relation whose field was never recorded, so it cannot be silently counted
as one that was.
"""

# %% one kind of thing a graph holds


@dataclass(frozen=True)
class RelationPattern(JsonRecord):
    """
    One kind of relation, said in classes rather than in objects.

    ``Cabinet`` holding a ``Drawer`` through ``drawers`` is one pattern however many
    cabinets a scene has, which is what makes two scenes comparable without pairing up
    their cabinets.
    """

    whole_class: str
    """
    What the holding object is.
    """

    field_name: str
    """
    The ontology field it holds the part in.
    """

    part_class: str
    """
    What the held object is.
    """

    relation: str
    """
    What the relation means.
    """

    def __str__(self) -> str:
        """
        :return: The pattern as one readable line.
        """
        return f"{self.whole_class} --{self.field_name}--> {self.part_class}"


@dataclass(frozen=True)
class Tally(JsonRecord):
    """
    How many of one thing each graph holds, and what that agreement comes to.
    """

    modelled: int
    """
    How many the modelled world holds.
    """

    predicted: int
    """
    How many the run built.
    """

    @property
    def agreeing(self) -> int:
        """
        :return: How many are matched by one in the other graph, which is the smaller
            count: a graph cannot be credited for more than the other holds.
        """
        return min(self.modelled, self.predicted)

    @property
    def precision(self) -> float:
        """
        :return: What fraction of what the run built is matched by the modelled world.
        """
        return 1.0 if self.predicted == 0 else self.agreeing / self.predicted

    @property
    def recall(self) -> float:
        """
        :return: What fraction of the modelled world is matched by what the run built.
        """
        return 1.0 if self.modelled == 0 else self.agreeing / self.modelled

    @property
    def f_score(self) -> float:
        """
        :return: The harmonic mean of the two, or nought where neither is above nought.
        """
        if self.precision + self.recall == 0.0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


@dataclass(frozen=True)
class CountedPattern(JsonRecord):
    """
    One kind of relation and how many of it each graph holds.
    """

    pattern: RelationPattern
    """
    The kind of relation counted.
    """

    counted: Tally
    """
    How many of it each graph holds.
    """


# %% counting a graph


def classes_in(objects: Iterable[ClassifiedObject]) -> Counter:
    """
    Count what a graph's objects are.

    :param objects: The objects in scope.
    :return: How many objects stand as each class.
    """
    return Counter(
        semantic_class
        for one in objects
        for semantic_class in (one.classes or [UNCLASSIFIED])
    )


def relation_patterns_in(
    relations: Iterable[TypedRelation], objects: Iterable[ClassifiedObject]
) -> Counter:
    """
    Count what kinds of relation a graph holds.

    :param relations: The relations in scope.
    :param objects: The objects those relations run between.
    :return: How many relations of each pattern the graph holds.
    """
    named = {
        one.name: (one.classes[0] if one.classes else UNCLASSIFIED) for one in objects
    }
    return Counter(
        RelationPattern(
            whole_class=named.get(one.whole, UNCLASSIFIED),
            field_name=one.field_name or UNNAMED_FIELD,
            part_class=named.get(one.part, UNCLASSIFIED),
            relation=one.relation,
        )
        for one in relations
    )


# %% what the two graphs came to


@dataclass(frozen=True)
class StructuralComparison(JsonRecord):
    """
    What two graphs are made of, side by side.

    ..note:: Every count here is identity-free. A run that builds the right number of
        relations of a kind scores the same whether or not they are in the right places,
        so these results say whether a run builds the right *kind* of world, not whether
        it built it where the modelled world did.
    """

    classes: Dict[str, Tally] = field(default_factory=dict)
    """
    How many objects of each class each graph holds.
    """

    relations: List[CountedPattern] = field(default_factory=list)
    """
    How many relations of each pattern each graph holds, commonest first.
    """

    @classmethod
    def between(
        cls,
        predicted_objects: Iterable[ClassifiedObject],
        predicted_relations: Iterable[TypedRelation],
        modelled_objects: Iterable[ClassifiedObject],
        modelled_relations: Iterable[TypedRelation],
    ) -> StructuralComparison:
        """
        Count both graphs and set the counts beside each other.

        :param predicted_objects: The run's objects in scope.
        :param predicted_relations: The run's relations in scope.
        :param modelled_objects: The modelled world's objects in scope.
        :param modelled_relations: The modelled world's relations in scope.
        :return: What each graph holds, by class and by relation pattern.
        """
        predicted_objects = list(predicted_objects)
        modelled_objects = list(modelled_objects)
        predicted_classes = classes_in(predicted_objects)
        modelled_classes = classes_in(modelled_objects)
        predicted_patterns = relation_patterns_in(
            predicted_relations, predicted_objects
        )
        modelled_patterns = relation_patterns_in(modelled_relations, modelled_objects)
        relations = [
            CountedPattern(
                pattern=pattern,
                counted=Tally(
                    modelled=modelled_patterns.get(pattern, 0),
                    predicted=predicted_patterns.get(pattern, 0),
                ),
            )
            for pattern in set(modelled_patterns) | set(predicted_patterns)
        ]
        return cls(
            classes={
                name: Tally(
                    modelled=modelled_classes.get(name, 0),
                    predicted=predicted_classes.get(name, 0),
                )
                for name in sorted(set(modelled_classes) | set(predicted_classes))
            },
            relations=sorted(
                relations,
                key=lambda row: (
                    -(row.counted.modelled + row.counted.predicted),
                    str(row.pattern),
                ),
            ),
        )

    @property
    def relation_totals(self) -> Tally:
        """
        :return: How many relations each graph holds altogether, from which the overall
            agreement on relations follows.
        """
        return Tally(
            modelled=sum(row.counted.modelled for row in self.relations),
            predicted=sum(row.counted.predicted for row in self.relations),
        )

    @property
    def class_totals(self) -> Tally:
        """
        :return: How many objects each graph holds altogether.
        """
        return Tally(
            modelled=sum(counted.modelled for counted in self.classes.values()),
            predicted=sum(counted.predicted for counted in self.classes.values()),
        )

    def relations_only_one_graph_holds(self) -> List[CountedPattern]:
        """
        :return: The relation patterns one graph holds and the other does not at all,
            which is where a disagreement about structure shows most plainly.
        """
        return [
            row
            for row in self.relations
            if row.counted.modelled == 0 or row.counted.predicted == 0
        ]
