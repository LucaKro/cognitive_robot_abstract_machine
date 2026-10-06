"""
How much structure each graph has, per class, without pairing objects up.

Two graphs can use the same classes and the same kinds of relation and still be built
differently. Once the ontology and the modelled world agree on vocabulary, agreement on
the *kind* of a relation saturates -- every relation the run asserts is of a kind the
modelled world holds -- and stops telling anyone anything. What still differs is how much
of each kind there is: one sink modelled against three found, or every modelled drawer
held by a cabinet against half the found ones held by nothing.

Both are read by counting per class, which needs no correspondence between the graphs and
so can be reported for a scene with no alignment.

..note:: Neither number says a drawer is in the *right* cabinet. They say how much
    structure of each kind was built, not where it was put.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from typing_extensions import Dict, Iterable, List, Optional

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.structure import (
    UNCLASSIFIED,
    ClassifiedObject,
    TypedRelation,
)

# %% one class, in both graphs


@dataclass(frozen=True)
class ClassComposition(JsonRecord):
    """
    How many objects of one class each graph holds, and how many of them are held.
    """

    semantic_class: str
    """
    The class being counted.
    """

    modelled_objects: int
    """
    How many the modelled world holds.
    """

    predicted_objects: int
    """
    How many the run built.
    """

    modelled_held: int
    """
    How many of the modelled world's are part of something.
    """

    predicted_held: int
    """
    How many of the run's are part of something.
    """

    @property
    def granularity(self) -> Optional[float]:
        """
        How many objects the run built for each one the modelled world holds.

        One is agreement. Three means the run cut into three what the modelled world
        holds as one, which is the failure every identity-free count of relations
        misses.

        :return: The ratio, or nothing where the modelled world holds none and there is
            no ratio to take.
        """
        if not self.modelled_objects:
            return None
        return self.predicted_objects / self.modelled_objects

    @property
    def modelled_share_held(self) -> Optional[float]:
        """
        :return: The share of the modelled world's objects of this class that are part of
            something, or nothing where it holds none.
        """
        return self._share(self.modelled_held, self.modelled_objects)

    @property
    def predicted_share_held(self) -> Optional[float]:
        """
        :return: The share of the run's objects of this class that are part of something,
            or nothing where it built none.
        """
        return self._share(self.predicted_held, self.predicted_objects)

    @staticmethod
    def _share(held: int, objects: int) -> Optional[float]:
        """
        :param held: How many are part of something.
        :param objects: How many there are.
        :return: The share, or nothing where there are none. A share of nothing is not
            zero, and zero would read as objects that were all left unheld.
        """
        return held / objects if objects else None


# %% both graphs, class by class


@dataclass(frozen=True)
class CompositionComparison(JsonRecord):
    """
    What each graph is made of, class by class.
    """

    by_class: Dict[str, ClassComposition] = field(default_factory=dict)
    """
    One entry per class either graph holds, so a class only one of them has is still
    reported rather than agreeing by omission.
    """

    @classmethod
    def between(
        cls,
        predicted_objects: Iterable[ClassifiedObject],
        predicted_relations: Iterable[TypedRelation],
        modelled_objects: Iterable[ClassifiedObject],
        modelled_relations: Iterable[TypedRelation],
    ) -> CompositionComparison:
        """
        Count both graphs class by class and set the counts beside each other.

        :param predicted_objects: The run's objects in scope.
        :param predicted_relations: The run's relations in scope.
        :param modelled_objects: The modelled world's objects in scope.
        :param modelled_relations: The modelled world's relations in scope.
        :return: What each holds, by class.
        """
        modelled = cls._counted(modelled_objects, modelled_relations)
        predicted = cls._counted(predicted_objects, predicted_relations)
        empty = (0, 0)
        return cls(
            by_class={
                name: ClassComposition(
                    semantic_class=name,
                    modelled_objects=modelled.get(name, empty)[0],
                    predicted_objects=predicted.get(name, empty)[0],
                    modelled_held=modelled.get(name, empty)[1],
                    predicted_held=predicted.get(name, empty)[1],
                )
                for name in sorted(set(modelled) | set(predicted))
            }
        )

    @staticmethod
    def _counted(
        objects: Iterable[ClassifiedObject], relations: Iterable[TypedRelation]
    ) -> Dict[str, tuple]:
        """
        :param objects: One graph's objects.
        :param relations: That graph's relations.
        :return: Per class, how many objects it has and how many of them are held.
        """
        objects = list(objects)
        held = {relation.part for relation in relations}
        counted: Counter = Counter()
        holding: Counter = Counter()
        for one in objects:
            name = one.classes[0] if one.classes else UNCLASSIFIED
            counted[name] += 1
            if one.name in held:
                holding[name] += 1
        return {name: (counted[name], holding[name]) for name in counted}

    @property
    def classes_the_run_divided_differently(self) -> List[ClassComposition]:
        """
        :return: The classes the two graphs cut into different numbers of objects, worst
            first, which is where over- and under-segmentation shows.
        """
        differing = [
            one
            for one in self.by_class.values()
            if one.granularity is not None and one.granularity != 1.0
        ]
        return sorted(differing, key=lambda one: -abs((one.granularity or 1.0) - 1.0))
