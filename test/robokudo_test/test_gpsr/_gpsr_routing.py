"""
Building blocks for testing which subtree the GPSR tree picks for a query.
"""

from __future__ import annotations

# %% Imports
from py_trees.behaviours import Success
from robokudo.analysis_engine import SubtreeInterface
from robokudo.tree_components.query_based_task_scheduler import QueryBasedScheduler
from robokudo_msgs.action import Query
from typing_extensions import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from py_trees.behaviour import Behaviour

# %% placeholder subtrees


class PlaceholderSubtree(SubtreeInterface):
    """
    Stands in for a GPSR subtree: it loads no detector and its behaviour succeeds at
    once, so the routing can be tested without the detector weights.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Accept and ignore the arguments the real subtree takes.
        """

    def name(self) -> str:
        return PlaceholderSubtree.__name__

    def implementation(self, *args: Any, **kwargs: Any) -> Behaviour:
        return Success(name=self.name())


# %% scheduler of the tree


def scheduler_of(tree: Behaviour) -> QueryBasedScheduler:
    """
    :param tree: The GPSR tree.
    :return: The scheduler of the GPSR tree, which picks a subtree for every query.
    """
    return next(
        behaviour
        for behaviour in tree.iterate()
        if isinstance(behaviour, QueryBasedScheduler)
    )


# %% queries


def gpsr_query(
    object_type: str,
    attribute: list[str] | None = None,
    location: str = "",
    description: list[str] | None = None,
    color: list[str] | None = None,
) -> Query.Goal:
    """
    :param object_type: The queried kind of thing.
    :param attribute: The queried attributes.
    :param location: The queried location.
    :param description: The queried descriptions.
    :param color: The queried colors.
    :return: A query asking for the given kind of thing.
    """
    query = Query.Goal()
    query.obj.type = object_type
    query.obj.attribute = attribute or []
    query.obj.location = location
    query.obj.description = description or []
    query.obj.color = color or []
    return query
