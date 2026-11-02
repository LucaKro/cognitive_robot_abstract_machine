"""
Tests for choosing the perception subtree that answers a GPSR query.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path

import pytest
from ament_index_python.packages import get_packages_with_prefixes
from py_trees.behaviours import Success
from robokudo.analysis_engine import SubtreeInterface
from robokudo.annotators.gpsr.pointing import HumanPointingAnnotator
from robokudo.descriptors.analysis_engines.gpsr import gpsr_implementation
from robokudo.descriptors.analysis_engines.gpsr.gpsr_implementation import (
    AnalysisEngine,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_near_object import (
    FURNITURE,
)
from robokudo.tree_components.query_based_task_scheduler import QueryBasedScheduler
from robokudo_msgs.action import Query
from typing_extensions import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from py_trees.behaviour import Behaviour
    from rclpy.node import Node

    from ._clip_mock import ScoringModelAndProcessor

# %% the GPSR tree with placeholder subtrees


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


@pytest.fixture
def gpsr_tree(node: Node, monkeypatch: pytest.MonkeyPatch) -> Behaviour:
    """
    The GPSR tree with its real routing, every subtree it routes to replaced by a
    :class:`PlaceholderSubtree`.
    """
    for module_attribute, value in list(vars(gpsr_implementation).items()):
        if isinstance(value, type) and issubclass(value, SubtreeInterface):
            monkeypatch.setattr(
                gpsr_implementation, module_attribute, PlaceholderSubtree
            )
    return AnalysisEngine().implementation()


def scheduler_of(tree: Behaviour) -> QueryBasedScheduler:
    """
    :return: The scheduler of the GPSR tree, which picks a subtree for every query.
    """
    return next(
        behaviour
        for behaviour in tree.iterate()
        if isinstance(behaviour, QueryBasedScheduler)
    )


@pytest.fixture
def scheduler(gpsr_tree: Behaviour) -> QueryBasedScheduler:
    """
    The scheduler of the GPSR tree with placeholder subtrees.
    """
    return scheduler_of(gpsr_tree)


@pytest.fixture
def ported_routes() -> list[str]:
    """
    The routes of the GPSR perception ported to CRAM so far.
    """
    return [
        "BeliefStateLookupPipeline",
        "DoorPipeline",
        "HumanActivity",
        "HumanActivityFiltered",
        "HumanActivityInRoomPipeline",
        "HumanInRoomPipeline",
        "HumanNearObjectPipeline",
        "HumanPosturePipeline",
        "ObjInRoomPipeline",
        "ObjectDetectionPipeline",
        "PointingPipeline",
        "WavingRaisingPipeline",
    ]


def gpsr_query(
    object_type: str,
    attribute: list[str] | None = None,
    location: str = "",
    description: list[str] | None = None,
    color: list[str] | None = None,
) -> Query.Goal:
    """
    :return: A query asking for the given kind of thing.
    """
    query = Query.Goal()
    query.obj.type = object_type
    query.obj.attribute = attribute or []
    query.obj.location = location
    query.obj.description = description or []
    query.obj.color = color or []
    return query


# %% the GPSR tree with its real subtrees


@pytest.fixture
def real_gpsr_tree(
    node: Node, scoring_clip: ScoringModelAndProcessor
) -> Iterator[Behaviour]:
    """
    The whole GPSR tree with its real subtrees, built with the CLIP stand-in, whose
    pointing nodes are destroyed after the test.

    .. note::
        The real subtrees load their detector weights from ``robokudo_suturo``, so the
        test is skipped where that package or its weights are not installed, as in CI.
    """
    suturo_prefix = get_packages_with_prefixes().get("robokudo_suturo")
    if (
        suturo_prefix is None
        or not (Path(suturo_prefix) / "share" / "robokudo_suturo" / "weights").is_dir()
    ):
        pytest.skip("robokudo_suturo with its weights is not installed")
    tree = AnalysisEngine().implementation()
    yield tree
    for behaviour in tree.iterate():
        if isinstance(behaviour, HumanPointingAnnotator):
            behaviour.node.destroy_node()


def test_gpsr_tree_builds_a_real_subtree_for_every_ported_route(
    real_gpsr_tree: Behaviour, ported_routes: list[str]
) -> None:
    routes = scheduler_of(real_gpsr_tree).tasks
    assert sorted(routes) == sorted(ported_routes)
    assert [
        route for route, subtree in routes.items() if isinstance(subtree, Success)
    ] == []


# %% the routes of the tree


def test_gpsr_tree_holds_a_subtree_for_every_ported_route(
    scheduler: QueryBasedScheduler, ported_routes: list[str]
) -> None:
    assert sorted(scheduler.tasks) == sorted(ported_routes)


# %% human queries


@pytest.mark.parametrize(
    "query, route",
    [
        (gpsr_query("human", ["waving"]), "WavingRaisingPipeline"),
        (gpsr_query("human", ["raising left arm"]), "WavingRaisingPipeline"),
        (gpsr_query("human", ["raising right arm"]), "WavingRaisingPipeline"),
        (gpsr_query("human", ["sitting"]), "HumanPosturePipeline"),
        (gpsr_query("human", ["standing"]), "HumanPosturePipeline"),
        (gpsr_query("human", ["laying"]), "HumanPosturePipeline"),
        (gpsr_query("human", ["pointing left"]), "PointingPipeline"),
        (gpsr_query("human", ["pointing right"]), "PointingPipeline"),
        (gpsr_query("human", ["sitting", "waving"]), "HumanActivityFiltered"),
        (gpsr_query("human"), "HumanActivity"),
        (gpsr_query("human", location="iai_kitchen"), "HumanInRoomPipeline"),
        (
            gpsr_query("human", ["waving"], location="living_room"),
            "HumanActivityInRoomPipeline",
        ),
        (
            gpsr_query("human", ["sitting"], location="bedroom"),
            "HumanActivityInRoomPipeline",
        ),
        (
            gpsr_query("human", ["pointing left"], location="office"),
            "HumanActivityInRoomPipeline",
        ),
        (gpsr_query("human", location="hallway"), "HumanInRoomPipeline"),
        (gpsr_query("human", location="kitchen"), "HumanActivity"),
    ],
    ids=[
        "waving",
        "raising left arm",
        "raising right arm",
        "sitting",
        "standing",
        "laying",
        "pointing left",
        "pointing right",
        "posture and gesture",
        "any activity",
        "in a room",
        "waving in a room",
        "sitting in a room",
        "pointing in a room",
        "in the hallway",
        "in a room of unknown name",
    ],
)
def test_human_query_takes_its_route(
    scheduler: QueryBasedScheduler, query: Query.Goal, route: str
) -> None:
    assert scheduler.filter_fn(query) == route


@pytest.mark.parametrize("furniture", list(FURNITURE))
def test_human_near_known_furniture_takes_the_furniture_route(
    scheduler: QueryBasedScheduler, furniture: str
) -> None:
    assert (
        scheduler.filter_fn(gpsr_query("human", location=furniture))
        == "HumanNearObjectPipeline"
    )


# %% object queries


@pytest.mark.parametrize(
    "query, route",
    [
        (gpsr_query("object"), "ObjectDetectionPipeline"),
        (gpsr_query("object", description=["Mug"]), "ObjectDetectionPipeline"),
        (gpsr_query("object", location="iai_kitchen"), "ObjInRoomPipeline"),
        (gpsr_query("object", ["red"]), None),
    ],
    ids=["any object", "object of a type", "object in a room", "unknown attribute"],
)
def test_object_query_takes_its_route(
    scheduler: QueryBasedScheduler, query: Query.Goal, route: str | None
) -> None:
    assert scheduler.filter_fn(query) == route


# %% other queries


@pytest.mark.parametrize(
    "query, route",
    [
        (gpsr_query("recall"), "BeliefStateLookupPipeline"),
        (gpsr_query("door"), "DoorPipeline"),
        (gpsr_query(""), None),
        (gpsr_query("animal"), None),
    ],
    ids=["recall", "door", "no type", "unknown type"],
)
def test_other_query_takes_its_route(
    scheduler: QueryBasedScheduler, query: Query.Goal, route: str | None
) -> None:
    assert scheduler.filter_fn(query) == route
