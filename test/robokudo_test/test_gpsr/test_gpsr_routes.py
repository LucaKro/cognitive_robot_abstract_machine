"""
Tests for choosing the perception subtree that answers a GPSR query.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path

import pytest
from ament_index_python.packages import get_packages_with_prefixes
from py_trees.behaviours import Success
from robokudo.annotators.gpsr.pointing import HumanPointingAnnotator
from robokudo.descriptors.analysis_engines.gpsr.gpsr_implementation import (
    AnalysisEngine,
    GpsrRoute,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_near_object import (
    FURNITURE,
)
from robokudo.tree_components.query_based_task_scheduler import QueryBasedScheduler
from robokudo_msgs.action import Query
from typing_extensions import TYPE_CHECKING

from ._gpsr_routing import gpsr_query, scheduler_of

if TYPE_CHECKING:
    from collections.abc import Iterator

    from py_trees.behaviour import Behaviour
    from rclpy.node import Node

    from ._clip_mock import ScoringModelAndProcessor

# %% ported routes


@pytest.fixture
def ported_routes() -> list[str]:
    """
    The routes of the GPSR perception ported to CRAM so far.
    """
    return [
        GpsrRoute.HUMAN_ATTRIBUTE.value,
        GpsrRoute.BELIEF_LOOKUP.value,
        GpsrRoute.DOOR_OPEN_VOCABULARY.value,
        GpsrRoute.DOOR.value,
        GpsrRoute.FACE.value,
        GpsrRoute.HUMAN_ACTIVITY.value,
        GpsrRoute.HUMAN_ACTIVITY_FILTERED.value,
        GpsrRoute.HUMAN_ACTIVITY_IN_ROOM.value,
        GpsrRoute.HUMAN_IN_ROOM.value,
        GpsrRoute.HUMAN_NEAR_OBJECT.value,
        GpsrRoute.HUMAN_POSTURE.value,
        GpsrRoute.OBJECT_IN_ROOM.value,
        GpsrRoute.OBJECT_DETECTION.value,
        GpsrRoute.OPEN_VOCABULARY.value,
        GpsrRoute.OPEN_VOCABULARY_IN_ROOM.value,
        GpsrRoute.POINTING.value,
        GpsrRoute.WAVING_RAISING.value,
    ]


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
        (gpsr_query("human", ["waving"]), GpsrRoute.WAVING_RAISING.value),
        (gpsr_query("human", ["raising left arm"]), GpsrRoute.WAVING_RAISING.value),
        (gpsr_query("human", ["raising right arm"]), GpsrRoute.WAVING_RAISING.value),
        (gpsr_query("human", ["sitting"]), GpsrRoute.HUMAN_POSTURE.value),
        (gpsr_query("human", ["standing"]), GpsrRoute.HUMAN_POSTURE.value),
        (gpsr_query("human", ["laying"]), GpsrRoute.HUMAN_POSTURE.value),
        (gpsr_query("human", ["pointing left"]), GpsrRoute.POINTING.value),
        (gpsr_query("human", ["pointing right"]), GpsrRoute.POINTING.value),
        (
            gpsr_query("human", ["sitting", "waving"]),
            GpsrRoute.HUMAN_ACTIVITY_FILTERED.value,
        ),
        (gpsr_query("human"), GpsrRoute.HUMAN_ACTIVITY.value),
        (gpsr_query("human", location="iai_kitchen"), GpsrRoute.HUMAN_IN_ROOM.value),
        (
            gpsr_query("human", ["waving"], location="living_room"),
            GpsrRoute.HUMAN_ACTIVITY_IN_ROOM.value,
        ),
        (
            gpsr_query("human", ["sitting"], location="bedroom"),
            GpsrRoute.HUMAN_ACTIVITY_IN_ROOM.value,
        ),
        (
            gpsr_query("human", ["pointing left"], location="office"),
            GpsrRoute.HUMAN_ACTIVITY_IN_ROOM.value,
        ),
        (gpsr_query("human", location="hallway"), GpsrRoute.HUMAN_IN_ROOM.value),
        (gpsr_query("human", location="kitchen"), GpsrRoute.HUMAN_ACTIVITY.value),
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
        == GpsrRoute.HUMAN_NEAR_OBJECT.value
    )


# %% object queries


@pytest.mark.parametrize(
    "query, route",
    [
        (gpsr_query("object"), GpsrRoute.OBJECT_DETECTION.value),
        (gpsr_query("object", description=["Mug"]), GpsrRoute.OBJECT_DETECTION.value),
        (gpsr_query("object", location="iai_kitchen"), GpsrRoute.OBJECT_IN_ROOM.value),
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
        (gpsr_query("recall"), GpsrRoute.BELIEF_LOOKUP.value),
        (gpsr_query("door"), GpsrRoute.DOOR.value),
        (gpsr_query(""), None),
        (gpsr_query("animal"), None),
    ],
    ids=["recall", "door", "no type", "unknown type"],
)
def test_other_query_takes_its_route(
    scheduler: QueryBasedScheduler, query: Query.Goal, route: str | None
) -> None:
    assert scheduler.filter_fn(query) == route
