"""
Tests for choosing the perception subtree that answers a GPSR query.
"""

from __future__ import annotations

# %% Imports
import pytest
from ament_index_python.packages import get_packages_with_prefixes
from py_trees.behaviours import Success
from robokudo.annotators.gpsr.pointing import HumanPointingAnnotator
from robokudo.annotators.gpsr.vision_language_annotator import VisionLanguageAnnotator
from robokudo.descriptors.analysis_engines.gpsr.gpsr_implementation import (
    AnalysisEngine,
    GpsrRoute,
)
from robokudo.descriptors.analysis_engines.gpsr.subtree_human_near_object import (
    Furniture,
)
from typing_extensions import TYPE_CHECKING, Iterator

from ._gpsr_routing import gpsr_query, scheduler_of

if TYPE_CHECKING:
    from robokudo_msgs.action import Query
    from robokudo.tree_components.query_based_task_scheduler import QueryBasedScheduler
    from py_trees.behaviour import Behaviour
    from rclpy.node import Node

    from ._clip_mock import ScoringModelAndProcessor

# %% ported routes


@pytest.fixture
def gpsr_routes() -> list[str]:
    """
    :return: The routes of the GPSR tree.
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
    The whole GPSR tree with its real subtrees, built with the CLIP stand-in.

    Its pointing nodes are destroyed after the test.

    .. note::
        The real subtrees load detector weights from their ROS package. The test is
        skipped where that package is not installed, as in CI.

    :param node: The shared ROS node.
    :param scoring_clip: The CLIP stand-in.
    :return: An iterator over the tree.
    """
    model_package = VisionLanguageAnnotator.Descriptor.Parameters().ros_package_name
    if model_package not in get_packages_with_prefixes():
        pytest.skip("GPSR model package is not installed")
    tree = AnalysisEngine().implementation()
    yield tree
    for behaviour in tree.iterate():
        if isinstance(behaviour, HumanPointingAnnotator):
            behaviour.node.destroy_node()


def test_gpsr_tree_builds_a_real_subtree_for_every_ported_route(
    real_gpsr_tree: Behaviour, gpsr_routes: list[str]
) -> None:
    """
    Build every route with its real subtree instead of a success placeholder.
    """
    routes = scheduler_of(real_gpsr_tree).tasks
    assert sorted(routes) == sorted(gpsr_routes)
    assert [
        route for route, subtree in routes.items() if isinstance(subtree, Success)
    ] == []


# %% the routes of the tree


def test_gpsr_tree_holds_a_subtree_for_every_ported_route(
    scheduler: QueryBasedScheduler, gpsr_routes: list[str]
) -> None:
    """
    Register every expected route in the scheduler.
    """
    assert sorted(scheduler.tasks) == sorted(gpsr_routes)


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
        "in iai_kitchen",
        "waving in living_room",
        "sitting in bedroom",
        "pointing left in office",
        "in the hallway",
        "unknown location kitchen",
    ],
)
def test_human_query_takes_its_route(
    scheduler: QueryBasedScheduler, query: Query.Goal, route: str
) -> None:
    """
    Select the expected pipeline for each person query.
    """
    assert scheduler.filter_fn(query) == route


@pytest.mark.parametrize("furniture", [member.value for member in Furniture])
def test_human_near_known_furniture_takes_the_furniture_route(
    scheduler: QueryBasedScheduler, furniture: str
) -> None:
    """
    Select furniture detection for each supported furniture location.
    """
    assert (
        scheduler.filter_fn(gpsr_query("human", location=furniture))
        == GpsrRoute.HUMAN_NEAR_OBJECT.value
    )


# %% combined query precedence


@pytest.mark.parametrize(
    "query, route",
    [
        (
            gpsr_query("human", ["sitting", "waving"], location="living_room"),
            GpsrRoute.HUMAN_IN_ROOM.value,
        ),
        (
            gpsr_query("human", ["waving"], location="chair"),
            GpsrRoute.HUMAN_NEAR_OBJECT.value,
        ),
        (
            gpsr_query("human", ["waving"], color=["red"], description=["shirt"]),
            GpsrRoute.OPEN_VOCABULARY.value,
        ),
        (
            gpsr_query("object", ["largest"], location="living_room"),
            GpsrRoute.OBJECT_IN_ROOM.value,
        ),
        (
            gpsr_query("object", ["largest", "smallest"]),
            GpsrRoute.OBJECT_DETECTION.value,
        ),
    ],
    ids=[
        "two activities in living_room select room pipeline",
        "waving near chair selects furniture pipeline",
        "red shirt and waving select open vocabulary",
        "largest object in living_room selects room pipeline",
        "multiple object attributes select object detection",
    ],
)
def test_combined_query_preserves_route_precedence(
    scheduler: QueryBasedScheduler, query: Query.Goal, route: str
) -> None:
    """
    Select the existing route when a query contains multiple conditions.
    """
    assert scheduler.filter_fn(query) == route


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
    """
    Select object detection, room filtering, or no route for each query.
    """
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
    """
    Select recall or door detection, or reject a missing or unknown type.
    """
    assert scheduler.filter_fn(query) == route
